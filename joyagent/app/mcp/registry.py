"""
Phase 8 Step 2 — MCPRegistry

MCP Registry 是多个 MCP Server 连接的集中管理中心。

职责：
  1. 注册并连接多个 MCP Server（按 MCPServerConfig 列表）
  2. 统一发现：聚合所有 Server 的工具列表
  3. 统一执行：根据 "server__tool" 全名路由到正确的 Server
  4. 生命周期管理：连接/重连/关闭所有 Server
  5. 状态监控：每个 Server 的连接状态和工具数量

面试要点：
  Q: "你的 MCP 系统如何管理多个 MCP Server？"
  A: "我们有一个 MCPRegistry，它管理所有 MCPClient 的连接生命周期。
      每个 MCP Server 通过 MCPServerConfig 注册，Registry 负责启动子进程、
      发现工具、执行工具。工具通过 'server_name__tool_name' 前缀避免重名。
      失败时自动重试（指数退避，最多 3 次），支持单 Server 隔离 —
      一个 Server 挂了不影响其他。"

与 MCPClient 的关系：
  MCPClient   — 管理 1 个 MCP Server 的连接（Step 1）
  MCPRegistry — 管理 N 个 MCPClient 的集合（Step 2）
  每个 MCPClient 持有自己的子进程、自己的工具缓存
"""

from __future__ import annotations

import asyncio

from app.mcp.client import MCPClient
from app.mcp.schemas import (
    MCPConnectionState,
    MCPServerConfig,
    MCPTool,
    MCPToolResult,
)

# ── 重试配置 ──
_MAX_RETRIES = 3             # 连接失败最大重试次数
_RETRY_BASE_DELAY = 1.0      # 重试基础延迟（秒），指数退避


# ═══════════════════════════════════════════════════════════════════════
# MCPRegistry
# ═══════════════════════════════════════════════════════════════════════

class MCPRegistry:
    """
    管理多个 MCP Server 连接。

    每个 Server 的内置 MCPClient 实例独立管理子进程、连接状态
    和工具缓存。Registry 提供统一的发现/执行/监控接口。

    典型用法::

        registry = MCPRegistry()

        # 注册 Server 配置（不立即连接）
        registry.add_server(filesystem_config)
        registry.add_server(github_config)

        # 启动所有注册的 Server（连接 + 发现工具）
        await registry.start_all()

        # 获取所有工具
        all_tools = registry.get_all_tools()

        # 执行工具（by full name: "github__search_repositories"）
        result = await registry.execute("github__search_repositories", query="fastapi")

        # 关闭所有连接
        await registry.shutdown()

    也可以作为上下文管理器::

        async with MCPRegistry() as registry:
            registry.add_server(fs_config)
            await registry.start_all()
            result = await registry.execute("filesystem__read_file", path="/tmp/x.txt")
    """

    def __init__(self):
        # ── Client 实例：server_name → MCPClient ──
        self._clients: dict[str, MCPClient] = {}

        # ── 配置缓存：server_name → MCPServerConfig ──
        self._configs: dict[str, MCPServerConfig] = {}

        # ── 是否已调用 start_all ──
        self._started = False

        # ── 统计 ──
        self._total_executions: int = 0
        self._failed_executions: int = 0

    # ── 注册 ──────────────────────────────────────────────

    def add_server(self, config: MCPServerConfig) -> None:
        """
        注册一个 MCP Server 配置（不立即连接）。

        如果同名 Server 已注册，覆盖旧配置（旧的 MCPClient 不会自动断开）。

        Args:
            config: MCP Server 的连接配置
        """
        self._configs[config.name] = config

    def add_servers(self, configs: list[MCPServerConfig]) -> None:
        """批量注册 MCP Server 配置。"""
        for cfg in configs:
            self._configs[cfg.name] = cfg

    # ── 启动 / 停止 ───────────────────────────────────────

    async def start_all(self) -> dict[str, bool]:
        """
        连接所有已注册但未连接的 MCP Server + 发现工具。

        对每个 auto_connect=True 的 Server，创建 MCPClient、
        调用 connect()、缓存 client。

        连接失败的 Server 会重试（指数退避，最多 3 次）。
        一个 Server 失败不影响其他 Server 的启动。

        Returns:
            dict[str, bool]: server_name → 是否连接成功
        """
        results: dict[str, bool] = {}

        for name, config in self._configs.items():
            if not config.auto_connect:
                continue

            if name in self._clients and self._clients[name].is_connected:
                results[name] = True
                continue

            # 创建 client 并尝试连接（带重试）
            client = MCPClient(config)
            connected = await self._connect_with_retry(client, name)

            if connected:
                self._clients[name] = client
            else:
                # 保留 client 引用（即使连接失败），方便后续重连
                self._clients[name] = client

            results[name] = connected

        self._started = True

        total = len(results)
        ok = sum(1 for v in results.values() if v)
        if total > 0:
            print(f"  [mcp:registry] {ok}/{total} servers connected", flush=True)

        return results

    async def stop_all(self) -> None:
        """
        断开所有 MCP Server 连接。

        先发 disconnect 再清空引用。断开失败的 Server 不会被阻塞——
        用 asyncio.gather 并发关闭所有。
        """
        if not self._clients:
            return

        async def _safe_disconnect(name: str, client: MCPClient) -> None:
            try:
                await client.disconnect()
            except Exception as e:
                print(f"  [mcp:registry] Error disconnecting '{name}': {e}",
                      flush=True)

        tasks = [
            _safe_disconnect(name, client)
            for name, client in self._clients.items()
        ]
        await asyncio.gather(*tasks)

        self._clients.clear()
        self._started = False
        print(f"  [mcp:registry] All servers disconnected", flush=True)

    async def shutdown(self) -> None:
        """优雅关闭所有连接（stop_all 的别名，语义更明确）。"""
        await self.stop_all()

    # ── 工具发现 ──────────────────────────────────────────

    def get_all_tools(self) -> list[MCPTool]:
        """
        获取所有已连接 MCP Server 提供的工具列表。

        只返回已连接成功的 Server 的工具。已注册但未连接的 Server
        会被跳过（它们的 _clients value 可能是未连接的 client）。

        Returns:
            list[MCPTool]: 所有可用 MCP 工具
        """
        tools: list[MCPTool] = []
        for client in self._clients.values():
            if client.is_connected:
                tools.extend(client.tools)
        return tools

    def get_tools_by_server(self, server_name: str) -> list[MCPTool]:
        """
        获取指定 Server 的工具列表。

        Args:
            server_name: Server 名称

        Returns:
            list[MCPTool]: 该 Server 的工具列表（未连接/不存在则空列表）
        """
        client = self._clients.get(server_name)
        if client and client.is_connected:
            return list(client.tools)
        return []

    def get_all_anthropic_schemas(self) -> list[dict]:
        """
        获取所有 MCP 工具的 Anthropic 原生 tool schema 列表。

        可直接传给 self.client.messages.create(tools=...) 使用。
        每个工具经过 MCPTool.to_anthropic_schema() 转换，
        包含 server 名前缀和 MCP 标识。
        """
        schemas: list[dict] = []
        for client in self._clients.values():
            if client.is_connected:
                schemas.extend(client.get_anthropic_tool_schemas())
        return schemas

    # ── 工具执行 ──────────────────────────────────────────

    async def execute(
        self,
        tool_full_name: str,
        **kwargs,
    ) -> MCPToolResult:
        """
        执行 MCP 工具（by full name: "server_name__tool_name"）。

        解析全名 → 找到对应 MCPClient → 调用 execute_tool()。

        Args:
            tool_full_name: 完整的工具名（含 Server 前缀），
                           如 "github__search_repositories"
            **kwargs:       工具输入参数

        Returns:
            MCPToolResult: 执行结果
        """
        self._total_executions += 1

        # ── 解析 server_name 和 tool_name ──
        if "__" not in tool_full_name:
            result = MCPToolResult(
                tool_name=tool_full_name,
                server_name="",
                success=False,
                error=f"Invalid tool name format: '{tool_full_name}'. "
                      f"Expected 'server_name__tool_name'.",
            )
            self._failed_executions += 1
            return result

        server_name, tool_name = tool_full_name.split("__", 1)

        # ── 找到对应 client ──
        client = self._clients.get(server_name)
        if client is None:
            self._failed_executions += 1
            return MCPToolResult(
                tool_name=tool_name,
                server_name=server_name,
                success=False,
                error=f"Unknown MCP server: '{server_name}'. "
                      f"Available: {list(self._clients.keys())}",
            )

        if not client.is_connected:
            self._failed_executions += 1
            return MCPToolResult(
                tool_name=tool_name,
                server_name=server_name,
                success=False,
                error=f"MCP server '{server_name}' is not connected",
            )

        # ── 执行 ──
        result = await client.execute_tool(tool_name, **kwargs)
        if not result.success:
            self._failed_executions += 1
        return result

    def execute_sync(self, tool_full_name: str, **kwargs) -> MCPToolResult:
        """
        同步版本（非 async 上下文中一次性调用）。

        如果当前在事件循环中运行，用 asyncio.run_coroutine_threadsafe()
        安排到事件循环中执行。否则直接跑一个新的 event loop。

        注意：生产环境中尽量用 async 版本 `execute()`。
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 没有运行中的事件循环 → 创建新的
            return asyncio.run(self.execute(tool_full_name, **kwargs))

        # 有事件循环 → 提交并等待
        future = asyncio.run_coroutine_threadsafe(
            self.execute(tool_full_name, **kwargs), loop
        )
        return future.result(timeout=120.0)

    # ── 单个 Server 管理 ──────────────────────────────────

    async def connect_server(self, server_name: str) -> bool:
        """
        连接（或重连）指定的 Server。

        Args:
            server_name: 要连接的 Server 名称

        Returns:
            是否连接成功
        """
        config = self._configs.get(server_name)
        if config is None:
            print(f"  [mcp:registry] Unknown server: '{server_name}'", flush=True)
            return False

        # 如果已有连接 → 先断开
        old = self._clients.get(server_name)
        if old and old.is_connected:
            await old.disconnect()

        client = MCPClient(config)
        connected = await self._connect_with_retry(client, server_name)
        self._clients[server_name] = client
        return connected

    async def disconnect_server(self, server_name: str) -> None:
        """
        断开指定的 Server 连接。

        Args:
            server_name: 要断开的 Server 名称
        """
        client = self._clients.pop(server_name, None)
        if client:
            try:
                await client.disconnect()
            except Exception:
                pass

    # ── 状态监控 ──────────────────────────────────────────

    def get_all_states(self) -> list[MCPConnectionState]:
        """
        获取所有 Server 的当前状态（用于监控 API）。

        Returns:
            list[MCPConnectionState]: 每个已注册 Server 一个状态
        """
        states = []
        for name, client in self._clients.items():
            state = client.state
            # 确保 server_name 在 state 中
            if not state.server_name:
                state.server_name = name
            states.append(state)

        # 已注册但未实例化 client 的 Server（从未连接成功过）
        registered = set(self._configs.keys())
        clients = set(self._clients.keys())
        for name in registered - clients:
            states.append(MCPConnectionState(
                server_name=name,
                connected=False,
                error_message="Not started",
            ))

        return states

    @property
    def server_names(self) -> list[str]:
        """已注册的所有 Server 名称。"""
        return list(self._configs.keys())

    @property
    def connected_server_names(self) -> list[str]:
        """已连接成功的 Server 名称。"""
        return [n for n, c in self._clients.items() if c.is_connected]

    @property
    def stats(self) -> dict:
        """Registry 统计快照。"""
        return {
            "started": self._started,
            "servers_registered": len(self._configs),
            "servers_connected": len(self.connected_server_names),
            "total_tools": len(self.get_all_tools()),
            "total_executions": self._total_executions,
            "failed_executions": self._failed_executions,
            "server_names": self.connected_server_names,
        }

    # ── 上下文管理器 ──────────────────────────────────────

    async def __aenter__(self):
        """进入上下文 — 自动 start_all。"""
        await self.start_all()
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        """退出上下文 — 自动 shutdown。"""
        await self.shutdown()
        return False

    # ═══════════════════════════════════════════════════════════
    # 内部
    # ═══════════════════════════════════════════════════════════

    async def _connect_with_retry(
        self,
        client: MCPClient,
        name: str,
        max_retries: int = _MAX_RETRIES,
    ) -> bool:
        """
        带指数退避的连接重试。

        Args:
            client:      MCPClient 实例
            name:        Server 名称（日志用）
            max_retries: 最大重试次数

        Returns:
            是否连接成功
        """
        for attempt in range(1, max_retries + 1):
            try:
                await client.connect()
                return True
            except Exception as e:
                wait = _RETRY_BASE_DELAY * (2 ** (attempt - 1))  # 1, 2, 4s
                print(
                    f"  [mcp:registry] '{name}' connection failed "
                    f"(attempt {attempt}/{max_retries}): {type(e).__name__}: {e} — "
                    f"retrying in {wait}s",
                    flush=True,
                )
                if attempt < max_retries:
                    await asyncio.sleep(wait)

        print(f"  [mcp:registry] '{name}' FAILED after {max_retries} attempts",
              flush=True)
        return False


# ═══════════════════════════════════════════════════════════════════════
# 全局单例
# ═══════════════════════════════════════════════════════════════════════

mcp_registry = MCPRegistry()
