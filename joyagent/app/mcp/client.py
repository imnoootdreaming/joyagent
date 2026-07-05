"""
Phase 8 Step 1 — MCPClient（纯 Python JSON-RPC over stdio 实现）

MCPClient 是 MCP (Model Context Protocol) 的 Client 端实现，
使用 asyncio 子进程 + JSON-RPC 2.0 over stdio 协议与
外部 MCP Server 进程通信。

核心职责：
  1. 启动 MCP Server 子进程（根据 MCPServerConfig）
  2. 发送 JSON-RPC 请求（initialize / tools/list / tools/call）
  3. 解析 JSON-RPC 响应
  4. 管理连接生命周期（connect → discover → execute → disconnect）

为什么自己实现而不是用 mcp SDK？
  - 不依赖第三方 mcp 包（离线环境、版本冲突）
  - 纯 Python + asyncio，代码 200 行，面试时能讲清楚每行
  - JSON-RPC 2.0 是 MCP 的传输层，自己实现加深理解
  - 更好的错误处理和超时控制（SDK 的默认行为不透明）

MCP 协议背景（面试必问）：
  MCP (Model Context Protocol) 是 Anthropic 提出的 LLM 与外部工具/数据源
  之间的标准协议。它基于 JSON-RPC 2.0，支持两种传输方式：
    - stdio（本实现使用）：Server 是子进程，通过 stdin/stdout 通信
    - Streamable HTTP（未来支持）：Server 是独立服务，通过 HTTP 通信

  协议流程：
    Client                         Server (子进程)
      │                                │
      │──── initialize ───────────────▶│  握手 + 获取 capabilities
      │◀─── {"serverInfo": ...} ───────│
      │                                │
      │──── tools/list ───────────────▶│  发现工具
      │◀─── {"tools": [...]} ──────────│
      │                                │
      │──── tools/call ───────────────▶│  执行工具
      │◀─── {"content": [...]} ────────│
      │                                │
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any

from app.mcp.schemas import (
    MCPConnectionState,
    MCPServerConfig,
    MCPTool,
    MCPToolResult,
)


# ═══════════════════════════════════════════════════════════════════════
# MCPClient
# ═══════════════════════════════════════════════════════════════════════

class MCPClient:
    """
    单个 MCP Server 的客户端 —— 管理与一个 MCP Server 的完整连接。

    生命周期：
      connect() → _discover_tools() → execute_tool() → disconnect()

    示例用法::

        config = MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        client = MCPClient(config)
        await client.connect()                        # 启动子进程 + 握手
        print(client.tools)                            # 发现的所有工具
        result = await client.execute_tool("read_file", path="/tmp/test.txt")
        await client.disconnect()
    """

    # ── 超时配置 ──────────────────────────────────────────
    _CONNECT_TIMEOUT = 15.0    # 子进程启动 + initialize 超时
    _RPC_TIMEOUT = 30.0        # 单个 RPC 调用超时
    _EXECUTE_TIMEOUT = 60.0    # 工具执行超时（可以更久）

    def __init__(self, config: MCPServerConfig):
        """
        Args:
            config: MCP Server 连接配置
        """
        self.config = config

        # ── 子进程引用 ──
        self._process: asyncio.subprocess.Process | None = None

        # ── 已发现的工具缓存 ──
        self.tools: list[MCPTool] = []

        # ── JSON-RPC request ID 计数器 ──
        self._request_id: int = 0

        # ── 连接状态 ──
        self._connected: bool = False
        self._last_error: str = ""

    # ── 公共 API ───────────────────────────────────────────

    async def connect(self) -> None:
        """
        建立与 MCP Server 的连接。

        步骤：
          1. 根据 config 启动 MCP Server 子进程
          2. 发送 initialize 请求（JSON-RPC 握手）
          3. 调用 tools/list 发现所有可用工具
          4. 缓存工具列表到 self.tools

        Raises:
            RuntimeError:  子进程启动失败
            TimeoutError:  连接/初始化超时
            ConnectionError: Server 返回错误响应
        """
        if self._connected:
            return

        # ── Step 1: 启动子进程 ──
        await self._start_process()

        # ── Step 2: initialize（握手） ──
        await self._initialize()

        # ── Step 3: 发现工具 ──
        await self._discover_tools()

        self._connected = True

        count = len(self.tools)
        names = [t.name for t in self.tools]
        print(f"  [mcp:{self.config.name}] Connected — {count} tools: {', '.join(names)}",
              flush=True)

    async def disconnect(self) -> None:
        """
        断开与 MCP Server 的连接。

        步骤：
          1. 关闭 stdin（通知 Server 结束输入）
          2. 等待子进程优雅退出（5s 超时）
          3. 超时则 SIGTERM → SIGKILL
          4. 清理管道和引用
        """
        if not self._connected or self._process is None:
            return

        self._connected = False
        self.tools.clear()

        try:
            # 关闭 stdin → Server 收到 EOF → 退出
            if self._process.stdin:
                self._process.stdin.close()
                await self._process.stdin.wait_closed()
        except Exception:
            pass

        try:
            # 等待子进程退出
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            # 优雅退出超时 → SIGTERM
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                # SIGTERM 无效 → SIGKILL
                try:
                    self._process.kill()
                    await self._process.wait()
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

        self._process = None
        print(f"  [mcp:{self.config.name}] Disconnected", flush=True)

    async def execute_tool(
        self,
        tool_name: str,
        **kwargs,
    ) -> MCPToolResult:
        """
        执行 MCP Server 上的一个工具。

        向 Server 发送 tools/call JSON-RPC 请求，等待并解析响应。

        Args:
            tool_name: 工具名称（不含 Server 前缀）
            **kwargs:  工具的输入参数

        Returns:
            MCPToolResult: 含 success / content / error 的结构化结果
        """
        if not self._connected:
            return MCPToolResult(
                tool_name=tool_name,
                server_name=self.config.name,
                success=False,
                error="Client not connected",
            )

        try:
            result = await self._call_rpc(
                method="tools/call",
                params={"name": tool_name, "arguments": kwargs},
                timeout=self._EXECUTE_TIMEOUT,
            )

            # MCP tools/call 响应格式：
            #   {"content": [{"type": "text", "text": "..."}]}
            #   {"content": [{"type": "resource", ...}]}
            #   {"isError": true, "content": [...]}

            is_error = result.get("isError", False)
            content_list = result.get("content", [])

            # 提取文本内容
            text_parts = []
            for item in content_list:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
                else:
                    text_parts.append(str(item))

            content = "\n".join(text_parts) if text_parts else str(result)

            return MCPToolResult(
                tool_name=tool_name,
                server_name=self.config.name,
                success=not is_error,
                content=content,
                error=content if is_error else None,
            )

        except Exception as e:
            return MCPToolResult(
                tool_name=tool_name,
                server_name=self.config.name,
                success=False,
                error=f"{type(e).__name__}: {e}",
            )

    # ── 连接状态 ──────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """是否已连上 MCP Server 且工具已发现。"""
        return self._connected and self._process is not None

    @property
    def state(self) -> MCPConnectionState:
        """当前连接状态（用于监控 API）。"""
        return MCPConnectionState(
            server_name=self.config.name,
            connected=self._connected,
            tool_count=len(self.tools),
            tool_names=[t.name for t in self.tools],
            error_message=self._last_error,
        )

    def get_anthropic_tool_schemas(self) -> list[dict]:
        """获取所有已发现工具的 Anthropic 原生 tool schema。"""
        return [t.to_anthropic_schema() for t in self.tools]

    # ═══════════════════════════════════════════════════════════
    # 内部：子进程管理
    # ═══════════════════════════════════════════════════════════

    async def _start_process(self) -> None:
        """
        启动 MCP Server 子进程。

        根据 config 的 command/args/env 创建 asyncio 子进程。
        stdin=PIPE, stdout=PIPE, stderr=PIPE（用于日志收集）。

        环境变量处理：
          - 继承当前进程的 os.environ
          - 合并 config.env（如果有）
        """
        # 构建环境变量
        env = os.environ.copy()
        if self.config.env:
            env.update(self.config.env)

        try:
            self._process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"MCP Server '{self.config.name}': "
                f"command not found: {self.config.command}"
            )
        except Exception as e:
            raise RuntimeError(
                f"MCP Server '{self.config.name}': "
                f"failed to start: {type(e).__name__}: {e}"
            )

        # 检查子进程是否立即退出（说明启动参数有误）
        await asyncio.sleep(0.3)
        if self._process.returncode is not None:
            stderr = ""
            if self._process.stderr:
                try:
                    stderr_bytes = await self._process.stderr.read()
                    stderr = stderr_bytes.decode("utf-8", errors="replace")
                except Exception:
                    pass
            raise RuntimeError(
                f"MCP Server '{self.config.name}' exited immediately "
                f"(code={self._process.returncode}): {stderr[:500]}"
            )

    # ═══════════════════════════════════════════════════════════
    # 内部：JSON-RPC 2.0
    # ═══════════════════════════════════════════════════════════

    async def _call_rpc(
        self,
        method: str,
        params: dict | None = None,
        timeout: float | None = None,
    ) -> Any:
        """
        发送 JSON-RPC 2.0 请求，等待并解析响应。

        JSON-RPC 2.0 请求格式：
          {"jsonrpc": "2.0", "id": 1, "method": "...", "params": {...}}

        JSON-RPC 2.0 响应格式（成功）：
          {"jsonrpc": "2.0", "id": 1, "result": {...}}

        JSON-RPC 2.0 响应格式（错误）：
          {"jsonrpc": "2.0", "id": 1, "error": {"code": -32600, "message": "..."}}

        Args:
            method:  RPC 方法名（如 "tools/list", "tools/call"）
            params:  方法参数
            timeout: 超时秒数（默认用 _RPC_TIMEOUT）

        Returns:
            result 字段的值（any JSON-serializable）

        Raises:
            RuntimeError:    Server 返回 JSON-RPC error
            TimeoutError:    RPC 调用超时
            ConnectionError: stdin/stdout 不可用
        """
        timeout = timeout or self._RPC_TIMEOUT

        if self._process is None or self._process.stdin is None:
            raise ConnectionError(f"MCP Server '{self.config.name}' not running")

        # ── 发送请求 ──
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }

        try:
            payload = json.dumps(request) + "\n"     # <-- 换行分隔
            self._process.stdin.write(payload.encode("utf-8"))
            await self._process.stdin.drain()
        except (BrokenPipeError, OSError) as e:
            self._connected = False
            raise ConnectionError(
                f"MCP Server '{self.config.name}': "
                f"write failed: {type(e).__name__}: {e}"
            )

        # ── 读取响应 ──
        try:
            line = await asyncio.wait_for(
                self._process.stdout.readline(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"MCP Server '{self.config.name}': "
                f"RPC '{method}' timed out ({timeout}s)"
            )

        if not line:
            # EOF → Server 崩溃
            self._connected = False
            raise ConnectionError(
                f"MCP Server '{self.config.name}' closed stdout "
                f"during RPC '{method}'"
            )

        # 收集 stderr 日志（异步后台任务，不阻塞 RPC）
        self._drain_stderr_background()

        # ── 解析响应 ──
        try:
            response = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"MCP Server '{self.config.name}': "
                f"invalid JSON response: {e}"
            )

        # ── 检查 JSON-RPC error ──
        if "error" in response:
            err = response["error"]
            err_msg = err.get("message", str(err))
            err_code = err.get("code", -1)
            raise RuntimeError(
                f"MCP Server '{self.config.name}' RPC error "
                f"(code={err_code}): {err_msg}"
            )

        # ── 返回 result ──
        return response.get("result", {})

    async def _initialize(self) -> None:
        """
        MCP initialize 握手。

        发送 initialize 请求，获取 Server 能力声明。
        Client 声明自己支持协议版本和能力。
        """
        try:
            result = await asyncio.wait_for(
                self._call_rpc(
                    method="initialize",
                    params={
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "joyagent",
                            "version": "0.7.0",
                        },
                    },
                    timeout=self._CONNECT_TIMEOUT,
                ),
                timeout=self._CONNECT_TIMEOUT,
            )

            server_info = result.get("serverInfo", {})
            server_name = server_info.get("name", self.config.name)
            server_version = server_info.get("version", "unknown")
            print(f"  [mcp:{self.config.name}] Handshake OK — "
                  f"{server_name} v{server_version}", flush=True)

        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"MCP Server '{self.config.name}': "
                f"initialize timeout ({self._CONNECT_TIMEOUT}s)"
            )

    async def _discover_tools(self) -> None:
        """
        通过 tools/list RPC 发现 MCP Server 的所有工具。

        将 MCP 协议返回的工具列表转换为 MCPTool 对象列表，
        缓存到 self.tools。
        """
        result = await self._call_rpc(
            method="tools/list",
            timeout=self._RPC_TIMEOUT,
        )

        raw_tools = result.get("tools", [])
        self.tools = [
            MCPTool.from_list_tools_result(t, self.config.name)
            for t in raw_tools
        ]

    def _drain_stderr_background(self) -> None:
        """
        后台收集 stderr（非阻塞）。

        MCP Server 的 stderr 用于自身日志输出，Client 不应阻塞等待。
        用 asyncio.create_task 后台读取，避免 stderr 缓冲区满导致
        Server 阻塞。
        """
        if self._process is None or self._process.stderr is None:
            return

        async def _drain():
            try:
                while True:
                    line = await asyncio.wait_for(
                        self._process.stderr.readline(),
                        timeout=0.5,
                    )
                    if not line:
                        break
                    # stderr 仅记录到 _last_error（前 500 字符）
                    try:
                        self._last_error = line.decode("utf-8", errors="replace")[:500]
                    except Exception:
                        pass
            except (asyncio.TimeoutError, Exception):
                pass

        asyncio.create_task(_drain())
