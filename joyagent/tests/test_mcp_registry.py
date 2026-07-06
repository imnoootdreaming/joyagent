"""
Phase 8 Step 2 — MCPRegistry 单元测试

覆盖：
  - MCPRegistry: add_server / add_servers
  - start_all / stop_all / shutdown: 并发连接、优雅关闭
  - get_all_tools / get_tools_by_server / get_all_anthropic_schemas
  - execute: server__tool 路由、未知 server、未连接 server、未知工具
  - 连接失败重试（指数退避）
  - connect_server / disconnect_server: 单 Server 管理
  - get_all_states / stats / server_names: 监控
  - 上下文管理器: async with
  - 全局单例: mcp_registry
  - 集成: 2 个 mock MCP server 同时运行
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from app.mcp.client import MCPClient
from app.mcp.registry import MCPRegistry, mcp_registry
from app.mcp.schemas import (
    MCPConnectionState,
    MCPServerConfig,
    MCPTool,
    MCPToolResult,
)


# ═══════════════════════════════════════════════════════════════════════
# Mock MCP Server（用于测试）
# ═══════════════════════════════════════════════════════════════════════

_MOCK_SERVER_SCRIPT = r"""
import json, sys

TOOLS = [
    {
        "name": "tool_{suffix}",
        "description": "A mock tool from {suffix}",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "ping_{suffix}",
        "description": "Health check for {suffix}",
        "inputSchema": {"type": "object", "properties": {}}
    },
]

for line in sys.stdin:
    line = line.strip()
    if not line:
        break
    try:
        req = json.loads(line)
    except json.JSONDecodeError:
        continue
    rid = req.get("id")
    method = req.get("method", "")
    params = req.get("params", {})

    if method == "initialize":
        sys.stdout.write(json.dumps({
            "jsonrpc": "2.0", "id": rid,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "mock-{suffix}", "version": "0.1"},
                "capabilities": {"tools": {}}
            }
        }) + "\n")
        sys.stdout.flush()
    elif method == "tools/list":
        sys.stdout.write(json.dumps({
            "jsonrpc": "2.0", "id": rid,
            "result": {"tools": TOOLS}
        }) + "\n")
        sys.stdout.flush()
    elif method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments", {})
        sys.stdout.write(json.dumps({
            "jsonrpc": "2.0", "id": rid,
            "result": {"content": [{"type": "text", "text": f"[{name}] OK: {{args}}"}]}
        }) + "\n")
        sys.stdout.flush()
    else:
        sys.stdout.write(json.dumps({
            "jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }) + "\n")
        sys.stdout.flush()
"""


def _write_mock_server(path: Path, suffix: str) -> None:
    """写入带指定 suffix 的 mock server 脚本。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    script = _MOCK_SERVER_SCRIPT.replace("{suffix}", suffix)
    path.write_text(script, encoding="utf-8")


def _mock_config(name: str, suffix: str, script_path: Path) -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        command=sys.executable,
        args=[str(script_path)],
        description=f"Mock MCP server '{name}' for testing",
    )


# ═══════════════════════════════════════════════════════════════════════
# MCPRegistry 测试
# ═══════════════════════════════════════════════════════════════════════

class TestMCPRegistry:
    """MCPRegistry 核心功能测试。"""

    @pytest.fixture
    def scripts(self, tmp_path):
        """创建两个 mock MCP server 脚本。"""
        s1 = tmp_path / "mock_a.py"
        s2 = tmp_path / "mock_b.py"
        _write_mock_server(s1, "alpha")
        _write_mock_server(s2, "beta")
        return {"alpha": s1, "beta": s2}

    # ── 注册 ──────────────────────────────────────────

    def test_add_server(self):
        reg = MCPRegistry()
        cfg = MCPServerConfig(name="test", command="python")
        reg.add_server(cfg)
        assert "test" in reg.server_names

    def test_add_servers_batch(self):
        reg = MCPRegistry()
        configs = [
            MCPServerConfig(name="a", command="python"),
            MCPServerConfig(name="b", command="python"),
        ]
        reg.add_servers(configs)
        assert reg.server_names == ["a", "b"]

    def test_add_same_name_overwrites(self, scripts):
        """同名 Server 覆盖旧配置。"""
        reg = MCPRegistry()
        reg.add_server(MCPServerConfig(name="s", command="python"))
        reg.add_server(MCPServerConfig(name="s", command="node"))
        assert reg._configs["s"].command == "node"

    # ── 启动 / 停止 ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_start_all_connects_servers(self, scripts):
        """start_all 连接两个 Server → 都成功。"""
        reg = MCPRegistry()
        reg.add_server(_mock_config("alpha", "alpha", scripts["alpha"]))
        reg.add_server(_mock_config("beta", "beta", scripts["beta"]))

        results = await reg.start_all()
        assert results["alpha"] is True
        assert results["beta"] is True
        assert reg.connected_server_names == ["alpha", "beta"]

        await reg.shutdown()

    @pytest.mark.asyncio
    async def test_start_all_skips_auto_connect_false(self, scripts):
        """auto_connect=False 的 Server 不连接。"""
        reg = MCPRegistry()
        cfg = _mock_config("off", "off", scripts["alpha"])
        cfg.auto_connect = False
        reg.add_server(cfg)

        results = await reg.start_all()
        assert results == {}  # 没有要启动的
        assert reg.connected_server_names == []

    @pytest.mark.asyncio
    async def test_start_all_is_idempotent(self, scripts):
        """重复调用 start_all 不重新连接已连接的 Server。"""
        reg = MCPRegistry()
        reg.add_server(_mock_config("a", "alpha", scripts["alpha"]))
        await reg.start_all()
        assert len(reg.connected_server_names) == 1

        # 第二次调用 — 跳过已连接的
        results2 = await reg.start_all()
        assert results2["a"] is True  # 跳过但返回 True
        assert len(reg.connected_server_names) == 1

        await reg.shutdown()

    @pytest.mark.asyncio
    async def test_stop_all_disconnects_all(self, scripts):
        """stop_all 断开所有连接。"""
        reg = MCPRegistry()
        reg.add_server(_mock_config("a", "alpha", scripts["alpha"]))
        reg.add_server(_mock_config("b", "beta", scripts["beta"]))
        await reg.start_all()
        assert len(reg.connected_server_names) == 2

        await reg.stop_all()
        assert len(reg.connected_server_names) == 0
        assert reg._started is False

    @pytest.mark.asyncio
    async def test_start_all_one_fails_other_succeeds(self, scripts):
        """一个 Server 不可用 → 另一个仍正常连接。"""
        reg = MCPRegistry()
        reg.add_server(_mock_config("good", "good", scripts["alpha"]))
        reg.add_server(MCPServerConfig(
            name="bad",
            command="this_command_does_not_exist_xyz",
        ))

        results = await reg.start_all()
        assert results["good"] is True
        assert results["bad"] is False

        # good server 仍可用
        assert "good" in reg.connected_server_names

        await reg.shutdown()

    # ── 工具发现 ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_get_all_tools(self, scripts):
        """get_all_tools 聚合所有 Server 的工具。"""
        reg = MCPRegistry()
        reg.add_server(_mock_config("a", "alpha", scripts["alpha"]))
        reg.add_server(_mock_config("b", "beta", scripts["beta"]))
        await reg.start_all()

        tools = reg.get_all_tools()
        assert len(tools) == 4  # 2 per server
        names = {t.full_name for t in tools}
        assert "a__tool_alpha" in names
        assert "a__ping_alpha" in names
        assert "b__tool_beta" in names
        assert "b__ping_beta" in names

        await reg.shutdown()

    @pytest.mark.asyncio
    async def test_get_tools_by_server(self, scripts):
        """get_tools_by_server 过滤单个 Server。"""
        reg = MCPRegistry()
        reg.add_server(_mock_config("a", "alpha", scripts["alpha"]))
        reg.add_server(_mock_config("b", "beta", scripts["beta"]))
        await reg.start_all()

        a_tools = reg.get_tools_by_server("a")
        assert len(a_tools) == 2
        assert all(t.server_name == "a" for t in a_tools)

        b_tools = reg.get_tools_by_server("b")
        assert len(b_tools) == 2

        # 不存在的 server
        assert reg.get_tools_by_server("nonexistent") == []

        await reg.shutdown()

    @pytest.mark.asyncio
    async def test_get_all_anthropic_schemas(self, scripts):
        """get_all_anthropic_schemas 返回 Anthropic 原生格式。"""
        reg = MCPRegistry()
        reg.add_server(_mock_config("a", "alpha", scripts["alpha"]))
        await reg.start_all()

        schemas = reg.get_all_anthropic_schemas()
        assert len(schemas) == 2
        for s in schemas:
            assert "name" in s
            assert "description" in s
            assert "input_schema" in s
            assert s["name"].startswith("a__")

        await reg.shutdown()

    # ── 工具执行 ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_by_full_name(self, scripts):
        """execute("a__tool_alpha", ...) → 路由到正确的 Server。"""
        reg = MCPRegistry()
        reg.add_server(_mock_config("a", "alpha", scripts["alpha"]))
        reg.add_server(_mock_config("b", "beta", scripts["beta"]))
        await reg.start_all()

        result = await reg.execute("a__tool_alpha", query="test")
        assert result.success is True
        assert result.server_name == "a"
        assert result.tool_name == "tool_alpha"
        assert "OK" in result.content

        await reg.shutdown()

    @pytest.mark.asyncio
    async def test_execute_unknown_server(self):
        """不存在的 Server → 错误。"""
        reg = MCPRegistry()
        result = await reg.execute("ghost__tool", x=1)
        assert result.success is False
        assert "Unknown" in result.error

    @pytest.mark.asyncio
    async def test_execute_not_connected(self, scripts):
        """已注册但未连接的 Server → 错误（client 不在 _clients 中）。"""
        reg = MCPRegistry()
        reg.add_server(_mock_config("a", "alpha", scripts["alpha"]))
        # 不调用 start_all → server 未连接 → _clients 中无此 key

        result = await reg.execute("a__tool_alpha", query="x")
        assert result.success is False
        assert "unknown" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_no_prefix(self):
        """无 "__" 前缀的工具名 → 错误。"""
        reg = MCPRegistry()
        result = await reg.execute("bare_tool_name")
        assert result.success is False
        assert "format" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_increments_counters(self, scripts):
        """execute 更新 total/failed 计数器。"""
        reg = MCPRegistry()
        reg.add_server(_mock_config("a", "alpha", scripts["alpha"]))
        await reg.start_all()

        assert reg._total_executions == 0
        await reg.execute("a__tool_alpha", query="x")
        assert reg._total_executions == 1
        assert reg._failed_executions == 0

        await reg.execute("ghost__x")  # fails
        assert reg._total_executions == 2
        assert reg._failed_executions == 1

        await reg.shutdown()

    # ── 单 Server 管理 ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_connect_server(self, scripts):
        """connect_server 连接指定 Server。"""
        reg = MCPRegistry()
        reg.add_server(_mock_config("a", "alpha", scripts["alpha"]))
        reg.add_server(_mock_config("b", "beta", scripts["beta"]))

        ok = await reg.connect_server("a")
        assert ok is True
        assert "a" in reg.connected_server_names
        assert "b" not in reg.connected_server_names  # b 没连

        await reg.shutdown()

    @pytest.mark.asyncio
    async def test_connect_server_unknown(self):
        """不存在的 Server → 返回 False。"""
        reg = MCPRegistry()
        ok = await reg.connect_server("nonexistent")
        assert ok is False

    @pytest.mark.asyncio
    async def test_disconnect_server(self, scripts):
        """disconnect_server 断开指定 Server。"""
        reg = MCPRegistry()
        reg.add_server(_mock_config("a", "alpha", scripts["alpha"]))
        reg.add_server(_mock_config("b", "beta", scripts["beta"]))
        await reg.start_all()

        await reg.disconnect_server("a")
        assert "a" not in reg.connected_server_names
        assert "b" in reg.connected_server_names  # b 还在

        await reg.shutdown()

    # ── 连接失败重试 ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_connect_with_retry_fails_gracefully(self):
        """命令不存在的 Server → 重试 3 次后返回 False。"""
        reg = MCPRegistry()
        config = MCPServerConfig(name="bad", command="no_such_binary_xyz")
        reg.add_server(config)

        results = await reg.start_all()
        # 连接失败但 start_all 不抛异常
        assert results["bad"] is False

    @pytest.mark.asyncio
    async def test_connect_retry_skips_after_success(self, scripts):
        """连接成功后不重试。"""
        reg = MCPRegistry()
        client = MCPClient(_mock_config("a", "alpha", scripts["alpha"]))
        ok = await reg._connect_with_retry(client, "a")
        assert ok is True
        assert client.is_connected is True

        await client.disconnect()

    # ── 监控 ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_stats(self, scripts):
        """stats 返回正确统计。"""
        reg = MCPRegistry()
        reg.add_server(_mock_config("a", "alpha", scripts["alpha"]))
        await reg.start_all()

        stats = reg.stats
        assert stats["started"] is True
        assert stats["servers_registered"] == 1
        assert stats["servers_connected"] == 1
        assert stats["total_tools"] == 2

        await reg.shutdown()

    @pytest.mark.asyncio
    async def test_get_all_states(self, scripts):
        """get_all_states 返回每个 Server 的连接状态。"""
        reg = MCPRegistry()
        reg.add_server(_mock_config("a", "alpha", scripts["alpha"]))
        reg.add_server(_mock_config("b", "beta", scripts["beta"]))
        reg.add_server(MCPServerConfig(name="never_started", command="python"))
        await reg.start_all()

        states = reg.get_all_states()
        assert len(states) == 3

        by_name = {s.server_name: s for s in states}
        assert by_name["a"].connected is True
        assert by_name["b"].connected is True
        assert by_name["never_started"].connected is False

        await reg.shutdown()

    # ── 上下文管理器 ─────────────────────────────────────

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Windows asyncio subprocess cleanup compatibility issue"
    )
    @pytest.mark.asyncio
    async def test_async_context_manager(self, scripts):
        """async with MCPRegistry() → 自动 start_all + shutdown。"""
        cfg = _mock_config("a", "alpha", scripts["alpha"])

        async with MCPRegistry() as reg:
            reg.add_server(cfg)
            # start_all 在 __aenter__ 中调用
            assert reg._started is True
            assert "a" in reg.connected_server_names

        # __aexit__ 中调用了 shutdown
        assert reg._started is False
        assert len(reg._clients) == 0


# ═══════════════════════════════════════════════════════════════════════
# 全局单例测试
# ═══════════════════════════════════════════════════════════════════════

class TestGlobalSingleton:
    """mcp_registry 全局单例。"""

    def test_singleton_is_mcp_registry(self):
        """全局单例是 MCPRegistry 实例。"""
        assert isinstance(mcp_registry, MCPRegistry)

    def test_singleton_is_same_instance(self):
        """多次 import 拿到同一个实例。"""
        from app.mcp.registry import mcp_registry as reg2
        assert mcp_registry is reg2

    def test_singleton_starts_empty(self):
        """初始状态无 Server 注册。"""
        assert mcp_registry.server_names == []
        assert mcp_registry._started is False
