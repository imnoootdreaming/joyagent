"""
Phase 8 Step 4 — Demo MCP Server 测试

覆盖：
  - Demo Server 启动 + initialize 握手
  - tools/list: 返回 3 个工具（get_weather / calculate / get_time）
  - tools/call: get_weather（模拟天气）
  - tools/call: calculate（安全表达式求值）
  - tools/call: get_time（时间输出）
  - tools/call: 未知工具返回 isError
  - 安全求值器: safe_eval 正常 + 拒绝危险输入
  - MCPClient 集成: Client ↔ Demo Server 完整链路
  - DEMO_MCP_CONFIG: 配置正确性
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from app.mcp.client import MCPClient
from app.mcp.demo_server import DEMO_MCP_CONFIG
from app.mcp.demo_server.server import safe_eval, DEMO_TOOLS, DemoMCPServer


# ═══════════════════════════════════════════════════════════════════════
# 安全求值器测试（不依赖子进程）
# ═══════════════════════════════════════════════════════════════════════

class TestSafeEval:
    """safe_eval() 数学表达式安全求值器。"""

    def test_simple_arithmetic(self):
        assert safe_eval("2 + 3") == 5
        assert safe_eval("10 - 4") == 6
        assert safe_eval("6 * 7") == 42
        assert safe_eval("100 / 4") == 25

    def test_precedence(self):
        assert safe_eval("2 + 3 * 4") == 14
        assert safe_eval("(2 + 3) * 4") == 20

    def test_math_functions(self):
        assert safe_eval("sqrt(144)") == 12
        assert abs(safe_eval("sin(0)")) < 0.0001
        assert safe_eval("abs(-5)") == 5
        assert safe_eval("round(3.7)") == 4

    def test_constants(self):
        assert abs(safe_eval("pi") - 3.14159) < 0.001
        assert abs(safe_eval("e") - 2.71828) < 0.001

    # ── 安全拒绝 ──

    def test_rejects_import(self):
        # __import__ hits the '__' check before 'import'
        with pytest.raises(ValueError):
            safe_eval("__import__('os')")

    def test_rejects_system_call(self):
        with pytest.raises(ValueError, match="system"):
            safe_eval("os.system('ls')")

    def test_rejects_underscore_dunder(self):
        with pytest.raises(ValueError, match="__"):
            safe_eval("''.__class__")

    def test_rejects_eval_nested(self):
        with pytest.raises(ValueError, match="eval"):
            safe_eval("eval('1+1')")

    def test_rejects_statements(self):
        with pytest.raises((ValueError, SyntaxError)):
            safe_eval("x = 5")

    def test_rejects_open(self):
        with pytest.raises(ValueError, match="open"):
            safe_eval("open('/etc/passwd')")

    def test_rejects_lambda(self):
        with pytest.raises(ValueError, match="lambda"):
            safe_eval("(lambda: 1)()")


# ═══════════════════════════════════════════════════════════════════════
# 工具定义测试
# ═══════════════════════════════════════════════════════════════════════

class TestDemoTools:
    """DEMO_TOOLS 列表验证。"""

    def test_three_tools_defined(self):
        assert len(DEMO_TOOLS) == 3
        names = {t["name"] for t in DEMO_TOOLS}
        assert names == {"get_weather", "calculate", "get_time"}

    def test_each_tool_has_required_fields(self):
        for tool in DEMO_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    def test_get_weather_requires_city(self):
        weather = [t for t in DEMO_TOOLS if t["name"] == "get_weather"][0]
        assert "city" in weather["inputSchema"]["required"]

    def test_calculate_requires_expression(self):
        calc = [t for t in DEMO_TOOLS if t["name"] == "calculate"][0]
        assert "expression" in calc["inputSchema"]["required"]


# ═══════════════════════════════════════════════════════════════════════
# DEMO_MCP_CONFIG 测试
# ═══════════════════════════════════════════════════════════════════════

class TestDemoConfig:
    """DEMO_MCP_CONFIG 配置验证。"""

    def test_config_name(self):
        assert DEMO_MCP_CONFIG.name == "joyagent-demo"

    def test_config_command_is_python(self):
        assert "python" in DEMO_MCP_CONFIG.command

    def test_config_auto_connect_false(self):
        """Demo server 默认不自动连接。"""
        assert DEMO_MCP_CONFIG.auto_connect is False


# ═══════════════════════════════════════════════════════════════════════
# MCPClient ↔ Demo Server 集成测试
# ═══════════════════════════════════════════════════════════════════════

class TestDemoServerIntegration:
    """用 MCPClient 连接 Demo Server，验证完整协议栈。"""

    @pytest.mark.asyncio
    async def test_connect_and_discover(self):
        """Client connect → 发现 3 个工具。"""
        client = MCPClient(DEMO_MCP_CONFIG)
        await client.connect()

        assert client.is_connected is True
        assert len(client.tools) == 3
        names = {t.name for t in client.tools}
        assert names == {"get_weather", "calculate", "get_time"}

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_execute_get_weather(self):
        """调用 get_weather → 返回天气数据。"""
        client = MCPClient(DEMO_MCP_CONFIG)
        await client.connect()

        result = await client.execute_tool("get_weather", city="Tokyo")
        assert result.success is True
        assert "Tokyo" in result.content
        assert "Temperature" in result.content

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_execute_calculate(self):
        """调用 calculate → 返回计算结果。"""
        client = MCPClient(DEMO_MCP_CONFIG)
        await client.connect()

        result = await client.execute_tool("calculate", expression="2 + 3 * 4")
        assert result.success is True
        assert "14" in result.content

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_execute_get_time_iso(self):
        """调用 get_time (iso) → 返回 ISO 格式时间。"""
        client = MCPClient(DEMO_MCP_CONFIG)
        await client.connect()

        result = await client.execute_tool("get_time", format="iso")
        assert result.success is True
        assert "T" in result.content  # ISO 8601 含 T
        assert len(result.content) > 10

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_execute_get_time_human(self):
        """调用 get_time (human) → 返回可读时间。"""
        client = MCPClient(DEMO_MCP_CONFIG)
        await client.connect()

        result = await client.execute_tool("get_time", format="human")
        assert result.success is True
        assert "current time" in result.content.lower()

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """调用不存在的工具 → isError=True。"""
        client = MCPClient(DEMO_MCP_CONFIG)
        await client.connect()

        result = await client.execute_tool("nonexistent_tool")
        assert result.success is False
        assert "Unknown" in result.error

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_multiple_calls_same_connection(self):
        """同一连接多次调用不同工具。"""
        client = MCPClient(DEMO_MCP_CONFIG)
        await client.connect()

        # 三次调用
        r1 = await client.execute_tool("get_weather", city="Paris")
        r2 = await client.execute_tool("calculate", expression="10**2")
        r3 = await client.execute_tool("get_time", format="unix")

        assert r1.success
        assert r2.success
        assert r3.success
        assert "Paris" in r1.content
        assert "100" in r2.content

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_and_reconnect(self):
        """断开后重新连接 → 仍可用。"""
        client = MCPClient(DEMO_MCP_CONFIG)
        await client.connect()
        assert client.is_connected
        await client.disconnect()
        assert not client.is_connected

        # 重新连接
        await client.connect()
        assert client.is_connected
        assert len(client.tools) == 3

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_to_anthropic_schema(self):
        """工具可转为 Anthropic schema（含前缀）。"""
        client = MCPClient(DEMO_MCP_CONFIG)
        await client.connect()

        schemas = client.get_anthropic_tool_schemas()
        assert len(schemas) == 3
        for s in schemas:
            assert s["name"].startswith("joyagent-demo__")
            assert "[MCP:joyagent-demo]" in s["description"]
            assert "input_schema" in s

        await client.disconnect()
