"""
Phase 8 Step 5 — MCPToolAdapter 单元测试 + register_mcp_tools 集成测试

覆盖：
  - MCPToolAdapter: name/description/input_schema/is_dangerous/server_name/to_schema
  - register_mcp_tools: Demo Server → Registry → ToolRegistry 注册
  - 端到端: Demo Server → Registry → Adapter.execute()
  - 多 Server 前缀无冲突
  - Server 断开后调用返回错误
"""

from __future__ import annotations

import pytest

from app.mcp.registry import MCPRegistry
from app.mcp.adapter import MCPToolAdapter, register_mcp_tools
from app.mcp.schemas import MCPTool


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

_SEARCH_TOOL = MCPTool(
    name="search_repos", description="Search GitHub repositories",
    parameters={"type": "object", "properties": {"query": {"type": "string"}},
                "required": ["query"]},
    server_name="github",
)


def _demo_config():
    from app.mcp.demo_server import DEMO_MCP_CONFIG
    return DEMO_MCP_CONFIG


# ═══════════════════════════════════════════════════════════════════════
# MCPToolAdapter 单元测试
# ═══════════════════════════════════════════════════════════════════════

class TestMCPToolAdapter:
    """MCPToolAdapter — MCPTool → BaseTool 适配器。"""

    def test_has_all_base_tool_attrs(self):
        adapter = MCPToolAdapter(_SEARCH_TOOL, MCPRegistry())
        for attr in ("name", "description", "input_schema",
                     "is_dangerous", "execute", "to_schema"):
            assert hasattr(adapter, attr), f"Missing: {attr}"

    def test_name_prefix(self):
        assert MCPToolAdapter(_SEARCH_TOOL, MCPRegistry()).name == "github__search_repos"

    def test_description_tag(self):
        desc = MCPToolAdapter(_SEARCH_TOOL, MCPRegistry()).description
        assert "[MCP:github]" in desc
        assert "Search GitHub" in desc

    def test_input_schema(self):
        schema = MCPToolAdapter(_SEARCH_TOOL, MCPRegistry()).input_schema
        assert schema["required"] == ["query"]

    def test_is_dangerous(self):
        assert MCPToolAdapter(_SEARCH_TOOL, MCPRegistry()).is_dangerous is False

    def test_server_name(self):
        assert MCPToolAdapter(_SEARCH_TOOL, MCPRegistry()).server_name == "github"

    def test_to_schema(self):
        s = MCPToolAdapter(_SEARCH_TOOL, MCPRegistry()).to_schema()
        assert s["name"] == "github__search_repos"
        assert "input_schema" in s
        assert s["input_schema"]["required"] == ["query"]

    def test_two_servers_no_conflict(self):
        a = MCPToolAdapter(MCPTool(name="read", description="A",
                                   server_name="srv_a"), MCPRegistry())
        b = MCPToolAdapter(MCPTool(name="read", description="B",
                                   server_name="srv_b"), MCPRegistry())
        assert a.name == "srv_a__read"
        assert b.name == "srv_b__read"
        assert a.name != b.name


# ═══════════════════════════════════════════════════════════════════════
# register_mcp_tools + Demo Server 集成测试
# ═══════════════════════════════════════════════════════════════════════

class TestRegisterMcpToolsE2E:
    """register_mcp_tools() — Demo Server 端到端集成。"""

    async def _setup_reg_with_demo(self):
        """Helper: create registry with demo server connected."""
        reg = MCPRegistry()
        reg.add_server(_demo_config())
        ok = await reg.connect_server("joyagent-demo")  # auto_connect=False, manual
        assert ok, "Demo server failed to connect"
        return reg

    @pytest.mark.asyncio
    async def test_adapter_execute_calculate(self):
        reg = await self._setup_reg_with_demo()
        tool = MCPTool(name="calculate", description="Calc",
                       server_name="joyagent-demo")
        result = await MCPToolAdapter(tool, reg).execute(expression="10*5+7")
        assert result.success is True
        assert "57" in result.message
        await reg.shutdown()

    @pytest.mark.asyncio
    async def test_adapter_execute_weather(self):
        reg = await self._setup_reg_with_demo()
        tool = MCPTool(name="get_weather", description="Weather",
                       server_name="joyagent-demo")
        result = await MCPToolAdapter(tool, reg).execute(city="Tokyo")
        assert result.success is True
        assert "Tokyo" in result.message
        await reg.shutdown()

    @pytest.mark.asyncio
    async def test_adapter_execute_get_time(self):
        reg = await self._setup_reg_with_demo()
        tool = MCPTool(name="get_time", description="Time",
                       server_name="joyagent-demo")
        result = await MCPToolAdapter(tool, reg).execute(format="iso")
        assert result.success is True
        assert "T" in result.message
        await reg.shutdown()

    @pytest.mark.asyncio
    async def test_adapter_error_on_disconnected(self):
        reg = await self._setup_reg_with_demo()
        await reg.shutdown()
        tool = MCPTool(name="calculate", description="Calc",
                       server_name="joyagent-demo")
        result = await MCPToolAdapter(tool, reg).execute(expression="1+1")
        assert result.success is False
        assert result.error is not None
