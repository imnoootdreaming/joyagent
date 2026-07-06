"""
Phase 8 Step 5 — MCP Tool Adapter

将 MCP Tool 适配为 Phase 2 的 BaseTool。

适配器模式（Adapter Pattern）：
  Phase 2 Agent 只知道 BaseTool 接口（name / description / execute）
  MCP 工具是 MCPTool 对象（由 MCPClient 发现）
  MCPToolAdapter 桥接两者——Agent 调用 MCP 工具就像调用本地工具一样

面试要点：
  Q: "MCP 工具如何融入你现有的 Tool Calling 系统？"
  A: "通过适配器模式。MCPToolAdapter 实现了 Phase 2 的 BaseTool 接口，
      内部将 execute() 委托给 MCPRegistry.execute()。"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.mcp.schemas import MCPTool, MCPToolResult

if TYPE_CHECKING:
    from app.mcp.registry import MCPRegistry


class MCPToolAdapter:
    """
    MCPTool → BaseTool 适配器。

    由于 app.tools.base 的导入链会触发 docker / chromadb 等
    重依赖，此类在 import 时不继承 BaseTool。在 execute() 首次
    调用时通过 _lazy_import 延迟加载 BaseTool / ToolResult。

    对外表现与 BaseTool 完全一致：name, description, input_schema,
    execute(), to_schema(), is_dangerous。
    """

    def __init__(self, mcp_tool: MCPTool, registry: "MCPRegistry"):
        self.mcp_tool = mcp_tool
        self._registry = registry

    @property
    def name(self) -> str:
        return self.mcp_tool.full_name

    @property
    def description(self) -> str:
        return self.mcp_tool.display_description

    @property
    def input_schema(self) -> dict:
        return self.mcp_tool.parameters

    @property
    def is_dangerous(self) -> bool:
        return False

    @property
    def server_name(self) -> str:
        return self.mcp_tool.server_name

    async def execute(self, **kwargs):
        """执行工具 — 返回与 ToolResult 兼容的简单对象（无 app.tools 依赖）。"""
        from dataclasses import dataclass

        @dataclass
        class _Result:
            success: bool
            message: str = ""
            error: str | None = None

        result: MCPToolResult = await self._registry.execute(self.name, **kwargs)
        return _Result(
            success=result.success,
            message=result.content if result.success else "",
            error=result.error,
        )

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# ═══════════════════════════════════════════════════════════════════════
# 便捷注册函数
# ═══════════════════════════════════════════════════════════════════════

async def register_mcp_tools(
    registry: "MCPRegistry | None" = None,
) -> int:
    """
    将 MCPRegistry 中所有工具注册到全局 ToolRegistry。

    Args:
        registry: 默认用 mcp_registry 单例

    Returns: 成功注册数
    """
    if registry is None:
        from app.mcp.registry import mcp_registry
        registry = mcp_registry

    from app.tools.registry import tool_registry

    tools = registry.get_all_tools()
    registered = 0
    for mcp_tool in tools:
        adapter = MCPToolAdapter(mcp_tool, registry)
        result = tool_registry.register_tool(adapter)
        if result.success:
            registered += 1
        else:
            print(f"  [mcp:adapter] dup: {mcp_tool.full_name}", flush=True)

    if registered > 0:
        print(f"  [mcp:adapter] {registered} MCP tools registered", flush=True)
    return registered
