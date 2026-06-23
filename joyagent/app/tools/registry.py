from app.tools.base import BaseTool, ToolResult

class ToolRegistry:
    """ 
    ToolRegistry is a singleton class
    Using this class to register and retrieve tools.
    """

    def __init__(self):
        self._tools : dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool) -> ToolResult:
        """Register a tool in the registry."""
        if tool.name in self._tools:
            return ToolResult(success=False, message="", error=f"Tool: '{tool.name}' is already registered.")
        self._tools[tool.name] = tool
        return ToolResult(success=True, message=f"Tool: '{tool.name}' registered successfully.", error=None)

    def get_tool(self, tool_name: str) -> BaseTool:
        """Retrieve a tool from the registry by name."""
        return self._tools.get(tool_name, None)
    
    def list_tools(self) -> list[str]:
        """List all registered tools' names."""
        return list(self._tools.keys())
    
    def get_tool_schemas(self) -> list[dict]:
        """Get the schemas of all registered tools."""
        return [tool.to_schema() for tool in self._tools.values()]
    
    def get_dangerous_tools(self) -> list[str]:
        """Get the names of all registered tools that are marked as dangerous."""
        return [tool.name for tool in self._tools.values() if tool.is_dangerous]
    
    async def execute(self, name: str, **kwargs) -> ToolResult:
        """根据名称执行工具"""
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(success=False, message="", error=f"Unknown tool: {name}")
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(success=False, message="", error=str(e))

# Create a singleton instance of ToolRegistry
tool_registry = ToolRegistry()