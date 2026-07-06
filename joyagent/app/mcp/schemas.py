"""
Phase 8 Step 1 — MCP 数据模型

定义 MCP Client 的核心数据结构：
  - MCPServerConfig: MCP Server 连接配置（命令、参数、环境变量）
  - MCPTool:         MCP Server 提供的工具（发现后缓存）
  - MCPToolResult:   工具执行结果

这些模型与 MCP JSON-RPC 协议一一对应：
  list_tools 响应 → list[MCPTool]
  call_tool  响应 → MCPToolResult

工具命名：为防止不同 MCP Server 的工具重名，使用前缀分隔符：
  server_name + "__" + tool_name  例如 "github__search_repositories"
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════
# MCP Server 连接配置
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MCPServerConfig:
    """
    MCP Server 的连接配置。

    描述如何启动一个 MCP Server 进程，包括它的命令行参数和
    所需环境变量。MCPClient 根据此配置启动子进程并通过
    stdio 进行 JSON-RPC 通信。

    Attributes:
        name:         逻辑名称（如 "github", "postgres"），用作工具前缀
        command:      启动命令（如 "npx", "python", "node"）
        args:         命令参数列表
        env:          环境变量 dict（如 GITHUB_TOKEN），会注入到子进程
        auto_connect: Agent 启动时是否自动连接此 Server
        description:  可读描述（日志用）

    Example:
        GITHUB_MCP_CONFIG = MCPServerConfig(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN")},
        )
    """
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    auto_connect: bool = True
    description: str = ""


# ═══════════════════════════════════════════════════════════════════════
# MCP 工具定义
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MCPTool:
    """
    MCP Server 提供的工具 —— 通过 list_tools RPC 发现。

    每个 MCPTool 对应 MCP Server 暴露的一个能力。Client 在
    connect() 后调用 list_tools 获取所有可用工具，缓存在
    MCPClient.tools 列表中。

    Attributes:
        name:        工具名称（MCP Server 定义的原始名称）
        description: 工具描述（含 Server 名前缀后注入 LLM context）
        parameters:  JSON Schema dict（工具的输入参数定义）
        server_name: 所属 MCP Server 的逻辑名称

    to_schema() 将 MCPTool 转为 Anthropic Messages API 原生工具格式，
    包括加前缀防冲突（如 github__search_repositories）。
    """

    name: str
    description: str
    parameters: dict = field(default_factory=dict)
    server_name: str = ""

    @property
    def full_name(self) -> str:
        """带前缀的完整工具名（防冲突）。"""
        if self.server_name:
            return f"{self.server_name}__{self.name}"
        return self.name

    @property
    def display_description(self) -> str:
        """LLM 可见的工具描述（含 Server 名标识）。"""
        if self.server_name:
            return f"[MCP:{self.server_name}] {self.description}"
        return self.description

    def to_anthropic_schema(self) -> dict:
        """
        转为 Anthropic Messages API 原生 tool schema。

        格式与 app.tools.base.BaseTool.to_schema() 保持一致，
        确保 MCP 工具和本地工具对 LLM 无感知差异。

        Returns:
            dict: {"name": "...", "description": "...", "input_schema": {...}}
        """
        return {
            "name": self.full_name,
            "description": self.display_description,
            "input_schema": self.parameters,
        }

    @classmethod
    def from_list_tools_result(
        cls,
        tool_data: dict,
        server_name: str,
    ) -> MCPTool:
        """
        从 MCP list_tools 响应中的单个工具条目构建 MCPTool。

        MCP 协议的 list_tools 返回格式：
          {"name": "search_repositories",
           "description": "Search GitHub repos",
           "inputSchema": {"type": "object", ...}}

        注意：MCP 协议用 "inputSchema"，Anthropic 用 "input_schema"，
        此处归一化为 "parameters"。

        Args:
            tool_data:   MCP 协议返回的单个 tool 对象
            server_name: 所属 Server 的名称

        Returns:
            MCPTool 实例
        """
        return cls(
            name=tool_data.get("name", ""),
            description=tool_data.get("description", ""),
            parameters=tool_data.get("inputSchema", tool_data.get("input_schema", {})),
            server_name=server_name,
        )


# ═══════════════════════════════════════════════════════════════════════
# MCP 工具执行结果
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MCPToolResult:
    """
    MCP 工具执行结果 —— 通过 call_tool RPC 获得。

    Attributes:
        tool_name:   工具名称（不含前缀）
        server_name: 所属 Server 名称
        success:     是否执行成功
        content:     工具返回的文本内容（成功时）
        error:       错误描述（失败时）

    Example:
        result = MCPToolResult(
            tool_name="search_repositories",
            server_name="github",
            success=True,
            content="Found 5 repos: ...",
        )
    """
    tool_name: str
    server_name: str
    success: bool
    content: str = ""
    error: str | None = None

    @property
    def full_name(self) -> str:
        """带前缀的完整工具名。"""
        if self.server_name:
            return f"{self.server_name}__{self.tool_name}"
        return self.tool_name

    @property
    def summary(self) -> str:
        """单行结果摘要（日志用）。"""
        status = "✓" if self.success else "✗"
        return f"[{self.server_name}] {status} {self.tool_name}"


# ═══════════════════════════════════════════════════════════════════════
# MCP 连接状态
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MCPConnectionState:
    """
    MCPClient 的连接状态。

    用于监控和调试——每个 MCPClient 实例有一个 state。
    外部（如 status API）可读取此状态判断哪些 Server 在线。
    """
    server_name: str = ""
    connected: bool = False
    tool_count: int = 0
    tool_names: list[str] = field(default_factory=list)
    error_message: str = ""
