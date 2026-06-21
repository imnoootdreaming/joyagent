# Phase 8：MCP Plugin System

> **返回主索引：** [0-1 coding.md](0-1%20coding.md)
> **上一阶段：** [Phase 7: Multi-Agent](phase-7-multi-agent.md)
> **下一阶段：** [Phase 9: 工程化](phase-9-engineering.md)

---

## 一、目标与定位

### 目标
实现 Claude Code 风格的 MCP (Model Context Protocol) 插件体系，支持外部能力的动态扩展。

### 范围调整 ⚠️（重要变更）

原文档计划“自建 GitHub/PostgreSQL/Browser 三个 MCP Server”。**调整为：**

1. **实现 MCP Client** — 接入官方已有的 MCP Server（GitHub、PostgreSQL、Filesystem）
2. **自建 1 个 Demo MCP Server** — 展示对 MCP 协议的深度理解
3. **不做** 完整的生产级 MCP Server（那是独立项目级别的工作量）

### MCP 协议模型

```
┌──────────────────┐         ┌──────────────────┐
│   MCP Client      │ ◀─────▶│   MCP Server      │
│   (我们的 Agent)   │  JSON   │   (外部能力提供)  │
│                   │  RPC    │                   │
│ - Tool Discovery  │────────▶│ - List Tools      │
│ - Tool Execution  │────────▶│ - Execute Tool    │
│ - Resource Access │────────▶│ - Read Resource   │
└──────────────────┘         └──────────────────┘
```

### 在整体架构中的位置
MCP 是 Tool Calling Layer 的**扩展机制**——Phase 2 的工具是硬编码在我们的代码中的，MCP 让 Agent 能发现和使用外部进程提供的工具。

### 本 Phase 不做什么
- ❌ 不做生产级 MCP Server（如完整的 GitHub Server）
- ❌ 不做 MCP 协议的 Streamable HTTP transport（Phase 8 仅 stdio）

---

## 二、前置依赖

| 依赖 | 用途 |
|------|------|
| Phase 2 完成 | Tool Calling Framework |
| mcp | MCP Python SDK |
| 官方 MCP Servers | GitHub、PostgreSQL、Filesystem |

```bash
uv add mcp
# 安装官方 MCP Server（作为外部进程运行，不需要作为 Python 依赖）
npm install -g @modelcontextprotocol/server-github
npm install -g @modelcontextprotocol/server-postgres
npm install -g @modelcontextprotocol/server-filesystem
```

---

## 三、目录结构

```text
app/
├── mcp/
│   ├── __init__.py
│   ├── client.py              # MCP Client：连接 MCP Server，发现+执行工具
│   ├── registry.py            # MCP Server Registry：管理多个 MCP Server 连接
│   │
│   ├── adapters/              # MCP Server 适配器（连接配置）
│   │   ├── __init__.py
│   │   ├── github.py          # GitHub MCP Server 连接配置
│   │   ├── postgres.py        # PostgreSQL MCP Server 连接配置
│   │   └── filesystem.py      # Filesystem MCP Server 连接配置
│   │
│   └── demo_server/           # 自建的 Demo MCP Server
│       ├── __init__.py
│       ├── server.py          # MCP Server 实现（stdio transport）
│       └── tools.py           # Demo Server 提供的工具
```

---

## 四、核心数据模型 / Schema 定义

### 4.1 MCP Client 抽象

```python
from dataclasses import dataclass
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

@dataclass
class MCPServerConfig:
    """MCP Server 连接配置"""
    name: str                      # 逻辑名称，如 "github"
    command: str                   # 启动命令，如 "npx"
    args: list[str]                # 命令参数，如 ["-y", "@modelcontextprotocol/server-github"]
    env: dict[str, str] | None     # 环境变量（如 GITHUB_TOKEN）
    auto_connect: bool = True      # Agent 启动时是否自动连接

@dataclass
class MCPTool:
    """MCP Server 提供的工具"""
    name: str
    description: str
    parameters: dict               # JSON Schema
    server_name: str               # 来自哪个 MCP Server
    
    def to_schema(self) -> dict:
        """转为 Anthropic 原生工具格式"""
        return {
            "name": f"{self.server_name}__{self.name}",  # 加前缀防冲突
            "description": f"[{self.server_name}] {self.description}",
            "input_schema": self.parameters,
        }

@dataclass
class MCPToolResult:
    """MCP 工具执行结果"""
    tool_name: str
    server_name: str
    success: bool
    content: str
    error: str | None = None
```

### 4.2 MCP Client 实现

```python
class MCPClient:
    """单个 MCP Server 的客户端"""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session: ClientSession | None = None
        self.tools: list[MCPTool] = []
    
    async def connect(self) -> None:
        """建立与 MCP Server 的连接"""
        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env,
        )
        
        # stdio_client 返回 read/write stream
        self.read, self.write = await stdio_client(server_params).__aenter__()
        self.session = await ClientSession(self.read, self.write).__aenter__()
        await self.session.initialize()
        
        # 发现 Server 提供的工具
        await self._discover_tools()
    
    async def _discover_tools(self) -> None:
        """发现 MCP Server 提供的所有工具"""
        tools_result = await self.session.list_tools()
        self.tools = [
            MCPTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema,
                server_name=self.config.name,
            )
            for tool in tools_result.tools
        ]
    
    async def execute_tool(self, tool_name: str, **kwargs) -> MCPToolResult:
        """执行 MCP Server 的工具"""
        try:
            result = await self.session.call_tool(tool_name, arguments=kwargs)
            return MCPToolResult(
                tool_name=tool_name,
                server_name=self.config.name,
                success=True,
                content=str(result.content),
            )
        except Exception as e:
            return MCPToolResult(
                tool_name=tool_name,
                server_name=self.config.name,
                success=False,
                content="",
                error=str(e),
            )
    
    async def disconnect(self) -> None:
        """断开连接"""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if hasattr(self, 'read'):
            await self.read.__aexit__(None, None, None)
```

### 4.3 MCP Registry

```python
class MCPRegistry:
    """管理多个 MCP Server 连接"""
    
    def __init__(self):
        self._clients: dict[str, MCPClient] = {}
    
    async def register_server(self, config: MCPServerConfig) -> None:
        """注册并连接一个 MCP Server"""
        client = MCPClient(config)
        await client.connect()
        self._clients[config.name] = client
    
    async def get_all_tools(self) -> list[MCPTool]:
        """获取所有 MCP Server 提供的工具"""
        tools = []
        for client in self._clients.values():
            tools.extend(client.tools)
        return tools
    
    async def execute(self, tool_full_name: str, **kwargs) -> MCPToolResult:
        """
        执行 MCP 工具。
        tool_full_name 格式: "server_name__tool_name"（如 "github__search_repositories"）
        """
        server_name, tool_name = tool_full_name.split("__", 1)
        client = self._clients.get(server_name)
        if not client:
            return MCPToolResult(tool_name=tool_name, server_name=server_name,
                                 success=False, content="", error=f"Unknown server: {server_name}")
        return await client.execute_tool(tool_name, **kwargs)
    
    async def shutdown(self) -> None:
        """断开所有连接"""
        for client in self._clients.values():
            await client.disconnect()

# 全局单例
mcp_registry = MCPRegistry()
```

---

## 五、详细开发清单（含 HOW）

### Step 1：实现 MCP Client（1.5 小时）
- 按 §4.2 实现 `MCPClient` 类
- 核心 API：`connect()` → `_discover_tools()` → `execute_tool()` → `disconnect()`
- 传输方式：stdio（标准输入/输出），MCP 最基础的 transport

### Step 2：实现 MCP Registry（30 分钟）
- 按 §4.3 实现 `MCPRegistry`
- 管理多个 MCP Client 的连接生命周期
- 提供统一的工具发现和执行接口

### Step 3：接入官方 MCP Server（1 小时）

**GitHub MCP Server：**
```python
# app/mcp/adapters/github.py
GITHUB_MCP_CONFIG = MCPServerConfig(
    name="github",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN")},
)
# 提供的工具：search_repositories, create_issue, create_pull_request, ...
```

**PostgreSQL MCP Server：**
```python
# app/mcp/adapters/postgres.py
POSTGRES_MCP_CONFIG = MCPServerConfig(
    name="postgres",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-postgres"],
    env={"DATABASE_URL": os.getenv("DATABASE_URL")},
)
# 提供的工具：query, list_tables, describe_table, ...
```

**Filesystem MCP Server：**
```python
# app/mcp/adapters/filesystem.py
FILESYSTEM_MCP_CONFIG = MCPServerConfig(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
)
# 提供的工具：read_file, write_file, list_directory, ...
```

### Step 4：自建 Demo MCP Server（1.5 小时）⭐ 核心

**`mcp/demo_server/server.py`：** 一个简单的 Weather MCP Server，展示 MCP 协议的完整实现。

```python
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationCapabilities
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 创建 MCP Server 实例
app = Server("joyagent-demo-server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """告诉 Client 我们有哪些工具"""
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        ),
        Tool(
            name="calculate",
            description="Evaluate a mathematical expression",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. '2 + 3 * 4'"
                    }
                },
                "required": ["expression"]
            }
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """执行工具"""
    if name == "get_weather":
        city = arguments["city"]
        # 模拟天气数据（实际应调用天气 API）
        return [TextContent(
            type="text",
            text=f"Weather in {city}: Sunny, 22°C, Humidity 45%"
        )]
    
    elif name == "calculate":
        expression = arguments["expression"]
        try:
            result = eval(expression)  # ⚠️ 仅 Demo，生产需要安全沙箱
            return [TextContent(type="text", text=f"Result: {result}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationCapabilities(
                sampling={},
                roots={},
                experimental={},
            ),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**配置自建 Server 的连接：**
```python
DEMO_MCP_CONFIG = MCPServerConfig(
    name="joyagent-demo",
    command="python",
    args=["-m", "app.mcp.demo_server.server"],
)
```

### Step 5：集成到 ToolRegistry（1 小时）

```python
class MCPToolAdapter(BaseTool):
    """将 MCP Tool 适配为 Phase 2 的 BaseTool"""
    
    def __init__(self, mcp_tool: MCPTool, registry: MCPRegistry):
        self.mcp_tool = mcp_tool
        self.mcp_registry = registry
    
    @property
    def name(self) -> str:
        return f"{self.mcp_tool.server_name}__{self.mcp_tool.name}"
    
    @property
    def description(self) -> str:
        return f"[MCP:{self.mcp_tool.server_name}] {self.mcp_tool.description}"
    
    @property
    def parameters(self) -> dict:
        return self.mcp_tool.parameters
    
    async def execute(self, **kwargs) -> ToolResult:
        result = await self.mcp_registry.execute(self.name, **kwargs)
        return ToolResult(
            success=result.success,
            output=result.content,
            error=result.error,
        )

async def register_mcp_tools():
    """将 MCP Server 的工具注册到 ToolRegistry"""
    for mcp_tool in await mcp_registry.get_all_tools():
        tool_registry.register(MCPToolAdapter(mcp_tool, mcp_registry))
```

### Step 6：启动时初始化（30 分钟）
- 在 `app/main.py` 的 startup event 中初始化 MCP Registry
- 连接所有配置的 MCP Server
- 注册 MCP 工具到 ToolRegistry

---

## 六、关键代码模式与伪代码

### 6.1 MCP 工具发现与注册全流程

```python
# 1. Agent 启动
await mcp_registry.register_server(GITHUB_MCP_CONFIG)
await mcp_registry.register_server(POSTGRES_MCP_CONFIG)
await mcp_registry.register_server(DEMO_MCP_CONFIG)

# 2. 发现所有 MCP 工具
mcp_tools = await mcp_registry.get_all_tools()
# mcp_tools: [
#   MCPTool(name="search_repositories", server="github", ...),
#   MCPTool(name="create_issue", server="github", ...),
#   MCPTool(name="query", server="postgres", ...),
#   MCPTool(name="get_weather", server="joyagent-demo", ...),
#   MCPTool(name="calculate", server="joyagent-demo", ...),
# ]

# 3. 适配为 BaseTool 并注册
for mcp_tool in mcp_tools:
    tool_registry.register(MCPToolAdapter(mcp_tool, mcp_registry))

# 4. Agent 使用时无感知——就像使用本地工具一样
# LLM sees: "github__search_repositories" tool available
```

---

## 七、完成标志

### 基本完成
- [ ] MCP Client 能连接至少 1 个官方 MCP Server（如 Filesystem）
- [ ] 官方 MCP Server 的工具能被发现并注册到 ToolRegistry
- [ ] Agent 能通过 MCP 工具执行操作（如通过 Filesystem Server 读写文件）
- [ ] 自建 Demo MCP Server 的 `list_tools` 和 `call_tool` 正常工作
- [ ] 自建 Demo Server 被 Agent 成功调用

### 自测用例

```bash
# 测试 1：MCP Filesystem Server
curl -X POST /api/chat -d '{
  "message": "列出 /workspace 目录下的所有文件"
}'
# 期望：Agent 通过 MCP filesystem__list_directory 工具获取文件列表

# 测试 2：自建 Demo MCP Server
curl -X POST /api/chat -d '{
  "message": "查询北京的天气，并计算 123 * 456"
}'
# 期望：Agent 调用 joyagent-demo__get_weather 和 joyagent-demo__calculate

# 测试 3：MCP + 本地工具混合
curl -X POST /api/chat -d '{
  "message": "读取 README.md（用 MCP filesystem server），然后创建一个备份文件 README_backup.md"
}'
```

---

## 八、文档差距分析

### 原文档缺失的内容及补充

| # | 原文档问题 | 实际开发需要 | 本文档补充位置 |
|---|-----------|-------------|-------------|
| 1 | **MCP 工作量严重低估** | 完整的 GitHub MCP Server 需要处理 GitHub API 的认证、分页、限流、Webhook，是一个独立项目。调整为接入官方 Server + 自建 1 个 Demo | §1 |
| 2 | 没说 MCP 传输协议 | MCP 支持 stdio（本 Phase 用）和 Streamable HTTP。stdio 适合本地进程间通信。 | §5 Step 1 |
| 3 | 没说 MCP Tool 如何与 Phase 2 的 ToolRegistry 集成 | 通过适配器模式：MCPToolAdapter implements BaseTool | §5 Step 5 |
| 4 | 没说 MCP Server 的 `list_tools` 和 `call_tool` 两个核心接口 | 这是 MCP 协议的核心：Server 声明能力 → Client 发现能力 → Client 调用能力 | §5 Step 4 |
| 5 | 没有 MCP Client 的生命周期管理 | 需要 connect → discover → execute → disconnect 完整流程 | §4.2 |
| 6 | 没说 MCP 工具的命名冲突问题 | 多个 MCP Server 可能提供同名工具 → 加前缀 `server_name__tool_name` | §4.1 |

### MCP 面试要点：MCP 与传统 Tool Calling 的区别

| 维度 | 传统 Tool Calling | MCP |
|------|------------------|-----|
| 工具定义 | 硬编码在 Agent 代码中 | Server 动态声明（`list_tools`） |
| 工具实现 | 与 Agent 同一进程 | 独立进程，通过 stdio/HTTP 通信 |
| 扩展方式 | 改代码 + 重新部署 | 启动新 MCP Server → 自动发现 |
| 语言限制 | 必须用 Agent 的语言 | 任意语言实现 Server |
| 生态 | 各自为政 | 统一协议，社区共享 Server |
| 安全 | 直接调用 | 进程隔离 + 权限由 Client 控制 |

---

## 九、面试考点映射

| 面试题 | 答案要点 | 本文档参考 |
|--------|---------|-----------|
| **MCP 是什么？** | Model Context Protocol：Anthropic 提出的 LLM 与外部工具/数据源之间的标准协议。让 Agent 能动态发现和调用外部能力，而不需要硬编码每个工具。 | §1, §8 |
| **MCP 与传统 Tool Calling 的区别？** | 传统 = 硬编码、同进程、改代码部署。MCP = 动态发现、独立进程、热插拔。MCP 是 Tool Calling 的标准化和生态化。 | §8 |
| **MCP Client 如何发现 Tool？** | 连接 MCP Server → 调用 `list_tools()` RPC → 获取 Tool 列表（含 name/description/inputSchema）→ 注册到本地 ToolRegistry。 | §4.2 |
| **MCP Server 如何注册能力？** | 实现 `list_tools`（声明能力）和 `call_tool`（执行能力）两个核心接口。通过 stdio 或 HTTP transport 暴露。 | §5 Step 4 |
| **为什么 MCP 会成为 Agent 生态标准？** | 1) 统一协议降低集成成本 2) 社区共享 Server 3) 语言无关 4) 动态扩展 5) Anthropic/OpenAI 双支持。 | §8 |
