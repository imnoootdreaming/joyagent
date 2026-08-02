"""
Zotero MCP Server 适配器

文献整理场景：接入 Zotero 文献库，让 Agent 能检索、归类与整理实验室文献。

复用项目自研的 MCPRegistry + MCPServerConfig（纯 Python stdio JSON-RPC 客户端，
不依赖第三方 SDK），以 stdio 方式启动 Zotero MCP Server 并自动发现其工具
（如 search_items / get_item / add_item / get_pdf_annotations）。

支持两种接入方式：
  1. 社区 Server（默认）：通过 npx 拉起 @mgmeyers/zotero-mcp-server
     （需本地已运行 Zotero + 启用 Zotero Local API 插件，且设置环境变量
      ZOTERO_LOCAL_API_KEY / ZOTERO_USER_ID）
  2. 自定义 Server：设置环境变量 ZOTERO_MCP_COMMAND / ZOTERO_MCP_ARGS
     覆盖启动命令（例如自建的文献整理 Server）

安全说明：
  Zotero API Key 仅通过环境变量注入子进程，不进代码仓库。
"""
import os

from app.mcp.schemas import MCPServerConfig

# 默认社区 Zotero MCP Server（需本地 Zotero 开启 Local API）
ZOTERO_DEFAULT_COMMAND = os.getenv("ZOTERO_MCP_COMMAND", "npx")
ZOTERO_DEFAULT_ARGS = os.getenv(
    "ZOTERO_MCP_ARGS",
    "-y @mgmeyers/zotero-mcp-server",
).split()


def build_zotero_config() -> MCPServerConfig:
    """构建 Zotero MCP 连接配置（凭据来自环境变量）。"""
    env = None
    api_key = os.getenv("ZOTERO_LOCAL_API_KEY")
    user_id = os.getenv("ZOTERO_USER_ID")
    if api_key or user_id:
        env = {}
        if api_key:
            env["ZOTERO_LOCAL_API_KEY"] = api_key
        if user_id:
            env["ZOTERO_USER_ID"] = user_id

    return MCPServerConfig(
        name="zotero",
        command=ZOTERO_DEFAULT_COMMAND,
        args=ZOTERO_DEFAULT_ARGS,
        env=env,
        auto_connect=bool(os.getenv("ZOTERO_ENABLED", "true").lower() == "true"),
        description="Zotero MCP Server — 文献检索、PDF 注解汇总、条目归档与打标签",
    )


# 导出默认配置实例（供 mcp_registry.add_server 使用）
ZOTERO_MCP_CONFIG = build_zotero_config()
