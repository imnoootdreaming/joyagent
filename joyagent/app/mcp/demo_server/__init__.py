"""
Phase 8 Step 4 — JoyAgent Demo MCP Server

一个展示 MCP 协议完整实现的 Demo Server。

启动方式：
  python -m app.mcp.demo_server.server

可通过 MCPClient 连接进行测试：
  client = MCPClient(DEMO_MCP_CONFIG)
  await client.connect()
"""

from app.mcp.schemas import MCPServerConfig

DEMO_MCP_CONFIG = MCPServerConfig(
    name="joyagent-demo",
    command="python",
    args=["-m", "app.mcp.demo_server.server"],
    auto_connect=True,
    description="JoyAgent Demo MCP Server — Weather + Calculator + Clock",
)

__all__ = ["DEMO_MCP_CONFIG"]
