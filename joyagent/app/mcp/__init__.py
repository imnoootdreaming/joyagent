"""
Phase 8 Step 1 — MCP Client System

MCP (Model Context Protocol) Client 端实现。
使用纯 Python JSON-RPC 2.0 over stdio 与外部 MCP Server 进程通信。

核心组件：
  - MCPServerConfig:  MCP Server 连接配置（命令/参数/环境变量）
  - MCPTool:          MCP Server 提供的工具（发现后缓存）
  - MCPToolResult:    工具执行结果
  - MCPClient:        单个 MCP Server 的客户端（connect → discover → execute → disconnect）
  - MCPConnectionState: 连接状态（监控/调试用）

后续 Steps 会加入：
  - MCPRegistry:      管理多个 MCP Server 连接
  - MCPToolAdapter:   将 MCPTool 适配为 Phase 2 的 BaseTool
  - adapters/:        官方 MCP Server 的适配器配置
  - demo_server/:     自建的 Demo MCP Server
"""

from app.mcp.schemas import (
    MCPConnectionState,
    MCPServerConfig,
    MCPTool,
    MCPToolResult,
)
from app.mcp.client import MCPClient

__all__ = [
    # ── 数据模型 ──
    "MCPServerConfig",
    "MCPTool",
    "MCPToolResult",
    "MCPConnectionState",
    # ── 客户端 ──
    "MCPClient",
]
