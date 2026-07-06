"""
Phase 8 — MCP (Model Context Protocol) Plugin System

MCP Client 端完整实现。纯 Python JSON-RPC 2.0 over stdio。

核心组件：
  - MCPServerConfig:   MCP Server 连接配置
  - MCPTool:           MCP Server 提供的工具
  - MCPClient:         单个 MCP Server 的客户端
  - MCPRegistry:       管理多个 MCP Server 连接
  - MCPToolAdapter:    将 MCPTool 适配为 Phase 2 BaseTool
  - adapters/:         官方 Server 连接配置
  - demo_server/:      自建 Demo Server
"""

from app.mcp.schemas import (
    MCPConnectionState,
    MCPServerConfig,
    MCPTool,
    MCPToolResult,
)
from app.mcp.client import MCPClient
from app.mcp.registry import MCPRegistry, mcp_registry
from app.mcp.adapter import MCPToolAdapter, register_mcp_tools
from app.mcp.demo_server import DEMO_MCP_CONFIG
from app.mcp import adapters

__all__ = [
    "MCPServerConfig",
    "MCPTool",
    "MCPToolResult",
    "MCPConnectionState",
    "MCPClient",
    "MCPRegistry",
    "mcp_registry",
    "MCPToolAdapter",
    "register_mcp_tools",
    "DEMO_MCP_CONFIG",
    "adapters",
]
