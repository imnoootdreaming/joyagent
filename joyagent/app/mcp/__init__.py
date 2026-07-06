"""
Phase 8 — MCP (Model Context Protocol) Plugin System

MCP Client 端实现。使用纯 Python JSON-RPC 2.0 over stdio 与
外部 MCP Server 进程通信。

核心组件：
  - MCPServerConfig:   MCP Server 连接配置
  - MCPTool:           MCP Server 提供的工具
  - MCPToolResult:     工具执行结果
  - MCPClient:         单个 MCP Server 的客户端
  - MCPRegistry:       管理多个 MCP Server 连接
  - mcp_registry:      全局单例
"""

from app.mcp.schemas import (
    MCPConnectionState,
    MCPServerConfig,
    MCPTool,
    MCPToolResult,
)
from app.mcp.client import MCPClient
from app.mcp.registry import MCPRegistry, mcp_registry

__all__ = [
    "MCPServerConfig",
    "MCPTool",
    "MCPToolResult",
    "MCPConnectionState",
    "MCPClient",
    "MCPRegistry",
    "mcp_registry",
]
