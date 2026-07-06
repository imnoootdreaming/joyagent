"""
Phase 8 — MCP (Model Context Protocol) Plugin System

MCP Client 端完整实现。使用纯 Python JSON-RPC 2.0 over stdio。

核心组件：
  - MCPServerConfig:   MCP Server 连接配置
  - MCPTool:           MCP Server 提供的工具（发现后缓存）
  - MCPToolResult:     工具执行结果
  - MCPClient:         单个 MCP Server 的客户端
  - MCPRegistry:       管理多个 MCP Server 连接
  - MCPToolAdapter:    将 MCPTool 适配为 Phase 2 BaseTool
  - mcp_registry:      全局单例
  - DEMO_MCP_CONFIG:   自建 Demo Server 的连接配置
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

__all__ = [
    # 数据模型
    "MCPServerConfig",
    "MCPTool",
    "MCPToolResult",
    "MCPConnectionState",
    # 客户端
    "MCPClient",
    # 注册中心
    "MCPRegistry",
    "mcp_registry",
    # 适配器
    "MCPToolAdapter",
    "register_mcp_tools",
    # Demo Server
    "DEMO_MCP_CONFIG",
]
