"""
Phase 8 Step 3 — MCP Server 适配器配置

预置的官方 MCP Server 连接配置：
  - GitHub MCP Server:    仓库搜索、Issue/PR 管理
  - PostgreSQL MCP Server: 数据库查询、表结构探索
  - Filesystem MCP Server: 文件读写、目录遍历

使用方式：
  from app.mcp.adapters import GITHUB_MCP_CONFIG
  mcp_registry.add_server(GITHUB_MCP_CONFIG)

注意：
  - 这些 Server 通过 npx（npm 包管理器）启动，需要 Node.js ≥ 18
  - GitHub 需要 GITHUB_PERSONAL_ACCESS_TOKEN 环境变量
  - PostgreSQL 需要 DATABASE_URL 环境变量
  - 每个 Server 独立进程运行，通过 stdio JSON-RPC 通信
"""

from app.mcp.adapters.github import GITHUB_MCP_CONFIG
from app.mcp.adapters.postgres import POSTGRES_MCP_CONFIG
from app.mcp.adapters.filesystem import FILESYSTEM_MCP_CONFIG

__all__ = [
    "GITHUB_MCP_CONFIG",
    "POSTGRES_MCP_CONFIG",
    "FILESYSTEM_MCP_CONFIG",
]
