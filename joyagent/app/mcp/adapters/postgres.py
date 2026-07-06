"""
Phase 8 Step 3 — PostgreSQL MCP Server 适配器

PostgreSQL MCP Server 提供数据库操作能力：
  - query:              执行只读 SQL 查询
  - list_tables:        列出所有表
  - describe_table:     查看表结构
  - list_schemas:       列出 Schema

前置条件：
  - Node.js ≥ 18
  - DATABASE_URL 环境变量（如 postgresql://user:pass@localhost:5432/dbname）

环境变量：
  DATABASE_URL — PostgreSQL 连接字符串
"""

import os

from app.mcp.schemas import MCPServerConfig

_DATABASE_URL = os.getenv("DATABASE_URL", "")

POSTGRES_MCP_CONFIG = MCPServerConfig(
    name="postgres",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-postgres"],
    env={
        "DATABASE_URL": _DATABASE_URL,
    },
    auto_connect=bool(_DATABASE_URL),
    description="PostgreSQL MCP Server — 数据库查询、表结构探索",
)


def is_postgres_configured() -> bool:
    """检查 PostgreSQL 连接是否已配置。"""
    return bool(os.getenv("DATABASE_URL"))


def get_postgres_status() -> str:
    """返回 PostgreSQL 配置状态。"""
    url = os.getenv("DATABASE_URL", "")
    if not url:
        return "NOT SET (set DATABASE_URL env var)"
    # 隐藏密码
    if "@" in url:
        parts = url.split("@")
        return f"configured ({parts[0].split(':')[0]}://...@{parts[1]})"
    return "configured"
