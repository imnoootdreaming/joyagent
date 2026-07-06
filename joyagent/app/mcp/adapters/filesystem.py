"""
Phase 8 Step 3 — Filesystem MCP Server 适配器

Filesystem MCP Server 提供文件系统操作能力：
  - read_file:         读取文件内容
  - write_file:        写入文件
  - list_directory:    列出目录内容
  - create_directory:  创建目录
  - move_file:         移动/重命名文件
  - search_files:      搜索文件
  - get_file_info:     获取文件元数据

前置条件：
  - Node.js ≥ 18
  - 第一个参数是要暴露给 Server 的目录路径

安全注意：
  Filesystem Server 只能访问 args 中指定的目录及其子目录。
  Server 通过 allowedDirectories 限制访问范围。
"""

import os

from app.mcp.schemas import MCPServerConfig

# 默认暴露 /workspace（sandbox 和主容器共享的工作目录）
_WORKSPACE = os.getenv("MCP_FILESYSTEM_ROOT", "/workspace")

FILESYSTEM_MCP_CONFIG = MCPServerConfig(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", _WORKSPACE],
    env=None,
    auto_connect=True,  # Filesystem server 总是可用（不需要外部凭证）
    description=f"Filesystem MCP Server — 文件读写、目录遍历（root={_WORKSPACE}）",
)
