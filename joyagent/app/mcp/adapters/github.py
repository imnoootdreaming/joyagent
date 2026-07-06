"""
Phase 8 Step 3 — GitHub MCP Server 适配器

GitHub MCP Server 提供 GitHub 平台的操作能力：
  - search_repositories:   搜索仓库
  - create_issue:          创建 Issue
  - create_pull_request:   创建 PR
  - list_issues:           列出 Issue
  - get_file_contents:     读取文件内容
  - ... 等 20+ 工具

前置条件：
  - Node.js ≥ 18（npx 命令可用）
  - GitHub Personal Access Token（repo + issues 权限）

Token 获取：
  GitHub → Settings → Developer settings → Personal access tokens →
  Fine-grained tokens → Generate new token
  权限：Repository access (selected) + Repository permissions (Contents: read,
         Issues: read/write, Pull requests: read/write)

环境变量：
  GITHUB_PERSONAL_ACCESS_TOKEN — 在宿主机 .env 或 docker run -e 中设置

启动方式：
  npx -y @modelcontextprotocol/server-github
  这个命令自动下载并启动官方 GitHub MCP Server
"""

import os

from app.mcp.schemas import MCPServerConfig

# ── 从环境变量读取 Token ──
_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")

GITHUB_MCP_CONFIG = MCPServerConfig(
    name="github",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={
        "GITHUB_PERSONAL_ACCESS_TOKEN": _TOKEN,
    },
    auto_connect=bool(_TOKEN),  # 有 Token 才自动连接
    description="GitHub MCP Server — 仓库搜索、Issue/PR 管理、文件读取",
)


# ═══════════════════════════════════════════════════════════════════════
# Token 检查工具
# ═══════════════════════════════════════════════════════════════════════

def is_github_configured() -> bool:
    """检查 GitHub Token 是否已配置。"""
    return bool(os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"))

def get_github_token_status() -> str:
    """返回 Token 状态（用于启动时日志）。"""
    token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    if not token:
        return "NOT SET (set GITHUB_PERSONAL_ACCESS_TOKEN env var)"
    if token.startswith("ghp_") or token.startswith("github_pat_"):
        return f"configured ({token[:8]}...)"
    return "configured (⚠ unexpected format)"
