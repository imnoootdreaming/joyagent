"""
Phase 8 Step 3 — MCP Adapter 配置测试

验证 3 个官方 MCP Server 的连接配置正确性：
  - GitHub:  config 结构、Token 检测、npx 参数
  - PostgreSQL: config 结构、DATABASE_URL 检测
  - Filesystem: config 结构、工作目录

不能做真正连接测试（需 npx + npm 包），只验证配置对象。
"""

import os

import pytest

from app.mcp.adapters import (
    GITHUB_MCP_CONFIG,
    POSTGRES_MCP_CONFIG,
    FILESYSTEM_MCP_CONFIG,
)
from app.mcp.adapters.github import (
    is_github_configured,
    get_github_token_status,
)
from app.mcp.adapters.postgres import is_postgres_configured
from app.mcp.schemas import MCPServerConfig


# ═══════════════════════════════════════════════════════════════════════
# GitHub Adapter
# ═══════════════════════════════════════════════════════════════════════

class TestGitHubAdapter:
    """GitHub MCP Server 连接配置。"""

    def test_config_is_mcpserverconfig(self):
        assert isinstance(GITHUB_MCP_CONFIG, MCPServerConfig)

    def test_name(self):
        assert GITHUB_MCP_CONFIG.name == "github"

    def test_command_is_npx(self):
        assert GITHUB_MCP_CONFIG.command == "npx"

    def test_args_correct(self):
        assert "-y" in GITHUB_MCP_CONFIG.args
        assert "@modelcontextprotocol/server-github" in GITHUB_MCP_CONFIG.args

    def test_env_contains_token_key(self):
        # env dict 中应有 GITHUB_PERSONAL_ACCESS_TOKEN key（值可为空）
        assert "GITHUB_PERSONAL_ACCESS_TOKEN" in GITHUB_MCP_CONFIG.env

    def test_auto_connect_respects_token(self):
        # 如果没设 Token，auto_connect 应为 False（避免启动时卡住）
        token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        if not token:
            assert GITHUB_MCP_CONFIG.auto_connect is False
        else:
            assert GITHUB_MCP_CONFIG.auto_connect is True

    def test_is_github_configured_no_token(self, monkeypatch):
        monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
        assert is_github_configured() is False

    def test_is_github_configured_with_token(self, monkeypatch):
        monkeypatch.setenv("GITHUB_PERSONAL_ACCESS_TOKEN", "ghp_test123")
        assert is_github_configured() is True

    def test_token_status_not_set(self, monkeypatch):
        monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
        assert "NOT SET" in get_github_token_status()

    def test_token_status_configured(self, monkeypatch):
        monkeypatch.setenv("GITHUB_PERSONAL_ACCESS_TOKEN", "ghp_abcd1234")
        status = get_github_token_status()
        assert "configured" in status
        assert "ghp_abcd" in status


# ═══════════════════════════════════════════════════════════════════════
# PostgreSQL Adapter
# ═══════════════════════════════════════════════════════════════════════

class TestPostgresAdapter:
    """PostgreSQL MCP Server 连接配置。"""

    def test_config_is_mcpserverconfig(self):
        assert isinstance(POSTGRES_MCP_CONFIG, MCPServerConfig)

    def test_name(self):
        assert POSTGRES_MCP_CONFIG.name == "postgres"

    def test_command_is_npx(self):
        assert POSTGRES_MCP_CONFIG.command == "npx"

    def test_args_correct(self):
        assert "@modelcontextprotocol/server-postgres" in POSTGRES_MCP_CONFIG.args

    def test_env_contains_database_url_key(self):
        assert "DATABASE_URL" in POSTGRES_MCP_CONFIG.env

    def test_auto_connect_respects_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        # 需要新创建 config（因为模块级已缓存了旧的 auto_connect 值）
        from app.mcp.schemas import MCPServerConfig as Cfg
        cfg = Cfg(
            name="test-pg", command="npx",
            args=["-y", "@modelcontextprotocol/server-postgres"],
            env={"DATABASE_URL": os.getenv("DATABASE_URL", "")},
            auto_connect=bool(os.getenv("DATABASE_URL", "")),
        )
        assert cfg.auto_connect is False

    def test_is_postgres_configured_no_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert is_postgres_configured() is False

    def test_is_postgres_configured_with_url(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
        assert is_postgres_configured() is True


# ═══════════════════════════════════════════════════════════════════════
# Filesystem Adapter
# ═══════════════════════════════════════════════════════════════════════

class TestFilesystemAdapter:
    """Filesystem MCP Server 连接配置。"""

    def test_config_is_mcpserverconfig(self):
        assert isinstance(FILESYSTEM_MCP_CONFIG, MCPServerConfig)

    def test_name(self):
        assert FILESYSTEM_MCP_CONFIG.name == "filesystem"

    def test_command_is_npx(self):
        assert FILESYSTEM_MCP_CONFIG.command == "npx"

    def test_args_correct(self):
        assert "@modelcontextprotocol/server-filesystem" in FILESYSTEM_MCP_CONFIG.args

    def test_always_auto_connect(self):
        """Filesystem server 不需要外部凭证，始终自动连接。"""
        assert FILESYSTEM_MCP_CONFIG.auto_connect is True

    def test_env_is_none(self):
        """Filesystem server 不需要环境变量。"""
        assert FILESYSTEM_MCP_CONFIG.env is None

    def test_root_in_args(self):
        """最后一个 arg 是工作目录路径。"""
        # 至少有一个路径参数
        path_args = [a for a in FILESYSTEM_MCP_CONFIG.args
                     if a.startswith("/")]
        assert len(path_args) >= 1, f"No root path in args: {FILESYSTEM_MCP_CONFIG.args}"


# ═══════════════════════════════════════════════════════════════════════
# 集成：Adapters 注册到 MCPRegistry
# ═══════════════════════════════════════════════════════════════════════

class TestAdaptersRegistry:
    """适配器注册到 MCPRegistry 的集成测试。"""

    def test_all_three_in_adapters_package(self):
        """adapters/__init__ 导出全部 3 个 config。"""
        from app.mcp import adapters
        names = [adapters.GITHUB_MCP_CONFIG.name,
                 adapters.POSTGRES_MCP_CONFIG.name,
                 adapters.FILESYSTEM_MCP_CONFIG.name]
        assert names == ["github", "postgres", "filesystem"]

    def test_add_servers_to_registry(self):
        """3 个 adapter 可以 add_server 到 registry（不连接）。"""
        from app.mcp.registry import MCPRegistry
        reg = MCPRegistry()
        reg.add_servers([
            GITHUB_MCP_CONFIG,
            POSTGRES_MCP_CONFIG,
            FILESYSTEM_MCP_CONFIG,
        ])
        assert reg.server_names == ["github", "postgres", "filesystem"]

    def test_filesystem_config_no_env_needed(self):
        """Filesystem config 可以在没有特殊环境变量的情况下创建。"""
        cfg = MCPServerConfig(
            name="fs-test",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            auto_connect=True,
        )
        assert cfg.name == "fs-test"
        assert cfg.auto_connect is True
        assert cfg.env is None
