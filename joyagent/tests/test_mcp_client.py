"""
Phase 8 Step 1 — MCPClient 单元测试

覆盖：
  - MCPServerConfig: 创建、默认值、环境变量
  - MCPTool: 创建、to_anthropic_schema、full_name、from_list_tools_result
  - MCPToolResult: 创建、full_name、summary
  - MCPClient: 构造、连接流程、工具发现、工具执行、断开、错误处理
  - 集成: 完整 connect → discover → execute → disconnect 周期
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from app.mcp.schemas import (
    MCPConnectionState,
    MCPServerConfig,
    MCPTool,
    MCPToolResult,
)
from app.mcp.client import MCPClient


# ═══════════════════════════════════════════════════════════════════════
# Mock MCP Server: 一个极简的 JSON-RPC stdio Server（用于测试）
# ═══════════════════════════════════════════════════════════════════════

_MOCK_SERVER_SCRIPT = r"""
import json, sys, os

def respond(id, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": id, "result": result}) + "\n")
    sys.stdout.flush()

def error(id, code, msg):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": msg}}) + "\n")
    sys.stdout.flush()

MOCK_TOOLS = [
    {
        "name": "read_file",
        "description": "Read contents of a file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "list_directory",
        "description": "List files in a directory",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dir_path": {"type": "string", "description": "Directory path"}
            },
            "required": ["dir_path"]
        }
    },
    {
        "name": "get_status",
        "description": "Get server status",
        "inputSchema": {"type": "object", "properties": {}}
    },
]

for line in sys.stdin:
    line = line.strip()
    if not line:
        break
    try:
        req = json.loads(line)
    except json.JSONDecodeError:
        continue
    rid = req.get("id")
    method = req.get("method", "")
    params = req.get("params", {})

    if method == "initialize":
        respond(rid, {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "mock-server", "version": "0.1.0"},
            "capabilities": {"tools": {}}
        })
    elif method == "tools/list":
        respond(rid, {"tools": MOCK_TOOLS})
    elif method == "tools/call":
        tool_name = params.get("name", "")
        args = params.get("arguments", {})
        if tool_name == "read_file":
            respond(rid, {"content": [{"type": "text", "text": f"[mock] Contents of {args.get('path', '?')}"}]})
        elif tool_name == "list_directory":
            respond(rid, {"content": [{"type": "text", "text": f"[mock] Files in {args.get('dir_path', '.')}: file1.py, file2.py"}]})
        elif tool_name == "get_status":
            respond(rid, {"content": [{"type": "text", "text": "[mock] Server is healthy"}]})
        else:
            respond(rid, {"isError": True, "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}]})
    else:
        error(rid, -32601, f"Method not found: {method}")
"""


def _write_mock_server(path: Path) -> None:
    """将 mock MCP server 脚本写入文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_MOCK_SERVER_SCRIPT, encoding="utf-8")


def _mock_server_config(script_path: Path) -> MCPServerConfig:
    """返回指向 mock server 脚本的 MCPServerConfig。"""
    return MCPServerConfig(
        name="mock-server",
        command=sys.executable,
        args=[str(script_path)],
        description="Mock MCP server for testing",
    )


# ═══════════════════════════════════════════════════════════════════════
# MCPServerConfig 测试
# ═══════════════════════════════════════════════════════════════════════

class TestMCPServerConfig:
    """MCPServerConfig 数据类测试。"""

    def test_create_minimal(self):
        """最小配置：只有 name + command。"""
        cfg = MCPServerConfig(name="test", command="python")
        assert cfg.name == "test"
        assert cfg.command == "python"
        assert cfg.args == []
        assert cfg.env is None
        assert cfg.auto_connect is True

    def test_create_full(self):
        """完整配置：含 args + env。"""
        cfg = MCPServerConfig(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "ghp_xxx"},
            auto_connect=False,
            description="GitHub MCP Server",
        )
        assert cfg.args == ["-y", "@modelcontextprotocol/server-github"]
        assert cfg.env == {"GITHUB_TOKEN": "ghp_xxx"}
        assert cfg.auto_connect is False
        assert cfg.description == "GitHub MCP Server"


# ═══════════════════════════════════════════════════════════════════════
# MCPTool 测试
# ═══════════════════════════════════════════════════════════════════════

class TestMCPTool:
    """MCPTool 数据类测试。"""

    def test_create_tool(self):
        tool = MCPTool(
            name="search_repos",
            description="Search GitHub repos",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            server_name="github",
        )
        assert tool.name == "search_repos"
        assert tool.server_name == "github"

    def test_full_name_with_prefix(self):
        """full_name 格式：server_name__tool_name。"""
        tool = MCPTool(name="read_file", description="Read a file",
                       server_name="filesystem")
        assert tool.full_name == "filesystem__read_file"

    def test_full_name_without_prefix(self):
        """无 server_name → full_name == name。"""
        tool = MCPTool(name="local_tool", description="A local tool")
        assert tool.full_name == "local_tool"

    def test_display_description(self):
        """display_description 含 Server 名前缀。"""
        tool = MCPTool(
            name="query",
            description="Execute SQL query",
            server_name="postgres",
        )
        assert tool.display_description == "[MCP:postgres] Execute SQL query"

    def test_to_anthropic_schema(self):
        """to_anthropic_schema 返回 Anthropic 原生格式。"""
        tool = MCPTool(
            name="read_file",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            server_name="filesystem",
        )
        schema = tool.to_anthropic_schema()
        assert schema["name"] == "filesystem__read_file"
        assert "[MCP:filesystem]" in schema["description"]
        assert schema["input_schema"] == tool.parameters
        assert "path" in schema["input_schema"]["properties"]

    def test_from_list_tools_result(self):
        """从 MCP list_tools 响应构建 MCPTool。"""
        tool_data = {
            "name": "create_issue",
            "description": "Create a GitHub issue",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["title"],
            },
        }
        tool = MCPTool.from_list_tools_result(tool_data, "github")
        assert tool.name == "create_issue"
        assert tool.server_name == "github"
        assert "title" in tool.parameters["properties"]

    def test_from_list_tools_result_inputSchema_camelcase(self):
        """MCP 协议用 'inputSchema'（驼峰），from_list_tools_result 兼容。"""
        tool_data = {
            "name": "test",
            "description": "test",
            "inputSchema": {"type": "object", "properties": {}},
        }
        tool = MCPTool.from_list_tools_result(tool_data, "test-srv")
        assert tool.parameters == {"type": "object", "properties": {}}


# ═══════════════════════════════════════════════════════════════════════
# MCPToolResult 测试
# ═══════════════════════════════════════════════════════════════════════

class TestMCPToolResult:
    """MCPToolResult 数据类测试。"""

    def test_success_result(self):
        r = MCPToolResult(
            tool_name="read_file",
            server_name="filesystem",
            success=True,
            content="Hello world",
        )
        assert r.success is True
        assert r.error is None
        assert r.content == "Hello world"

    def test_error_result(self):
        r = MCPToolResult(
            tool_name="read_file",
            server_name="filesystem",
            success=False,
            error="Permission denied",
        )
        assert r.success is False
        assert r.error == "Permission denied"

    def test_full_name(self):
        r = MCPToolResult(tool_name="read_file", server_name="fs", success=True)
        assert r.full_name == "fs__read_file"

    def test_summary(self):
        r = MCPToolResult(tool_name="read_file", server_name="fs", success=True)
        assert "✓" in r.summary
        r2 = MCPToolResult(tool_name="bad", server_name="fs", success=False)
        assert "✗" in r2.summary


# ═══════════════════════════════════════════════════════════════════════
# MCPClient 集成测试（使用 mock MCP server 子进程）
# ═══════════════════════════════════════════════════════════════════════

class TestMCPClient:
    """MCPClient 集成测试 — 用 mock MCP server 子进程。"""

    @pytest.fixture
    def tmp_script(self, tmp_path):
        """写入 mock server 脚本到临时路径。"""
        script = tmp_path / "mock_mcp_server.py"
        _write_mock_server(script)
        return script

    @pytest.mark.asyncio
    async def test_connect_and_discover_tools(self, tmp_script):
        """连接 → 发现工具 → 缓存到 self.tools。"""
        config = _mock_server_config(tmp_script)
        client = MCPClient(config)

        await client.connect()
        assert client.is_connected is True
        assert len(client.tools) == 3

        names = {t.name for t in client.tools}
        assert names == {"read_file", "list_directory", "get_status"}

        await client.disconnect()
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, tmp_script):
        """执行已发现的工具 → 返回成功结果。"""
        config = _mock_server_config(tmp_script)
        client = MCPClient(config)
        await client.connect()

        result = await client.execute_tool("read_file", path="/tmp/test.txt")
        assert result.success is True
        assert result.tool_name == "read_file"
        assert result.server_name == "mock-server"
        assert "mock" in result.content.lower()
        assert "/tmp/test.txt" in result.content

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_returns_error(self, tmp_script):
        """执行不存在的工具 → isError=True → success=False。"""
        config = _mock_server_config(tmp_script)
        client = MCPClient(config)
        await client.connect()

        result = await client.execute_tool("nonexistent_tool")
        assert result.success is False
        assert "Unknown" in result.error

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_clears_tools(self, tmp_script):
        """disconnect 后 self.tools 被清空。"""
        config = _mock_server_config(tmp_script)
        client = MCPClient(config)
        await client.connect()
        assert len(client.tools) == 3

        await client.disconnect()
        assert len(client.tools) == 0
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_is_idempotent(self, tmp_script):
        """已连接时再次 connect 不重复初始化。"""
        config = _mock_server_config(tmp_script)
        client = MCPClient(config)
        await client.connect()
        count = len(client.tools)

        # 第二次 connect 直接返回
        await client.connect()
        assert len(client.tools) == count

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_execute_without_connect_returns_error(self, tmp_script):
        """未连接时 execute_tool 返回错误（不抛异常）。"""
        config = _mock_server_config(tmp_script)
        client = MCPClient(config)

        result = await client.execute_tool("read_file", path="/tmp/x.txt")
        assert result.success is False
        assert "not connected" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_anthropic_tool_schemas(self, tmp_script):
        """get_anthropic_tool_schemas 返回 Anthropic 原生格式。"""
        config = _mock_server_config(tmp_script)
        client = MCPClient(config)
        await client.connect()

        schemas = client.get_anthropic_tool_schemas()
        assert len(schemas) == 3
        for s in schemas:
            assert "name" in s
            assert "description" in s
            assert "input_schema" in s
            assert "__" in s["name"]  # server prefix

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_state_property(self, tmp_script):
        """MCPClient.state 返回连接状态。"""
        config = _mock_server_config(tmp_script)
        client = MCPClient(config)

        s0 = client.state
        assert s0.connected is False
        assert s0.tool_count == 0

        await client.connect()
        s1 = client.state
        assert s1.connected is True
        assert s1.tool_count == 3
        assert "read_file" in s1.tool_names

        await client.disconnect()
        s2 = client.state
        assert s2.connected is False
        assert s2.tool_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_twice_is_safe(self, tmp_script):
        """重复 disconnect 不抛异常。"""
        config = _mock_server_config(tmp_script)
        client = MCPClient(config)
        await client.connect()
        await client.disconnect()
        await client.disconnect()  # 第二次不抛异常

    @pytest.mark.asyncio
    async def test_command_not_found(self):
        """命令不存在的 Server → RuntimeError。"""
        config = MCPServerConfig(
            name="nonexistent",
            command="this_command_does_not_exist_xyz",
        )
        client = MCPClient(config)

        with pytest.raises(RuntimeError, match="command not found"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_server_exits_immediately(self):
        """Server 立即退出（参数错误） → RuntimeError。"""
        config = MCPServerConfig(
            name="bad",
            command=sys.executable,
            args=["-c", "import sys; sys.exit(1)"],
        )
        client = MCPClient(config)

        with pytest.raises(RuntimeError, match="exited immediately"):
            await client.connect()


# ═══════════════════════════════════════════════════════════════════════
# MCPConnectionState 测试
# ═══════════════════════════════════════════════════════════════════════

class TestMCPConnectionState:
    """MCPConnectionState 数据类测试。"""

    def test_default_state(self):
        s = MCPConnectionState()
        assert s.connected is False
        assert s.tool_count == 0
        assert s.tool_names == []
        assert s.error_message == ""

    def test_connected_state(self):
        s = MCPConnectionState(
            server_name="github",
            connected=True,
            tool_count=5,
            tool_names=["search", "create_issue", "create_pr"],
        )
        assert s.tool_count == 5
        assert len(s.tool_names) == 3
