"""
Phase 2: 工具注册入口。

在 FastAPI 启动时调用 register_all_tools()，将所有工具注册到全局 ToolRegistry。
Agent 通过 tool_registry.get_tool_schemas() 获取工具列表传给 Anthropic API，
通过 tool_registry.execute(name, **input) 执行工具。
"""

from app.tools.registry import tool_registry

# ── 文件工具 ──
from app.tools.file_read import FileReadTool
from app.tools.file_write import FileWriteTool

# ── Shell 工具 ──
from app.tools.shell.execute import ShellExecuteTool

# ── Git 工具 ──
from app.tools.git.status import GitStatusTool
from app.tools.git.diff import GitDiffTool
from app.tools.git.log import GitLogTool
from app.tools.git.branch import GitBranchTool
from app.tools.git.commit import GitCommitTool


def register_all_tools():
    """
    注册所有工具到全局 ToolRegistry。

    调用时机：FastAPI startup 事件（app/main.py）。
    幂等性：同名工具重复注册会抛 ValueError，避免意外覆盖。

    工具清单（Phase 2）：
      - 文件操作:   read_file (只读), write_file (写入, ⚠️危险)
      - Shell 执行: execute_shell (⚠️危险)
      - Git 只读:   git_status, git_diff, git_log, git_branch
      - Git 写入:   git_commit (⚠️危险)
    """
    # 文件工具
    tool_registry.register_tool(FileReadTool())
    tool_registry.register_tool(FileWriteTool())

    # Shell 工具
    tool_registry.register_tool(ShellExecuteTool())

    # Git 只读工具
    tool_registry.register_tool(GitStatusTool())
    tool_registry.register_tool(GitDiffTool())
    tool_registry.register_tool(GitLogTool())
    tool_registry.register_tool(GitBranchTool())

    # Git 写入工具
    tool_registry.register_tool(GitCommitTool())

    print(f"  [OK] Registered {len(tool_registry.list_tools())} tools: "
          f"{', '.join(tool_registry.list_tools())}")
    print(f"  [!!] Dangerous tools (require confirmation): "
          f"{', '.join(tool_registry.get_dangerous_tools())}")
