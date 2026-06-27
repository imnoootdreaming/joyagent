"""
Phase 2: 工具注册入口。

在 FastAPI 启动时调用 register_all_tools()，将所有工具注册到全局 ToolRegistry。
Agent 通过 tool_registry.get_tool_schemas() 获取工具列表传给 Anthropic API，
通过 tool_registry.execute(name, **input) 执行工具。

Phase 2 Step 8: 同时注册 ToolStatsCollector Hook，实现工具调用可观测性。
"""

from app.tools.registry import tool_registry  # 全局工具注册中心单例

# ── 工具类 ──
from app.tools.file_read import FileReadTool        # 文件读取（只读）
from app.tools.file_write import FileWriteTool       # 文件写入（危险）
from app.tools.shell.execute import ShellExecuteTool  # Shell 命令执行（危险）
from app.tools.git.status import GitStatusTool       # Git 状态查询（只读）
from app.tools.git.diff import GitDiffTool           # Git 差异查看（只读）
from app.tools.git.log import GitLogTool             # Git 日志查看（只读）
from app.tools.git.branch import GitBranchTool       # Git 分支列表（只读）
from app.tools.git.commit import GitCommitTool       # Git 提交（危险）

# ── Coding 工具（Phase 4 Step 6） ──
from app.tools.coding.load_repo import LoadRepoTool         # 仓库结构加载（只读）
from app.tools.coding.search_code import SearchCodeTool     # 代码搜索（只读）
from app.tools.coding.analyze_code import AnalyzeCodeTool   # AST 分析（只读）
from app.tools.coding.generate_diff import GenerateDiffTool # Diff 生成（只读）
from app.tools.coding.apply_patch import ApplyPatchTool     # Patch 应用（危险）

# ── Hook 统计收集器（Phase 2 Step 8） ──
from app.tools.hooks import ToolStatsCollector  # 工具调用统计收集器

# 全局统计收集器实例 —— 供外部 API 查询（如 GET /api/tools/stats）
# 每 10 次工具调用后自动在控制台输出统计摘要
tool_stats = ToolStatsCollector(log_interval=10)


def register_all_tools():
    """
    注册所有工具 + Hook 到全局 ToolRegistry。

    调用时机：FastAPI startup 事件（app/main.py）。
    幂等性：同名工具重复注册会抛 ValueError，避免意外覆盖。

    注册清单（Phase 2）：
      工具 (8):
        - 文件:  read_file (只读), write_file (写入, 危险)
        - Shell: execute_shell (危险)
        - Git:   git_status, git_diff, git_log, git_branch (只读)
        - Git:   git_commit (写入, 危险)
      Hook (1):
        - ToolStatsCollector — 每 10 次调用输出一次统计摘要
    """
    # ── 注册全部 8 个工具 ──

    # 文件工具
    tool_registry.register_tool(FileReadTool())      # 只读，不危险
    tool_registry.register_tool(FileWriteTool())      # 写入，危险

    # Shell 工具
    tool_registry.register_tool(ShellExecuteTool())   # 命令执行，危险

    # Git 只读工具
    tool_registry.register_tool(GitStatusTool())      # 查看工作区状态
    tool_registry.register_tool(GitDiffTool())        # 查看差异内容
    tool_registry.register_tool(GitLogTool())         # 查看提交历史
    tool_registry.register_tool(GitBranchTool())      # 查看分支列表

    # Git 写入工具
    tool_registry.register_tool(GitCommitTool())      # 提交变更，危险

    # Coding 工具 — 只读（搜索/分析/加载/diff）
    tool_registry.register_tool(LoadRepoTool())       # 仓库结构概览
    tool_registry.register_tool(SearchCodeTool())     # 代码搜索
    tool_registry.register_tool(AnalyzeCodeTool())    # AST 结构分析
    tool_registry.register_tool(GenerateDiffTool())   # Diff 生成

    # Coding 工具 — 写入（patch 应用）
    tool_registry.register_tool(ApplyPatchTool())     # 应用 diff（危险）

    # ── 注册统计收集器 Hook ──
    tool_registry.register_hook(tool_stats)

    # ── 启动日志 ──
    print(f"  [OK] Registered {len(tool_registry.list_tools())} tools: "
          f"{', '.join(tool_registry.list_tools())}")
    print(f"  [!!] Dangerous tools (require confirmation): "
          f"{', '.join(tool_registry.get_dangerous_tools())}")
    print(f"  [OK] Registered 1 hook: ToolStatsCollector "
          f"(log every {tool_stats.log_interval} calls)")
