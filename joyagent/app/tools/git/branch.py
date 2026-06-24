# ── 标准库导入 ──
import subprocess                  # 调用 git CLI（封装 git branch）

# ── 项目内导入 ──
from app.tools.base import BaseTool, ToolResult


class GitBranchTool(BaseTool):
    """
    Phase 2: Git 分支管理工具（只读）。
    封装 git branch -a -v，显示本地和远程分支列表及最近一次提交。
    只读操作，自动执行无需用户确认。

    为什么 Phase 2 不做 git branch -d / 创建分支？
      - 删除/创建分支会修改仓库状态，需要用户确认
      - 先用只读版本让 Agent 了解分支结构，后续 Phase 可扩展写入操作
    """

    # ─── 工具标识 ───
    name = "git_branch"                            # LLM 调用的工具名

    description = (
        "List all git branches (local and remote) with their latest commit. "
        "Use 'all=true' to include remote branches (default true). "
        "Use 'verbose=false' to omit latest commit info for brevity."
    )

    # ─── JSON Schema 定义 ───
    @property
    def input_schema(self) -> dict:
        """
        输入参数（均为可选）：
          - all:     是否显示远程分支（默认 True）
          - verbose: 是否显示最近一次提交（默认 True）
        """
        return {
            "type": "object",                       # JSON Schema 根类型
            "properties": {
                "all": {
                    "type": "boolean",              # 布尔类型
                    "description": "Include remote tracking branches (-a flag). Default true.",
                },
                "verbose": {
                    "type": "boolean",              # 布尔类型
                    "description": "Show latest commit hash and subject for each branch (-v flag). Default true.",
                },
            },
            "required": [],                         # 无必填参数
        }

    # ─── 安全标记 ───
    @property
    def is_dangerous(self) -> bool:
        """
        只列分支，不修改任何内容。安全。
        """
        return False

    # ─── 核心执行逻辑 ───
    async def execute(self, all: bool = True, verbose: bool = True, **kwargs) -> ToolResult:
        """
        执行 git branch [-a] [-v] 并返回结果。

        参数默认值：
          all=True     → 同时显示本地和远程分支
          verbose=True → 显示每个分支最新 commit 的 hash 和 subject

        输出示例：
          * phase-2-tool-calling  297192b feat(phase-2): add BaseTool class
            main                  d685c8f merge phase-1-agent-demo into main
            remotes/origin/main   d685c8f merge phase-1-agent-demo into main

        当前分支以 * 开头，方便 LLM 识别所在位置。
        """
        try:
            # 1. 校验仓库前置条件
            check = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True,                # 捕获 stdout/stderr
                text=True,                          # 文本而非字节
                timeout=5,                          # 5 秒超时
            )
            if check.returncode != 0:
                return ToolResult(
                    success=False,
                    message="Error: Not in a git repository.",
                    error="NotAGitRepo",
                )

            # 2. 构建 git branch 参数列表
            cmd = ["git", "branch"]                 # 基础命令：git branch

            #   -a: 包含远程跟踪分支（remotes/origin/*）
            if all:
                cmd.append("-a")

            #   -v: 详细模式——每个分支后附加最新 commit 的 hash 和 subject
            #   无 -v: 只列分支名（更简洁但信息量少）
            if verbose:
                cmd.append("-v")

            # 3. 执行 git branch
            result = subprocess.run(
                cmd,
                capture_output=True,                # 捕获 stdout/stderr
                text=True,                          # 自动 decode
                timeout=10,                         # 10 秒超时
            )

            # 4. 解析输出
            output = result.stdout.strip()           # 去掉首尾空行
            if not output:
                output = "No branches found."       # 空仓库没有分支

            return ToolResult(
                success=True,                        # 执行成功
                message=output,                      # 分支列表给 LLM
                metadata={
                    "all": all,                     #   - 是否含远程分支
                    "verbose": verbose,             #   - 是否详细模式
                    "exit_code": result.returncode, #   - git 退出码
                    "branch_count": len(output.split("\n")),  #   - 分支数量
                },
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                message="Error: git is not installed on this system.",
                error="GitNotInstalled",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                message="Error: git branch timed out after 10 seconds.",
                error="GitBranchTimeout",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error: git branch failed: {e}",
                error=str(e),
            )
