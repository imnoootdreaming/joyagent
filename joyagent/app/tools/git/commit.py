# ── 标准库导入 ──
import subprocess                  # 调用 git CLI（封装 git commit）

# ── 项目内导入 ──
from app.tools.base import BaseTool, ToolResult


class GitCommitTool(BaseTool):
    """
    Phase 2: Git 提交工具（写入操作）。
    封装 git commit -m <message> [-a]。
    写入操作 is_dangerous = True——每次执行前需要用户确认。

    安全设计：
      - is_dangerous = True，Phase 2 先打黄色警告
      - Phase 9 接入 Human-in-the-Loop 审批弹窗
      - 只做 commit，不做 push（push 是另一个工具，未来扩展）

    为什么不做 amend / --no-verify？
      - --amend 会改写历史，风险更高，暂不实装
      - --no-verify 会绕过 pre-commit hooks，安全风险更大
      - 渐进式安全：从最简单的 commit 开始
    """

    # ─── 工具标识 ───
    name = "git_commit"                            # LLM 调用的工具名

    description = (
        "Create a git commit with the given message. "
        "Use 'all=true' to stage all modified files before committing (git commit -a). "
        "This is a WRITE operation — it modifies the git repository."
    )

    # ─── JSON Schema 定义 ───
    @property
    def input_schema(self) -> dict:
        """
        输入参数：
          - message: 提交信息（必填——无信息的 commit 不可追溯）
          - all:     是否先暂存所有变更再提交（默认 False，只提交已 git add 的文件）
        """
        return {
            "type": "object",                       # JSON Schema 根类型
            "properties": {
                "message": {
                    "type": "string",               # 字符串类型
                    "description": "The commit message. Required — every commit must have a message.",
                },
                "all": {
                    "type": "boolean",              # 布尔类型
                    "description": "Stage all modified tracked files before committing (equivalent to git commit -a). Default false.",
                },
            },
            "required": ["message"],                # message 必填，all 可选
        }

    # ─── 安全标记 ───
    @property
    def is_dangerous(self) -> bool:
        """
        git commit 修改仓库历史——这是写入操作，必须用户确认。
        Phase 2: ToolRegistry.execute() 在控制台打印黄色警告。
        Phase 9: 每次 commit 前弹确认对话框，展示将要提交的文件列表。
        """
        return True

    # ─── 核心执行逻辑 ───
    async def execute(self, message: str, all: bool = False, **kwargs) -> ToolResult:
        """
        执行 git commit -m <message> [-a]。

        参数：
          message → git commit -m "xxx"
          all     → 是否加 -a（自动暂存所有已跟踪文件的变更）

        ⚠️ 注意：此工具不会执行 git push——push 是独立的风险操作，
        需要单独的审批流程（未来扩展 git/push.py）。
        """
        # 0. 参数校验
        if not message or not message.strip():
            # 空提交信息——git 也拒绝，但我们提前校验给更好的错误提示
            return ToolResult(
                success=False,
                message="Error: Commit message cannot be empty.",
                error="EmptyCommitMessage",
            )

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

            # 2. 检查是否有变更可提交
            status = subprocess.run(
                ["git", "status", "--porcelain"],   # --porcelain: 机器可读格式
                capture_output=True,
                text=True,
                timeout=10,
            )
            if not status.stdout.strip():
                # 没有变更——commit 会失败（nothing to commit）
                return ToolResult(
                    success=False,
                    message="Error: Nothing to commit. Working tree is clean.",
                    error="NothingToCommit",
                )

            # 3. 构建 git commit 参数列表
            cmd = ["git", "commit"]                 # 基础命令

            #   -m <message>: 提交信息
            cmd.extend(["-m", message.strip()])     # strip() 去掉首尾空白

            #   -a: 自动暂存所有已跟踪文件的变更（不包括 untracked）
            if all:
                cmd.append("-a")

            # 4. 执行 git commit
            result = subprocess.run(
                cmd,
                capture_output=True,                # 捕获 stdout/stderr
                text=True,                          # 自动 decode
                timeout=30,                         # 大仓库的 pre-commit hooks 可能慢
            )

            # 5. 解析输出
            output = result.stdout.strip()           # git commit 成功时输出到 stdout
            stderr_output = result.stderr.strip()    # 某些 hooks 的消息在 stderr

            # 合并 stdout 和 stderr：pre-commit hooks 的输出通常在 stderr
            full_output = output
            if stderr_output:
                full_output += "\n" + stderr_output

            return ToolResult(
                success=result.returncode == 0,      # returncode=0 表示提交成功
                message=full_output,                 # 给 LLM 的提交结果
                metadata={
                    "message": message,             #   - 提交信息
                    "all": all,                     #   - 是否用了 -a
                    "exit_code": result.returncode, #   - git 退出码
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
                message="Error: git commit timed out after 30 seconds.",
                error="GitCommitTimeout",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error: git commit failed: {e}",
                error=str(e),
            )
