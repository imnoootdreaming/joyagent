# ── 标准库导入 ──
import subprocess                  # 调用 git CLI（封装 git diff）

# ── 项目内导入 ──
from app.tools.base import BaseTool, ToolResult


class GitDiffTool(BaseTool):
    """
    Phase 2: Git 差异查看工具。
    封装 git diff（未暂存变更）和 git diff --staged（已暂存变更）。
    只读操作，自动执行无需用户确认。

    与 git status 的分工：
      - status → "哪些文件变了"（文件级别概览）
      - diff   → "变了什么内容"（行级别差异）
    先调 status 定位文件，再调 diff 查看详情。
    """

    # ─── 工具标识 ───
    name = "git_diff"                              # LLM 调用的工具名

    description = (
        "Show git diff output — line-by-line changes in the working tree. "
        "Use 'staged=true' to view staged changes (git diff --staged), "
        "use 'path' to limit diff to a specific file or directory."
    )

    # ─── JSON Schema 定义 ───
    @property
    def input_schema(self) -> dict:
        """
        输入参数（均为可选）：
          - staged: 是否查看已暂存（git add 后）的变更（默认 False，即未暂存）
          - path:   限定到某个文件或目录的 diff（默认空，即全部文件）
        """
        return {
            "type": "object",                       # JSON Schema 根类型
            "properties": {
                "staged": {
                    "type": "boolean",              # 布尔类型
                    "description": "If true, show staged changes (git add'ed). Default false (unstaged).",
                },
                "path": {
                    "type": "string",               # 可选的文件路径限定
                    "description": "Limit diff to a specific file or directory. If omitted, diff all files.",
                },
            },
            "required": [],                         # 无必填参数——默认行为就是 git diff 全部未暂存
        }

    # ─── 安全标记 ───
    @property
    def is_dangerous(self) -> bool:
        """
        git diff 是纯只读操作，不产生任何副作用。
        """
        return False

    # ─── 核心执行逻辑 ───
    async def execute(self, staged: bool = False, path: str = None, **kwargs) -> ToolResult:
        """
        执行 git diff [--staged] [--] [path] 并返回差异内容。

        参数默认值：
          staged=False → 查看未暂存变更（工作区 vs 暂存区）
          staged=True  → 查看已暂存变更（暂存区 vs HEAD）
          path=None    → 所有文件
          path="x.py"  → 只看 x.py 的差异

        为什么用 "--" 分隔符？
          防止文件名被误解析为 git 选项。例如 path="--help" 时，
          git diff -- --help 不会被当成参数，而是安全地作为文件路径。
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
                # 不在 git 仓库中
                return ToolResult(
                    success=False,
                    message="Error: Not in a git repository.",
                    error="NotAGitRepo",
                )

            # 2. 构建 git diff 参数列表
            cmd = ["git", "diff"]                   # 基础命令：git diff

            #   --staged: 查看已暂存变更（等价于 git diff --cached）
            if staged:
                cmd.append("--staged")

            #   -- 分隔符：明确告诉 git "后面的都是文件路径，不是选项"
            #   再追加 path（如果有）
            cmd.append("--")
            if path:
                cmd.append(path)

            # 3. 执行 git diff
            result = subprocess.run(
                cmd,
                capture_output=True,                # 捕获 stdout 和 stderr
                text=True,                          # 自动 decode
                timeout=30,                         # 大仓库 diff 可能很慢，30 秒超时
            )

            # 4. 解析输出
            output = result.stdout.strip()           # 去掉首尾空行
            if not output:
                # 无差异时 git diff 输出为空
                scope = "staged" if staged else "unstaged"
                output = f"No {scope} changes."     # 给 LLM 友好的提示

            # 5. 截断过长输出——大型 diff 会填满 LLM 上下文
            was_truncated = len(output) > 8000
            truncated_output = output[:8000]         # 最多返回 8000 字符

            return ToolResult(
                success=True,                        # 执行成功
                message=truncated_output,            # 差异内容给 LLM
                metadata={
                    "staged": staged,               #   - 是否查看暂存区
                    "path": path,                   #   - 限定路径
                    "exit_code": result.returncode, #   - git 退出码
                    "truncated": was_truncated,     #   - 是否被截断
                    "original_length": len(output), #   - 原始输出长度
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
                message="Error: git diff timed out after 30 seconds.",
                error="GitDiffTimeout",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error: git diff failed: {e}",
                error=str(e),
            )
