# ── 标准库导入 ──
import subprocess                  # 调用 git CLI（封装 git status --short）
import os                          # 检查当前目录是否为 git 仓库

# ── 项目内导入 ──
from app.tools.base import BaseTool, ToolResult


class GitStatusTool(BaseTool):
    """
    Phase 2: Git 状态查询工具。
    封装 git status --short，只读操作，自动执行无需用户确认。

    为什么不直接用 gitpython 库？
      - subprocess 零额外依赖，调用 git CLI 最可靠
      - gitpython 是大仓，额外引入解析开销
      - git CLI 的输出格式稳定，不会被库版本影响
    """

    # ─── 工具标识 ───
    name = "git_status"                            # LLM 调用的工具名

    description = (
        "Show the current git working tree status using 'git status --short'. "
        "Returns the list of changed, staged, and untracked files."
    )

    # ─── JSON Schema 定义 ───
    @property
    def input_schema(self) -> dict:
        """
        无参数工具——只需要声明 type: object，required 为空即可。
        这是 Anthropic API 对无参工具的要求格式。
        """
        return {
            "type": "object",                      # JSON Schema 根类型
            "properties": {},                      # 无参数 → 空 properties
            "required": [],                        # 无必填参数
        }

    # ─── 安全标记 ───
    @property
    def is_dangerous(self) -> bool:
        """
        git status 是纯只读操作——不修改仓库、不影响工作区。
        无需用户确认，Agent 可自由调用。
        """
        return False

    # ─── 核心执行逻辑 ───
    async def execute(self, **kwargs) -> ToolResult:
        """
        执行 git status --short 并返回结果。

        为什么签名只有 **kwargs？
          GitStatusTool 无参数，不需要显式声明。**kwargs 兼容 BaseTool 的
          通用签名（ToolRegistry.execute(name, **kwargs) 解包调用）。
        """
        try:
            # 1. 检查当前目录是否在 git 仓库中
            #    git status 在非 git 目录下会报错，我们先做预检查
            check = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True,               # 捕获 stdout 和 stderr
                text=True,                         # 以文本而非字节返回
                timeout=5,                         # 5 秒超时（git rev-parse 很快）
            )
            if check.returncode != 0:
                # 当前目录不是 git 仓库
                return ToolResult(
                    success=False,
                    message="Error: Not in a git repository.",
                    error="NotAGitRepo",
                )

            # 2. 执行 git status --short
            result = subprocess.run(
                ["git", "status", "--short"],      # --short: 简洁输出（每行一个文件）
                capture_output=True,               # 捕获标准输出
                text=True,                         # 自动 decode 为字符串
                timeout=10,                        # 10 秒超时
            )

            # 3. 解析输出
            output = result.stdout.strip()          # 去掉首尾空行
            if not output:
                output = "Working tree clean"       # 干净的工作区，给 LLM 友好的提示

            return ToolResult(
                success=True,                       # git status 执行成功
                message=output,                     # 给 LLM 的状态文本
                metadata={
                    "exit_code": result.returncode, #   - git 退出码
                    "clean": output == "Working tree clean",  #   - 工作区是否干净
                },
            )

        except FileNotFoundError:
            # git 没有安装在系统上
            return ToolResult(
                success=False,
                message="Error: git is not installed on this system.",
                error="GitNotInstalled",
            )
        except subprocess.TimeoutExpired:
            # git status 超时（大仓库可能慢）
            return ToolResult(
                success=False,
                message="Error: git status timed out after 10 seconds.",
                error="GitStatusTimeout",
            )
        except Exception as e:
            # 兜底：所有其他异常
            return ToolResult(
                success=False,
                message=f"Error: git status failed: {e}",
                error=str(e),
            )
