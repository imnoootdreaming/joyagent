# ── 标准库导入 ──
import subprocess                  # 调用 git CLI（封装 git log）

# ── 项目内导入 ──
from app.tools.base import BaseTool, ToolResult


class GitLogTool(BaseTool):
    """
    Phase 2: Git 日志查看工具。
    封装 git log --oneline，支持 --graph 拓扑图，可限制返回条数。
    只读操作，自动执行无需用户确认。

    与 git status/diff 的分工：
      - status → 当前工作区快照
      - diff   → 未提交的内容差异
      - log    → 历史提交沿革
    """

    # ─── 工具标识 ───
    name = "git_log"                               # LLM 调用的工具名

    description = (
        "Show git commit history in compact one-line format. "
        "Use 'n' to limit the number of commits (default 20), "
        "use 'graph=true' to show branch topology with --graph."
    )

    # ─── JSON Schema 定义 ───
    @property
    def input_schema(self) -> dict:
        """
        输入参数（均为可选）：
          - n:     返回的提交条数（默认 20）
          - graph: 是否显示分支拓扑图（默认 True，更直观）
        """
        return {
            "type": "object",                       # JSON Schema 根类型
            "properties": {
                "n": {
                    "type": "integer",              # 整数类型
                    "description": "Number of recent commits to show. Default 20.",
                },
                "graph": {
                    "type": "boolean",              # 布尔类型
                    "description": "Show branch/merge graph topology. Default true.",
                },
            },
            "required": [],                         # 无必填参数——默认显示最近 20 条
        }

    # ─── 安全标记 ───
    @property
    def is_dangerous(self) -> bool:
        """
        git log 是只读操作，只查看历史记录，不修改任何内容。
        """
        return False

    # ─── 核心执行逻辑 ───
    async def execute(self, n: int = 20, graph: bool = True, **kwargs) -> ToolResult:
        """
        执行 git log --oneline [-n <n>] [--graph] 并返回结果。

        参数默认值：
          n=20    → 最近 20 条提交（防止默认返回全部历史撑爆上下文）
          graph=True → 默认开启 --graph，方便 LLM 理解分支结构

        --oneline 格式示例：
          d685c8f merge phase-1-agent-demo into main
          297192b (HEAD -> phase-2-tool-calling) feat(phase-2): add BaseTool
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

            # 2. 构建 git log 参数列表
            cmd = ["git", "log", "--oneline"]       # 基础命令：单行格式

            #   --graph: 在左侧显示 ASCII 分支拓扑图
            #   例如:
            #   *   297192b feat(phase-2): add BaseTool
            #   |\
            #   | * 320726c feat(phase-1): add tools
            #   |/
            #   * d685c8f merge phase-1
            if graph:
                cmd.append("--graph")

            #   -n: 限制返回条数
            #   用 f"-{n}" 构造 "-20" 这样的参数
            cmd.append(f"-{n}")

            # 3. 执行 git log
            result = subprocess.run(
                cmd,
                capture_output=True,                # 捕获 stdout/stderr
                text=True,                          # 自动 decode
                timeout=10,                         # 10 秒超时
            )

            # 4. 解析输出
            output = result.stdout.strip()           # 去掉首尾空行
            if not output:
                output = "No commits yet."          # 空仓库

            return ToolResult(
                success=True,                        # 执行成功
                message=output,                      # 提交历史给 LLM
                metadata={
                    "n": n,                         #   - 请求的条数
                    "graph": graph,                 #   - 是否显示拓扑
                    "exit_code": result.returncode, #   - git 退出码
                    "line_count": len(output.split("\n")),  #   - 实际返回的提交数
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
                message="Error: git log timed out after 10 seconds.",
                error="GitLogTimeout",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error: git log failed: {e}",
                error=str(e),
            )
