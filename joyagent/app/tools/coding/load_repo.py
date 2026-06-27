"""
Phase 4 Step 6: load_repo tool — 仓库结构加载工具。

将 RepositoryLoader 包装为 BaseTool，让 LLM Agent 了解仓库整体结构：
有哪些文件、各语言的分布、总规模等。这是 Coding Agent 理解项目的第一步。
"""

from app.tools.base import BaseTool, ToolResult         # 工具基类和返回格式
from app.coding.repository_loader import RepositoryLoader  # 仓库加载器


class LoadRepoTool(BaseTool):
    """
    仓库加载工具 —— 加载并返回仓库的整体结构概览。

    Agent 收到编程任务后，应先调用此工具了解仓库结构，
    再决定是否需要 search_code / analyze_code 深入分析。
    """

    name = "load_repo"

    description = (
        "Load the repository structure overview. Returns file counts by language, "
        "total code size, and directory summary. Use this first before searching/analyzing code. "
        "Use 'refresh=true' to force reload (e.g. after applying patches)."
    )

    @property
    def is_dangerous(self) -> bool:
        return False                     # 只读加载，不修改文件

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "refresh": {
                    "type": "boolean",
                    "description": "Force reload the repo (clear cache). Default: false.",
                },
                "language": {
                    "type": "string",
                    "description": "Filter to a specific language (e.g. 'python', 'javascript'). Omit for all.",
                },
            },
            "required": [],
        }

    async def execute(self, refresh: bool = False, language: str = "",
                      **kwargs) -> ToolResult:
        # ── 每次都重新加载（load 很快，且保证始终是最新状态） ──
        loader = RepositoryLoader(".")
        repo = loader.load()

        try:
            if language:
                # ── 按语言查看 ──
                files = repo.get_files_by_language(language)
                total_kb = repo.get_total_size_bytes(language) / 1024
                lines = [
                    f"## {language.title()} files: {len(files)} ({total_kb:.1f} KB)",
                    "",
                ]
                for f in sorted(files, key=lambda x: -x.size_bytes)[:30]:
                    lines.append(f"  {f.path:50s} {f.size_bytes:>6d} bytes")
                return ToolResult(success=True, message="\n".join(lines), metadata={"count": len(files)})
            else:
                # ── 全局概览 ──
                formatted = repo.summarize()
                return ToolResult(success=True, message=formatted, metadata={"total_files": len(repo.files)})
        except Exception as e:
            return ToolResult(success=False, message="", error=str(e))
