"""
Phase 4 Step 6: search_code tool — 代码搜索工具。

将 CodeSearcher 的搜索能力包装为 BaseTool，让 LLM Agent 可以通过
Tool Calling 直接搜索代码。支持正则搜索和关键词搜索两种模式。
"""

from app.tools.base import BaseTool, ToolResult         # 工具基类和返回格式
from app.coding.repository_loader import RepositoryLoader  # 仓库加载器
from app.coding.code_search import CodeSearcher         # 搜索引擎


# ── 全局仓库缓存（同一会话内复用，避免重复加载大仓库） ──
_repo_cache: dict[str, tuple] = {}                       # {root: (RepoContext,)}


def _get_or_load_repo(root_path: str = "."):
    """懒加载仓库上下文 —— 首次加载后缓存，后续调用复用。"""
    import os
    abs_path = os.path.abspath(root_path)
    if abs_path not in _repo_cache:
        loader = RepositoryLoader(abs_path)
        _repo_cache[abs_path] = loader.load()
    return _repo_cache[abs_path]


class SearchCodeTool(BaseTool):
    """
    代码搜索工具 —— 在仓库中搜索代码。

    使用方式（LLM 调用）：
      - 正则搜索：search_code(pattern="async\\s+def\\s+\\w+")
      - 关键词搜索：search_code(keyword="ToolRegistry")
    """

    name = "search_code"

    description = (
        "Search the codebase for patterns or keywords. "
        "Use 'pattern' for regex search, or 'keyword' for simple text search. "
        "Use 'file_pattern' to limit to specific file types (e.g. '*.py'). "
        "Returns matching lines with file paths and line numbers."
    )

    @property
    def is_dangerous(self) -> bool:
        return False                     # 只读搜索，不修改任何文件

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python regex pattern to search for. Mutually exclusive with 'keyword'.",
                },
                "keyword": {
                    "type": "string",
                    "description": "Simple text/keyword to search (no regex). Mutually exclusive with 'pattern'.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Limit search to files matching this glob (e.g. '*.py', 'app/**/*.py'). Default: all files.",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["pattern", "keyword", "definition", "callers"],
                    "description": "Search type. 'definition' finds def/class, 'callers' finds call sites. Default: auto-detected.",
                },
            },
            "required": [],
        }

    async def execute(self, pattern: str = "", keyword: str = "",
                      file_pattern: str = "*", search_type: str = "",
                      **kwargs) -> ToolResult:
        repo = _get_or_load_repo()
        searcher = CodeSearcher(repo)

        try:
            # ── 根据 search_type 选择搜索方法 ──
            if search_type == "definition":
                name = keyword or pattern
                if not name:
                    return ToolResult(success=False, message="", error="Need 'keyword' or 'pattern' for definition search.")
                result = searcher.search_by_name(name, file_pattern=file_pattern)
            elif search_type == "callers":
                name = keyword or pattern
                if not name:
                    return ToolResult(success=False, message="", error="Need 'keyword' or 'pattern' for callers search.")
                result = searcher.search_callers(name, file_pattern=file_pattern)
            elif keyword:
                result = searcher.search_by_content(keyword, file_pattern=file_pattern)
            elif pattern:
                result = searcher.search_by_pattern(pattern, file_pattern=file_pattern)
            else:
                return ToolResult(success=False, message="", error="Either 'pattern' or 'keyword' is required.")

            # ── 格式化为 LLM 友好输出 ──
            formatted = result.format_for_llm(max_items=30)
            return ToolResult(
                success=True,
                message=formatted,
                metadata={"total_found": result.total_found, "truncated": result.truncated},
            )
        except Exception as e:
            return ToolResult(success=False, message="", error=str(e))
