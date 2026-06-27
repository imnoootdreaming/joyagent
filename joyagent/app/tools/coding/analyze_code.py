"""
Phase 4 Step 6: analyze_code tool — Python AST 分析工具。

将 ASTAnalyzer 结构化分析能力包装为 BaseTool，让 LLM Agent 可以
精确了解 Python 文件的函数/类/导入结构，而不用盲读整个文件。
"""

from app.tools.base import BaseTool, ToolResult         # 工具基类和返回格式
from app.coding.ast_analyzer import ASTAnalyzer         # AST 分析器
from app.coding.repository_loader import RepositoryLoader  # 仓库加载器


class AnalyzeCodeTool(BaseTool):
    """
    代码分析工具 —— 分析 Python 文件的 AST 结构。

    输出结构化摘要：函数/类/方法的位置、参数、装饰器等。
    比 read_file 更高效（不读取整个文件内容，只输出结构）。
    """

    name = "analyze_code"

    description = (
        "Analyze a Python file's structure using AST. "
        "Returns structured info: imports, functions (with args/decorators/docstrings), "
        "classes (with methods and base classes), and module-level variables. "
        "Use 'path' to specify which file, or omit to analyze all Python files."
    )

    @property
    def is_dangerous(self) -> bool:
        return False                     # 只读分析，不修改文件

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to analyze (relative to repo root). Leave empty to analyze ALL Python files.",
                },
                "detail": {
                    "type": "string",
                    "enum": ["summary", "full", "functions", "classes", "imports"],
                    "description": "Detail level. 'summary' = compact, 'full' = everything. Default: 'summary'.",
                },
            },
            "required": [],
        }

    async def execute(self, path: str = "", detail: str = "summary",
                      **kwargs) -> ToolResult:
        loader = RepositoryLoader(".")
        repo = loader.load()
        analyzer = ASTAnalyzer()

        try:
            if path:
                # ── 单文件分析 ──
                content = repo.get_content(path)
                if content is None:
                    return ToolResult(success=False, message="", error=f"File not found in repo: {path}")

                analysis = analyzer.analyze_file(path, content)
                if analysis.has_syntax_error:
                    return ToolResult(
                        success=False,
                        message="",
                        error=f"Syntax error in {path}: {analysis.syntax_error_msg}",
                    )

                if detail == "full":
                    formatted = analysis.format_summary(max_items=50)
                elif detail == "functions":
                    lines = [f"## Functions in {path}"]
                    for f in analysis.functions:
                        async_kw = "async " if f.is_async else ""
                        args_str = ", ".join(f.args)
                        lines.append(f"  L{f.start_line}-{f.end_line}: {async_kw}def {f.name}({args_str})")
                        if f.docstring:
                            lines.append(f"    docstring: {f.docstring[:100]}")
                    for c in analysis.classes:
                        for m in c.methods:
                            async_kw = "async " if m.is_async else ""
                            args_str = ", ".join(m.args)
                            lines.append(f"  L{m.start_line}-{m.end_line}: {c.name}.{m.name}({args_str})")
                    formatted = "\n".join(lines)
                elif detail == "classes":
                    lines = [f"## Classes in {path}"]
                    for c in analysis.classes:
                        bases = f"({', '.join(c.base_classes)})" if c.base_classes else ""
                        lines.append(f"  L{c.start_line}-{c.end_line}: class {c.name}{bases}")
                        for m in c.methods[:10]:
                            lines.append(f"    L{m.start_line}: def {m.name}(...)")
                    formatted = "\n".join(lines)
                elif detail == "imports":
                    lines = [f"## Imports in {path}"]
                    for imp in analysis.imports:
                        lines.append(f"  {imp}")
                    formatted = "\n".join(lines)
                else:
                    formatted = analysis.format_summary(max_items=20)
            else:
                # ── 全局分析 ──
                analyses = analyzer.analyze_repo(repo)
                total_funcs = sum(len(a.functions) + sum(len(c.methods) for c in a.classes) for a in analyses.values())
                total_classes = sum(len(a.classes) for a in analyses.values())

                lines = [f"## Repo Analysis ({len(analyses)} Python files)"]
                lines.append(f"  Functions: {total_funcs}, Classes: {total_classes}")
                lines.append(f"\n### Per-File Summary\n")

                # 按函数数量排序（最重要的文件排前面）
                sorted_files = sorted(
                    analyses.items(),
                    key=lambda kv: len(kv[1].functions) + sum(len(c.methods) for c in kv[1].classes),
                    reverse=True,
                )
                for file_path, a in sorted_files[:20]:
                    f_count = len(a.functions) + sum(len(c.methods) for c in a.classes)
                    c_count = len(a.classes)
                    lines.append(
                        f"  {file_path:45s}  {f_count:>3d} funcs, {c_count:>2d} classes"
                    )

                formatted = "\n".join(lines)

            return ToolResult(success=True, message=formatted, metadata={})
        except Exception as e:
            return ToolResult(success=False, message="", error=str(e))
