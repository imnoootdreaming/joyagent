"""
Phase 4: Coding Agent - 代码理解与跨文件修改工具包。

模块结构（按开发顺序）：
  repository_loader.py  — Step 1: 加载仓库文件树，过滤 ignore + 二进制
  code_search.py        — Step 2: 代码搜索：grep + 文件名匹配
  ast_analyzer.py       — Step 3: Python AST 分析：函数/类/导入查询
  diff_generator.py     — Step 4: Diff 生成：使用 difflib.unified_diff
  patch_apply.py        — Step 5: Patch 应用：使用系统 patch 命令

数据模型（Step 1）：
  FileInfo   — 单文件信息 (path, language, size_bytes, is_binary)
  RepoContext — 仓库上下文 (files + file_contents 映射 + 便捷查询)

使用示例：
  from app.coding.repository_loader import RepositoryLoader

  loader = RepositoryLoader("/path/to/repo")
  repo = loader.load()
  print(repo.summarize())
  print(f"Python files: {len(repo.get_python_files())}")
"""

from app.coding.repository_loader import (
    RepositoryLoader,
    FileInfo,
    RepoContext,
)

from app.coding.code_search import (
    CodeSearcher,
    SearchMatch,
    SearchResult,
)

from app.coding.ast_analyzer import (
    ASTAnalyzer,
    ASTAnalysis,
    FunctionInfo,
    ClassInfo,
    VariableInfo,
)

from app.coding.diff_generator import (
    DiffGenerator,
    DiffResult,
    DiffHunk,
)

__all__ = [
    "RepositoryLoader", "FileInfo", "RepoContext",
    "CodeSearcher", "SearchMatch", "SearchResult",
    "ASTAnalyzer", "ASTAnalysis", "FunctionInfo", "ClassInfo", "VariableInfo",
    "DiffGenerator", "DiffResult", "DiffHunk",
]
