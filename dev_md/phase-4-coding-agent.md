# Phase 4：Coding Agent

> **返回主索引：** [0-1 coding.md](0-1%20coding.md)
> **上一阶段：** [Phase 3: LangGraph Workflow](phase-3-langgraph-workflow.md)
> **下一阶段：** [Phase 5: Docker Sandbox + Auto Testing](phase-5-sandbox-testing.md)

---

## 一、目标与定位

### 目标
使 Agent 具备**代码理解**和**跨文件修改**能力：加载整个仓库、搜索代码、分析 AST、生成 Diff、应用 Patch。

### 范围限定 ⚠️
**AST 分析仅支持 Python。** 其他语言的代码搜索通过 grep/正则完成，不做 AST 级别的结构分析。

### 在整体架构中的位置
Phase 4 是 Agent 从"通用对话助手"到"真正的 Coding Agent"的关键升级。它插在 LangGraph Workflow 的 Executor Node 中，让 Executor 在修改代码时使用 code_search + AST 分析而不是盲目 read_file。

```
Phase 3 Executor:  read_file → LLM 看 → write_file（盲目）
Phase 4 Executor:  code_search → AST 分析 → 定位修改点 → 生成 Diff → 应用 Patch（精准）
```

### 本 Phase 不做什么
- ❌ 不做多语言 AST（Java/JS/TS/etc.）——仅 Python
- ❌ 不做自动测试（Phase 5）
- ❌ 不做语义搜索（embedding，Phase 6）

---

## 二、前置依赖

| 依赖 | 用途 |
|------|------|
| Phase 3 完成 | LangGraph Workflow 框架 |
| Python `ast` 模块 | 标准库，Python AST 解析 |
| Python `difflib` 模块 | 标准库，统一 Diff 生成 |
| `pathspec` | .gitignore 规则匹配 |

```bash
uv add pathspec
```

---

## 三、目录结构

```text
app/
├── coding/
│   ├── __init__.py
│   ├── repository_loader.py   # 仓库加载：遍历文件树，过滤 ignore
│   ├── code_search.py         # 代码搜索：grep + 简单 AST 查询
│   ├── ast_analyzer.py        # Python AST 分析：函数/类/导入查询
│   ├── diff_generator.py      # Diff 生成：使用 difflib.unified_diff
│   └── patch_apply.py         # Patch 应用：使用 patch 命令或手动应用
│
├── tools/
│   └── coding/                # 新增：Coding 相关工具
│       ├── search_code.py     # code_search 工具（供 Agent 调用）
│       ├── analyze_code.py    # AST 分析工具
│       └── apply_patch.py     # 应用 Diff 工具
```

---

## 四、核心数据模型 / Schema 定义

### 4.1 仓库结构模型

```python
from dataclasses import dataclass

@dataclass
class FileInfo:
    """仓库中单个文件的信息"""
    path: str               # 相对路径
    language: str           # "python" | "javascript" | "unknown"
    size_bytes: int
    is_binary: bool

@dataclass  
class RepoContext:
    """加载后的仓库上下文"""
    root_path: str
    files: list[FileInfo]
    file_contents: dict[str, str]  # path → content (仅文本文件)
    
    def get_python_files(self) -> list[FileInfo]:
        return [f for f in self.files if f.language == "python"]
```

### 4.2 AST 分析结果模型

```python
@dataclass
class FunctionInfo:
    name: str
    start_line: int
    end_line: int
    args: list[str]
    decorators: list[str]
    docstring: str | None

@dataclass
class ClassInfo:
    name: str
    start_line: int
    end_line: int
    methods: list[FunctionInfo]
    base_classes: list[str]

@dataclass
class ASTAnalysis:
    """单个 Python 文件的 AST 分析结果"""
    file_path: str
    imports: list[str]           # ["import os", "from typing import List"]
    functions: list[FunctionInfo]
    classes: list[ClassInfo]
    top_level_code: str | None   # 模块级别的非函数/类代码
```

### 4.3 Diff/Patch 模型

```python
@dataclass
class DiffResult:
    """Diff 生成结果"""
    file_path: str
    original_content: str
    modified_content: str
    diff_text: str                # unified_diff 格式
    
    def can_apply_cleanly(self) -> bool:
        """检查 diff 是否可以干净应用"""
        return len(self.diff_text) > 0

@dataclass 
class PatchResult:
    """Patch 应用结果"""
    success: bool
    file_path: str
    hunks_applied: int
    hunks_failed: int
    error_message: str | None
```

---

## 五、详细开发清单（含 HOW）

### Step 1：Repository Loader（1 小时）

**`coding/repository_loader.py`：**
```python
import os
from pathspec import PathSpec

class RepositoryLoader:
    """加载仓库文件树，过滤 .gitignore 和二进制文件"""
    
    # 已知的文本文件扩展名
    TEXT_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
                       ".json", ".yaml", ".yml", ".md", ".txt", ".toml", ".cfg",
                       ".ini", ".sh", ".bat", ".Makefile", ".env", ".gitignore"}
    
    def __init__(self, root_path: str):
        self.root_path = root_path
        self._load_gitignore()
    
    def _load_gitignore(self):
        """加载 .gitignore 规则"""
        gitignore_path = os.path.join(self.root_path, ".gitignore")
        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r") as f:
                self.spec = PathSpec.from_lines("gitwildmatch", f.readlines())
        else:
            self.spec = PathSpec.from_lines("gitwildmatch", [])
    
    def load(self) -> RepoContext:
        """遍历仓库，返回 RepoContext"""
        files = []
        contents = {}
        
        for dirpath, dirnames, filenames in os.walk(self.root_path):
            # 跳过隐藏目录和常见忽略目录
            dirnames[:] = [d for d in dirnames 
                          if not d.startswith(".") 
                          and d not in ("node_modules", "__pycache__", "venv", ".git")]
            
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                relpath = os.path.relpath(filepath, self.root_path)
                
                # 检查 .gitignore
                if self.spec.match_file(relpath):
                    continue
                
                # 判断是否为文本文件
                ext = os.path.splitext(filename)[1]
                if ext not in self.TEXT_EXTENSIONS and filename not in ("Makefile", "Dockerfile"):
                    continue
                
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                except (UnicodeDecodeError, PermissionError):
                    continue
                
                files.append(FileInfo(
                    path=relpath,
                    language=self._detect_language(ext),
                    size_bytes=len(content),
                    is_binary=False,
                ))
                contents[relpath] = content
        
        return RepoContext(root_path=self.root_path, files=files, file_contents=contents)
    
    def _detect_language(self, ext: str) -> str:
        MAPPING = {".py": "python", ".js": "javascript", ".ts": "typescript",
                   ".jsx": "javascript", ".tsx": "typescript", ".html": "html",
                   ".css": "css", ".json": "json", ".md": "markdown"}
        return MAPPING.get(ext, "unknown")
```

### Step 2：Code Search（1 小时）

**`coding/code_search.py`：**
```python
import re

class CodeSearcher:
    """代码搜索引擎：基于正则 + 文件名匹配"""
    
    def __init__(self, repo: RepoContext):
        self.repo = repo
    
    def search_by_pattern(self, pattern: str, file_pattern: str = "*.py") -> list[dict]:
        """
        按正则搜索代码内容。
        返回：[{file, line_number, line_content, match}]
        """
        results = []
        regex = re.compile(pattern)
        
        for file_path, content in self.repo.file_contents.items():
            if not self._match_file_pattern(file_path, file_pattern):
                continue
            
            for i, line in enumerate(content.split("\n"), 1):
                match = regex.search(line)
                if match:
                    results.append({
                        "file": file_path,
                        "line_number": i,
                        "line_content": line.strip(),
                        "match": match.group(),
                    })
        
        return results[:50]  # 限制结果数量
    
    def search_by_name(self, name: str) -> list[dict]:
        """搜索函数/类定义（简单正则匹配）"""
        pattern = rf"(def|class)\s+{re.escape(name)}\b"
        return self.search_by_pattern(pattern)
    
    def _match_file_pattern(self, file_path: str, pattern: str) -> bool:
        import fnmatch
        return fnmatch.fnmatch(file_path, pattern)
```

### Step 3：AST Analyzer（1.5 小时）⭐ 核心

**`coding/ast_analyzer.py`：**
```python
import ast

class ASTAnalyzer:
    """Python AST 分析器。只解析 Python 文件。"""
    
    def analyze_file(self, file_path: str, source: str) -> ASTAnalysis:
        """分析单个 Python 文件，返回结构化信息"""
        tree = ast.parse(source)
        
        imports = self._extract_imports(tree)
        functions = self._extract_functions(tree)
        classes = self._extract_classes(tree)
        
        return ASTAnalysis(
            file_path=file_path,
            imports=imports,
            functions=functions,
            classes=classes,
            top_level_code=None,
        )
    
    def _extract_imports(self, tree: ast.Module) -> list[str]:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                names = ", ".join(a.name for a in node.names)
                imports.append(f"from {node.module} import {names}")
        return imports
    
    def _extract_functions(self, tree: ast.Module) -> list[FunctionInfo]:
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(FunctionInfo(
                    name=node.name,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    args=[arg.arg for arg in node.args.args],
                    decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                    docstring=ast.get_docstring(node),
                ))
        return functions
    
    def _extract_classes(self, tree: ast.Module) -> list[ClassInfo]:
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(FunctionInfo(
                            name=item.name,
                            start_line=item.lineno,
                            end_line=item.end_lineno or item.lineno,
                            args=[arg.arg for arg in item.args.args],
                            decorators=[],
                            docstring=ast.get_docstring(item),
                        ))
                classes.append(ClassInfo(
                    name=node.name,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    methods=methods,
                    base_classes=[self._get_name(b) for b in node.bases],
                ))
        return classes
    
    def find_function(self, analysis: ASTAnalysis, func_name: str) -> FunctionInfo | None:
        """在分析结果中查找函数"""
        for func in analysis.functions:
            if func.name == func_name:
                return func
        return None
    
    def _get_decorator_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        return "unknown"
    
    def _get_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        return "unknown"
```

**面试要点：** `ast.parse()` 是 Python 标准库；`ast.walk()` 遍历 AST 节点；通过 `isinstance` 判断节点类型。

### Step 4：Diff Generator（30 分钟）

**`coding/diff_generator.py`：**
```python
import difflib

class DiffGenerator:
    """生成 unified diff"""
    
    def generate(self, file_path: str, original: str, modified: str) -> DiffResult:
        diff_lines = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        ))
        
        return DiffResult(
            file_path=file_path,
            original_content=original,
            modified_content=modified,
            diff_text="".join(diff_lines),
        )
```

### Step 5：Patch Apply（30 分钟）

**`coding/patch_apply.py`：**
```python
import subprocess
import tempfile
import os

class PatchApplier:
    """应用 unified diff patch"""
    
    def apply(self, diff_text: str, working_dir: str) -> PatchResult:
        """使用系统 patch 命令应用 diff"""
        # 写入临时 diff 文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            f.write(diff_text)
            diff_path = f.name
        
        try:
            result = subprocess.run(
                ["patch", "-p0", "-i", diff_path],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return PatchResult(
                success=result.returncode == 0,
                file_path=working_dir,
                hunks_applied=1 if result.returncode == 0 else 0,
                hunks_failed=0 if result.returncode == 0 else 1,
                error_message=result.stderr if result.returncode != 0 else None,
            )
        finally:
            os.unlink(diff_path)
```

### Step 6：注册 Coding 工具到 ToolRegistry（30 分钟）
- 将 `search_code` / `analyze_code` / `apply_patch` 包装为 BaseTool
- 注册到 Phase 2 的 ToolRegistry
- 更新 System Prompt 告知 Agent 可以使用代码分析工具

---

## 六、关键代码模式与伪代码

### 代码修改流程

```python
async def modify_code_with_understanding(request: str, repo: RepoContext) -> PatchResult:
    """
    Agent 修改代码的标准流程：
    1. 理解需求
    2. 搜索相关代码（grep + AST）
    3. 读取相关文件
    4. LLM 生成修改方案
    5. 生成 Diff
    6. 应用 Patch
    """
    
    # Step 1-2: 搜索
    searcher = CodeSearcher(repo)
    analyzer = ASTAnalyzer()
    
    # 用 LLM 从需求中提取搜索关键词
    search_terms = await llm_extract_search_terms(request)
    matches = searcher.search_by_pattern(search_terms.pattern)
    
    # Step 3: 读取相关文件 + AST 分析
    affected_files = set(m["file"] for m in matches)
    for file_path in affected_files:
        content = repo.file_contents[file_path]
        ast_info = analyzer.analyze_file(file_path, content)
        # 将 AST 信息注入 LLM 上下文
    
    # Step 4: LLM 生成修改
    context = build_context(matches, ast_infos)
    modified_code = await llm_generate_fix(context, request)
    
    # Step 5-6: Diff + Patch
    diff = DiffGenerator().generate(file_path, content, modified_code)
    patch_result = PatchApplier().apply(diff.diff_text, repo.root_path)
    
    return patch_result
```

---

## 七、完成标志

### 基本完成
- [ ] `RepositoryLoader` 能正确加载仓库文件树（排除 .gitignore 和二进制）
- [ ] `CodeSearcher` 能搜索代码（正则 + 文件名匹配）
- [ ] `ASTAnalyzer` 能解析 Python 文件，提取函数/类/导入信息
- [ ] `DiffGenerator` 能生成 unified diff
- [ ] `PatchApplier` 能应用 diff
- [ ] Agent 能执行"在 XX 函数中添加 YY 参数"这类跨文件修改

### 自测用例

```bash
# 测试 1：搜索代码
curl -X POST /api/chat -d '{"message": "在项目中找到所有定义 main 函数的地方"}'

# 测试 2：AST 分析
curl -X POST /api/chat -d '{"message": "分析 app/main.py 中有哪些函数和类"}'

# 测试 3：跨文件修改
curl -X POST /api/chat -d '{
  "message": "在 api/agent.py 的 chat 函数中添加一个新参数 timeout，默认值 30"
}'
```

---

## 八、文档差距分析

### 原文档缺失的内容及补充

| # | 原文档问题 | 实际开发需要 | 本文档补充位置 |
|---|-----------|-------------|-------------|
| 1 | **"AST 分析"范围未限定** | 仅 Python AST 可控；多语言需 tree-sitter，工作量×5。 | §1 |
| 2 | 没说 Repository Loader 的具体行为 | 需要文件过滤（.gitignore、二进制、隐藏目录、node_modules），否则 LLM Context 被垃圾文件淹没 | §5 Step 1 |
| 3 | 没说代码搜索的策略 | 需要同时支持正则搜索 + AST 结构化搜索 + 文件名搜索，三种互补 | §5 Step 2-3 |
| 4 | `diff_generator` vs `patch_apply` 没说用什么实现 | 使用 Python 标准库 `difflib.unified_diff` 生成 diff，系统 `patch` 命令应用 | §5 Step 4-5 |
| 5 | 没有说明修改流程 | Agent 修改代码不是一次性操作，是 Search→Read→Analyze→Generate→Diff→Apply 六步 | §6 |
| 6 | 没有提到文件编码处理 | `UnicodeDecodeError` 处理二进制/非 UTF-8 文件 | §5 Step 1 |
| 7 | 没有提到搜索结果截断 | 搜索结果必须限制数量，否则 LLM Context Window 溢出 | §5 Step 2 |

### 为什么不支持多语言 AST？

| 语言 | AST 工具 | 复杂度 |
|------|---------|--------|
| Python | `ast` (标准库) | ⭐ 低 |
| JavaScript/TypeScript | tree-sitter / babel | ⭐⭐⭐ 中 |
| Java | tree-sitter / javalang | ⭐⭐⭐ 中 |
| Go | tree-sitter / go/parser | ⭐⭐⭐ 中 |

**设计决策：** 先做 Python（面试够用），通过 grep 搜索支持其他语言的基本查找。如果需要，后续引入 tree-sitter 统一多语言（但这是独立 Phase 的工作量）。

---

## 九、面试考点映射

| 面试题 | 答案要点 | 本文档参考 |
|--------|---------|-----------|
| **AST 是什么？Agent 为什么需要它？** | AST = Abstract Syntax Tree，把源代码解析为树形结构。Agent 通过 AST 能**理解代码结构**（而不是盲目的文本匹配）——找到函数定义、参数列表、调用关系。Python 标准库 `ast` 模块可直接使用。 | §5 Step 3 |
| **Diff 生成和 Patch 应用的区别？** | Diff = 计算两个版本的差异（unified format）；Patch = 将差异应用到实际文件。Agent 先让 LLM 生成新代码，再用 diff 计算差异，最后 patch 应用。 | §5 Step 4-5 |
| **如何处理大仓库？** | 1) 文件过滤（.gitignore, 二进制）2) 搜索结果截断（top 50）3) LLM 上下文窗口管理（只给 Agent 看相关文件）。不做全量扫描。 | §5 Step 1-2 |
| **为什么不做语义搜索？** | 语义搜索（embedding）在 Phase 6 的 Memory System 中实现。Phase 4 先做基于正则+AST 的结构化搜索，两者互补：正则适合精确匹配，语义适合模糊匹配。 | §1 |
