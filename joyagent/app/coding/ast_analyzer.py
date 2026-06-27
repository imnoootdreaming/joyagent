"""
Phase 4 Step 3: AST Analyzer — Python AST 结构化代码分析。

ASTAnalyzer 是 Coding Agent 的"理解引擎"——它把 Python 源代码解析为
抽象语法树（AST），从中提取函数、类、导入、变量等结构化信息。
这种结构化理解远超 grep/正则的文本匹配能力。

核心能力：
  1. 解析单个 Python 文件 → 提取 导入/函数/类/顶层变量
  2. 仓库级批量分析 → 一次性分析所有 .py 文件
  3. 结构化查询     → 按名称查函数/类、按行号定位函数
  4. 调用链分析     → 找出函数内部调用了哪些其他函数
  5. LLM 上下文格式化 → 将分析结果转为紧凑文本供 LLM 消费

为什么只用 Python 标准库 `ast`？
  - 零依赖：ast 是 Python 内置模块
  - 精确：ast.parse() 是 CPython 解释器使用的同一解析器
  - 够用：Java/JS/TS 等语言通过 grep 搜索（Phase 4 §1 范围限定）

面试要点：
  - ast.parse(source): 源码 → AST 树
  - ast.walk(tree):    遍历所有节点（DFS，含嵌套）
  - isinstance(node, ast.Xxx): 判断节点类型
  - node.lineno / node.end_lineno: 行号定位
  - ast.get_docstring(node): 提取 docstring
  - Python ≥ 3.8 自动设置 end_lineno
"""

# ── Python 标准库 ──
import ast                             # 标准库 AST 解析器（CPython 同款）
from dataclasses import dataclass, field  # 数据类装饰器

# ── 项目内导入 ──
from app.coding.repository_loader import RepoContext, FileInfo
# RepoContext: 仓库文件列表 + 内容映射
# FileInfo:    单文件语言/大小信息


# ═══════════════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FunctionInfo:
    """
    单个函数/方法的结构化信息。

    通过 AST 解析提取，不是简单的正则匹配。包含函数签名、
    位置（起止行号）、装饰器、文档字符串等完整元数据。

    字段：
      name        — 函数名（如 "chat", "agent_loop"）
      start_line  — def 关键字所在行（1-based）
      end_line    — 函数体最后一行（ast 自动设置 end_lineno）
      args        — 参数名列表（如 ["self", "user_message", "history"]）
      decorators  — 装饰器列表（如 ["@router.post", "@staticmethod"]）
      docstring   — 文档字符串内容（无则为 None）
      is_async    — 是否为 async def
      is_method   — 是否在类内部定义（method）
      class_name  — 所属类名（仅 method 时有值，顶层函数为 None）
    """
    name: str                           # 函数名
    start_line: int                     # 定义首行（def 行）
    end_line: int                       # 定义末行（函数体结束）
    args: list[str]                     # 参数名列表
    decorators: list[str]               # 装饰器（如 "@app.get('/')"）
    docstring: str | None = None        # 文档字符串（无则为 None）
    is_async: bool = False              # async def?
    is_method: bool = False             # 类方法?
    class_name: str | None = None       # 所属类名（顶层函数为 None）


@dataclass
class ClassInfo:
    """
    单个类的结构化信息。

    包含类名、基类列表、位置信息、所有方法（含 __init__）等。
    """
    name: str                           # 类名（如 "Agent", "ToolRegistry"）
    start_line: int                     # class 关键字所在行
    end_line: int                       # 类体最后一行
    methods: list[FunctionInfo]         # 类内所有方法（含 __init__, dunder 等）
    base_classes: list[str]             # 基类名称列表（如 ["ABC", "BaseTool"]）
    decorators: list[str] = field(default_factory=list)  # 类装饰器（如 @dataclass）
    docstring: str | None = None        # 类文档字符串


@dataclass
class VariableInfo:
    """
    模块级（顶层）变量的结构化信息。

    变量在 AST 中是 ast.Assign 节点，提取 target 和简单值。
    仅记录顶层变量（不在函数/类内部）。
    """
    name: str                           # 变量名（如 "DEFAULT_MAX_TOKENS"）
    line_number: int                    # 赋值所在行
    value_preview: str                  # 值预览（截断，如 "4096", "\"hello\""）
    is_constant: bool                   # 是否常量——全大写蛇形命名


@dataclass
class ASTAnalysis:
    """
    单个 Python 文件的完整 AST 分析结果。

    这是 analyze_file() 的返回类型，是 CodeSearcher 的"结构化补充"——
    grep 告诉你"第 N 行匹配了正则"，AST 告诉你"第 N 行是 chat 函数的参数定义"。
    """
    file_path: str                      # 文件相对路径
    imports: list[str]                  # import 语句列表
    functions: list[FunctionInfo]       # 顶层函数列表
    classes: list[ClassInfo]            # 类列表
    variables: list[VariableInfo]       # 顶层变量列表
    has_syntax_error: bool = False      # 文件是否有语法错误
    syntax_error_msg: str = ""          # 语法错误描述（如有）

    # ── 便捷查询 ──────────────────────────────────────────────

    def find_function(self, name: str) -> FunctionInfo | None:
        """查找顶层函数。"""
        for f in self.functions:
            if f.name == name:
                return f
        return None

    def find_class(self, name: str) -> ClassInfo | None:
        """查找类。"""
        for c in self.classes:
            if c.name == name:
                return c
        return None

    def find_method(self, class_name: str, method_name: str) -> FunctionInfo | None:
        """查找指定类的指定方法。"""
        for c in self.classes:
            if c.name == class_name:
                for m in c.methods:
                    if m.name == method_name:
                        return m
        return None

    def get_function_at_line(self, line_number: int) -> FunctionInfo | None:
        """
        按行号查找包含该行的函数/方法。

        当 Agent 从 CodeSearcher 获得行号后，可以用此方法
        快速知道该行属于哪个函数。
        """
        for f in self.functions:
            if f.start_line <= line_number <= f.end_line:
                return f
        for c in self.classes:
            for m in c.methods:
                if m.start_line <= line_number <= m.end_line:
                    return m
        return None

    def find_import(self, module_name: str) -> str | None:
        """
        查找指定模块的 import 语句。

        Args: module_name 如 "fastapi"、"os.path"
        Returns: 完整的 import 语句 或 None
        """
        for imp in self.imports:
            if module_name in imp:
                return imp
        return None

    def get_all_function_names(self) -> list[str]:
        """获取文件中所有函数/方法的名称列表。"""
        names = [f.name for f in self.functions]
        for c in self.classes:
            names.extend(m.name for m in c.methods)
        return names

    def format_summary(self, max_items: int = 15) -> str:
        """
        将分析结果格式化为 LLM 可读的摘要文本。

        只输出摘要信息（名称 + 位置），不输出完整代码。
        Agent 可以基于此摘要决定需要进一步 read_file 查看哪些函数。
        """
        lines = [f"## AST Analysis: {self.file_path}"]

        # 导入
        if self.imports:
            lines.append(f"\n### Imports ({len(self.imports)})")
            for imp in self.imports[:5]:
                lines.append(f"  {imp}")
            if len(self.imports) > 5:
                lines.append(f"  ... and {len(self.imports) - 5} more")

        # 模块变量
        if self.variables:
            lines.append(f"\n### Variables ({len(self.variables)})")
            for v in self.variables[:5]:
                const_flag = " (const)" if v.is_constant else ""
                lines.append(f"  L{v.line_number:>4d}: {v.name} = {v.value_preview}{const_flag}")

        # 顶层函数
        if self.functions:
            lines.append(f"\n### Top-Level Functions ({len(self.functions)})")
            for f in self.functions[:max_items]:
                async_flag = "async " if f.is_async else ""
                args_str = ", ".join(f.args)
                lines.append(
                    f"  L{f.start_line:>4d}-{f.end_line:<4d}: "
                    f"{async_flag}def {f.name}({args_str})"
                )

        # 类 — 每个类列出方法
        if self.classes:
            lines.append(f"\n### Classes ({len(self.classes)})")
            for cls in self.classes[:max_items]:
                bases = f"({', '.join(cls.base_classes)})" if cls.base_classes else ""
                lines.append(f"  L{cls.start_line:>4d}-{cls.end_line:<4d}: "
                             f"class {cls.name}{bases}")
                for m in cls.methods[:8]:
                    async_flag = "async " if m.is_async else ""
                    lines.append(f"      L{m.start_line:>4d}: {async_flag}def {m.name}(...)")
                if len(cls.methods) > 8:
                    lines.append(f"      ... and {len(cls.methods) - 8} more methods")

        # 语法错误
        if self.has_syntax_error:
            lines.append(f"\n### ⚠ Syntax Error: {self.syntax_error_msg}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# ASTAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

class ASTAnalyzer:
    """
    Python AST 结构化分析器。

    只解析 Python 文件（其他语言的代码分析由 CodeSearcher 的 grep 搜索覆盖）。

    核心方法：
      analyze_file()    → 单文件分析，返回 ASTAnalysis
      analyze_repo()    → 仓库级批量分析，返回 dict[path, ASTAnalysis]
      get_callees()     → 提取函数内部调用的所有其他函数名

    安全设计：
      - ast.parse() 对无效语法抛出 SyntaxError → 捕获后标记 has_syntax_error
      - 不处理 10MB+ 的巨型文件（可在调用侧限制，这里不做检查）
      - Python 版本需 ≥ 3.8（end_lineno 需要）

    使用方式：
      from app.coding.ast_analyzer import ASTAnalyzer
      from app.coding.repository_loader import RepositoryLoader

      repo = RepositoryLoader(".").load()
      analyzer = ASTAnalyzer()
      analysis = analyzer.analyze_file("app/api/agent.py", repo.get_content("app/api/agent.py"))
      print(analysis.format_summary())
    """

    def analyze_file(self, file_path: str, source: str) -> ASTAnalysis:
        """
        分析单个 Python 文件。

        这是一次性解析——把源码送给 ast.parse() 得到的 tree
        用于提取所有信息（导入、函数、类、变量），不做重复遍历。

        Args:
            file_path: 文件相对路径（用于 ASTAnalysis.file_path）
            source:    Python 源代码字符串

        Returns:
            ASTAnalysis: 结构化分析结果。语法错误时 has_syntax_error=True。
        """
        # ── 0. 解析源码为 AST 树 ──────────────────────────────
        # ast.parse(source) 是 CPython 解释器实际使用的解析器
        # 对语法错误抛出 SyntaxError（我们捕获后标记而非崩溃）
        try:
            tree = ast.parse(source)       # 返回 ast.Module 根节点
        except SyntaxError as e:
            return ASTAnalysis(
                file_path=file_path,
                imports=[],
                functions=[],
                classes=[],
                variables=[],
                has_syntax_error=True,
                syntax_error_msg=f"{e.msg} (line {e.lineno})",
            )

        # ── 1. 提取各类信息 ──────────────────────────────────
        # 对同一棵 AST 树做四次遍历（各提取一种信息）
        # 复杂度 O(4N) = O(N)，对 Python 文件完全可接受
        imports = self._extract_imports(tree)      # import / from-import
        functions = self._extract_functions(tree)   # 顶层 async def / def
        classes = self._extract_classes(tree)       # class 定义
        variables = self._extract_variables(tree)   # 顶层变量赋值

        return ASTAnalysis(
            file_path=file_path,
            imports=imports,
            functions=functions,
            classes=classes,
            variables=variables,
        )

    def analyze_repo(self, repo: RepoContext) -> dict[str, ASTAnalysis]:
        """
        批量分析仓库中所有 Python 文件。

        只处理 repo.get_python_files() 返回的文件（已知为 Python），
        跳过其他语言文件。

        Args:
            repo: 已加载的仓库上下文

        Returns:
            dict {file_path: ASTAnalysis} — 按文件路径索引的分析结果
            语法错误的文件也在其中（has_syntax_error=True）

        Example:
            repo = RepositoryLoader(".").load()
            analyses = ASTAnalyzer().analyze_repo(repo)
            for path, analysis in analyses.items():
                print(f"{path}: {len(analysis.functions)} functions")
        """
        results: dict[str, ASTAnalysis] = {}
        for py_file in repo.get_python_files():
            content = repo.get_content(py_file.path)
            if content is None:
                continue                 # 文件无内容 → 跳过
            results[py_file.path] = self.analyze_file(py_file.path, content)
        return results

    def get_callees(self, analysis: ASTAnalysis,
                    func_name: str) -> list[str]:
        """
        提取指定函数内部调用的所有函数名。

        这是"调用链分析"的核心——当 Agent 需要修改函数签名时，
        先通过 get_callees 了解该函数调用了谁，再通过 CodeSearcher.search_callers
        了解谁调用了它。综合两者完成完整的"影响分析"。

        Args:
            analysis:  该文件的分析结果
            func_name: 目标函数名

        Returns:
            该函数内部直接调用的函数名称列表（去重，按字母序）

        Example:
            # Agent 想知道 chat() 函数的内部调用了哪些函数
            callees = analyzer.get_callees(analysis, "chat")
            # → ["_build_initial_state", "_extract_final_response", "_should_use_workflow"]
        """
        # 1. 查找函数位置
        func = analysis.find_function(func_name)
        if func is None:
            # 可能是类方法 → 在所有类的 methods 中查找
            for cls in analysis.classes:
                for m in cls.methods:
                    if m.name == func_name:
                        func = m
                        break
                if func:
                    break

        if func is None:
            return []                    # 函数不存在

        # 2. 从源码中提取函数体并解析为 AST
        #    更精确的做法：直接从原始 AST 树的对应节点提取 Call 节点
        #    但考虑到 analyze_file 不保存原始 tree，这里用行号范围重新解析
        source = self._get_source_by_lines(analysis.file_path, func.start_line, func.end_line)
        if source is None:
            return []

        try:
            tree = ast.parse(source)     # 只解析函数体
        except SyntaxError:
            return []

        # 3. 遍历 AST，提取所有 ast.Call 节点的函数名
        callees: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = self._get_call_name(node)
                if name:
                    callees.add(name)

        # 排除自身（递归调用不计入）
        callees.discard(func_name)
        return sorted(callees)

    # ── 私有：提取方法 ──────────────────────────────────────

    def _extract_imports(self, tree: ast.Module) -> list[str]:
        """
        提取所有 import 语句。

        处理两种格式：
          import module              → "import module"
          from module import a, b    → "from module import a, b"
        """
        imports: list[str] = []
        for node in ast.walk(tree):
            # ast.Import = "import os, sys" 格式
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # alias.name = 模块名, alias.asname = 别名（如 "import numpy as np"）
                    if alias.asname:
                        imports.append(f"import {alias.name} as {alias.asname}")
                    else:
                        imports.append(f"import {alias.name}")

            # ast.ImportFrom = "from X import Y" 格式
            elif isinstance(node, ast.ImportFrom):
                names = ", ".join(
                    f"{a.name} as {a.asname}" if a.asname else a.name
                    for a in node.names
                )
                # node.module 是 "from X" 中的 X（None 表示相对导入）
                module_prefix = node.module or "?"
                imports.append(f"from {module_prefix} import {names}")

        return imports

    def _extract_functions(self, tree: ast.Module) -> list[FunctionInfo]:
        """
        提取所有顶层函数定义（不含类方法）。

        只取 ast.Module.body 的直接子节点中的函数——类方法由 _extract_classes 处理。
        使用 ast.iter_child_nodes 而非 ast.walk，避免递归到类内部。

        处理 async def（ast.AsyncFunctionDef）和普通 def（ast.FunctionDef）。
        """
        functions: list[FunctionInfo] = []
        # iter_child_nodes: 只遍历直接子节点，不递归
        # Module.body = [Func1, Class1, Func2, Assign1, ...]
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(self._build_function_info(node))
        return functions

    def _extract_classes(self, tree: ast.Module) -> list[ClassInfo]:
        """
        提取所有类定义及其内部方法。

        对每个类：提取名称、基类、行号、装饰器、docstring，
        然后遍历类体提取所有方法（含 __init__、@property 等）。
        """
        classes: list[ClassInfo] = []
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.ClassDef):
                continue                 # 非类定义 → 跳过

            # ── 提取方法 ──
            methods: list[FunctionInfo] = []
            for item in node.body:       # node.body 是类体的语句列表
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fi = self._build_function_info(item, is_method=True,
                                                  class_name=node.name)
                    methods.append(fi)

            # ── 提取基类名称 ──
            bases = [self._get_name_str(b) for b in node.bases]

            # ── 提取类装饰器 ──
            decorators = [
                self._get_decorator_str(d) for d in node.decorator_list
            ]

            classes.append(ClassInfo(
                name=node.name,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                methods=methods,
                base_classes=bases,
                decorators=decorators,
                docstring=ast.get_docstring(node),
            ))

        return classes

    def _extract_variables(self, tree: ast.Module) -> list[VariableInfo]:
        """
        提取模块级顶层变量赋值。

        只提取 ast.Assign 节点（如 `x = 1`），不提取函数/类内变量。
        包含 AnnAssign（类型注解赋值 `x: int = 1`）。

        值预览策略：
          - 简单字面量 → 直接显示（如 "4096", "\"hello\"", "True"）
          - 表达式 → 显示为 "<expression>"
          - 函数调用 → 显示为 "<call:func()>"
        """
        variables: list[VariableInfo] = []
        for node in ast.iter_child_nodes(tree):
            # ast.Assign: x = value
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    name = self._get_name_str(target)
                    if name:
                        variables.append(VariableInfo(
                            name=name,
                            line_number=node.lineno,
                            value_preview=self._preview_value(node.value),
                            is_constant=name.isupper() and "_" in name,
                        ))

            # ast.AnnAssign: x: int = value (类型注解赋值)
            elif isinstance(node, ast.AnnAssign):
                name = self._get_name_str(node.target)
                if name:
                    variables.append(VariableInfo(
                        name=name,
                        line_number=node.lineno,
                        value_preview=(
                            self._preview_value(node.value)
                            if node.value else "<no value>"
                        ),
                        is_constant=name.isupper() and "_" in name,
                    ))

        return variables

    # ── 私有：辅助方法 ──────────────────────────────────────

    def _build_function_info(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_method: bool = False,
        class_name: str | None = None,
    ) -> FunctionInfo:
        """
        从 AST 函数节点构建 FunctionInfo。

        统一处理普通函数、async 函数、类方法。
        """
        # 提取参数名：只取显式参数，跳过 *args 和 **kwargs
        args = []
        for arg in node.args.args:
            args.append(arg.arg)         # arg.arg 是参数名字符串（如 "self", "path"）

        # 如果有 *args (vararg)
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")

        # 如果有 **kwargs (kwarg)
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        # 提取装饰器名称
        decorators = [
            self._get_decorator_str(d) for d in node.decorator_list
        ]

        return FunctionInfo(
            name=node.name,
            start_line=node.lineno,      # def 关键字所在行
            end_line=node.end_lineno or node.lineno,
            args=args,
            decorators=decorators,
            docstring=ast.get_docstring(node),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=is_method,
            class_name=class_name,
        )

    def _get_decorator_str(self, node) -> str:
        """
        将装饰器 AST 节点转为字符串表示。

        处理三种常见装饰器格式：
          @name             → "@name"
          @name()           → "@name()"
          @name(arg="val")  → "@name(arg=val)"

        复杂表达式（如 @app.get("/path")）通过 ast.unparse 还原源码文本。
        Python ≥ 3.9 有 ast.unparse()，更早版本降级为 "unknown"。
        """
        # 简单装饰器：@staticmethod, @abstractmethod
        if isinstance(node, ast.Name):
            return f"@{node.id}"

        # 带参数的装饰器：@app.get("/path"), @router.post("...")
        if isinstance(node, ast.Call):
            func_name = self._get_name_str(node.func)
            return f"@{func_name}(...)"

        # 属性访问装饰器：@decorator.arg
        if isinstance(node, ast.Attribute):
            return f"@{self._get_attr_path(node)}"

        # 通用降级：尝试 Python 3.9+ 的 ast.unparse()
        try:
            return f"@{ast.unparse(node)}"
        except (AttributeError, Exception):
            return "@unknown"

    def _get_name_str(self, node) -> str:
        """
        从 AST 节点提取名称字符串。

        覆盖常见的名称载体：
          ast.Name(id="chat")     → "chat"
          ast.Attribute(...)      → "module.attr"
          ast.Subscript(...)      → "dict[key]"
          其他                    → ""
        """
        if isinstance(node, ast.Name):
            return node.id               # 简单名称：如 "x", "MyClass"
        if isinstance(node, ast.Attribute):
            return self._get_attr_path(node)  # 属性链：如 "self.name"
        if isinstance(node, ast.Subscript):
            # 下标访问：如 d["key"] → 取被下标的变量名
            return self._get_name_str(node.value)
        if isinstance(node, ast.Call):
            # 函数调用作为 target（少见，如 func() = x）
            return self._get_name_str(node.func)
        return ""

    def _get_attr_path(self, node: ast.Attribute) -> str:
        """
        还原属性访问链为字符串。

        ast.Attribute(value=Name(id="os"), attr="path") → "os.path"
        递归处理多层嵌套：a.b.c → "a.b.c"
        """
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        # 最后一项（最底层）
        if isinstance(current, ast.Name):
            parts.append(current.id)
        parts.reverse()                  # 从根到叶
        return ".".join(parts)

    def _get_call_name(self, node: ast.Call) -> str | None:
        """
        从 ast.Call 节点提取被调用的函数名。

        处理三种调用格式：
          func()        → "func"
          obj.method()  → "obj.method"
          module.func() → "module.func"
        """
        func = node.func
        if isinstance(func, ast.Name):
            return func.id               # 直接调用：func()
        if isinstance(func, ast.Attribute):
            return self._get_attr_path(func)  # 方法调用：obj.method()
        return None

    def _preview_value(self, node) -> str:
        """
        生成值的简短预览文本。

        用于 VariableInfo.value_preview，让 Agent 快速了解变量值。
        不会生成超过 50 字符的预览。
        """
        if node is None:
            return "None"
        # 字面量常量
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, str):
                if len(val) > 40:
                    return repr(val[:37] + "...")  # 长字符串截断
                return repr(val)         # repr 保留引号
            return str(val)              # 数字/布尔/None
        # 函数调用
        if isinstance(node, ast.Call):
            name = self._get_call_name(node) or "?"
            return f"<call:{name}()>"
        # 列表字面量
        if isinstance(node, ast.List):
            return f"<list[{len(node.elts)}]>"
        # 字典字面量
        if isinstance(node, ast.Dict):
            return f"<dict[{len(node.keys)}]>"
        # 其他表达式
        return "<expression>"

    def _get_source_by_lines(self, file_path: str,
                             start: int, end: int) -> str | None:
        """
        从仓库中提取指定行号范围的源码。

        用于 get_callees() 方法——只解析函数体部分。

        Args:
            file_path: 文件路径
            start:     起始行号（1-based）
            end:       结束行号（1-based）

        Returns:
            指定行范围的源码字符串，或 None（文件不存在）
        """
        # 注：ASTAnalyzer 不持有 RepoContext 引用，所以这里用文件系统读取
        # 实际使用中应由调用方传入 repo 或 content
        # 但为了独立性，这里回退到文件系统读取
        import os
        abs_path = os.path.join(os.getcwd(), file_path)
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
            # 行号是 1-based → 列表索引是 0-based
            selected = all_lines[start - 1:end]
            return "".join(selected)
        except (FileNotFoundError, PermissionError):
            return None
