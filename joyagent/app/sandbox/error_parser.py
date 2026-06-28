"""
Phase 5 Step 4: Error Parser — 结构化解析 Python traceback。

ErrorParser 是 Fix Loop 的"诊断引擎"——它将 pytest 输出的原始 traceback
文本解析为结构化的 TestFailure 列表，并按错误类型分类，为不同错误类型
生成有针对性的修复 Prompt。

为什么需要结构化解析？
  1. 原始 traceback 可能有几百行，LLM 直接看难以快速定位根因
  2. 不同错误类型需要不同的修复策略（Syntax → 重写, Import → 安装依赖, ...）
  3. 结构化信息让 Fix Loop 可以做精准决策——先修哪类错误、用什么 Prompt

错误分类 → 修复策略映射：
  syntax    — SyntaxError / IndentationError        → LLM 重写代码
  import    — ImportError / ModuleNotFoundError     → pip install 或修正导入路径
  assertion — AssertionError                        → LLM 分析逻辑差异
  runtime   — NameError / TypeError / AttributeError → LLM 上下文补全
  timeout   — TimeoutError                          → 优化性能或拆分测试
  unknown   — 其他未分类错误                          → 通用修复 Prompt

使用方式：
  parser = ErrorParser()
  failures = parser.parse(stderr_text)
  for f in failures:
      category = parser.categorize_error(f)
      fix_prompt = parser.generate_fix_prompt(f, category, source_code)
"""
from __future__ import annotations

# ── Python 标准库 ──
import re                              # 正则解析 traceback 行
from dataclasses import dataclass      # 数据类

# ── 项目内导入 ──
from app.sandbox.security import TestFailure  # 测试失败数据模型


# ═══════════════════════════════════════════════════════════════════════════════
# ErrorParser — traceback 结构化解析器
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorParser:
    """
    Python traceback 结构化解析器。

    职责：
      1. 从 pytest/raw stderr 中解析出所有错误 → list[TestFailure]
      2. 按错误类型分类 → 确定修复策略
      3. 生成分类修复 Prompt → 传给 Fix Loop 的 LLM 修复步骤
      4. 错误模式检测 → 识别重复/同一根因的多个错误

    支持的 traceback 来源：
      - pytest -v 输出（每个测试一行状态 + FAILURES/ERRORS 区块）
      - 原始 Python traceback（python test.py 崩溃输出）
      - Docker 容器 stderr（可能包含 ANSI 转义 + 容器元信息）
    """

    # ── traceback 解析正则 — 多个模式覆盖不同格式 ──────────────

    # 模式 1: Python 标准 traceback
    # File "path/to/file.py", line 42, in function_name
    #     broken_code()
    # ErrorType: error message
    TRACEBACK_PATTERN = re.compile(
        r'File\s+"(?P<file>[^"]+)",\s+line\s+(?P<line>\d+)'
        r'(?:,\s+in\s+(?P<func>\w+))?'
        r'\n\s*(?P<code>[^\n]+)'               # 代码行（单行，不用 DOTALL）
        r'\n(?P<error_type>\w+(?:Error|Exception|Warning|Interrupt)):'
        r'\s*(?P<message>[^\n]*)',             # 消息行（单行）
    )

    # 模式 2: pytest FAILURES 区块
    # FAILED tests/test_x.py::test_name - AssertionError: message
    PYTEST_FAILURE_PATTERN = re.compile(
        r'(?P<status>FAILED|ERROR)\s+'
        r'(?P<file>[^ ]*)::(?P<test_name>[^ ]+)'
        r'\s*-\s*'
        r'(?P<error_type>\w+(?:Error|Exception|Warning|Error|Interrupt)):?'
        r'\s*(?P<message>.*)',
    )

    # 模式 3: 简化单行错误
    # ErrorType: message  (无 traceback)
    SIMPLE_ERROR_PATTERN = re.compile(
        r'(?P<error_type>\w+(?:Error|Exception|Warning)):\s*(?P<message>.*)',
    )

    # 模式 4: 行号提取（宽松版）
    # line 42  /  :42  /  , line 42
    LINE_NUMBER_PATTERN = re.compile(
        r'(?:line\s+|:|\s+L)(\d+)',
        re.IGNORECASE,
    )

    # ── 错误分类映射 ──
    # 每种错误类型 → 分类标签 + 修复策略描述

    ERROR_CATEGORIES: dict[str, tuple[str, str]] = {
        # ── 语法类：需要重写代码 ──
        "SyntaxError": ("syntax", "代码存在语法错误，需要检查括号匹配、缩进、冒号等。"),
        "IndentationError": ("syntax", "代码缩进错误——混用了空格和 Tab，或缩进层级不对。"),
        "TabError": ("syntax", "混用了 Tab 和空格缩进，需要统一。"),

        # ── 导入类：依赖缺失或路径错误 ──
        "ImportError": ("import", "模块导入失败——检查模块名是否正确、是否在正确的 Python 环境中。"),
        "ModuleNotFoundError": ("import", "模块未安装或路径错误——需要 pip install 或修正 sys.path。"),

        # ── 断言类：逻辑错误 ──
        "AssertionError": ("assertion", "测试断言失败——期望值和实际值不一致，需要分析逻辑差异。"),

        # ── 运行时类 ──
        "NameError": ("runtime", "变量/函数未定义——可能拼写错误或缺少 import。"),
        "TypeError": ("runtime", "类型错误——传入了错误类型的参数，或对错误类型调用了方法。"),
        "AttributeError": ("runtime", "属性不存在——对象没有该属性/方法，可能需要用其他 API。"),
        "ValueError": ("runtime", "值错误——传入的值合法但不合理（如负数传给正数参数）。"),
        "IndexError": ("runtime", "索引越界——访问了列表/元组不存在的位置。"),
        "KeyError": ("runtime", "键不存在——字典中没有指定的 key。"),
        "ZeroDivisionError": ("runtime", "除零错误——需要添加分母非零检查。"),
        "FileNotFoundError": ("runtime", "文件不存在——需要检查路径或先创建文件。"),
        "PermissionError": ("runtime", "权限不足——用非 root 用户操作需要权限的文件。"),

        # ── 超时类 ──
        "TimeoutError": ("timeout", "代码执行超时——可能包含死循环或处理数据量过大。"),

        # ── 其他 ──
        "RecursionError": ("runtime", "递归深度超限——可能是无限递归或递归终止条件有问题。"),
        "MemoryError": ("runtime", "内存不足——代码分配了过多内存，需要优化。"),
        "StopIteration": ("runtime", "迭代器耗尽——在已耗尽的迭代器上调用了 next()。"),
    }

    # ── 公共 API ──────────────────────────────────────────────

    def parse(self, stderr: str) -> list[TestFailure]:
        """
        从 stderr 文本中解析出所有测试失败。

        解析流程（由精确到宽松）：
          1. 标准 traceback 模式 (File "..." line N → ErrorType: message)
          2. pytest 摘要模式 (FAILED path::name - ErrorType: message)
          3. 纯错误消息（无 traceback，只有 ErrorType: message）
          4. 无法匹配任何模式 → 作为单个 "Unknown" 错误返回

        Args:
            stderr: 完整的 stderr 文本（pytest 输出或 Python traceback）

        Returns:
            list[TestFailure] — 按出现顺序排列的失败详情列表
            如果未识别到任何错误，返回空列表

        Example:
            parser = ErrorParser()
            failures = parser.parse(pytest_output.stderr)
            for f in failures:
                print(f"{f.test_name}: {f.error_type} at L{f.line_number}")
        """
        if not stderr or not stderr.strip():
            return []

        # ── 1. 清除 ANSI 转义序列 ──
        clean = self._strip_ansi(stderr)

        # ── 2. 尝试标准 traceback 模式 ──
        failures = self._parse_traceback(clean)

        # ── 3. 如果标准模式没匹配到，尝试 pytest 摘要模式 ──
        if not failures:
            failures = self._parse_pytest_summary(clean)

        # ── 4. 兜底：纯文本字符串匹配 ──
        if not failures:
            failures = self._parse_simple_errors(clean)

        return failures

    def categorize_error(self, failure: TestFailure) -> str:
        """
        将错误分类为修复策略类别。

        Args:
            failure: 单个测试失败详情

        Returns:
            分类标签: "syntax" | "import" | "assertion" | "runtime" | "timeout" | "unknown"

        Fix Loop 根据分类标签选择不同的修复策略：
          - syntax    → LLM 重写代码（检查括号/缩进/语法）
          - import    → 提示 LLM 安装依赖或修正导入
          - assertion → LLM 分析逻辑差异
          - runtime   → LLM 上下文补全（变量定义/类型检查）
          - timeout   → 提示 LLM 优化代码或拆分测试
          - unknown   → 通用修复 Prompt
        """
        error_type = failure.error_type
        if error_type in self.ERROR_CATEGORIES:
            return self.ERROR_CATEGORIES[error_type][0]
        return "unknown"

    def get_error_description(self, failure: TestFailure) -> str:
        """获取错误的详细修复建议描述。"""
        error_type = failure.error_type
        if error_type in self.ERROR_CATEGORIES:
            return self.ERROR_CATEGORIES[error_type][1]
        return "未知错误类型，需要人工分析。"

    def categorize_batch(
        self,
        failures: list[TestFailure],
    ) -> dict[str, list[TestFailure]]:
        """
        将一批失败测试按类别分组。

        Returns:
            {"syntax": [...], "import": [...], "assertion": [...], ...}

        Fix Loop 可以据此决定修复优先级（先修 syntax → 因为它会阻塞所有测试）。
        """
        groups: dict[str, list[TestFailure]] = {
            "syntax": [], "import": [], "assertion": [],
            "runtime": [], "timeout": [], "unknown": [],
        }
        for f in failures:
            cat = self.categorize_error(f)
            groups[cat].append(f)
        return groups

    def generate_fix_prompt(
        self,
        failure: TestFailure,
        source_code: str,
    ) -> str:
        """
        为单个错误生成针对性的 LLM 修复 Prompt。

        不同错误类型使用不同的 Prompt 模板，引导 LLM
        使用正确的修复策略。

        Args:
            failure:     单个测试失败
            source_code: 相关的源代码（用于 LLM 修复上下文）

        Returns:
            分类修复 Prompt 文本
        """
        category = self.categorize_error(failure)
        description = self.get_error_description(failure)

        base = (
            f"The following test failed:\n\n"
            f"  Test:      {failure.test_name}\n"
            f"  Error:     {failure.error_type}: {failure.error_message}\n"
            f"  File:      {failure.file_path}\n"
            f"  Line:      {failure.line_number}\n"
            f"  Category:  {category} — {description}\n"
        )

        if source_code:
            base += f"\n## Source Code\n```python\n{source_code}\n```\n"

        # ── 分类提示 ──
        if category == "syntax":
            base += (
                "\n## Fix Strategy (syntax)\n"
                "This is a syntax/indentation error. "
                "Please fix the syntax: check brackets, colons, indentation consistency. "
                "Output ONLY the corrected code — no explanation needed."
            )
        elif category == "import":
            base += (
                "\n## Fix Strategy (import)\n"
                "This is an import error. Two possible fixes:\n"
                "1. If it's a standard/third-party package → add to requirements.txt "
                "or use execute_shell to pip install.\n"
                "2. If it's a project module → check the import path and fix the "
                "module reference.\n"
                "Output the fix and explain which approach you took."
            )
        elif category == "assertion":
            base += (
                "\n## Fix Strategy (assertion)\n"
                "This is an assertion failure — expected vs actual mismatch.\n"
                "1. Read the test to understand what it expects.\n"
                "2. Read the source code to understand what it actually produces.\n"
                "3. Determine: is the code wrong or is the test wrong?\n"
                "4. Fix whichever is incorrect.\n"
                "Output the fixed code and briefly explain the logic issue."
            )
        elif category == "runtime":
            base += (
                "\n## Fix Strategy (runtime)\n"
                "This is a runtime error ({error_type}).\n"
                "Check: variable definitions, type correctness, edge cases (None, empty list, etc.). "
                "Add necessary guards (if/else, try/except) and fix the root cause."
            ).format(error_type=failure.error_type)
        elif category == "timeout":
            base += (
                "\n## Fix Strategy (timeout)\n"
                "The code timed out — likely infinite loop or excessive computation.\n"
                "Check: loop termination conditions, input size, algorithm complexity. "
                "Optimize or split into smaller test cases."
            )
        else:
            base += (
                "\n## Fix Strategy (unknown)\n"
                "Analyze the error message and traceback, then fix the root cause. "
                "Output the fixed code."
            )

        return base

    def generate_batch_fix_prompt(
        self,
        failures: list[TestFailure],
        source_code: str,
    ) -> str:
        """
        为一组错误生成批量修复 Prompt。

        按修复优先级排序（syntax → import → runtime → assertion → timeout），
        同类错误合并为一个修复任务。

        Args:
            failures:    失败列表（通常是一次 pytest run 的全部失败）
            source_code: 相关源代码

        Returns:
            批量修复 Prompt 文本
        """
        groups = self.categorize_batch(failures)
        total = len(failures)

        prompt_parts = [
            f"## Fix Task: {total} test failure(s) to fix\n",
        ]

        # ── 按优先级输出 ──
        priority_order = ["syntax", "import", "runtime", "assertion", "timeout", "unknown"]
        for cat in priority_order:
            cat_failures = groups.get(cat, [])
            if not cat_failures:
                continue

            prompt_parts.append(f"### {cat.upper()} errors ({len(cat_failures)})\n")
            for f in cat_failures:
                prompt_parts.append(
                    f"  - {f.test_name}: {f.error_type}: {f.error_message[:120]}\n"
                )
            prompt_parts.append("")

        if source_code:
            prompt_parts.append(f"## Source Code\n```python\n{source_code}\n```\n")

        prompt_parts.append(
            "## Instructions\n"
            "Fix ALL failures. Prioritize by category:\n"
            "1. SYNTAX first (blocking all other fixes)\n"
            "2. IMPORT next (unblock dependencies)\n"
            "3. RUNTIME next (fix variable/type issues)\n"
            "4. ASSERTION last (logic fixes — easier after other errors are cleared)\n"
            "\nOutput the fixed code and list what you changed."
        )

        return "".join(prompt_parts)

    def detect_root_cause(
        self,
        failures: list[TestFailure],
    ) -> str | None:
        """
        检测是否存在同一根因导致多个测试失败。

        判断规则：
          - 同一文件中多个相同类型的错误 → 可能是同一根因
          - 例如：修改函数签名后，所有调用它的测试都抛 TypeError

        Returns:
            根因分析文本（如 "3/5 failures in test_calc.py are AttributeError — "
            "likely caused by a missing/renamed method."），无明确根因时返回 None
        """
        if len(failures) <= 1:
            return None

        # 按 (file, error_type) 分组
        groups: dict[tuple, list] = {}
        for f in failures:
            key = (f.file_path, f.error_type)
            groups.setdefault(key, []).append(f)

        # 最大组
        max_key, max_group = max(groups.items(), key=lambda kv: len(kv[1]))
        ratio = len(max_group) / len(failures)

        if ratio >= 0.5 and len(max_group) >= 2:
            return (
                f"{len(max_group)}/{len(failures)} failures in "
                f"'{max_key[0]}' are {max_key[1]} — "
                f"likely caused by a single root cause."
            )

        return None

    # ── 私有解析器 ──────────────────────────────────────────

    def _parse_traceback(self, stderr: str) -> list[TestFailure]:
        """使用标准 Python traceback 模式解析。"""
        failures: list[TestFailure] = []

        for match in self.TRACEBACK_PATTERN.finditer(stderr):
            file_path = match.group("file")
            line_number = int(match.group("line"))
            func_name = match.group("func") or ""
            code = match.group("code").strip()
            error_type = match.group("error_type")
            error_message = match.group("message").strip()

            # 从函数名或文件名推断测试名
            test_name = func_name if func_name else file_path

            failures.append(TestFailure(
                test_name=test_name,
                error_type=error_type,
                error_message=error_message,
                file_path=file_path,
                line_number=line_number,
                traceback=self._extract_relevant_traceback(
                    stderr, file_path, line_number, error_type
                ),
            ))

        return failures

    def _parse_pytest_summary(self, stderr: str) -> list[TestFailure]:
        """
        从 pytest 的 FAILURES/ERRORS 摘要中解析。

        pytest -v 输出的典型格式（在 test status line 之后）：
        _____ test_name _____
        ...
        E   ErrorType: message

        或者：
        FAILED path::test_name - ErrorType: message
        """
        failures: list[TestFailure] = []

        # ── 模式 A: pytest --tb=short 摘要 ──
        for match in self.PYTEST_FAILURE_PATTERN.finditer(stderr):
            file_path = match.group("file")
            test_name = match.group("test_name")
            error_type = match.group("error_type")
            error_message = match.group("message").strip()

            # 提取行号（从上下文中搜索）
            line_number = self._extract_line_number(stderr, file_path)

            failures.append(TestFailure(
                test_name=test_name,
                error_type=error_type,
                error_message=error_message,
                file_path=file_path,
                line_number=line_number,
                traceback=self._extract_relevant_traceback(
                    stderr, file_path, line_number, error_type
                ),
            ))

        # ── 模式 B: pytest --tb=long 格式 ──
        if not failures:
            failures = self._parse_pytest_long_format(stderr)

        return failures

    def _parse_pytest_long_format(self, stderr: str) -> list[TestFailure]:
        """
        解析 pytest --tb=long 格式的长 traceback。

        格式：
          _____ test_name _____
          ...
          file.py:42: AssertionError
          E   assert 5 == 4
        """
        failures: list[TestFailure] = []
        # 找到所有 test section
        test_sections = re.split(r'(?:_{3,}|={3,})\s*(\S+)\s*(?:_{3,}|={3,})', stderr)
        # test_sections[0] 是第一个 section 之前的内容
        # test_sections[1], [3], [5] ... 是 test names
        # test_sections[2], [4], [6] ... 是 section content

        for i in range(1, len(test_sections) - 1, 2):
            test_name = test_sections[i].strip()
            content = test_sections[i + 1] if i + 1 < len(test_sections) else ""

            # 在 content 中搜索错误信息
            error_match = re.search(
                r'(?:E\s+)?(\w+(?:Error|Exception|Warning)):\s*(.+)',
                content,
                re.MULTILINE,
            )
            if not error_match:
                continue

            error_type = error_match.group(1)
            error_message = error_match.group(2).strip()

            # 提取文件路径和行号
            file_match = re.search(
                r'(?:^\s*|E\s+)([^\s:]+\.py):(\d+):',
                content,
                re.MULTILINE,
            )
            file_path = file_match.group(1) if file_match else ""
            line_number = int(file_match.group(2)) if file_match else None

            failures.append(TestFailure(
                test_name=test_name,
                error_type=error_type,
                error_message=error_message[:500],
                file_path=file_path,
                line_number=line_number,
                traceback=content[:2000],
            ))

        return failures

    def _parse_simple_errors(self, stderr: str) -> list[TestFailure]:
        """
        兜底解析：纯错误消息文本（无完整 traceback）。

        处理 LLM 可能只输出 "ErrorType: message" 而没有堆栈的情况。
        或处理 Docker 环境的简化错误输出。
        """
        failures: list[TestFailure] = []

        for match in self.SIMPLE_ERROR_PATTERN.finditer(stderr):
            error_type = match.group("error_type")
            error_message = match.group("message").strip()

            line_number = self._extract_line_number(stderr)

            failures.append(TestFailure(
                test_name=f"({error_type})",
                error_type=error_type,
                error_message=error_message[:500],
                file_path="",
                line_number=line_number,
                traceback=stderr[:2000],
            ))

        # 如果完全没有任何错误匹配，但有内容 → 作为 Unknown
        if not failures and stderr.strip():
            failures.append(TestFailure(
                test_name="(unknown)",
                error_type="Unknown",
                error_message=stderr.strip()[:500],
                file_path="",
                line_number=None,
                traceback=stderr[:2000],
            ))

        return failures

    # ── 辅助方法 ──────────────────────────────────────────────

    def _extract_line_number(
        self,
        text: str,
        file_path: str | None = None,
    ) -> int | None:
        """
        从文本中提取行号。如果在 context 中指定了 file_path，
        优先匹配该文件附近的行号。
        """
        if file_path:
            # 搜索 "file_path:NN" 或 "file_path\", line NN"
            escaped = re.escape(file_path)
            m = re.search(
                rf'{escaped}[:\"],\s*line\s+(\d+)',
                text,
                re.IGNORECASE,
            )
            if m:
                return int(m.group(1))

        # 通用搜索
        m = self.LINE_NUMBER_PATTERN.search(text)
        return int(m.group(1)) if m else None

    def _extract_relevant_traceback(
        self,
        full_text: str,
        file_path: str,
        line_number: int | None,
        error_type: str,
    ) -> str:
        """
        从完整 stderr 中提取与此错误相关的 traceback 片段。

        不返回完整 stderr（太长），而是提取以该文件路径为中心的上下文。
        包含：File "..." 行 → 代码行 → 错误行。
        """
        if not file_path:
            return full_text[:2000]   # 无文件路径 → 返回全文前 2000 字符

        # 找到该文件的 traceback 起始位置
        escaped = re.escape(file_path)
        start_idx = full_text.find(f'File "{file_path}"')

        if start_idx < 0:
            # 尝试其他格式
            pattern = re.compile(rf'{escaped}[:\"]\D*(\d+)')
            m = pattern.search(full_text)
            if m:
                start_idx = m.start()
            else:
                # 找不到 → 返回错误附近的文本
                err_idx = full_text.find(error_type)
                if err_idx >= 0:
                    return full_text[max(0, err_idx - 200):err_idx + 1800]
                return full_text[:2000]

        # 从 File "..." 行开始，找到下一个 File "..." 或错误结束
        end_idx = start_idx + 2000
        next_file = full_text.find('\nFile "', start_idx + 1)
        if next_file > 0 and next_file < end_idx:
            end_idx = next_file

        return full_text[start_idx:end_idx]

    @staticmethod
    def _strip_ansi(text: str) -> str:
        """
        清除 ANSI 转义序列（终端颜色码）。

        Docker 日志或 pytest 输出可能包含 ANSI 颜色码，
        这些序列在正则匹配中会产生噪音。
        """
        ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
        return ansi_escape.sub('', text)
