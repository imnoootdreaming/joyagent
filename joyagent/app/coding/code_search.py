"""
Phase 4 Step 2: Code Search — 代码搜索引擎。

CodeSearcher 是 Coding Agent 的"搜索引擎"——在 RepoContext 加载的仓库内容中，
用正则/关键词/文件名匹配来定位相关代码片段。

与 Phase 1-2 的 read_file 工具的区别：
  - read_file:    盲读整个文件 → 大量无用内容进入 LLM 上下文
  - CodeSearcher: 精准搜索 → 只返回命中的行 + 上下文，上下文用量降低 10-50×

搜索方法（按使用频率排列）：
  search_by_pattern   — 正则搜索（最灵活，支持复杂模式）
  search_by_name      — 函数/类/变量定义搜索
  search_by_content   — 简单子串/关键词搜索（无需正则知识）
  search_callers      — 查找指定函数的所有调用点
  search_imports      — 查找指定模块的 import 语句
  search_files        — 按文件名 glob 匹配

结果格式：每条结果包含 file + line_number + line_content + match + context_before + context_after
"""

# ── Python 标准库 ──
import re                              # 正则表达式匹配（核心搜索引擎）
import fnmatch                         # Unix shell 风格的文件名模式匹配（如 "*.py"）
from dataclasses import dataclass, field  # 数据类装饰器

# ── 项目内导入 ──
from app.coding.repository_loader import RepoContext, FileInfo
# RepoContext: 仓库的文件列表 + 内容映射
# FileInfo:    单个文件的元信息（语言、大小等）


# ═══════════════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchMatch:
    """
    单个搜索命中结果。

    每个实例代表代码中的一行匹配。包含命中行本身、上下文行、文件位置等信息。
    Agent 可以通过这些字段快速判断匹配是否相关、是否值得进一步 read_file。
    """
    file_path: str                     # 匹配所在文件的相对路径
    language: str                      # 文件语言（用于判断是否可做 AST 分析）
    line_number: int                   # 匹配所在行号（1-based，与编辑器一致）
    line_content: str                  # 匹配行的完整内容（去除首尾空白）
    match_text: str                    # 实际命中的文本（正则的 match.group()）
    context_before: list[str]          # 命中的前 N 行（用于理解上下文）
    context_after: list[str]           # 命中的后 N 行


@dataclass
class SearchResult:
    """
    一次搜索的完整结果集。

    包含所有匹配项的列表 + 搜索元信息。提供格式化方法用于 LLM 上下文注入。

    字段：
      matches        — 匹配项列表（已截断，不超过 max_results）
      total_found    — 实际匹配总数（截断前的真实数量）
      truncated      — 是否因超出 max_results 而被截断
      pattern        — 使用的搜索模式（方便 Agent 追溯搜索依据）
      searched_files — 搜索覆盖的文件数量
    """
    matches: list[SearchMatch]         # 匹配结果列表
    total_found: int                   # 实际匹配总数（截断前）
    truncated: bool                    # 结果是否被截断
    pattern: str                       # 搜索模式（正则或关键词）
    searched_files: int                # 搜索覆盖的文件数量

    def format_for_llm(self, max_items: int = 20) -> str:
        """
        将搜索结果格式化为 LLM 可读的文本块。

        相当于把结构化结果"展开"为 Agent 对话中的一段文本，
        让 LLM 不需要解析 JSON 就能理解搜索结果。

        输出格式：
          Found 15 matches for "def chat" in 12 files (showing top 20)
          ─────────────────────────────────────────
          [1] app/api/agent.py:245
            → @router.post("/chat", ...)
            → async def chat(request: ChatRequest):

          [2] app/agent/agent.py:37
            → async def agent_loop(self, user_message: str, ...):
          ─────────────────────────────────────────

        Args:
            max_items: 最多显示多少条匹配（默认 20）
        """
        if not self.matches:
            return f'No matches found for "{self.pattern}" in {self.searched_files} files.'

        lines = [
            f'Found {self.total_found} match(es) for "{self.pattern}" '
            f'in {self.searched_files} file(s)'
            + (f' (showing top {max_items})' if len(self.matches) < self.total_found else ''),
            "─" * 55,
        ]

        for i, m in enumerate(self.matches[:max_items], 1):
            # 拼接上下文行（前缀标记区分命中行和上下文行）
            context_lines = []
            for cl in m.context_before:
                context_lines.append(f"      | {cl.strip()[:90]}")
            # 命中行用 → 标记
            context_lines.append(
                f"  → {m.line_content.strip()[:100]}"
            )
            for cl in m.context_after:
                context_lines.append(f"      | {cl.strip()[:90]}")

            lines.append(
                f"\n[{i}] {m.file_path}:{m.line_number} ({m.language})"
            )
            lines.extend(context_lines)

        lines.append("\n" + "─" * 55)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CodeSearcher
# ═══════════════════════════════════════════════════════════════════════════════

class CodeSearcher:
    """
    代码搜索引擎 — 在 RepoContext 中搜索代码。

    支持的搜索类型：
      - 正则匹配：   search_by_pattern(r"def\s+chat")
      - 名称搜索：   search_by_name("chat")          → 自动匹配 def chat / class chat
      - 关键词搜索：  search_by_content("timeout")    → 简单子串查找（不要求正则）
      - 调用点搜索：  search_callers("chat")          → 找到所有 chat(...) 调用
      - 导入搜索：   search_imports("fastapi")       → 找到 import fastapi 的位置
      - 文件搜索：   search_files("*.py")            → 按文件名 glob 查找

    搜索结果控制：
      - max_results: 最多返回 N 条匹配（默认 50，防止 LLM 上下文溢出）
      - context_lines: 每个匹配前后各 N 行上下文（默认 2，帮助 LLM 理解）
      - file_pattern: 限制搜索范围（如只搜 "*.py"）

    使用方式：
      from app.coding.repository_loader import RepositoryLoader
      from app.coding.code_search import CodeSearcher

      repo = RepositoryLoader(".").load()
      searcher = CodeSearcher(repo)

      # 查找所有定义 chat 函数/类的地方
      results = searcher.search_by_name("chat")
      print(results.format_for_llm())

      # 正则搜索所有 async def 开头的函数
      results = searcher.search_by_pattern(r"async\s+def\s+\w+")
    """

    def __init__(self, repo: RepoContext):
        """
        Args:
            repo: 已加载的仓库上下文（RepositoryLoader.load() 的返回值）
        """
        self.repo = repo                  # 仓库上下文（files + contents）

    # ── 核心搜索方法 ─────────────────────────────────────────────

    def search_by_pattern(
        self,
        pattern: str,
        file_pattern: str = "*",
        max_results: int = 50,
        context_lines: int = 2,
    ) -> SearchResult:
        """
        使用正则表达式搜索代码内容。

        这是 CodeSearcher 的底层引擎 —— 其他所有搜索方法最终都调用此方法。

        Args:
            pattern:       Python 正则表达式（如 r"def\s+chat"）
            file_pattern:  文件 glob 过滤（如 "*.py" 只搜 Python 文件，默认 "*" 全部）
            max_results:   最多返回多少条结果（默认 50）
            context_lines: 每个匹配前后各取多少行上下文（默认 2）

        Returns:
            SearchResult: 包含匹配列表 + 搜索元信息

        正则编译安全：
          LLM 可能生成有问题的正则（括号不匹配、非法转义等），
          这里用 try/except 捕获 re.error 并返回空结果，不抛异常。

        Examples:
          # 找到所有 async 函数定义
          searcher.search_by_pattern(r"async\s+def\s+\w+")

          # 只在 JavaScript 文件中搜索
          searcher.search_by_pattern(r"import\s+.*\s+from", file_pattern="*.js")
        """
        # ── 0. 编译正则（安全编译，LLM 生成的正则可能有语法错误） ──
        try:
            regex = re.compile(pattern)    # 编译正则，加速后续多次 match
        except re.error as e:
            # 正则语法错误 → 返回空结果（不抛异常，避免中断 Agent 工作流）
            return SearchResult(
                matches=[],
                total_found=0,
                truncated=False,
                pattern=pattern,
                searched_files=0,
            )

        # ── 1. 遍历仓库所有文件的每一行 ──
        all_matches: list[SearchMatch] = []  # 收集所有命中
        searched_count = 0                   # 实际搜索的文件数

        for file_path, content in self.repo.file_contents.items():
            # 1a. 文件名 glob 过滤（fnmatch 支持 * ? [seq]）
            if not self._match_glob(file_path, file_pattern):
                continue                     # 文件名不匹配 → 跳过整个文件
            searched_count += 1

            # 1b. 获取文件语言（用于 SearchMatch 的 language 字段）
            file_info = self._get_file_info(file_path)
            language = file_info.language if file_info else "unknown"

            # 1c. 逐行扫描
            lines = content.split("\n")      # 按换行分割（保留空行）
            for i, line in enumerate(lines):
                # line_number 从 1 开始（与编辑器一致）
                line_number = i + 1

                # 正则搜索当前行
                match = regex.search(line)
                if not match:
                    continue                 # 未命中 → 下一行

                # 1d. 提取上下文行（命中行的前后 N 行）
                ctx_before = self._get_context(
                    lines, line_number, before=True, count=context_lines
                )
                ctx_after = self._get_context(
                    lines, line_number, before=False, count=context_lines
                )

                # 1e. 构建 SearchMatch
                all_matches.append(SearchMatch(
                    file_path=file_path,
                    language=language,
                    line_number=line_number,
                    line_content=line.strip(),    # 去除首尾空白
                    match_text=match.group(),      # 实际命中的文本
                    context_before=ctx_before,
                    context_after=ctx_after,
                ))

        # ── 2. 截断结果（防止 LLM 上下文溢出） ──
        total_found = len(all_matches)
        truncated = total_found > max_results
        final_matches = all_matches[:max_results]

        return SearchResult(
            matches=final_matches,
            total_found=total_found,
            truncated=truncated,
            pattern=pattern,
            searched_files=searched_count,
        )

    def search_by_name(self, name: str, file_pattern: str = "*",
                       max_results: int = 30) -> SearchResult:
        """
        搜索函数/类/变量的定义位置。

        自动构造正则：匹配 def name / class name / name = 等定义模式。

        Args:
            name:  要搜索的函数名/类名/变量名
            file_pattern: 文件 glob（默认 "*" 全部）
            max_results:  最多返回数（默认 30）

        Returns:
            SearchResult（注意 pattern 字段会显示实际使用的正则）

        Examples:
          searcher.search_by_name("chat")     → 找到 def chat / class chat
          searcher.search_by_name("__init__") → 找到 def __init__
        """
        # 转义特殊字符（如 name="chat.test" 中的 "."）
        escaped = re.escape(name)
        # 匹配定义模式：
        #   (?:async\s+)?     — 可选的 async 前缀
        #   (?:def|class)\s+  — def 或 class 关键字
        #   {}                — 函数/类名称
        #   \b                — 词边界（避免 "chat_impl" 误匹配 "chat"）
        pattern = rf"(?:async\s+)?(?:def|class)\s+{escaped}\b"

        result = self.search_by_pattern(
            pattern, file_pattern=file_pattern, max_results=max_results
        )
        # 覆盖 pattern 为可读的名称（而非内部正则）
        result.pattern = f"name:{name}"
        return result

    def search_by_content(
        self,
        keyword: str,
        file_pattern: str = "*",
        max_results: int = 50,
        case_sensitive: bool = False,
    ) -> SearchResult:
        """
        简单关键词/子串搜索（不需要正则知识）。

        适合 LLM 在不知道精确语法时做模糊搜索。
        与 search_by_pattern 的区别：此方法转义所有特殊字符，当作字面文本搜索。

        Args:
            keyword:       要搜索的关键词（如 "timeout"、"FastAPI"）
            file_pattern:  文件 glob（默认 "*"）
            max_results:   最多返回数（默认 50）
            case_sensitive: 是否区分大小写（默认 False = 不区分）

        Returns:
            SearchResult

        Examples:
          searcher.search_by_content("uvicorn")   → 找到所有包含 uvicorn 的行
          searcher.search_by_content("TODO")      → 找到所有 TODO 注释
        """
        # 转义所有正则特殊字符（如 . * + ? 等 → \. \* \+ \?）
        escaped = re.escape(keyword)
        # 编译正则（可选忽略大小写）
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(escaped, flags)
        except re.error:
            return SearchResult([], 0, False, keyword, 0)

        # 底层仍用 search_by_pattern，但跳过它的正则编译步骤
        all_matches: list[SearchMatch] = []
        searched_count = 0

        for file_path, content in self.repo.file_contents.items():
            if not self._match_glob(file_path, file_pattern):
                continue
            searched_count += 1

            file_info = self._get_file_info(file_path)
            language = file_info.language if file_info else "unknown"
            lines = content.split("\n")

            for i, line in enumerate(lines):
                if regex.search(line):
                    line_number = i + 1
                    all_matches.append(SearchMatch(
                        file_path=file_path,
                        language=language,
                        line_number=line_number,
                        line_content=line.strip(),
                        match_text=keyword,              # 用原始关键词
                        context_before=self._get_context(lines, line_number, True, 2),
                        context_after=self._get_context(lines, line_number, False, 2),
                    ))

        total_found = len(all_matches)
        truncated = total_found > max_results
        return SearchResult(
            matches=all_matches[:max_results],
            total_found=total_found,
            truncated=truncated,
            pattern=f'content:"{keyword}"',
            searched_files=searched_count,
        )

    def search_callers(
        self,
        func_name: str,
        file_pattern: str = "*.py",
        max_results: int = 30,
    ) -> SearchResult:
        """
        查找指定函数的所有调用点（call sites）。

        构造正则匹配 func_name( 模式，自动排除定义行（def func_name）。
        这对跨文件修改至关重要 —— 改函数签名时需要知道所有调用方。

        Args:
            func_name:    被调用的函数名
            file_pattern: 文件 glob（默认 "*.py"）
            max_results:  最多返回数（默认 30）

        Returns:
            SearchResult — 每条匹配代表一个调用点

        Examples:
          searcher.search_callers("chat")
          → [app/api/agent.py:360] result = await simple_agent.agent_loop(...)
          → [app/main.py:15]     chat(request)
        """
        escaped = re.escape(func_name)
        # 匹配 func_name( 但排除 def func_name(
        # 用负向后顾 (?<!def\s) 排除定义行
        pattern = rf"(?<!\bdef\s)(?<!\bclass\s){escaped}\s*\("

        result = self.search_by_pattern(
            pattern, file_pattern=file_pattern, max_results=max_results
        )
        result.pattern = f"callers:{func_name}"
        return result

    def search_imports(
        self,
        module_name: str,
        file_pattern: str = "*.py",
        max_results: int = 30,
    ) -> SearchResult:
        """
        查找指定模块的所有 import 语句。

        支持两种 Python import 格式：
          import module_name          → "import os"
          from module_name import X   → "from os.path import join"

        Args:
            module_name:  模块名（如 "fastapi"、"os.path"）
            file_pattern: 文件 glob（默认 "*.py"）
            max_results:  最多返回数（默认 30）

        Returns:
            SearchResult

        Examples:
          searcher.search_imports("fastapi")
          → [app/main.py:1] from fastapi import FastAPI
          → [app/api/agent.py:20] from fastapi import APIRouter
        """
        escaped = re.escape(module_name)
        # 匹配两种格式：
        #   import module_name
        #   from module_name import ...
        pattern = rf"(?:import\s+{escaped}\b|from\s+{escaped}\s+import)"

        result = self.search_by_pattern(
            pattern, file_pattern=file_pattern, max_results=max_results
        )
        result.pattern = f"imports:{module_name}"
        return result

    def search_files(self, glob_pattern: str) -> list[str]:
        """
        按文件名 glob 匹配文件列表。

        不搜索文件内容，只返回匹配的文件路径列表。
        适合 LLM 先用此方法了解仓库结构，再决定 search_by_pattern 的目标文件。

        Args:
            glob_pattern: Unix shell 风格的通配符（如 "*.py"、"app/**/*.py"）

        Returns:
            匹配的文件路径列表

        Examples:
          searcher.search_files("*.py")        → ["main.py", "app/agent/agent.py", ...]
          searcher.search_files("app/api/*")   → ["app/api/agent.py"]
        """
        results = []
        for file_path in self.repo.file_contents.keys():
            if self._match_glob(file_path, glob_pattern):
                results.append(file_path)
        return sorted(results)               # 按字母排序，输出稳定

    # ── 内部辅助方法 ─────────────────────────────────────────────

    @staticmethod
    def _match_glob(file_path: str, pattern: str) -> bool:
        """
        检查文件路径是否匹配 glob 模式。

        使用 fnmatch（Unix shell 风格）：
          *      匹配任意字符（不含路径分隔符）
          ?      匹配单个字符
          [seq]  匹配字符集合
          **/    匹配任意层级目录（需自行扩展，fnmatch 不支持 **）

        Args:
            file_path: 文件相对路径（如 "app/api/agent.py"）
            pattern:   glob 模式（如 "*.py"、"app/**/*.py"）

        Returns:
            True → 路径匹配模式
        """
        # fnmatch 原生不支持 **（递归匹配），这里做简单扩展：
        # ** 转换为 *，使其匹配任意层级的目录
        if "**" in pattern:
            # 将 **/ 或 /** 替换为可匹配任意路径的模式
            import re as _re
            regex_pattern = _re.escape(pattern)
            # \*\*  →  .* （匹配任意字符序列，包括 /）
            regex_pattern = regex_pattern.replace(r"\*\*", ".*")
            try:
                return bool(_re.match(regex_pattern + "$", file_path))
            except _re.error:
                pass
            return False

        return fnmatch.fnmatch(file_path, pattern)

    def _get_file_info(self, file_path: str) -> FileInfo | None:
        """
        根据文件路径从 RepoContext.files 中查找 FileInfo。

        Args:
            file_path: 文件相对路径
        Returns:
            FileInfo 或 None（文件不在仓库中）
        """
        for f in self.repo.files:
            if f.path == file_path:
                return f
        return None

    @staticmethod
    def _get_context(
        lines: list[str],
        line_number: int,
        before: bool,
        count: int,
    ) -> list[str]:
        """
        提取某一行前后的上下文行。

        不包含命中行本身 —— 上下文只取相邻行。

        Args:
            lines:       文件的所有行
            line_number: 目标行号（1-based）
            before:      True=前行, False=后行
            count:       取几行

        Returns:
            上下文行列表（原始字符串，不去除空白）
        """
        if before:
            # 前行：line_number - count 到 line_number - 1
            start = max(0, line_number - 1 - count)  # line_number 是 1-based
            end = line_number - 1
            return lines[start:end]
        else:
            # 后行：line_number 到 line_number + count
            # （line_number 是 1-based，对应索引 line_number）
            start = line_number                      # 跳过命中行本身
            end = min(len(lines), line_number + count)
            return lines[start:end]
