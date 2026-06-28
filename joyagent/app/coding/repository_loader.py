"""
Phase 4 Step 1: Repository Loader — 加载仓库文件树，过滤无关文件。

RepositoryLoader 是 Coding Agent 的"眼睛"——在 Agent 修改代码之前，
它需要了解整个仓库的结构：有哪些文件、它们的语言是什么、内容是什么。

设计要点（为什么不是简单的 os.walk）：
  1. .gitignore 过滤 — 避免把 node_modules/.venv/build 等构建产物装入 LLM 上下文
  2. 二进制检测 — 图片/.pyc/.so 等文件对 Agent 无用，跳过可节省大量上下文空间
  3. 语言识别 — 后续 AST 分析只解析 Python 文件，需要预先分类
  4. 目录跳过 — .git/.venv/__pycache__/node_modules 直接跳过，不做文件遍历
  5. 编码容错 — 有些"文本文件"可能不是 UTF-8，需要安全跳过而非崩溃
"""
from __future__ import annotations

# ── Python 标准库 ──
import os                              # 文件系统遍历 (os.walk)、路径拼接 (os.path)
import json                            # JSON 文件大小计算（已废弃，改为 len(content)）
from dataclasses import dataclass, field  # 数据类装饰器

# ── 第三方库 ──
from pathspec import PathSpec          # .gitignore 规则解析 (gitwildmatch 语法)


# ═══════════════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FileInfo:
    """
    仓库中单个文件的结构化信息。

    这是 RepoContext.files 列表的元素，Agent 和工具通过它了解：
      - 文件在哪里 (path)
      - 是什么语言 (language) — 决定是否可以做 AST 分析
      - 有多大 (size_bytes) — 决定是否值得放入 LLM 上下文
      - 是否二进制 (is_binary) — 二进制文件跳过不读

    Example:
      FileInfo(
          path="app/agent/agent.py",
          language="python",
          size_bytes=4521,
          is_binary=False,
      )
    """
    path: str                           # 相对路径（相对于仓库根目录）
    language: str                       # 语言标识: "python" | "javascript" | "typescript" | "html" | "css" | "json" | "markdown" | "unknown"
    size_bytes: int                     # 文件字节数（用于 LLM 上下文预算分配）
    is_binary: bool                     # 是否为二进制文件（True 则无 content）


@dataclass
class RepoContext:
    """
    加载后的仓库完整上下文。

    这是 RepositoryLoader.load() 的返回类型，是后续所有代码分析操作的数据来源。
    CodeSearcher / ASTAnalyzer / DiffGenerator 都通过 RepoContext 访问仓库内容。

    字段：
      root_path     — 仓库根目录的绝对路径
      files         — 所有被加载的文件信息列表（已过滤 ignore/二进制）
      file_contents — path → content 的映射（仅文本文件，二进制文件不在此）
    """
    root_path: str                      # 仓库根目录绝对路径
    files: list[FileInfo]               # 文件信息列表（按遍历顺序排列）
    file_contents: dict[str, str]       # path(str) → content(str) 映射

    # ── 便捷查询方法 ──────────────────────────────────────────────

    def get_python_files(self) -> list[FileInfo]:
        """获取所有 Python 文件的 FileInfo 列表。"""
        return [f for f in self.files if f.language == "python"]

    def get_files_by_language(self, language: str) -> list[FileInfo]:
        """按语言过滤文件列表。language 参数如 'python'、'javascript'。"""
        return [f for f in self.files if f.language == language]

    def get_content(self, path: str) -> str | None:
        """按路径获取文件内容，文件不存在时返回 None。"""
        return self.file_contents.get(path)

    def get_file_count(self, language: str = None) -> int:
        """
        获取文件数量。
        不传 language → 返回总文件数；传入 language → 返回该语言的文件数。
        """
        if language is None:
            return len(self.files)
        return len(self.get_files_by_language(language))

    def get_total_size_bytes(self, language: str = None) -> int:
        """获取文件总大小（字节）。可选按语言过滤。"""
        target = self.files if language is None else self.get_files_by_language(language)
        return sum(f.size_bytes for f in target)

    def summarize(self) -> str:
        """
        生成仓库结构摘要（用于注入 LLM 上下文）。

        返回格式如：
          Repo: /home/user/project
          Files: 42 total (18 python, 8 javascript, 6 markdown, ...)
          Total size: 234.5 KB
        """
        lines = [f"Repository: {self.root_path}"]
        lines.append(f"Files: {self.get_file_count()} total")

        # 按语言统计文件数（从高到低排列）
        lang_counts: dict[str, int] = {}
        for f in self.files:
            lang_counts[f.language] = lang_counts.get(f.language, 0) + 1

        lang_parts = []
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
            lang_parts.append(f"{count} {lang}")
        lines.append(f"  ({', '.join(lang_parts)})")

        total_kb = self.get_total_size_bytes() / 1024
        if total_kb >= 1024:
            lines.append(f"Total size: {total_kb / 1024:.1f} MB")
        else:
            lines.append(f"Total size: {total_kb:.1f} KB")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# RepositoryLoader
# ═══════════════════════════════════════════════════════════════════════════════

class RepositoryLoader:
    """
    仓库文件加载器。

    职责：
      1. 遍历文件树 (os.walk)
      2. 跳过无关目录 (.git / .venv / __pycache__ / node_modules / 隐藏目录)
      3. 应用 .gitignore 规则过滤 (pathspec.PathSpec)
      4. 跳过二进制/非 UTF-8 文件
      5. 识别文件语言 (.py → python, .js → javascript, ...)
      6. 构建 RepoContext（包含 files 列表 + file_contents 映射）

    使用方式：
      loader = RepositoryLoader("/path/to/repo")
      repo: RepoContext = loader.load()
      print(repo.summarize())           # 仓库结构摘要
      print(len(repo.get_python_files()))  # Python 文件数量
    """

    # ── 已知文本文件扩展名 ──────────────────────────────────────────
    # 只有这些扩展名的文件才会被读入 file_contents。
    # 不在白名单中的文件（如 .pyc .so .png）直接跳过。
    # 注意：Makefile / Dockerfile / .env 没有扩展名，在遍历时单独处理。
    TEXT_EXTENSIONS: set[str] = {
        # 编程语言
        ".py", ".pyi",                   # Python（含 stub）
        ".js", ".jsx", ".ts", ".tsx",    # JavaScript/TypeScript
        ".java", ".go", ".rs", ".rb",    # Java / Go / Rust / Ruby
        ".c", ".cpp", ".h", ".hpp",      # C/C++
        ".swift", ".kt", ".scala",       # Swift / Kotlin / Scala
        # Web 前端
        ".html", ".htm", ".css", ".scss", ".less",
        ".vue", ".svelte",
        # 配置 & 数据
        ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
        ".xml", ".csv",
        # 文档
        ".md", ".markdown", ".rst", ".txt",
        # Shell & 构建
        ".sh", ".bat", ".ps1",           # Shell 脚本
        ".cmake", ".mk",                 # 构建系统
        ".dockerfile",                   # 少见的 Dockerfile 扩展名
        # 其他常用
        ".sql", ".graphql", ".proto",
    }

    # ── 无扩展名但应加载的文件名 ────────────────────────────────────
    NAME_WHITELIST: set[str] = {
        "Makefile", "Dockerfile", ".env", ".gitignore",
        ".dockerignore", ".editorconfig",
        "LICENSE", "CONTRIBUTING",
    }

    # ── 遍历时跳过的目录名 ──────────────────────────────────────────
    # 这些目录包含构建产物、第三方依赖、缓存——对 Agent 无价值
    SKIP_DIRECTORIES: set[str] = {
        ".git", ".svn", ".hg",           # 版本控制
        "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "node_modules", "venv", ".venv", "env", ".env",
        ".tox", ".nox",                  # 测试虚拟环境
        "build", "dist", ".eggs",        # 构建产出
        ".idea", ".vscode",              # IDE 配置
        ".ipynb_checkpoints",            # Jupyter 缓存
        "target",                        # Rust 构建
        ".next", ".nuxt",                # Next.js / Nuxt
        "coverage", "htmlcov",           # 测试覆盖率报告
    }

    def __init__(self, root_path: str):
        """
        初始化仓库加载器。

        Args:
            root_path: 仓库根目录路径（绝对或相对均可，内部会转为绝对路径）
        """
        self.root_path = os.path.abspath(root_path)  # 统一为绝对路径
        # 检查路径是否存在
        if not os.path.isdir(self.root_path):
            raise NotADirectoryError(
                f"Root path does not exist or is not a directory: {self.root_path}"
            )
        # 加载 .gitignore 规则（如果存在）
        self._gitignore_spec = self._load_gitignore()

    # ── .gitignore 加载 ───────────────────────────────────────────

    def _load_gitignore(self) -> PathSpec:
        """
        加载仓库根目录的 .gitignore 文件。

        如果 .gitignore 不存在，返回空的 PathSpec（不过滤任何文件）。

        为什么用 pathspec.PathSpec 而不是手写 glob？
          - PathSpec 完全兼容 git 的 .gitignore 语法（包括 **、[]、否定模式 !）
          - 避免重复造轮子——.gitignore 解析有大量边界情况
          - gitwildmatch 是 git 官方使用的匹配算法
        """
        gitignore_path = os.path.join(self.root_path, ".gitignore")
        if os.path.isfile(gitignore_path):
            try:
                with open(gitignore_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                return PathSpec.from_lines("gitwildmatch", lines)
            except Exception:
                # .gitignore 损坏或无法读取 → 不报错，继续加载
                pass
        # 没有 .gitignore → 返回空规则集（匹配不到任何文件）
        return PathSpec.from_lines("gitwildmatch", [])

    # ── 主加载逻辑 ────────────────────────────────────────────────

    def load(self) -> RepoContext:
        """
        加载整个仓库的文件树和内容。

        遍历流程（对每个文件）：
          1. 目录过滤 —— 跳过 SKIP_DIRECTORIES 和隐藏目录
          2. .gitignore 检查 —— 匹配 .gitignore 规则的文件跳过
          3. 类型判断 —— 扩展名/文件名是否在可加载白名单中
          4. 编码尝试 —— 尝试 UTF-8 读取，失败则跳过
          5. 语言识别 —— 根据扩展名判断编程语言
          6. 记录 FileInfo + 存储 content

        Returns:
            RepoContext: 包含所有被成功加载的文件信息 + 内容映射

        性能注意：
          对于大型仓库（> 5000 文件），本函数会一次性把所有文本文件读入内存。
          后续 Phase 可优化为懒加载（按需读取），Phase 4 MVP 先全量加载。
        """
        files: list[FileInfo] = []         # 文件信息列表
        contents: dict[str, str] = {}      # path → content

        # os.walk 自顶向下遍历：先根，后子目录
        for dirpath, dirnames, filenames in os.walk(
            self.root_path,
            topdown=True,                  # 自顶向下——可以原地修改 dirnames 来跳过目录
        ):
            # ── 1. 目录过滤 ──────────────────────────────────────
            # 原地修改 dirnames（os.walk 在 topdown=True 时支持）
            # 从 dirnames 中移除需要跳过的目录——os.walk 后续不会进入这些目录
            dirnames[:] = [
                d for d in dirnames
                if not self._should_skip_directory(d)
            ]

            # ── 2. 处理当前目录下的所有文件 ──────────────────────
            for filename in filenames:
                # 2a. 构建绝对路径和相对路径
                abs_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(abs_path, self.root_path)
                # 统一使用正斜杠（Windows 兼容 + git 规范）
                rel_path = rel_path.replace("\\", "/")

                # 2b. .gitignore 规则匹配
                if self._matches_gitignore(rel_path):
                    continue               # 被 gitignore 排除

                # 2c. 扩展名检查 + 类型判断
                ext = os.path.splitext(filename)[1].lower()
                is_text = (
                    ext in self.TEXT_EXTENSIONS or
                    filename in self.NAME_WHITELIST
                )
                if not is_text:
                    # 非白名单扩展名 → 尝试检测是否为二进制文件
                    # （有些文件如 .cfg/.conf 没在白名单但实际是文本）
                    if self._is_likely_text(abs_path):
                        pass               # 继续尝试读取
                    else:
                        # 真正的二进制文件 → 只记录 FileInfo，不读内容
                        try:
                            size = os.path.getsize(abs_path)
                        except OSError:
                            size = 0
                        files.append(FileInfo(
                            path=rel_path,
                            language="binary",
                            size_bytes=size,
                            is_binary=True,
                        ))
                        continue

                # 2d. 尝试读取文件内容（UTF-8）
                try:
                    with open(abs_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # 不是 UTF-8 文本 → 按二进制处理
                    try:
                        size = os.path.getsize(abs_path)
                    except OSError:
                        size = 0
                    files.append(FileInfo(
                        path=rel_path,
                        language="unknown",
                        size_bytes=size,
                        is_binary=True,
                    ))
                    continue
                except (PermissionError, OSError):
                    # 无权限或文件系统错误 → 跳过
                    continue

                # 2e. 识别语言
                language = self._detect_language(ext, filename)

                # 2f. 记录结果
                files.append(FileInfo(
                    path=rel_path,
                    language=language,
                    size_bytes=len(content.encode("utf-8")),  # 字节数
                    is_binary=False,
                ))
                contents[rel_path] = content

        return RepoContext(
            root_path=self.root_path,
            files=files,
            file_contents=contents,
        )

    # ── 内部辅助方法 ──────────────────────────────────────────────

    def _should_skip_directory(self, dirname: str) -> bool:
        """
        判断是否应跳过某个目录。

        跳过的目录：
          - 以 . 开头的隐藏目录（如 .git / .venv）
          - SKIP_DIRECTORIES 中列出的常见无关目录
          - 注意：以 . 开头但有特殊意义的目录（如 .github）不应跳过
            ——当前默认跳过所有隐藏目录，后续可扩展白名单
        """
        if dirname.startswith("."):
            # 例外：.github / .circleci 目录通常很小且含 CI 配置
            if dirname in (".github", ".circleci"):
                return False
            return True
        if dirname in self.SKIP_DIRECTORIES:
            return True
        return False

    def _matches_gitignore(self, rel_path: str) -> bool:
        """
        检查文件路径是否匹配 .gitignore 规则。

        Args:
            rel_path: 相对于仓库根目录的文件路径（使用 / 分隔符）
        Returns:
            True → 被 .gitignore 排除，不加载
        """
        return self._gitignore_spec.match_file(rel_path)

    def _is_likely_text(self, abs_path: str) -> bool:
        """
        快速检测文件是否"看起来是文本文件"。

        策略（轻量级）：
          读取前 1024 字节，检查是否全是可打印字符。
          如果包含 NULL 字节 (\x00)，则为二进制文件。

        这用于 .cfg / .conf / .rc 等没有标准扩展名但实际是文本的文件。
        """
        try:
            with open(abs_path, "rb") as f:
                chunk = f.read(1024)
            if not chunk:
                return True               # 空文件 → 当文本处理
            # 二进制文件特征：包含 NULL 字节
            if b"\x00" in chunk:
                return False
            # 尝试用 UTF-8 解码看是否可行
            try:
                chunk.decode("utf-8")
                return True
            except UnicodeDecodeError:
                return False
        except (PermissionError, OSError):
            return False

    def _detect_language(self, ext: str, filename: str) -> str:
        """
        根据文件扩展名识别编程语言。

        识别优先级：
          1. 扩展名映射表 → 精确匹配
          2. 特殊文件名（如 Dockerfile / Makefile）→ 独立判断
          3. 兜底 → "unknown"
        """
        MAPPING: dict[str, str] = {
            # Python
            ".py": "python", ".pyi": "python",
            # JavaScript / TypeScript
            ".js": "javascript", ".jsx": "javascript",
            ".ts": "typescript", ".tsx": "typescript",
            ".mjs": "javascript", ".cjs": "javascript",
            # Web
            ".html": "html", ".htm": "html",
            ".css": "css", ".scss": "scss", ".less": "less",
            ".vue": "vue", ".svelte": "svelte",
            # Java / Go / Rust / Ruby
            ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby",
            # C/C++
            ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
            # Swift / Kotlin / Scala
            ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
            # 配置
            ".json": "json", ".yaml": "yaml", ".yml": "yaml",
            ".toml": "toml", ".cfg": "cfg", ".ini": "ini",
            ".xml": "xml", ".csv": "csv",
            # 文档
            ".md": "markdown", ".markdown": "markdown",
            ".rst": "rst", ".txt": "text",
            # Shell
            ".sh": "shell", ".bat": "batch", ".ps1": "powershell",
            # 数据库
            ".sql": "sql",
            # 其他
            ".cmake": "cmake", ".mk": "makefile",
            ".proto": "protobuf",
        }

        if ext in MAPPING:
            return MAPPING[ext]

        # ── 无扩展名的特殊文件 ──
        if filename == "Makefile" or filename.endswith(".mk"):
            return "makefile"
        if filename == "Dockerfile" or filename.startswith("Dockerfile."):
            return "dockerfile"

        return "unknown"
