from __future__ import annotations
"""
RAG 文档加载器 —— 把文件 / 上传字节 / 目录转成 RagDocument[]。

三类入口：
  load_bytes(data, filename)  —— 前端文件上传
  load_file(path, source)     —— 本地单个文件
  load_dir(root)              —— 本地目录批量导入（带白名单与规模上限）

解析策略：
  文本类（.md/.txt/.py/...）→ 直接 UTF-8 解码
  .pdf  → pypdf（可选依赖，未安装时抛出友好提示）
  .docx → python-docx（可选依赖）
  其余二进制            → 跳过（load_dir）/ 报错（单文件）

安全：
  load_dir 会对根目录做存在性校验 + 可选白名单校验（Config.RAG_ALLOWED_DIRS），
  防止接口被利用来读取服务器任意路径的文件。
"""

import os
from typing import Optional

from app.rag.models import RagDocument

# 复用 app/coding/repository_loader.py 里维护好的扩展名白名单与目录黑名单，
# 避免两处各维护一份（DRY）。取值失败时退回内置默认值，保证 rag 包可独立工作。
try:
    from app.coding.repository_loader import RepositoryLoader as _RepoLoader
    TEXT_EXTENSIONS: set[str] = set(_RepoLoader.TEXT_EXTENSIONS)
    SKIP_DIRECTORIES: set[str] = set(_RepoLoader.SKIP_DIRECTORIES)
except Exception:                                    # pragma: no cover
    TEXT_EXTENSIONS = {".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml"}
    SKIP_DIRECTORIES = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}

# 支持解析的富文本扩展名（需可选依赖）
PDF_EXTENSIONS = {".pdf"}
DOCX_EXTENSIONS = {".docx"}

# 目录导入的规模上限，防止一次任务把 CPU 打满
MAX_DIR_FILES = 500
MAX_FILE_BYTES = 8 * 1024 * 1024          # 单文件 8 MB


class UnsupportedDocumentError(ValueError):
    """文档类型不支持 / 缺少可选解析依赖。"""


# ═══════════════════════════════════════════════════════════════════
# 公开入口
# ═══════════════════════════════════════════════════════════════════

def load_bytes(data: bytes, filename: str, source: Optional[str] = None) -> RagDocument:
    """
    从内存字节加载文档（对应前端 multipart 上传）。

    Args:
        data:     原始字节
        filename: 上传时的文件名（决定解析策略与默认 source）
        source:   知识库中的来源标识，默认用 filename

    Raises:
        UnsupportedDocumentError: 类型不支持或缺少可选依赖
    """
    src = source or os.path.basename(filename) or "uploaded"
    ext = _ext_of(src)

    if len(data) > MAX_FILE_BYTES:
        raise UnsupportedDocumentError(
            f"文件过大（{len(data) / 1024 / 1024:.1f} MB），上限 {MAX_FILE_BYTES // 1024 // 1024} MB"
        )

    content = _parse(data, ext, src)
    return RagDocument(source=src, content=content, ext=ext, size_bytes=len(data))


def load_file(path: str, source: Optional[str] = None) -> RagDocument:
    """
    从本地路径加载单个文件。

    Args:
        path:   文件绝对路径 / 相对路径
        source: 知识库来源标识，默认用相对路径（见 _default_source）
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"不是文件或不存在: {path}")

    size = os.path.getsize(path)
    if size > MAX_FILE_BYTES:
        raise UnsupportedDocumentError(f"文件过大: {path}")

    with open(path, "rb") as f:
        data = f.read()

    src = source or _default_source(path)
    ext = _ext_of(src) or _ext_of(path)
    content = _parse(data, ext, src)
    return RagDocument(source=src, content=content, ext=ext, size_bytes=size)


def load_dir(
    root: str,
    allowed_dirs: Optional[list[str]] = None,
    max_files: int = MAX_DIR_FILES,
) -> tuple[list[RagDocument], list[str]]:
    """
    批量加载目录下的所有可解析文件。

    Args:
        root:         目录路径（绝对/相对均可）
        allowed_dirs: 白名单绝对路径前缀列表。非空时 root 必须位于其中。
        max_files:    最多加载多少个文件

    Returns:
        (documents, skipped) —— skipped 是被跳过文件的相对路径列表

    Raises:
        ValueError: 路径不存在 / 不是目录 / 不在白名单内
    """
    abs_root = os.path.abspath(root)
    if not os.path.isdir(abs_root):
        raise ValueError(f"路径不存在或不是目录: {abs_root}")

    if allowed_dirs:
        if not _is_under_any(abs_root, allowed_dirs):
            raise ValueError(
                f"目录不在白名单内: {abs_root}（允许: {allowed_dirs}）"
            )

    documents: list[RagDocument] = []
    skipped: list[str] = []

    for dirpath, dirnames, filenames in os.walk(abs_root):
        # 原地过滤目录黑名单 + 隐藏目录
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRECTORIES and not d.startswith(".")
        ]

        for name in sorted(filenames):
            if len(documents) >= max_files:
                skipped.append("...（达到文件数上限，已停止扫描）")
                return documents, skipped

            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, abs_root)
            ext = _ext_of(name)

            if ext not in TEXT_EXTENSIONS and ext not in PDF_EXTENSIONS and ext not in DOCX_EXTENSIONS:
                continue

            try:
                documents.append(load_file(path, source=rel))
            except UnsupportedDocumentError as e:
                skipped.append(f"{rel} —— {e}")
            except Exception as e:                    # 编码失败 / 权限问题等
                skipped.append(f"{rel} —— {type(e).__name__}: {e}")

    return documents, skipped


# ═══════════════════════════════════════════════════════════════════
# 内部：按扩展名解析
# ═══════════════════════════════════════════════════════════════════

def _parse(data: bytes, ext: str, source: str) -> str:
    """按扩展名把字节解析为纯文本。"""
    if ext in PDF_EXTENSIONS:
        return _parse_pdf(data)
    if ext in DOCX_EXTENSIONS:
        return _parse_docx(data)
    return _decode_text(data, source)


def _decode_text(data: bytes, source: str) -> str:
    """UTF-8 解码，失败回退到 GBK，再失败则忽略非法字节。"""
    if _looks_binary(data):
        raise UnsupportedDocumentError(f"疑似二进制文件，无法解析: {source}")
    for enc in ("utf-8", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _looks_binary(data: bytes, probe: int = 8192) -> bool:
    """用 NUL 字节判断是否为二进制文件（与 git 的启发式一致）。"""
    return b"\x00" in data[:probe]


def _parse_pdf(data: bytes) -> str:
    """解析 PDF（需要可选依赖 pypdf）。"""
    try:
        import io
        import pypdf
    except ImportError as e:
        raise UnsupportedDocumentError(
            "解析 .pdf 需要可选依赖，请执行: pip install pypdf"
        ) from e

    reader = pypdf.PdfReader(io.BytesIO(data))
    parts = [(page.extract_text() or "") for page in reader.pages]
    text = "\n\n".join(p.strip() for p in parts if p.strip())
    if not text:
        raise UnsupportedDocumentError("PDF 无可选文本层（可能是扫描件，需 OCR）")
    return text


def _parse_docx(data: bytes) -> str:
    """解析 DOCX（需要可选依赖 python-docx）。"""
    try:
        import io
        import docx
    except ImportError as e:
        raise UnsupportedDocumentError(
            "解析 .docx 需要可选依赖，请执行: pip install python-docx"
        ) from e

    document = docx.Document(io.BytesIO(data))
    parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    # 表格内容也纳入
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# 内部：路径工具
# ═══════════════════════════════════════════════════════════════════

def _ext_of(name: str) -> str:
    """取小写扩展名（含点）；无扩展名返回 ""。"""
    base = os.path.basename(name)
    if "." not in base:
        return ""
    return "." + base.rsplit(".", 1)[-1].lower()


def _default_source(path: str) -> str:
    """单文件导入时的默认来源名：优先相对 CWD 的路径，否则文件名。"""
    try:
        rel = os.path.relpath(os.path.abspath(path), os.getcwd())
        return rel if not rel.startswith("..") else os.path.basename(path)
    except Exception:                                # pragma: no cover
        return os.path.basename(path)


def _is_under_any(path: str, prefixes: list[str]) -> bool:
    """判断 path 是否位于任一白名单前缀之下。"""
    abs_path = os.path.abspath(path)
    for prefix in prefixes:
        abs_prefix = os.path.abspath(prefix)
        if abs_path == abs_prefix or abs_path.startswith(abs_prefix + os.sep):
            return True
    return False
