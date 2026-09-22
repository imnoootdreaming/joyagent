from __future__ import annotations
"""
RAG 文本切分器 —— 把 RagDocument 切成适合 embedding 的 RagChunk[]。

为什么不引入 langchain_text_splitters：
  1. 项目消息格式是 Anthropic 原生 dict，引入 LangChain 会带来无谓的类型转换负担
  2. 自研约 200 行，可控性更好，也不用给 5.8GB 的镜像再增重

三级策略（由结构化到粗暴）：
  1. Markdown —— 按 # ~ ###### 标题做结构切分，保留标题路径（heading_path）
  2. 代码 —— 按空行 / def|class 边界切块，尽量不切断函数
  3. 退化 —— 固定字符窗口 + overlap，并在句末/行末优先断句

所有路径最终都会走 _pack_windows() 做长度兜底，保证：
  - 单 chunk 不超过 chunk_size（除非单个不可分割的 token 超长）
  - 相邻 chunk 之间保留 overlap 字符，避免语义在边界处被切断
  - 任何 chunk 都不超过 EmbeddingService.MAX_TEXT_LENGTH（8192）
"""

import re
from typing import Optional

from app.rag.models import RagChunk, RagDocument

# EmbeddingService 的文本上限，超过会被截断 → 这里硬性再切
MAX_TEXT_LENGTH = 8192

# Markdown 标题行（# ~ ######），以及 ``` 代码块围栏
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

# 代码中的顶层/缩进定义起始行
_DEF_RE = re.compile(
    r"^\s*(?:async\s+)?(?:def|class|function|func|fn|public|private|"
    r"static|export|const|let|var|interface|struct|impl)\b"
)

# 句末标点（用于优先断句）
_SENTENCE_END_RE = re.compile(r"[。！？；!?;\n]")


# ═══════════════════════════════════════════════════════════════════
# 公开入口
# ═══════════════════════════════════════════════════════════════════

def split_document(
    doc: RagDocument,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[RagChunk]:
    """
    把单个文档切成 chunks。

    Args:
        doc:        RagDocument（含 source / content / doc_hash）
        chunk_size: 每片目标字符数
        overlap:    相邻片的重叠字符数（必须 < chunk_size）

    Returns:
        RagChunk 列表（按 chunk_index 升序）。文档为空时返回 []。
    """
    if doc.is_empty:
        return []

    chunk_size = max(64, min(int(chunk_size), MAX_TEXT_LENGTH))
    overlap = max(0, min(int(overlap), chunk_size // 2))

    ext = (doc.ext or "").lower()
    if ext in (".md", ".markdown", ".rst"):
        blocks = _split_markdown(doc.content, chunk_size)
    elif ext in _CODE_EXTENSIONS:
        blocks = _split_code(doc.content, chunk_size)
    else:
        blocks = _split_plain(doc.content, chunk_size)

    # 兜底：对超长 block 再切窗口（Markdown/代码切分可能仍留下超长块）
    texts: list[tuple[str, str]] = []      # (heading_path, text)
    for heading_path, block in blocks:
        if len(block) <= chunk_size:
            texts.append((heading_path, block))
        else:
            for piece in _pack_windows(block, chunk_size, overlap):
                texts.append((heading_path, piece))

    # 过滤空片
    texts = [(h, t) for h, t in texts if t.strip()]
    if not texts:
        return []

    total = len(texts)
    return [
        RagChunk.build(
            source=doc.source,
            content=text,
            chunk_index=idx,
            doc_id=doc.doc_id,
            doc_hash=doc.doc_hash,
            heading_path=heading,
            ext=doc.ext,
            chunk_count=total,
        )
        for idx, (heading, text) in enumerate(texts)
    ]


def split_documents(
    docs: list[RagDocument],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[RagChunk]:
    """批量切分（多个文档的 chunk_index 各自从 0 开始）。"""
    chunks: list[RagChunk] = []
    for doc in docs:
        chunks.extend(split_document(doc, chunk_size, overlap))
    return chunks


# ═══════════════════════════════════════════════════════════════════
# 策略 1：Markdown 结构切分
# ═══════════════════════════════════════════════════════════════════

def _split_markdown(text: str, chunk_size: int) -> list[tuple[str, str]]:
    """
    按标题层级切分，返回 [(heading_path, block)]。

    heading_path 形如 "项目架构 > 模块划分"，作为 metadata 存下来，
    检索命中后前端可以展示"这段来自文档的哪一节"。

    代码块（``` 围栏）不会被标题切断 —— 围栏内的 # 不当作标题。
    """
    lines = text.splitlines()
    blocks: list[tuple[str, str]] = []

    heading_stack: list[tuple[int, str]] = []   # [(level, title)]
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal buf
        body = "\n".join(buf).strip()
        buf = []
        if body:
            blocks.append((" > ".join(t for _, t in heading_stack), body))

    for line in lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence

        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            flush()                                  # 标题出现 → 上一节结束
            level = len(m.group(1))
            title = m.group(2).strip()[:80]
            # 弹出同级或更深级的旧标题，保持栈的层级正确
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            continue

        buf.append(line)

    flush()
    return blocks or [("", text)]


# ═══════════════════════════════════════════════════════════════════
# 策略 2：代码切分
# ═══════════════════════════════════════════════════════════════════

_CODE_EXTENSIONS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs",
    ".rb", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt", ".scala",
    ".php", ".sh", ".sql", ".vue", ".proto", ".graphql",
}


def _split_code(text: str, chunk_size: int) -> list[tuple[str, str]]:
    """
    按空行分块，遇到 def/class 等定义行强制开新块（尽量不切断函数）。

    之后由 _greedy_pack 把小块贪心合并到接近 chunk_size，
    避免"一个 5 行的函数=一个 chunk"导致的碎片化。
    """
    lines = text.splitlines()
    blocks: list[str] = []
    buf: list[str] = []

    def flush_buf() -> None:
        nonlocal buf
        if buf:
            body = "\n".join(buf).strip()
            if body:
                blocks.append(body)
            buf = []

    for line in lines:
        # 定义行 + 当前 buf 非空 → 先收尾，让定义单独起块
        if _DEF_RE.match(line) and buf and buf[-1].strip() != "":
            flush_buf()
        buf.append(line)
        # 空行且 buf 已经够大 → 收尾
        if line.strip() == "" and sum(len(x) for x in buf) >= chunk_size:
            flush_buf()

    flush_buf()

    if not blocks:
        return [("", text)]

    return [("", b) for b in _greedy_pack(blocks, chunk_size)]


def _greedy_pack(blocks: list[str], chunk_size: int) -> list[str]:
    """把小块贪心合并，使每组合计长度尽量接近 chunk_size。"""
    packed: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for b in blocks:
        b_len = len(b)
        if cur and cur_len + b_len + 1 > chunk_size:
            packed.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(b)
        cur_len += b_len + 1
        # 单块就超长 → 立即收尾（后续由 _pack_windows 再切）
        if cur_len >= chunk_size:
            packed.append("\n".join(cur))
            cur, cur_len = [], 0

    if cur:
        packed.append("\n".join(cur))
    return packed


# ═══════════════════════════════════════════════════════════════════
# 策略 3：纯文本切分
# ═══════════════════════════════════════════════════════════════════

def _split_plain(text: str, chunk_size: int) -> list[tuple[str, str]]:
    """按段落（空行）分块，再贪心合并。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    return [("", b) for b in _greedy_pack(paragraphs, chunk_size)]


# ═══════════════════════════════════════════════════════════════════
# 兜底：固定字符窗口
# ═══════════════════════════════════════════════════════════════════

def _pack_windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    固定窗口滑动切分，断句时优先在句末标点处断开。

    Args:
        text:       待切文本
        chunk_size: 窗口大小
        overlap:    重叠字符数

    Returns:
        文本片段列表
    """
    if len(text) <= chunk_size:
        return [text]

    pieces: list[str] = []
    step = max(1, chunk_size - overlap)
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        # 不在末尾 → 尝试回退到最近的句末标点，避免切断句子
        if end < n:
            window = text[start:end]
            cut = _last_sentence_boundary(window, chunk_size)
            if cut > chunk_size // 2:          # 回退不能太多，否则窗口过短
                end = start + cut

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= n:
            break
        start = max(end - overlap, start + 1)   # 保证一定前进，避免死循环

    return pieces


def _last_sentence_boundary(window: str, chunk_size: int) -> int:
    """
    在 window 中找最后一个句末标点位置，返回其后的偏移（切断点）。

    找不到返回 0（调用方判断 0 > chunk_size//2 不成立 → 不回退）。
    只在窗口后半段找，避免把 chunk 切得太短。
    """
    search_from = len(window) // 2
    idx = -1
    for m in _SENTENCE_END_RE.finditer(window, search_from):
        idx = m.end()
    return idx if idx > 0 else 0
