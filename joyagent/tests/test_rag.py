"""
RAG 模块测试。

分三类：
  1. 纯逻辑（切分 / 模型 / 加载器 / 上下文拼装）—— 不需要向量库，随时可跑
  2. 接口契约（请求模型默认值）—— 保证向后兼容
  3. 集成（真实 ChromaDB + embedding 往返）—— 需要设置 RAG_INTEGRATION=1 才会执行

运行：
  pytest tests/test_rag.py
  RAG_INTEGRATION=1 pytest tests/test_rag.py     # 含真实向量库往返
"""

import os
import tempfile

import pytest

# RAG 依赖（chromadb / sentence-transformers）缺失时整模块跳过，
# 避免在没有向量库依赖的环境里误报失败。
try:
    from app.rag.loaders import (
        UnsupportedDocumentError,
        load_bytes,
        load_dir,
        load_file,
    )
    from app.rag.models import RagChunk, RagDocument, RagHit, compute_chunk_id
    from app.rag.retriever import format_context, to_sources
    from app.rag.splitters import split_document, split_documents
except ImportError as e:                                   # pragma: no cover
    pytest.skip(f"RAG 依赖未安装，跳过: {e}", allow_module_level=True)


# ═══════════════════════════════════════════════════════════════════
# 1. 切分器
# ═══════════════════════════════════════════════════════════════════

SAMPLE_MD = """# 项目架构

## 模块划分

JoyAgent 分为 API 层、Agent 层、Memory 层。
每一层职责单一，通过依赖注入解耦。

## 数据流

请求进入 API 层后，由 Agent 层编排，Memory 层负责上下文。
"""


def _md_doc() -> RagDocument:
    return RagDocument(source="docs/arch.md", content=SAMPLE_MD, ext=".md")


def test_markdown_split_keeps_heading_path():
    """Markdown 按标题切分，并保留标题路径作为 metadata。"""
    chunks = split_document(_md_doc(), chunk_size=200, overlap=40)

    assert len(chunks) >= 2
    paths = [c.metadata["heading_path"] for c in chunks]
    assert any("模块划分" in p for p in paths)
    assert any("数据流" in p for p in paths)
    # 标题路径应带上级标题
    assert any("项目架构" in p for p in paths)


def test_chunk_index_and_count_are_consistent():
    """chunk_index 从 0 连续递增，chunk_count 等于总数。"""
    chunks = split_document(_md_doc(), chunk_size=120, overlap=20)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["chunk_count"] == len(chunks) for c in chunks)


def test_chunk_content_respects_size_limit():
    """任何切片都不应明显超过 chunk_size（超长单块会被兜底再切）。"""
    huge = "x" * 5000 + "。\n" + "y" * 5000
    doc = RagDocument(source="huge.md", content=huge, ext=".md")

    chunks = split_document(doc, chunk_size=800, overlap=120)
    assert chunks
    assert all(len(c.content) <= 800 for c in chunks)


def test_chunk_id_is_content_addressed():
    """同一文档切两次得到相同 ID —— 重复入库可覆盖而非翻倍。"""
    a = split_document(_md_doc(), chunk_size=200, overlap=40)
    b = split_document(_md_doc(), chunk_size=200, overlap=40)

    assert [c.id for c in a] == [c.id for c in b]
    # ID 由 doc_id + index + content 决定
    assert compute_chunk_id(a[0].doc_id, 0, a[0].content) == a[0].id


def test_empty_document_yields_no_chunk():
    assert split_document(RagDocument(source="e.md", content="", ext=".md")) == []
    assert split_document(RagDocument(source="e.md", content="   \n ", ext=".md")) == []


def test_doc_hash_detects_content_change():
    """内容变更 → doc_hash 与 doc_id 变化（热更新的依据）。"""
    d1 = RagDocument(source="a.md", content="版本一")
    d2 = RagDocument(source="a.md", content="版本二")

    assert d1.doc_hash != d2.doc_hash
    assert d1.doc_id != d2.doc_id


def test_split_documents_handles_multiple_files():
    docs = [
        RagDocument(source="a.md", content="# A\n\n内容 A", ext=".md"),
        RagDocument(source="b.txt", content="内容 B", ext=".txt"),
    ]
    chunks = split_documents(docs, chunk_size=500, overlap=50)

    sources = {c.source for c in chunks}
    assert sources == {"a.md", "b.txt"}
    # 每个文档的 chunk_index 各自从 0 开始
    assert min(c.chunk_index for c in chunks) == 0


# ═══════════════════════════════════════════════════════════════════
# 2. 加载器
# ═══════════════════════════════════════════════════════════════════

def test_load_bytes_reads_utf8():
    doc = load_bytes("你好，JoyAgent。\n第二行".encode("utf-8"), "note.txt")

    assert doc.source == "note.txt"
    assert doc.ext == ".txt"
    assert "JoyAgent" in doc.content
    assert doc.doc_hash


def test_load_bytes_rejects_binary():
    with pytest.raises(UnsupportedDocumentError):
        load_bytes(b"\x00\x01\x02\x03" * 10, "blob.bin")


def test_load_file_and_dir(tmp_path):
    """目录导入：跳过黑名单目录与非白名单扩展名。"""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "__pycache__").mkdir()

    (tmp_path / "a.md").write_text("# 标题\n内容一", encoding="utf-8")
    (tmp_path / "sub" / "b.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    (tmp_path / "sub" / "__pycache__" / "c.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "d.bin").write_bytes(b"\x00\x01\x02")

    docs, _skipped = load_dir(str(tmp_path))
    sources = sorted(d.source for d in docs)

    assert os.path.join("sub", "b.py") in sources
    assert "a.md" in sources
    # __pycache__ 与 .bin 都不应出现
    assert not any("__pycache__" in s for s in sources)
    assert not any(s.endswith(".bin") for s in sources)


def test_load_dir_enforces_whitelist(tmp_path):
    outside = str(tmp_path / "outside")
    with pytest.raises(ValueError):
        load_dir(str(tmp_path), allowed_dirs=[outside])

    # 白名单命中时正常返回
    docs, _ = load_dir(str(tmp_path), allowed_dirs=[str(tmp_path)])
    assert isinstance(docs, list)


def test_load_dir_rejects_missing_path(tmp_path):
    with pytest.raises(ValueError):
        load_dir(str(tmp_path / "nope"))


def test_load_file_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_file(str(tmp_path / "missing.md"))


# ═══════════════════════════════════════════════════════════════════
# 3. 上下文拼装 / 引用来源
# ═══════════════════════════════════════════════════════════════════

def _hit(source: str, content: str, score: float) -> RagHit:
    return RagHit(
        chunk=RagChunk.build(
            source=source, content=content, chunk_index=0,
            doc_id="abc123", heading_path="项目架构 > 模块划分", ext=".md",
        ),
        score=score,
    )


def test_format_context_empty_hits():
    assert format_context([]) == ""


def test_format_context_includes_source_and_content():
    hits = [_hit("docs/arch.md", "JoyAgent 分为三层。", 0.87)]
    ctx = format_context(hits)

    assert "docs/arch.md" in ctx
    assert "JoyAgent 分为三层。" in ctx
    assert "0.87" in ctx


def test_format_context_respects_max_chars():
    hits = [_hit(f"doc{i}.md", "内容" * 500, 0.9 - i * 0.01) for i in range(10)]
    ctx = format_context(hits, max_chars=1000)

    assert len(ctx) <= 1400          # 至少塞一条，允许最后一条略微超出
    assert ctx.count("### [") <= 10


def test_to_sources_shape():
    hits = [_hit("docs/arch.md", "片段内容", 0.75)]
    sources = to_sources(hits)

    assert len(sources) == 1
    s = sources[0]
    assert s["source"] == "docs/arch.md"
    assert s["content"] == "片段内容"
    assert s["score"] == 0.75
    assert s["heading_path"] == "项目架构 > 模块划分"
    assert "doc_id" in s


# ═══════════════════════════════════════════════════════════════════
# 4. 接口契约（向后兼容）
# ═══════════════════════════════════════════════════════════════════

def test_chat_request_rag_fields_default_off():
    """新字段必须带默认值 —— 老的调用方行为完全不变。"""
    pytest.importorskip("fastapi")
    from app.api.agent import ChatRequest

    req = ChatRequest(message="读取 main.py")
    assert req.use_rag is False
    assert req.rag_collection is None
    assert req.rag_top_k is None

    req2 = ChatRequest(message="x", use_rag=True, rag_collection="kb2", rag_top_k=8)
    assert req2.use_rag is True
    assert req2.rag_collection == "kb2"
    assert req2.rag_top_k == 8


def test_rag_search_request_defaults():
    pytest.importorskip("fastapi")
    from app.api.rag import SearchRequest

    req = SearchRequest(query="项目有哪些模块")
    assert req.collection is None
    assert req.top_k is None
    assert req.score_threshold is None


# ═══════════════════════════════════════════════════════════════════
# 5. 集成：真实向量库往返（需 RAG_INTEGRATION=1）
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not os.getenv("RAG_INTEGRATION"),
    reason="需要真实 ChromaDB + embedding 模型，设置 RAG_INTEGRATION=1 启用",
)
@pytest.mark.asyncio
async def test_store_ingest_and_search_roundtrip():
    """入库 → 检索 → 命中同一文档；重复入库不翻倍。"""
    from app.rag.pipeline import ingest_documents
    from app.rag.store import KnowledgeStore

    with tempfile.TemporaryDirectory() as tmp:
        store = KnowledgeStore(persist_dir=tmp)

        docs = [RagDocument(source="arch.md", content=SAMPLE_MD, ext=".md")]
        result = await ingest_documents(docs, collection="test_kb", store=store)
        assert result.chunks > 0

        # 幂等：再次入库不应该让切片数翻倍
        count_after_first = store.count("test_kb")
        await ingest_documents(docs, collection="test_kb", store=store)
        assert store.count("test_kb") == count_after_first

        # 检索
        from app.rag.retriever import retrieve as rag_retrieve
        hits = await rag_retrieve("项目有哪些模块", collection="test_kb",
                                  top_k=3, store=store)
        assert hits
        assert all(h.source == "arch.md" for h in hits)

        # 删除来源
        assert store.delete_source("arch.md", collection="test_kb") > 0
        assert store.count("test_kb") == 0
