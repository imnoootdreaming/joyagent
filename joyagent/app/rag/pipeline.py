from __future__ import annotations
"""
RAG 入库流水线 —— 把"文档 → 切分 → embedding → 写入"串成一步。

  loaders.load_*  →  splitters.split_documents  →  store.ingest

抽出来是为了让 API 层保持薄：路由只负责解析请求和组装响应，
真正的编排逻辑（去重、跳过、统计耗时）放在这里，也方便单测。
"""

import asyncio
import time
from dataclasses import dataclass, field

from app.core.config import Config
from app.rag.models import RagChunk, RagDocument
from app.rag.splitters import split_documents
from app.rag.store import KnowledgeStore, get_knowledge_store


@dataclass
class IngestResult:
    """一次入库操作的统计结果（直接作为 HTTP 响应体返回）。"""
    documents: int = 0                       # 成功入库的文档数
    chunks: int = 0                          # 写入的切片数
    skipped: list[str] = field(default_factory=list)   # 被跳过的文件及原因
    collection: str = ""
    elapsed_ms: int = 0


async def ingest_documents(
    docs: list[RagDocument],
    collection: str = None,
    chunk_size: int = None,
    overlap: int = None,
    store: KnowledgeStore = None,
    skipped: list[str] = None,
) -> IngestResult:
    """
    把一批文档切分并写入知识库。

    Args:
        docs:        RagDocument 列表（已由 loaders 加载完成）
        collection:  目标知识库，默认 Config.RAG_COLLECTION
        chunk_size:  切片大小，默认 Config.RAG_CHUNK_SIZE
        overlap:     切片重叠，默认 Config.RAG_CHUNK_OVERLAP
        store:       KnowledgeStore，默认全局单例
        skipped:     上游（目录扫描）记录的跳过项，会合并进结果

    Returns:
        IngestResult
    """
    collection = collection or Config.RAG_COLLECTION
    chunk_size = chunk_size or Config.RAG_CHUNK_SIZE
    overlap = overlap or Config.RAG_CHUNK_OVERLAP

    start = time.time()
    result = IngestResult(collection=collection, skipped=list(skipped or []))

    # 过滤空文档（例如只有空白字符的 txt）
    valid_docs = [d for d in docs if not d.is_empty]
    result.skipped.extend(
        f"{d.source} —— 内容为空" for d in docs if d.is_empty
    )

    if not valid_docs:
        result.elapsed_ms = int((time.time() - start) * 1000)
        return result

    # 切分是纯 CPU 字符串处理，量大时也会阻塞事件循环 → 丢到线程
    chunks: list[RagChunk] = await asyncio.to_thread(
        split_documents, valid_docs, chunk_size, overlap
    )

    if not chunks:
        result.elapsed_ms = int((time.time() - start) * 1000)
        return result

    store = store or get_knowledge_store()
    written = await store.ingest(chunks, collection=collection)

    result.documents = len(valid_docs)
    result.chunks = written
    result.elapsed_ms = int((time.time() - start) * 1000)
    return result
