from __future__ import annotations
"""
app.rag —— RAG（检索增强生成）能力包。

与 app.memory 平级：
  app.memory —— Agent 的三级记忆（短期 / 长期 / 反思），服务"记住做过什么"
  app.rag    —— 外部文档知识库，服务"引用外部资料回答问题"

两者共享 ChromaDB 与 EmbeddingService，但使用不同的集合，互不干扰。

公开 API：
  from app.rag import (
      ingest_documents,       # 入库流水线
      retrieve,               # 检索（async）
      format_context,         # 命中 → prompt 上下文
      to_sources,             # 命中 → 前端引用来源
      get_knowledge_store,    # 知识库单例
      RagDocument, RagChunk, RagHit,
  )
"""

from app.rag.models import (
    RagChunk,
    RagDocument,
    RagHit,
    compute_chunk_id,
    compute_doc_hash,
    compute_doc_id,
)
from app.rag.pipeline import IngestResult, ingest_documents
from app.rag.retriever import (
    format_context,
    retrieve,
    retrieve_sync,
    to_sources,
)
from app.rag.splitters import split_document, split_documents
from app.rag.store import KnowledgeStore, get_knowledge_store, reset_knowledge_store

__all__ = [
    # 数据模型
    "RagChunk",
    "RagDocument",
    "RagHit",
    "compute_chunk_id",
    "compute_doc_hash",
    "compute_doc_id",
    # 入库 / 切分
    "ingest_documents",
    "IngestResult",
    "split_document",
    "split_documents",
    # 检索
    "retrieve",
    "retrieve_sync",
    "format_context",
    "to_sources",
    # 存储
    "KnowledgeStore",
    "get_knowledge_store",
    "reset_knowledge_store",
]
