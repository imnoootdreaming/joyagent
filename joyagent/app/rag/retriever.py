from __future__ import annotations
"""
RAG 检索器 —— 对话链路与 HTTP API 共用的检索入口。

三个纯函数式入口：
  retrieve(query, collection, top_k, score_threshold) -> list[RagHit]
  format_context(hits, max_chars)                     -> str   （注入 prompt）
  to_sources(hits)                                    -> list[dict]（返回前端）

设计原则：**RAG 是增强，不是依赖**。
  集合不存在 / ChromaDB 不可用 / embedding 失败 → 一律返回空值，
  只打一条 warning 日志，绝不让对话因为 RAG 挂掉而中断。
"""

import asyncio
from typing import Optional

from app.core.config import Config
from app.memory.embeddings import get_embedding_service
from app.rag.models import RagHit
from app.rag.store import KnowledgeStore, get_knowledge_store

# 上下文块的标题（与 retrieve_relevant_context 的 Markdown 风格保持一致）
_CONTEXT_HEADER = "## Retrieved Knowledge Base Context"


async def retrieve(
    query: str,
    collection: str = None,
    top_k: int = None,
    score_threshold: float = None,
    store: Optional[KnowledgeStore] = None,
) -> list[RagHit]:
    """
    检索知识库，返回相关性达标的命中结果。

    Args:
        query:           检索Query（通常直接用用户的问题）
        collection:      知识库逻辑名，默认 Config.RAG_COLLECTION
        top_k:           召回条数，默认 Config.RAG_TOP_K
        score_threshold: 相关性阈值（0~1），低于此值的命中被丢弃
        store:           KnowledgeStore 实例，默认用全局单例

    Returns:
        RagHit 列表（按 score 降序，已过阈值）。任何异常都返回 []。
    """
    collection = collection or Config.RAG_COLLECTION
    top_k = top_k or Config.RAG_TOP_K
    score_threshold = (
        Config.RAG_SCORE_THRESHOLD if score_threshold is None else score_threshold
    )

    if not query or not query.strip():
        return []

    try:
        store = store or get_knowledge_store()
    except Exception as e:
        print(f"  [rag] store unavailable — skipping retrieval: {e}")
        return []

    # 空库直接返回，省掉一次无谓的 embedding
    if store.count(collection) == 0:
        return []

    try:
        emb = get_embedding_service()
        # sentence-transformers 推理是 CPU 密集同步调用 → 脱离事件循环
        query_vec = await asyncio.to_thread(emb.embed, query)
    except Exception as e:
        print(f"  [rag] embedding failed — skipping retrieval: {e}")
        return []

    hits = store.search(query_vec, collection=collection, top_k=top_k)

    filtered = [h for h in hits if h.score >= score_threshold]
    if hits and not filtered:
        print(f"  [rag] {len(hits)} hit(s) below threshold {score_threshold} — ignored")
    return filtered


def format_context(hits: list[RagHit], max_chars: int = None) -> str:
    """
    把命中结果拼成可注入 prompt 的 Markdown 上下文块。

    格式与 app/memory/long_term.py::retrieve_relevant_context() 保持一致，
    这样 LLM 侧看到的"记忆上下文"和"知识库上下文"风格统一。

    Args:
        hits:      检索命中
        max_chars: 总长度上限（保护 LLM 上下文窗口），默认 Config 值

    Returns:
        Markdown 文本；无命中时返回 ""
    """
    if not hits:
        return ""

    max_chars = max_chars or Config.RAG_MAX_CONTEXT_CHARS

    lines = [
        _CONTEXT_HEADER,
        "",
        "以下是从知识库中检索到的相关内容，请优先依据这些内容回答；",
        "如果以下内容不足以回答问题，再结合你自己的知识回答。",
        "",
    ]

    used = sum(len(l) for l in lines)
    for idx, hit in enumerate(hits, start=1):
        heading = hit.chunk.metadata.get("heading_path", "")
        location = f"{hit.source}"
        if heading:
            location = f"{hit.source} § {heading}"
        block = (
            f"### [{idx}] {location} (similarity: {hit.score:.2f})\n"
            f"```\n{hit.content}\n```\n"
        )
        if used + len(block) > max_chars:
            # 至少塞进第一条，避免上下文全空
            if used <= sum(len(l) for l in lines):
                lines.append(_truncate(block, max_chars - used))
            break
        lines.append(block)
        used += len(block)

    return "\n".join(lines).strip()


def to_sources(hits: list[RagHit]) -> list[dict]:
    """
    把命中结果转成响应体 / 前端可直接渲染的引用来源列表。

    Returns:
        [{"source", "score", "chunk_index", "content", "heading_path", "doc_id"}, ...]
    """
    return [h.to_source_dict() for h in hits]


def retrieve_sync(
    query: str,
    collection: str = None,
    top_k: int = None,
) -> list[RagHit]:
    """同步版本（供非 async 上下文使用，如工具函数、脚本）。"""
    try:
        return asyncio.run(retrieve(query, collection, top_k))
    except RuntimeError:
        # 已在事件循环中运行时（如 Jupyter）无法用 asyncio.run —— 退化为直接检索
        return _retrieve_blocking(query, collection, top_k)


def _retrieve_blocking(
    query: str,
    collection: Optional[str],
    top_k: Optional[int],
) -> list[RagHit]:
    """绕过事件循环直接做一次同步检索。"""
    collection = collection or Config.RAG_COLLECTION
    top_k = top_k or Config.RAG_TOP_K
    try:
        store = get_knowledge_store()
        vec = get_embedding_service().embed(query)
        hits = store.search(vec, collection=collection, top_k=top_k)
        return [h for h in hits if h.score >= Config.RAG_SCORE_THRESHOLD]
    except Exception as e:
        print(f"  [rag] sync retrieval failed: {e}")
        return []


def _truncate(text: str, limit: int) -> str:
    """按字符截断（limit <= 0 时返回空串）。"""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n…（已截断）"
