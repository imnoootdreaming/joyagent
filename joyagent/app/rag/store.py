from __future__ import annotations
"""
RAG 知识库存储层 —— ChromaDB 上的独立集合管理。

为什么独立成 KnowledgeStore，而不是往 LongTermMemory.COLLECTIONS 里加一项：
  LongTermMemory 的三个集合（code/conversation/task）是 Agent 的"记忆语义空间"，
  retrieve_relevant_context() 会对它们做三路并行召回并注入 prompt。
  把外部文档混进去会污染记忆召回结果（搜文档时召回对话摘要）。
  所以知识库用 kb_ 前缀的独立集合，互不干扰，可单独 reset。

与 LongTermMemory 共享同一份 ChromaDB 持久化目录（默认 ./data/chroma）：
  PersistentClient 在相同 path + settings 下复用缓存的 System 实例，
  不会产生 SQLite 锁冲突。

使用方式：
  store = get_knowledge_store()
  await store.ingest(chunks)                     # 写入（含 embedding）
  hits = store.search("knowledge", query_vec, 4) # 检索
"""

import asyncio
import os
import threading
import time
from typing import Optional

import chromadb
from chromadb.config import Settings

from app.memory.embeddings import EmbeddingService, get_embedding_service
from app.rag.models import RagChunk, RagHit


class KnowledgeStore:
    """
    ChromaDB 上的 RAG 知识库。

    集合命名：逻辑名 "knowledge" → 实际 ChromaDB 集合名 "kb_knowledge"。
    加前缀是为了一眼区分知识库与记忆集合。
    """

    COLLECTION_PREFIX = "kb_"
    DEFAULT_PERSIST_DIR = "./data/chroma"

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        连接 ChromaDB（集合按需懒创建）。

        Raises:
            RuntimeError: ChromaDB 连接失败
        """
        self.persist_dir = os.path.abspath(
            persist_dir or os.getenv("CHROMA_PERSIST_DIR", self.DEFAULT_PERSIST_DIR)
        )
        self._embedding_service = embedding_service or get_embedding_service()

        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB at '{self.persist_dir}': {e}\n"
                f"Tips: set CHROMA_PERSIST_DIR env var to change path"
            ) from e

        self._lock = threading.Lock()
        self._collections: dict[str, chromadb.Collection] = {}
        self._known: set[str] = set()          # 已确认存在的集合，避免每次检索都探活

        print(f"  [rag] KnowledgeStore online (persist_dir={self.persist_dir})")

    # ── 集合管理 ────────────────────────────────────────────────

    def collection(self, name: str) -> chromadb.Collection:
        """获取（或创建）知识库集合，cosine 空间。"""
        with self._lock:
            if name not in self._collections:
                self._collections[name] = self.client.get_or_create_collection(
                    name=f"{self.COLLECTION_PREFIX}{name}",
                    embedding_function=self._embedding_service.ef,
                    metadata={"description": f"RAG knowledge base: {name}",
                              "hnsw:space": "cosine"},
                )
                self._known.add(name)
            return self._collections[name]

    def has_collection(self, name: str) -> bool:
        """集合是否已存在（不触发创建）。结果会被缓存，避免每次检索都探活。"""
        if name in self._known:
            return True
        try:
            self.client.get_collection(
                name=f"{self.COLLECTION_PREFIX}{name}",
                embedding_function=self._embedding_service.ef,
            )
            self._known.add(name)
            return True
        except Exception:
            return False

    def collection_names(self) -> list[str]:
        """列出所有知识库集合的**逻辑名**（去掉 kb_ 前缀）。"""
        try:
            # 不同 chromadb 版本返回 Collection 对象或纯名字 —— 两种都兼容
            raw = [getattr(c, "name", c) for c in self.client.list_collections()]
        except Exception:
            raw = []
        prefix = self.COLLECTION_PREFIX
        return [n[len(prefix):] for n in raw if n.startswith(prefix)]

    # ── 写入 ────────────────────────────────────────────────────

    async def ingest(
        self,
        chunks: list[RagChunk],
        collection: str = "knowledge",
    ) -> int:
        """
        批量写入切片（含 embedding），并按来源做幂等覆盖。

        幂等策略（热更新）：
          按 source 分组 → 先 delete(where={"source": s}) 再 add。
          这样同一文档重新入库不会残留旧切片，也不会翻倍。
          chunk id 是内容寻址的，内容未变时 ChromaDB 会覆盖同一 id。

        CPU 密集的 embedding 通过 asyncio.to_thread 脱离事件循环，
        避免入库期间卡死 FastAPI（包括 WebSocket 心跳）。

        Returns:
            实际写入的切片数
        """
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        start = time.time()

        embeddings = await asyncio.to_thread(
            self._embedding_service.embed_batch, texts
        )

        col = self.collection(collection)

        with self._lock:
            # 先删后增：按来源清理旧切片
            sources = sorted({c.source for c in chunks})
            for src in sources:
                try:
                    col.delete(where={"source": src})
                except Exception:
                    pass            # 首次入库时无旧数据，忽略

            col.add(
                ids=[c.id for c in chunks],
                documents=texts,
                metadatas=[c.metadata for c in chunks],
                embeddings=list(embeddings),
            )

        elapsed_ms = int((time.time() - start) * 1000)
        print(f"  [rag] ingest {len(chunks)} chunks from {len(sources)} source(s) "
              f"into '{collection}' ({elapsed_ms} ms)")
        return len(chunks)

    # ── 检索 ────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        collection: str = "knowledge",
        top_k: int = 4,
    ) -> list[RagHit]:
        """
        向量检索（同步）。

        Args:
            query_embedding: 查询向量
            collection:      知识库逻辑名
            top_k:           返回条数

        Returns:
            RagHit 列表（按相似度降序）。集合不存在 / 出错时返回 []。
        """
        if not self.has_collection(collection):
            return []

        try:
            col = self.collection(collection)
            total = col.count()
            if total == 0:
                return []

            results = col.query(
                query_embeddings=[list(query_embedding)],
                n_results=min(top_k, total),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"  [rag] search failed on '{collection}': {e}")
            return []

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        hits: list[RagHit] = []
        for idx, doc in enumerate(documents):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            dist = distances[idx] if idx < len(distances) else 2.0
            # cosine distance → 相似度（与 app/memory/long_term.py 一致）
            score = max(0.0, min(1.0, 1.0 - float(dist) / 2.0))
            hits.append(RagHit(
                chunk=RagChunk(
                    id=f"{collection}_{idx}",
                    doc_id=str(meta.get("doc_id", "")),
                    source=str(meta.get("source", "unknown")),
                    content=doc or "",
                    chunk_index=int(meta.get("chunk_index", idx) or 0),
                    metadata=dict(meta or {}),
                ),
                score=round(score, 4),
            ))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    # ── 管理 / 统计 ──────────────────────────────────────────────

    def count(self, collection: str = "knowledge") -> int:
        """集合内的切片数量。"""
        if not self.has_collection(collection):
            return 0
        try:
            return self.collection(collection).count()
        except Exception:
            return 0

    def list_sources(self, collection: str = "knowledge") -> list[dict]:
        """
        列出集合内所有来源及其切片数（供前端展示"知识库里有哪些文档"）。

        Returns:
            [{"source": "docs/a.md", "chunks": 12, "doc_hash": "...", "ext": ".md"}, ...]
        """
        if not self.has_collection(collection):
            return []
        try:
            data = self.collection(collection).get(include=["metadatas"])
        except Exception:
            return []

        agg: dict[str, dict] = {}
        for meta in (data.get("metadatas") or []):
            meta = meta or {}
            src = str(meta.get("source", "unknown"))
            if src not in agg:
                agg[src] = {
                    "source": src,
                    "chunks": 0,
                    "doc_hash": str(meta.get("doc_hash", "")),
                    "ext": str(meta.get("ext", "")),
                    "ingested_at": str(meta.get("ingested_at", "")),
                }
            agg[src]["chunks"] += 1

        return sorted(agg.values(), key=lambda d: d["source"])

    def delete_source(self, source: str, collection: str = "knowledge") -> int:
        """
        按来源删除切片（热更新 / 用户删除文档）。

        Returns:
            删除的切片数（集合为空或不存在时返回 0）
        """
        if not self.has_collection(collection):
            return 0
        try:
            col = self.collection(collection)
            before = col.count()
            with self._lock:
                col.delete(where={"source": source})
            after = col.count()
            removed = max(0, before - after)
            print(f"  [rag] deleted {removed} chunks of '{source}' from '{collection}'")
            return removed
        except Exception as e:
            print(f"  [rag] delete_source failed: {e}")
            return 0

    def reset(self, collection: str = "knowledge") -> bool:
        """清空整个知识库集合（不可逆）。"""
        if not self.has_collection(collection):
            return False
        try:
            col = self.collection(collection)
            with self._lock:
                ids = col.get(include=[])["ids"]
                if ids:
                    col.delete(ids=ids)
            print(f"  [rag] reset collection '{collection}'")
            return True
        except Exception as e:
            print(f"  [rag] reset failed: {e}")
            return False

    def health_check(self, collection: str = "knowledge") -> dict:
        """
        健康检查 + 统计。

        Returns:
            {"ok": True, "collection": "knowledge", "chunks": N, "documents": M, ...}
            或 {"ok": False, "error": "..."}
        """
        try:
            chunks = self.count(collection)
            sources = self.list_sources(collection)
            return {
                "ok": True,
                "persist_dir": self.persist_dir,
                "collection": collection,
                "chunks": chunks,
                "documents": len(sources),
                "sources": [s["source"] for s in sources],
                "embedding_model": self._embedding_service.model_name,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# 全局单例
# ═══════════════════════════════════════════════════════════════════

_knowledge_store: Optional[KnowledgeStore] = None
_store_lock = threading.Lock()


def get_knowledge_store(
    persist_dir: Optional[str] = None,
    embedding_service: Optional[EmbeddingService] = None,
) -> KnowledgeStore:
    """获取全局 KnowledgeStore 单例（线程安全懒加载）。"""
    global _knowledge_store
    if _knowledge_store is not None:
        return _knowledge_store

    with _store_lock:
        if _knowledge_store is None:
            _knowledge_store = KnowledgeStore(
                persist_dir=persist_dir,
                embedding_service=embedding_service,
            )
        return _knowledge_store


def reset_knowledge_store() -> None:
    """重置单例（测试 / 数据迁移用）。不删除磁盘数据。"""
    global _knowledge_store
    with _store_lock:
        _knowledge_store = None
