from __future__ import annotations
"""
Phase 6 Step 4: Long-term Memory — ChromaDB 向量存储 + 语义检索。

LongTermMemory 是三级记忆系统的持久化层 —— 让 Agent 在会话重启后仍能"记住"
之前做过的事、遇到的错误、写过的代码。它把自然语言内容转为向量存入 ChromaDB，
通过语义相似度检索而非关键词匹配来找到相关历史。

与 Short-term Memory 的区别：
  ┌──────────────────┬─────────────────────┬────────────────────────┐
  │                  │ Short-term Memory   │ Long-term Memory       │
  ├──────────────────┼─────────────────────┼────────────────────────┤
  │ 生命周期          │ 单会话              │ 跨会话（持久化到磁盘） │
  │ 存储后端          │ 内存列表            │ ChromaDB (SQLite)      │
  │ 检索方式          │ 顺序追加 + 滑动窗口 │ 向量语义检索            │
  │ 容量              │ ~8K tokens          │ 50K+ entries           │
  │ 典型用途          │ 当前对话上下文      │ 历史代码/对话/错误经验  │
  └──────────────────┴─────────────────────┴────────────────────────┘

ChromaDB 集合设计（为什么分 collection）：
  "code"         — 代码片段、文件内容、diff 记录 (高精度检索)
  "conversation" — 对话摘要、任务描述 (中等精度)
  "task"         — 任务执行记录、错误经验、修复方案 (涵盖 Reflection Memory)
  分开存储让同类型内容在同一向量空间中比较，减少跨类型噪音。

使用方式：
  from app.memory.long_term import LongTermMemory, MemoryEntry, get_long_term_memory

  ltm = get_long_term_memory()

  # 存储
  entry = MemoryEntry(
      id="code_main_py_001",
      content="class User: ...",
      embedding=es.embed("class User: ..."),
      memory_type="code",
      metadata={"file_path": "main.py", "tags": ["model", "user"]},
      created_at="2026-06-27T10:00:00",
  )
  await ltm.store(entry)

  # 检索
  results = await ltm.search("User model class", memory_type="code", top_k=5)
  for r in results:
      print(f"相似度 {r.similarity_score:.2f}: {r.entry.content[:100]}")

  # 遗忘（隐私合规）
  await ltm.forget("code_main_py_001", "code")
"""

# ── Python 标准库 ──
import os
import uuid
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

# ── 第三方库 ──
import chromadb
from chromadb.config import Settings

# ── 项目内导入 ──
from app.memory.embeddings import get_embedding_service, EmbeddingService


# ═══════════════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryEntry:
    """
    存入 Long-term Memory 的记录。

    每一条记忆都包含原始内容 + 向量表示 + 元数据标签。
    向量表示 (embedding) 是语义检索的基础 —— 让"按意思搜索"成为可能。

    memory_type 分类：
      "code"         — 代码片段、文件内容、类/函数定义
      "conversation" — 对话摘要、用户请求、Agent 响应
      "task"         — 任务执行记录、错误经验、修复方案、工具调用历史

    id 生成建议：
      使用有意义的 ID 前缀便于调试，如 "code_main_py_20260627"。
      传入空字符串时自动生成 UUID。
    """
    id: str = ""
    content: str = ""               # 原始内容（被检索和展示的文本）
    embedding: list[float] = field(default_factory=list)  # 向量表示 (384 维)
    memory_type: str = "code"       # "code" | "conversation" | "task"
    metadata: dict = field(default_factory=dict)  # {file_path, task_id, timestamp, tags, ...}
    created_at: str = ""            # ISO 8601 时间戳

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class MemoryQueryResult:
    """
    记忆检索结果。

    similarity_score 范围 0.0 ~ 1.0（由 ChromaDB distance 转换而来）。
    1.0 = 完全匹配，0.0 = 完全无关。

    注意：entry.embedding 在检索结果中始终为空列表（节省带宽）。
    需要 embedding 时请重新调用 EmbeddingService.embed()。
    """
    entry: MemoryEntry
    similarity_score: float         # 0.0 ~ 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 内部工具
# ═══════════════════════════════════════════════════════════════════════════════

def _sanitize_metadata(metadata: dict | None, fallback_id: str) -> dict:
    """
    确保 ChromaDB metadata 非空（ChromaDB 1.5.x 强制要求至少一个键）。
    空 dict 传入 ChromaDB 会触发 ValueError。

    处理策略：
      - 非空 dict → 原样返回
      - None / {} → 注入 {"entry_id": fallback_id} 作为最小占位
    """
    if metadata and len(metadata) > 0:
        return dict(metadata)
    return {"entry_id": fallback_id}


def _is_empty_embedding(emb) -> bool:
    """
    判断 embedding 是否为空（兼容 numpy array、Python list、None）。

    sentence-transformers 返回的是 numpy.ndarray，`if not emb` 会触发
    "The truth value of an array is ambiguous" 错误。
    所以用 len() 来判断。
    """
    if emb is None:
        return True
    try:
        return len(emb) == 0
    except TypeError:
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# LongTermMemory — ChromaDB 持久化记忆
# ═══════════════════════════════════════════════════════════════════════════════

class LongTermMemory:
    """
    基于 ChromaDB 的长期记忆 —— 跨会话的语义知识库。

    职责：
      1. 存储记忆（代码、对话、任务）到持久化向量数据库
      2. 语义检索：按意思搜索最相关的历史记忆
      3. 遗忘：支持按 ID 或按元数据条件删除（隐私合规）
      4. 生命周期管理：健康检查、统计、重置

    ChromaDB 后端说明：
      - PersistentClient + SQLite：数据持久化到磁盘，重启不丢失
      - HNSW 索引：近似最近邻搜索，百万级向量也能快速检索
      - 嵌入函数：SentenceTransformerEmbeddingFunction（与 EmbeddingService 共享）

    线程安全：
      ChromaDB 的 PersistentClient 内部使用 SQLite 的 WAL 模式，
      多线程读安全，写操作有文件级锁。本类的 _lock 保护集合创建等初始化操作。
    """

    # ── 集合定义 ──
    COLLECTIONS: dict[str, str] = {
        "code":         "存储代码片段和文件内容",
        "conversation": "存储对话摘要和任务描述",
        "task":         "存储任务执行记录和错误经验",
    }

    # 默认持久化目录（相对于项目根目录）
    DEFAULT_PERSIST_DIR = "./data/chroma"

    def __init__(
        self,
        persist_dir: str = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        初始化 Long-term Memory —— 连接 ChromaDB 并确保集合存在。

        Args:
            persist_dir:      持久化目录路径。默认 ./data/chroma。
                              可通过环境变量 CHROMA_PERSIST_DIR 覆盖。
            embedding_service: EmbeddingService 实例。None 时使用全局单例。

        Raises:
            RuntimeError: ChromaDB 连接失败时抛出。
        """
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", self.DEFAULT_PERSIST_DIR)
        self.persist_dir = os.path.abspath(persist_dir)

        # ── 嵌入服务 ──
        self._embedding_service = embedding_service or get_embedding_service()

        # ── 创建 ChromaDB 客户端 ──
        start_time = time.time()
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB at '{self.persist_dir}': {e}\n"
                f"Tips:\n"
                f"  1. Check disk space and write permissions\n"
                f"  2. Set CHROMA_PERSIST_DIR env var to change path"
            ) from e

        # ── 确保集合存在 ──
        self._lock = threading.Lock()
        self.collections: dict[str, chromadb.Collection] = {}
        self._ensure_collections()

        init_time = time.time() - start_time

        print(
            f"  [long_term_memory] persist_dir={self.persist_dir}, "
            f"collections={list(self.collections.keys())}, "
            f"init_time={init_time:.2f}s"
        )

    def _ensure_collections(self) -> None:
        """
        确保 ChromaDB 集合存在（幂等操作）。

        每个集合关联 SentenceTransformerEmbeddingFunction，
        这样 ChromaDB 可以在 add()/query() 时自动计算 embedding。
        """
        with self._lock:
            for name, description in self.COLLECTIONS.items():
                self.collections[name] = self.client.get_or_create_collection(
                    name=name,
                    embedding_function=self._embedding_service.ef,
                    metadata={"description": description, "hnsw:space": "cosine"},
                )

    # ── 核心 CRUD ────────────────────────────────────────────────

    async def store(self, entry: MemoryEntry) -> str:
        """
        存储单条记忆到对应集合。

        如果 entry.embedding 为空，自动调用 EmbeddingService 计算。

        Args:
            entry: MemoryEntry 实例

        Returns:
            记忆 ID

        Raises:
            ValueError: memory_type 不在已知集合中时抛出。
        """
        collection = self.collections.get(entry.memory_type)
        if collection is None:
            raise ValueError(
                f"Unknown memory_type '{entry.memory_type}'. "
                f"Must be one of: {list(self.COLLECTIONS.keys())}"
            )

        # ── 自动计算 embedding（如果未提供；兼容 numpy array） ──
        embedding = entry.embedding
        if _is_empty_embedding(embedding):
            embedding = self._embedding_service.embed(entry.content)
            entry.embedding = embedding

        # ── 确保 metadata 非空（ChromaDB 强制要求） ──
        meta = _sanitize_metadata(entry.metadata, entry.id)

        collection.add(
            ids=[entry.id],
            documents=[entry.content],
            metadatas=[meta],
            embeddings=[embedding],
        )

        return entry.id

    async def store_batch(self, entries: list[MemoryEntry]) -> list[str]:
        """
        批量存储记忆（比逐条 store 快 3-5 倍）。

        自动按 memory_type 分组，同一类型的条目一次性写入。

        Args:
            entries: MemoryEntry 列表

        Returns:
            所有记忆 ID 列表
        """
        if not entries:
            return []

        # ── 按 memory_type 分组 ──
        groups: dict[str, list[MemoryEntry]] = {}
        for entry in entries:
            groups.setdefault(entry.memory_type, []).append(entry)

        stored_ids = []

        for mem_type, group in groups.items():
            collection = self.collections.get(mem_type)
            if collection is None:
                continue

            ids = []
            documents = []
            metadatas = []
            embeddings = []

            for entry in group:
                ids.append(entry.id)
                documents.append(entry.content)
                metadatas.append(_sanitize_metadata(entry.metadata, entry.id))

                emb = entry.embedding
                if _is_empty_embedding(emb):
                    emb = self._embedding_service.embed(entry.content)
                    entry.embedding = emb
                embeddings.append(emb)

            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            stored_ids.extend(ids)

        return stored_ids

    async def search(
        self,
        query: str,
        memory_type: str = "code",
        top_k: int = 5,
    ) -> list[MemoryQueryResult]:
        """
        语义搜索 —— 找到与 query 最相似的历史记忆。

        使用 ChromaDB 的 query() 进行向量相似度检索。
        ChromaDB 返回 distance（欧氏距离平方），转换为 similarity = 1 - distance。

        Args:
            query:       搜索查询（自然语言）
            memory_type: 限定搜索的集合（"code" | "conversation" | "task"）
            top_k:       返回最多多少条结果

        Returns:
            MemoryQueryResult 列表，按相似度降序排列

        Example:
            results = await ltm.search("User class with email field", "code", top_k=3)
            for r in results:
                print(f"{r.similarity_score:.2f}: {r.entry.metadata.get('file_path')}")
        """
        collection = self.collections.get(memory_type)
        if not collection:
            return []

        # 空集合时 query 会报错 → 返回空列表
        if collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB 返回格式：
        #   results["ids"]       = [["id1", "id2", ...]]
        #   results["documents"] = [["doc1", "doc2", ...]]
        #   results["metadatas"] = [[{...}, {...}, ...]]
        #   results["distances"] = [[0.12, 0.45, ...]]
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output = []
        for i in range(len(ids)):
            distance = distances[i] if i < len(distances) else 1.0
            # ChromaDB 使用余弦距离（在 cosine space 下），范围 [0, 2]
            # 转换为相似度：similarity = 1 - distance / 2
            similarity = max(0.0, min(1.0, 1.0 - distance / 2.0))

            metadata = metadatas[i] if i < len(metadatas) else {}

            output.append(MemoryQueryResult(
                entry=MemoryEntry(
                    id=ids[i],
                    content=documents[i] if i < len(documents) else "",
                    embedding=[],                          # 不返回 embedding 节省带宽
                    memory_type=memory_type,
                    metadata=metadata or {},
                    created_at=metadata.get("created_at", ""),
                ),
                similarity_score=round(similarity, 4),
            ))

        return output

    async def search_all(
        self,
        query: str,
        top_k_per_collection: int = 3,
    ) -> dict[str, list[MemoryQueryResult]]:
        """
        跨所有集合搜索 —— 同时检索 code、conversation、task。

        Args:
            query:                 搜索查询
            top_k_per_collection:  每个集合返回的最大结果数

        Returns:
            {"code": [...], "conversation": [...], "task": [...]}
        """
        results: dict[str, list[MemoryQueryResult]] = {}
        for mem_type in self.COLLECTIONS:
            results[mem_type] = await self.search(query, mem_type, top_k_per_collection)
        return results

    async def forget(self, memory_id: str, memory_type: str) -> bool:
        """
        删除单条记忆（用于隐私合规 / 被遗忘权）。

        Args:
            memory_id:   记忆 ID
            memory_type: 所属集合

        Returns:
            True 如果成功删除，False 如果记忆不存在或集合无效
        """
        collection = self.collections.get(memory_type)
        if not collection:
            return False

        try:
            collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    async def forget_by_filter(
        self,
        memory_type: str,
        metadata_filter: dict,
    ) -> int:
        """
        按元数据条件批量删除记忆。

        Args:
            memory_type:     目标集合
            metadata_filter: 元数据过滤条件，如 {"file_path": "deleted_file.py"}

        Returns:
            删除的记忆条数

        Example:
            # 删除所有与 deleted_file.py 相关的记忆
            deleted = await ltm.forget_by_filter("code", {"file_path": "deleted_file.py"})
        """
        collection = self.collections.get(memory_type)
        if not collection or collection.count() == 0:
            return 0

        # 先查询匹配的 IDs
        try:
            results = collection.get(
                where=metadata_filter,
                include=[],
            )
            ids = results.get("ids", [])
            if ids:
                collection.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

    # ── 统计与查询 ───────────────────────────────────────────────

    def count(self, memory_type: str = None) -> int:
        """
        统计记忆数量。

        Args:
            memory_type: 集合名称。None 返回所有集合的总数。

        Returns:
            记忆条目数
        """
        if memory_type:
            collection = self.collections.get(memory_type)
            return collection.count() if collection else 0

        return sum(c.count() for c in self.collections.values())

    def count_all(self) -> dict[str, int]:
        """统计每个集合的记忆数量。"""
        return {name: c.count() for name, c in self.collections.items()}

    def list_recent(
        self,
        memory_type: str = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """
        列出最近存储的记忆（不涉及语义搜索，纯元数据查询）。

        Args:
            memory_type: 限定集合。None = 所有集合。
            limit:       返回条数上限。

        Returns:
            MemoryEntry 列表（不含 embedding）
        """
        entries = []

        collections_to_check = (
            [self.collections[memory_type]] if memory_type
            else self.collections.values()
        )

        for collection in collections_to_check:
            if collection.count() == 0:
                continue
            # 获取最近添加的条目（ChromaDB 按插入顺序存储）
            results = collection.get(
                include=["documents", "metadatas"],
                limit=limit,
            )
            ids = results.get("ids", [])
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            for i in range(len(ids)):
                entries.append(MemoryEntry(
                    id=ids[i],
                    content=documents[i] if i < len(documents) else "",
                    memory_type=memory_type or "",
                    metadata=metadatas[i] if i < len(metadatas) else {},
                ))

        # 按 ID 排序（通常是时间顺序）并截断
        entries.sort(key=lambda e: e.id, reverse=True)
        return entries[:limit]

    # ── 健康检查 ────────────────────────────────────────────────

    def health_check(self) -> dict:
        """
        健康检查：验证 ChromaDB 连接和各集合状态。

        Returns:
            {"ok": True, "persist_dir": "...", "collections": {...}, "total_entries": N}
            或
            {"ok": False, "error": "..."}
        """
        try:
            collection_info = {}
            total = 0
            for name, col in self.collections.items():
                c = col.count()
                total += c
                collection_info[name] = c

            return {
                "ok": True,
                "persist_dir": self.persist_dir,
                "collections": collection_info,
                "total_entries": total,
                "embedding_model": self._embedding_service.model_name,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_stats(self) -> dict:
        """获取详细统计信息。"""
        info = self.health_check()
        info["embedding_cache_hit_rate"] = self._embedding_service.cache_hit_rate
        return info

    # ── 重置 ─────────────────────────────────────────────────────

    def reset_collection(self, memory_type: str) -> bool:
        """
        清空指定集合（不可逆！）。

        Args:
            memory_type: 要清空的集合名称

        Returns:
            True 成功，False 集合不存在
        """
        collection = self.collections.get(memory_type)
        if not collection:
            return False

        # 获取所有 ID 并删除
        ids = collection.get(include=[])["ids"]
        if ids:
            collection.delete(ids=ids)
        return True

    def reset_all(self) -> None:
        """清空所有集合（不可逆！仅用于测试或完全重启）。"""
        for name in self.COLLECTIONS:
            self.reset_collection(name)
        print(
            f"[long_term_memory] All collections reset. "
            f"persist_dir={self.persist_dir}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 记忆检索上下文构建器 (§6.2)
# ═══════════════════════════════════════════════════════════════════════════════

async def retrieve_relevant_context(
    query: str,
    ltm: LongTermMemory,
    top_k_code: int = 3,
    top_k_conv: int = 2,
    top_k_task: int = 3,
) -> str:
    """
    在开始新任务前，检索相关历史上下文（§6.2 完整实现）。

    从三个维度同时搜索：
      1. 代码维度  — 相关的文件、类、函数
      2. 对话维度  — 之前的讨论和决策
      3. 错误维度  — 历史上的类似错误和修复方案

    将结果格式化为可直接注入 LLM 对话的文本块。

    Args:
        query:       搜索查询（通常是用户的新请求）
        ltm:         LongTermMemory 实例
        top_k_code:  代码检索返回的最大条数
        top_k_conv:  对话检索返回的最大条数
        top_k_task:  任务/错误检索返回的最大条数

    Returns:
        格式化的"相关历史上下文"文本块，可直接追加到 LLM messages 中。
        如果没有找到相关记忆，返回空字符串。

    Example:
        from app.memory.long_term import retrieve_relevant_context, get_long_term_memory

        ltm = get_long_term_memory()
        context = await retrieve_relevant_context("add email field to User", ltm)
        if context:
            messages.insert(0, {"role": "user", "content": context})
    """
    # ── 并行搜索所有三个维度 ──
    import asyncio

    code_task = ltm.search(query, "code", top_k=top_k_code)
    conv_task = ltm.search(query, "conversation", top_k=top_k_conv)
    task_task = ltm.search(query, "task", top_k=top_k_task)

    code_results, conv_results, task_results = await asyncio.gather(
        code_task, conv_task, task_task
    )

    # ── 构建上下文文本 ──
    has_code = any(r.similarity_score > 0.3 for r in code_results)
    has_conv = any(r.similarity_score > 0.3 for r in conv_results)
    has_task = any(r.similarity_score > 0.3 for r in task_results)

    # 所有结果都不够相关 → 返回空
    if not (has_code or has_conv or has_task):
        return ""

    lines = ["## Relevant Historical Context\n"]

    # ── 代码相关 ──
    if has_code:
        lines.append("### Related code from past sessions:")
        for r in code_results:
            if r.similarity_score < 0.3:
                continue
            file_path = r.entry.metadata.get("file_path", "unknown")
            snippet = r.entry.content[:300]
            # 截断过长的代码
            if len(r.entry.content) > 300:
                snippet += "\n... (truncated)"
            lines.append(
                f"- **{file_path}** (similarity: {r.similarity_score:.2f})\n"
                f"  ```\n  {snippet}\n  ```"
            )
        lines.append("")

    # ── 对话相关 ──
    if has_conv:
        lines.append("### Related conversations:")
        for r in conv_results:
            if r.similarity_score < 0.3:
                continue
            lines.append(
                f"- (similarity: {r.similarity_score:.2f}) "
                f"{r.entry.content[:300]}"
            )
        lines.append("")

    # ── 错误经验 ──
    if has_task:
        error_results = [
            r for r in task_results
            if r.similarity_score > 0.3
            and r.entry.metadata.get("type") == "error_experience"
        ]
        task_results_non_error = [
            r for r in task_results
            if r.similarity_score > 0.3
            and r.entry.metadata.get("type") != "error_experience"
        ]

        if task_results_non_error:
            lines.append("### Related tasks:")
            for r in task_results_non_error:
                lines.append(
                    f"- (similarity: {r.similarity_score:.2f}) "
                    f"{r.entry.content[:300]}"
                )
            lines.append("")

        if error_results:
            lines.append("### Past errors & fixes (learn from history):")
            for r in error_results:
                error_type = r.entry.metadata.get("error_type", "UnknownError")
                fix_success = r.entry.metadata.get("fix_success", "unknown")
                icon = "✅" if fix_success == "True" else "⚠️"
                lines.append(
                    f"- {icon} **{error_type}** "
                    f"(similarity: {r.similarity_score:.2f})\n"
                    f"  {r.entry.content[:300]}"
                )
            lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 全局单例
# ═══════════════════════════════════════════════════════════════════════════════

_long_term_memory: Optional[LongTermMemory] = None
_ltm_lock = threading.Lock()


def get_long_term_memory(
    persist_dir: str = None,
    embedding_service: Optional[EmbeddingService] = None,
) -> LongTermMemory:
    """
    获取全局 LongTermMemory 单例（线程安全懒加载）。

    首次调用时创建 ChromaDB 连接并初始化集合，后续调用返回同一实例。
    所有 Memory 模块（Reflection 等）共享同一个 Long-term Memory 实例。

    Args:
        persist_dir:      ChromaDB 持久化目录（仅首次调用时生效）
        embedding_service: EmbeddingService 实例（仅首次调用时生效）

    Returns:
        全局单例 LongTermMemory

    Example:
        from app.memory.long_term import get_long_term_memory

        ltm = get_long_term_memory()
        results = await ltm.search("User class", "code", top_k=5)
    """
    global _long_term_memory

    if _long_term_memory is not None:
        return _long_term_memory

    with _ltm_lock:
        if _long_term_memory is not None:
            return _long_term_memory

        print(
            f"[long_term_memory] Initializing global singleton "
            f"(persist_dir={persist_dir or LongTermMemory.DEFAULT_PERSIST_DIR})..."
        )
        _long_term_memory = LongTermMemory(
            persist_dir=persist_dir,
            embedding_service=embedding_service,
        )
        return _long_term_memory


def reset_long_term_memory() -> None:
    """
    重置全局单例（用于测试或数据迁移）。

    注意：这不会删除磁盘上的 ChromaDB 数据，只是释放客户端连接。
    下次调用 get_long_term_memory() 会重新连接。
    """
    global _long_term_memory
    with _ltm_lock:
        if _long_term_memory is not None:
            print(
                f"[long_term_memory] Resetting singleton "
                f"(persist_dir={_long_term_memory.persist_dir})"
            )
            _long_term_memory = None
