from __future__ import annotations
"""
Phase 6 Step 2: Embedding Service — 文本向量化服务。

EmbeddingService 是 Memory System 的语义引擎——它把自然语言文本转换为高维向量，
让机器能够"按意思搜索"而非仅仅"按关键词匹配"。Long-term Memory 和 Reflection Memory
都依赖 embedding 来实现语义检索。

为什么需要 Embedding Service 而不是简单的关键词匹配？
  1. 语义相似（semantic similarity）："修复 import 错误" 和 "解决模块导入问题"
     在关键词层面完全不同，但在语义空间中是相邻的向量
  2. 跨语言理解：中英文混合的查询也能找到相关结果
  3. 代码+自然语言混合搜索：错误信息、代码片段、修复描述都在同一向量空间中
  4. 排序精度：cosine similarity 能给出精确的 0.0~1.0 相似度分数

模型选型说明（面试用）：
  all-MiniLM-L6-v2 (本地默认)  — 80MB，384 维，轻量快速，适合 MVP + 本地运行
  all-mpnet-base-v2 (本地备选)  — 420MB，768 维，质量更好但更慢
  text-embedding-3-small (OpenAI) — 512 维，API 调用，质量最好但需付费
  text-embedding-3-large (OpenAI) — 3072 维，最高质量，成本最高

嵌入缓存策略：
  相同文本的 embedding 不会重复计算——使用内存 LRU 缓存避免浪费 CPU/GPU 资源。
  在 Fix Loop 场景中，同一个错误可能被反复分析，缓存能显著加速。

使用方式：
  from app.memory.embeddings import EmbeddingService, get_embedding_service

  es = get_embedding_service()
  vec = es.embed("how to fix ImportError")         # → list[float] (384 维)
  vecs = es.embed_batch(["error A", "error B"])    # → list[list[float]]
  dim = es.dimension                                # → 384
"""

# ── Python 标准库 ──
import time
import hashlib
import threading
from collections import OrderedDict
from typing import Optional

# ── 第三方库 ──
from chromadb.utils import embedding_functions  # ChromaDB 内置的 embedding 封装

# ── 项目内导入 ──
# (Config 由各调用方按需导入，本模块仅依赖环境变量 EMBEDDING_MODEL)


# ═══════════════════════════════════════════════════════════════════════════════
# 模型注册表
# ═══════════════════════════════════════════════════════════════════════════════

# 已知模型及其属性
_MODEL_REGISTRY: dict[str, dict] = {
    # ── Sentence-Transformers 本地模型（免费、离线可用） ──
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "max_seq_length": 256,
        "size_mb": 80,
        "provider": "sentence-transformers",
        "description": "轻量通用模型，适合 MVP 和快速迭代",
    },
    "all-mpnet-base-v2": {
        "dimension": 768,
        "max_seq_length": 384,
        "size_mb": 420,
        "provider": "sentence-transformers",
        "description": "质量更好的模型，适合对精度要求更高的场景",
    },
    "multi-qa-MiniLM-L6-cos-v1": {
        "dimension": 384,
        "max_seq_length": 512,
        "size_mb": 80,
        "provider": "sentence-transformers",
        "description": "针对问答场景优化的模型，适合语义搜索",
    },
    "all-distilroberta-v1": {
        "dimension": 768,
        "max_seq_length": 512,
        "size_mb": 290,
        "provider": "sentence-transformers",
        "description": "通用模型，速度与质量的平衡",
    },
}

# 默认模型（可通过环境变量 EMBEDDING_MODEL 覆盖）
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ═══════════════════════════════════════════════════════════════════════════════
# LRU 嵌入缓存
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingCache:
    """
    线程安全的 LRU 嵌入缓存。

    为什么需要缓存？
      1. 同一段文本（如错误信息）可能在短时间内被多次查询
      2. embedding 计算是 CPU 密集型操作——缓存能减少 90%+ 的重复计算
      3. 在 Fix Loop 场景中，同一个 error 可能被反复分析

    缓存失效策略：
      - LRU（最近最少使用）驱逐
      - 默认最大 5000 条（约 5000 × 384 × 4 bytes ≈ 7.7 MB 内存）
      - 文本内容通过 SHA256 哈希作为键

    线程安全：
      使用 threading.Lock 保护读写操作，适合多线程 Agent 并发场景。
    """

    def __init__(self, max_size: int = 5000):
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _hash(self, text: str) -> str:
        """用 SHA256 对文本生成缓存键（比存原文省内存 + 防止 Key 过长）。"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[list[float]]:
        """查询缓存。命中返回 embedding，未命中返回 None。"""
        key = self._hash(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)  # LRU → 移到末尾（最近使用）
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, text: str, embedding: list[float]) -> None:
        """存入缓存。如果超过 max_size，驱逐最久未使用的条目。"""
        key = self._hash(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = embedding
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)  # 弹出第一个（最久未使用）

    @property
    def hit_rate(self) -> float:
        """缓存命中率（0.0 ~ 1.0）。"""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def size(self) -> int:
        """当前缓存条目数。"""
        return len(self._cache)

    def clear(self) -> None:
        """清空缓存。"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# ═══════════════════════════════════════════════════════════════════════════════
# EmbeddingService — 文本向量化引擎
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingService:
    """
    文本向量化服务——Memory System 的语义搜索基石。

    职责：
      1. 将任意文本转换为固定维度的向量（embedding）
      2. 支持单条和批量 embedding（批量更高效）
      3. 透明缓存：相同文本不重复计算
      4. 管理模型生命周期（加载、健康检查）

    模型支持：
      - 本地模型（sentence-transformers）：免费、离线、隐私安全
      - 生产环境可扩展 OpenAI / Cohere 等 API embedding（见下方设计说明）

    输入约束：
      - 单次 batch 最大文本数：256 条（超过会分批处理）
      - 文本最大长度：受模型 max_seq_length 限制（超长文本自动截断）
      - 空字符串返回零向量（而非报错）

    设计说明 — 为什么 MVP 用本地模型：
      - 零依赖外部 API → 不花钱、不依赖网络
      - 80MB 模型 → 第一次加载 2-3 秒，之后常驻内存
      - 384 维 → 与 ChromaDB 的 HNSW 索引配合良好
      - 生产切换方案：只需改一行 model_name="text-embedding-3-small"
    """

    # ── 批处理限制 ──
    MAX_BATCH_SIZE = 256          # 单次 batch 最大文本数（防止 OOM）
    MAX_TEXT_LENGTH = 8192        # 预处理阶段截断到 8K 字符

    def __init__(
        self,
        model_name: str = None,
        cache_size: int = 5000,
        device: str = "cpu",
    ):
        """
        初始化 EmbeddingService —— 加载模型并预热。

        Args:
            model_name: 模型名称（如 "all-MiniLM-L6-v2"）。
                        为 None 时从环境变量 EMBEDDING_MODEL 读取，
                        仍未设置则使用 DEFAULT_EMBEDDING_MODEL。
            cache_size: 嵌入缓存最大条目数。
            device:     运行设备（"cpu" / "cuda"）。默认 "cpu"（Mac/Windows 友好）。

        Raises:
            ImportError: 如果 sentence-transformers 未安装。
            RuntimeError: 如果模型加载失败（网络问题 / 磁盘空间不足等）。
        """
        model_name = model_name or _resolve_model_name()
        self.model_name = model_name
        self.device = device

        # ── 获取模型元信息 ──
        model_info = _MODEL_REGISTRY.get(model_name, {})
        self.dimension = model_info.get("dimension", 384)
        self.max_seq_length = model_info.get("max_seq_length", 256)

        # ── 初始化嵌入缓存 ──
        self._cache = EmbeddingCache(max_size=cache_size)

        # ── 创建 ChromaDB 兼容的 embedding function ──
        # SentenceTransformerEmbeddingFunction 内部封装了 sentence-transformers 的加载逻辑
        start_time = time.time()
        try:
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name,
                device=device,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}': {e}\n"
                f"Tips:\n"
                f"  1. Check network connection (first load downloads ~80MB)\n"
                f"  2. Try 'pip install sentence-transformers' if not installed\n"
                f"  3. Set EMBEDDING_MODEL env var to switch model"
            ) from e

        load_time = time.time() - start_time

        # ── 日志 ──
        self._metrics = {
            "total_embeddings": 0,
            "total_batches": 0,
            "total_time_ms": 0.0,
        }

        print(
            f"  [embedding_service] model={model_name}, "
            f"dim={self.dimension}, "
            f"device={device}, "
            f"load_time={load_time:.1f}s, "
            f"cache_size={cache_size}"
        )

    # ── 核心 Embedding 方法 ────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """
        将单段文本转换为向量。

        Args:
            text: 任意文本（中文、英文、代码均可）

        Returns:
            384 维浮点数向量（维度取决于模型）

        Example:
            es = get_embedding_service()
            vec = es.embed("fix ImportError in main.py")  # → [0.012, -0.034, ...]
        """
        if not text or not text.strip():
            return [0.0] * self.dimension

        # ── 查缓存 ──
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        # ── 预处理：截断过长文本 ──
        processed = text[:self.MAX_TEXT_LENGTH]

        # ── 计算 embedding ──
        start = time.time()
        try:
            # ChromaDB 的 embedding function 接受 List[str] 返回 List[List[float]]
            result = self.ef([processed])
            embedding = result[0]
        except Exception as e:
            raise RuntimeError(
                f"Embedding failed for text '{text[:100]}...': {e}"
            ) from e
        elapsed = (time.time() - start) * 1000

        # ── 更新指标 ──
        self._metrics["total_embeddings"] += 1
        self._metrics["total_time_ms"] += elapsed

        # ── 缓存 & 返回 ──
        self._cache.put(text, embedding)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        批量将多段文本转换为向量（比逐条调用 embed 快 2-5 倍）。

        超大 batch 会自动分片处理，不会 OOM。

        Args:
            texts: 文本列表（最大 256 条/批次，超过自动分片）

        Returns:
            向量列表，与输入顺序一一对应。

        Example:
            es = get_embedding_service()
            vecs = es.embed_batch(["error A", "error B", "function C"])
            # → [[0.01, ...], [0.02, ...], [-0.01, ...]]
        """
        if not texts:
            return []

        results: list[Optional[list[float]]] = [None] * len(texts)
        uncached_texts: list[tuple[int, str]] = []

        # ── 第 1 遍：查缓存，区分命中/未命中 ──
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = [0.0] * self.dimension
                continue
            cached = self._cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_texts.append((i, text[:self.MAX_TEXT_LENGTH]))

        if not uncached_texts:
            return results  # 全部命中缓存

        # ── 第 2 遍：分批计算未命中的 embedding ──
        indices = [t[0] for t in uncached_texts]
        raw_texts = [t[1] for t in uncached_texts]

        all_new_embeddings: list[list[float]] = []

        for chunk_start in range(0, len(raw_texts), self.MAX_BATCH_SIZE):
            chunk = raw_texts[chunk_start:chunk_start + self.MAX_BATCH_SIZE]
            start = time.time()
            try:
                chunk_result = self.ef(chunk)
            except Exception as e:
                raise RuntimeError(
                    f"Batch embedding failed at chunk {chunk_start}: {e}\n"
                    f"Chunk size: {len(chunk)}, total texts: {len(texts)}"
                ) from e
            elapsed = (time.time() - start) * 1000

            all_new_embeddings.extend(chunk_result)

            # 更新指标
            self._metrics["total_embeddings"] += len(chunk)
            self._metrics["total_batches"] += 1
            self._metrics["total_time_ms"] += elapsed

        # ── 第 3 遍：写回缓存 + 填充结果 ──
        for idx_in_uncached, (original_idx, original_text) in enumerate(uncached_texts):
            emb = all_new_embeddings[idx_in_uncached]
            results[original_idx] = emb
            self._cache.put(original_text, emb)

        return results  # type: ignore[return-value]

    # ── 相似度计算（纯数学，无需模型推理） ──────────────────────────

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """
        计算两个向量的余弦相似度。

        返回值范围 0.0 ~ 1.0（因为 embedding 向量通常是非负的）。
        1.0 = 完全相同方向，0.0 = 正交（完全无关）。

        用于：
          - 排序搜索结果
          - 判断两段文本的语义相似程度
          - 去重：同一错误的变体检测
        """
        if len(a) != len(b):
            raise ValueError(
                f"Vector dimension mismatch: {len(a)} vs {len(b)}"
            )

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b) # 点积 * 范数乘积

    @staticmethod
    def euclidean_distance(a: list[float], b: list[float]) -> float:
        """
        计算两个向量的欧氏距离。

        ChromaDB 默认使用欧氏距离的平方进行相似度排序。
        距离越小 → 越相似。
        """
        if len(a) != len(b):
            raise ValueError(
                f"Vector dimension mismatch: {len(a)} vs {len(b)}"
            )
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    # ── 缓存管理 ──────────────────────────────────────────────────

    @property
    def cache_hit_rate(self) -> float:
        """缓存命中率。"""
        return self._cache.hit_rate

    @property
    def cache_size(self) -> int:
        """当前缓存条目数。"""
        return self._cache.size

    def clear_cache(self) -> None:
        """清空嵌入缓存。"""
        self._cache.clear()

    # ── 指标与健康检查 ───────────────────────────────────────────

    @property
    def metrics(self) -> dict:
        """获取服务运行指标。"""
        m = dict(self._metrics)
        m["cache_hit_rate"] = self.cache_hit_rate
        m["cache_size"] = self.cache_size
        if self._metrics["total_embeddings"] > 0:
            m["avg_time_ms"] = round(
                self._metrics["total_time_ms"] / self._metrics["total_embeddings"], 2
            )
        else:
            m["avg_time_ms"] = 0.0
        return m

    def health_check(self) -> dict:
        """
        健康检查：验证模型是否正常工作。

        Returns:
            {"ok": True, "model": "all-MiniLM-L6-v2", "dimension": 384, ...}
            或
            {"ok": False, "error": "..."}
        """
        try:
            test_text = "health check"
            vec = self.embed(test_text)
            return {
                "ok": True,
                "model": self.model_name,
                "dimension": self.dimension,
                "vector_len": len(vec),
                "cache_hit_rate": round(self.cache_hit_rate, 3),
                "total_embeddings": self._metrics["total_embeddings"],
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_model_info(self) -> dict:
        """获取当前模型的详细信息。"""
        info = _MODEL_REGISTRY.get(self.model_name, {})
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_seq_length": self.max_seq_length,
            "device": self.device,
            "model_info": info,
            "cache": {
                "size": self.cache_size,
                "hit_rate": round(self.cache_hit_rate, 3),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_model_name() -> str:
    """
    解析模型名称：环境变量 → 默认值。

    优先级：
      1. 环境变量 EMBEDDING_MODEL
      2. 默认值 DEFAULT_EMBEDDING_MODEL ("all-MiniLM-L6-v2")
    """
    import os
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


# ═══════════════════════════════════════════════════════════════════════════════
# 全局单例
# ═══════════════════════════════════════════════════════════════════════════════

_embedding_service: Optional[EmbeddingService] = None
_service_lock = threading.Lock()


def get_embedding_service(
    model_name: str = None,
    cache_size: int = 5000,
) -> EmbeddingService:
    """
    获取全局 EmbeddingService 单例（线程安全懒加载）。

    首次调用时创建实例（加载模型 ~2-3 秒），后续调用返回同一实例。
    所有 Memory 模块（Long-term、Reflection）共享同一个 embedding 服务。

    Args:
        model_name: 模型名称（仅首次调用时生效）。
                    为 None 时从环境变量读取。
        cache_size: 嵌入缓存大小（仅首次调用时生效）。

    Returns:
        全局单例 EmbeddingService

    Example:
        from app.memory.embeddings import get_embedding_service

        es = get_embedding_service()
        vec = es.embed("some text")
    """
    global _embedding_service

    if _embedding_service is not None:
        return _embedding_service

    with _service_lock:
        # 双重检查（double-checked locking）
        if _embedding_service is not None:
            return _embedding_service

        model = model_name or _resolve_model_name()
        print(
            f"[embedding_service] Initializing global singleton "
            f"(model={model}, cache_size={cache_size})..."
        )
        _embedding_service = EmbeddingService(
            model_name=model,
            cache_size=cache_size,
        )
        return _embedding_service


def reset_embedding_service() -> None:
    """
    重置全局单例（用于测试或模型切换）。

    注意：这会释放当前模型占用的内存，下次调用 get_embedding_service()
    会重新加载模型。
    """
    global _embedding_service
    with _service_lock:
        if _embedding_service is not None:
            print(
                f"[embedding_service] Resetting singleton "
                f"(model={_embedding_service.model_name})"
            )
            _embedding_service.clear_cache()
            _embedding_service = None
