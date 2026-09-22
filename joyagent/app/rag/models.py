from __future__ import annotations
"""
RAG 数据模型 —— 文档 / 切片 / 检索命中。

三层结构：
  RagDocument（一个原始文件）  --split-->  RagChunk[]（带向量的切片）
  RagChunk + score            ------->    RagHit（检索结果）

与 app/memory/long_term.py 的 MemoryEntry 的区别：
  MemoryEntry 服务于 Agent 的"长期记忆"（code/conversation/task），
  RagChunk 服务于外部文档知识库，两者存放在不同的 ChromaDB 集合，
  避免外部文档污染记忆的语义空间。
"""

import hashlib
import time
from dataclasses import dataclass, field


def compute_doc_hash(content: str) -> str:
    """对文档全文取 SHA256 —— 用于判断内容是否变更（跳过重复 embedding）。"""
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def compute_doc_id(source: str, doc_hash: str) -> str:
    """
    文档稳定 ID：同一来源 + 同一内容 → 同一 doc_id。

    只取前 12 位 hex：足够区分，同时让 ChromaDB 的 id 索引更紧凑。
    """
    return hashlib.sha256(f"{source}::{doc_hash}".encode("utf-8")).hexdigest()[:12]


def compute_chunk_id(doc_id: str, chunk_index: int, content: str) -> str:
    """
    切片 ID：内容寻址（content-addressed）。

    同一文档同一位置同一内容 → 同一 id，重复入库时 ChromaDB 会覆盖而非翻倍。
    """
    raw = f"{doc_id}::{chunk_index}::{content}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


@dataclass
class RagDocument:
    """
    一个已加载的原始文档。

    source 是知识库中的唯一标识（用于按来源删除 / 去重）：
      - 文件上传：上传时的文件名
      - 目录导入：相对于导入根目录的相对路径
    """
    source: str                          # 唯一来源标识（文件名 / 相对路径）
    content: str                         # 文档全文（已解码为文本）
    ext: str = ""                        # 扩展名（含点），如 ".md"
    doc_hash: str = ""                   # 全文哈希（内容变更检测）
    size_bytes: int = 0                  # 原始字节数

    def __post_init__(self):
        if not self.doc_hash:
            self.doc_hash = compute_doc_hash(self.content)
        if not self.ext and "." in self.source:
            self.ext = "." + self.source.rsplit(".", 1)[-1].lower()

    @property
    def doc_id(self) -> str:
        return compute_doc_id(self.source, self.doc_hash)

    @property
    def is_empty(self) -> bool:
        return not self.content.strip()


@dataclass
class RagChunk:
    """
    一个待入库的文档切片。

    metadata 会写入 ChromaDB（ChromaDB 要求非空 dict，且值只支持
    str/int/float/bool），所以这里全部用标量字段。
    """
    id: str = ""                         # 内容寻址 ID（见 compute_chunk_id）
    doc_id: str = ""                     # 所属文档 ID
    source: str = ""                     # 来源标识（冗余存一份，便于按来源删除）
    content: str = ""                    # 切片文本（被 embedding 与检索的内容）
    chunk_index: int = 0                 # 在文档内的序号（从 0 开始）
    metadata: dict = field(default_factory=dict)
    # metadata 固定键：doc_hash / heading_path / ext / ingested_at / chunk_count

    def __post_init__(self):
        if not self.id:
            self.id = compute_chunk_id(self.doc_id, self.chunk_index, self.content)
        # ChromaDB 强制要求 metadata 非空
        if not self.metadata:
            self.metadata = {"source": self.source or "unknown"}

    @staticmethod
    def build(
        source: str,
        content: str,
        chunk_index: int,
        doc_id: str = "",
        doc_hash: str = "",
        heading_path: str = "",
        ext: str = "",
        chunk_count: int = 0,
    ) -> "RagChunk":
        """构造一个切片（自动填充 ID 与标准 metadata）。"""
        meta = {
            "source": source,
            "doc_id": doc_id,
            "doc_hash": doc_hash,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "heading_path": heading_path,
            "ext": ext,
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return RagChunk(
            id=compute_chunk_id(doc_id, chunk_index, content),
            doc_id=doc_id,
            source=source,
            content=content,
            chunk_index=chunk_index,
            metadata=meta,
        )


@dataclass
class RagHit:
    """
    检索命中结果。

    score 由 ChromaDB 的 cosine distance 换算而来：
        similarity = 1 - distance / 2     （与 app/memory/long_term.py 保持一致）
    范围 0.0 ~ 1.0，越大越相关。
    """
    chunk: RagChunk
    score: float = 0.0                   # 相似度 0.0 ~ 1.0

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def content(self) -> str:
        return self.chunk.content

    def to_source_dict(self) -> dict:
        """转成响应体 / 前端可直接消费的 dict（不含向量）。"""
        return {
            "source": self.chunk.source,
            "score": round(self.score, 4),
            "chunk_index": self.chunk.chunk_index,
            "content": self.chunk.content,
            "heading_path": self.chunk.metadata.get("heading_path", ""),
            "doc_id": self.chunk.doc_id,
        }
