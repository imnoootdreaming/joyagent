"""
Phase 7 Step 1 — 消息持久化

消息持久化层，支持多种后端：
  - MemoryPersistence: 不做持久化（MVP 默认）
  - FilePersistence: 消息序列化到本地 JSON 文件

未来可扩展到 Redis、Kafka 等消息队列。
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.agent.mailbox.message import MailboxMessage

if TYPE_CHECKING:
    from app.agent.mailbox.inbox import AgentInbox


# ═══════════════════════════════════════════════════════════════════════
# 抽象基类
# ═══════════════════════════════════════════════════════════════════════

class MailboxPersistence(ABC):
    """消息持久化抽象基类。"""

    @abstractmethod
    async def save(self, inbox: "AgentInbox") -> None:
        """保存 Inbox 中所有待处理消息。"""
        ...

    @abstractmethod
    async def load(self, owner: str) -> list[dict] | None:
        """加载指定 owner 的持久化消息（返回消息 envelope 列表）。"""
        ...

    @abstractmethod
    async def delete(self, owner: str) -> None:
        """删除指定 owner 的持久化数据。"""
        ...


# ═══════════════════════════════════════════════════════════════════════
# MVP：纯内存（不做持久化）
# ═══════════════════════════════════════════════════════════════════════

class MemoryPersistence(MailboxPersistence):
    """
    MVP 默认后端：不做持久化。

    Agent 重启后所有未处理消息丢失。
    适用于开发阶段和单元测试。
    """

    async def save(self, inbox: "AgentInbox") -> None:
        """不保存任何数据。"""
        pass

    async def load(self, owner: str) -> list[dict] | None:
        """永远返回 None（无持久化数据可加载）。"""
        return None

    async def delete(self, owner: str) -> None:
        """无操作。"""
        pass


# ═══════════════════════════════════════════════════════════════════════
# 进阶：本地文件持久化
# ═══════════════════════════════════════════════════════════════════════

class FilePersistence(MailboxPersistence):
    """
    消息持久化到本地 JSON 文件。

    Agent 重启后可以从文件中恢复未处理消息。
    每个 Agent 的消息保存到独立文件：{path}/{owner}.json

    注意：
      - 只持久化未处理的消息（status != PROCESSED）
      - 已处理/已失败的消息不保存（减少文件体积）
      - 线程安全由 asyncio 保证（单线程事件循环）
    """

    def __init__(self, path: str = "data/mailbox"):
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)

    @property
    def storage_path(self) -> str:
        """持久化文件目录路径。"""
        return str(self._path)

    def _file_path(self, owner: str) -> Path:
        """获取指定 owner 的持久化文件路径。"""
        return self._path / f"{owner}.json"

    async def save(self, inbox: "AgentInbox") -> None:
        """
        保存 Inbox 中所有未处理的消息到 JSON 文件。

        只持久化 DELIVERED / READ 状态的消息（未处理完成的）。
        """
        unprocessed = inbox.fetch_unread()
        if not unprocessed:
            return

        envelopes = [msg.to_envelope() for msg in unprocessed]
        file_path = self._file_path(inbox.owner)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(envelopes, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"  [mailbox.persistence] Failed to save inbox for '{inbox.owner}': {e}")

    async def load(self, owner: str) -> list[dict] | None:
        """
        从 JSON 文件加载持久化消息。

        Returns:
            消息 envelope 列表，如果文件不存在或损坏则返回 None。
        """
        file_path = self._file_path(owner)
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                envelopes = json.load(f)
            if not isinstance(envelopes, list):
                return None
            return envelopes
        except (IOError, json.JSONDecodeError) as e:
            print(f"  [mailbox.persistence] Failed to load inbox for '{owner}': {e}")
            return None

    async def delete(self, owner: str) -> None:
        """删除指定 owner 的持久化文件。"""
        file_path = self._file_path(owner)
        try:
            if file_path.exists():
                file_path.unlink()
        except IOError as e:
            print(f"  [mailbox.persistence] Failed to delete inbox for '{owner}': {e}")


# ═══════════════════════════════════════════════════════════════════════
# Redis 配置
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RedisConfig:
    """
    Redis 连接配置。

    Attributes:
        host: Redis 主机地址
        port: Redis 端口
        password: Redis 密码（可选，生产环境必填）
        db: Redis 数据库编号（默认 0）
        prefix: 所有 key 的命名空间前缀
        ttl_seconds: 持久化消息的 TTL（秒），0 = 永不过期
        connect_timeout: 连接超时秒数
    """
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    prefix: str = "joyagent:mailbox"
    ttl_seconds: int = 0          # key 过期时间，0 = 永不过期
    connect_timeout: float = 5.0


# ═══════════════════════════════════════════════════════════════════════
# 生产级：Redis 持久化
# ═══════════════════════════════════════════════════════════════════════

class RedisPersistence(MailboxPersistence):
    """
    Redis 后端持久化 —— 使用 Redis List 做可靠队列 + Pub/Sub 做广播。

    设计决策（为什么用 List 而不是 Pub/Sub 做主队列）：
      - Pub/Sub 是 fire-and-forget：发布时如果无 subscriber 在线则消息丢失
      - Redis List（LPUSH / LRANGE）是持久化队列：消息写入后持久存在直到被消费
      - Pub/Sub 仅用于 BROADCAST 类型消息（系统通知、状态同步），
        因为这些消息本身就允许丢失

    Key 命名约定：
      joyagent:mailbox:inbox:{owner}  ← 每个 Agent 的 List（消息队列）
      joyagent:mailbox:broadcast      ← Pub/Sub channel（广播消息）

    消息格式：
      每条消息存储为 JSON 字符串（msg.to_envelope()）。

    使用方式：
        cfg = RedisConfig(host="localhost", port=6379, prefix="joyagent:mailbox")
        persistence = RedisPersistence(config=cfg)
        await persistence.save(coder_inbox)
        envelopes = await persistence.load("coder")
    """

    def __init__(self, config: RedisConfig | None = None):
        """
        Args:
            config: Redis 连接配置。默认连接 localhost:6379。
        """
        self._config = config or RedisConfig()
        self._client: object | None = None   # redis.asyncio.Redis（延迟连接）
        self._pubsub: object | None = None   # Pub/Sub 客户端

    # ── 连接管理 ──────────────────────────────────────────

    async def _get_client(self) -> object:
        """
        获取或创建 Redis 异步客户端（懒加载 + 连接池复用）。

        Returns:
            redis.asyncio.Redis 客户端实例。
        """
        if self._client is not None:
            return self._client

        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError(
                "redis package is required for RedisPersistence. "
                "Install it with: pip install redis>=8.0.1"
            )

        self._client = aioredis.Redis(
            host=self._config.host,
            port=self._config.port,
            password=self._config.password or None,
            db=self._config.db,
            socket_connect_timeout=self._config.connect_timeout,
            decode_responses=True,  # 自动 bytes → str
        )
        return self._client

    async def close(self) -> None:
        """关闭 Redis 连接（在 shutdown 时调用）。"""
        if self._pubsub is not None:
            try:
                await self._pubsub.unsubscribe()
            except Exception:
                pass
            self._pubsub = None

        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None

    # ── Key 工具 ──────────────────────────────────────────

    def _inbox_key(self, owner: str) -> str:
        """获取指定 owner 的 Inbox List key。"""
        return f"{self._config.prefix}:inbox:{owner}"

    # ── 持久化接口 ─────────────────────────────────────────

    async def save(self, inbox: "AgentInbox") -> None:
        """
        保存 Inbox 中所有未处理的消息到 Redis List。

        使用 LPUSH 将每条消息推入对应 Agent 的 List。
        如果设置了 ttl_seconds，还会为该 key 设置过期时间。

        注意：
          - 先清空旧数据（DELETE），再写入新数据（避免重复）
          - 只保存 DELIVERED / READ 状态的消息
        """
        unprocessed = inbox.fetch_unread()
        owner = inbox.owner
        key = self._inbox_key(owner)

        client = await self._get_client()

        # 清空旧数据
        await client.delete(key)

        if not unprocessed:
            return

        # 写入新数据（LPUSH 多条消息）
        envelopes = [json.dumps(msg.to_envelope()) for msg in unprocessed]
        await client.lpush(key, *envelopes)

        # 设置 TTL
        if self._config.ttl_seconds > 0:
            await client.expire(key, self._config.ttl_seconds)

    async def load(self, owner: str) -> list[dict] | None:
        """
        从 Redis List 加载指定 owner 的持久化消息。

        使用 LRANGE 读取所有消息，然后删除已加载的 key
        （返回的消息将由 AgentInbox.load_envelopes() 恢复到内存）。

        Returns:
            list[dict]: 消息 envelope 列表，如果没有数据则返回 None。
        """
        key = self._inbox_key(owner)
        client = await self._get_client()

        # 检查 key 是否存在
        exists = await client.exists(key)
        if not exists:
            return None

        # 读取所有消息（LRANGE 0 -1）
        raw_messages = await client.lrange(key, 0, -1)
        if not raw_messages:
            return None

        # 解析 JSON
        envelopes = []
        for raw in raw_messages:
            try:
                envelopes.append(json.loads(raw))
            except json.JSONDecodeError:
                continue

        # 删除已加载的 key（消息已恢复到内存）
        await client.delete(key)

        return envelopes if envelopes else None

    async def delete(self, owner: str) -> None:
        """删除指定 owner 的 Redis key。"""
        key = self._inbox_key(owner)
        client = await self._get_client()
        await client.delete(key)

    # ── Redis Info ─────────────────────────────────────────

    @property
    def config(self) -> RedisConfig:
        """当前 Redis 配置（只读）。"""
        return self._config

    @property
    def is_connected(self) -> bool:
        """是否已建立 Redis 连接。"""
        return self._client is not None
