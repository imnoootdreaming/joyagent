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
