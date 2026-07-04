"""
Phase 7 Step 1 — AgentInbox 收件箱

Agent 的收件箱。每个 Agent 拥有独立的 Inbox 实例。

职责：
  1. 接收 MailboxManager 投递的消息
  2. 按优先级排序
  3. 提供 fetch_unread() / fetch_by_correlation() 等查询接口
  4. 注册消息处理器（按 MessageType 分发）
  5. 自动清理过期消息 → 死信队列

使用方式：
  inbox = AgentInbox(owner="coder")
  inbox.register_handler(MessageType.TASK_ASSIGNMENT, coder.handle_task)
  inbox.register_handler(MessageType.REVIEW_FEEDBACK, coder.handle_review)

  # Agent 主循环
  while True:
      msg = await inbox.fetch_next()   # 阻塞直到有新消息
      await inbox.process(msg)         # 按类型分发到注册的 handler
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Callable, Awaitable

from app.agent.mailbox.message import (
    MailboxMessage,
    MessagePriority,
    MessageStatus,
    MessageType,
)

# ── 消息处理回调类型 ──
# 返回 True = 处理成功, False = 处理失败（触发重试）
MessageHandler = Callable[[MailboxMessage], Awaitable[bool]]


class AgentInbox:
    """
    Agent 的收件箱。

    职责：
      1. 接收 MailboxManager 投递的消息
      2. 按优先级排序
      3. 提供 fetch_unread() / fetch_by_correlation() 等查询接口
      4. 注册消息处理器（按 MessageType 分发）
      5. 自动清理过期消息 → 死信队列

    使用方式：
      inbox = AgentInbox(owner="coder")
      inbox.register_handler(MessageType.TASK_ASSIGNMENT, coder.handle_task)
      inbox.register_handler(MessageType.REVIEW_FEEDBACK, coder.handle_review)

      # Agent 主循环
      while True:
          msg = await inbox.fetch_next()   # 阻塞直到有新消息
          await inbox.process(msg)         # 按类型分发到注册的 handler
    """

    def __init__(self, owner: str, max_size: int = 100):
        self.owner = owner
        self.max_size = max_size

        # ── 消息存储 ──
        # 按优先级分组存储（URGENT → HIGH → NORMAL → LOW）
        self._messages: dict[MessagePriority, list[MailboxMessage]] = {
            p: [] for p in MessagePriority
        }
        self._handler_map: dict[MessageType | str, MessageHandler] = {}

        # ── 事件通知 ──
        self._new_message_event = asyncio.Event()  # 新消息到达时 set

        # ── 死信队列 ──
        self._dead_letter: list[MailboxMessage] = []

        # ── 统计 ──
        self._stats: dict[str, int] = {
            "received": 0, "processed": 0, "failed": 0,
            "expired": 0, "dead_lettered": 0,
        }

    # ── 处理器注册 ──────────────────────────────────────

    def register_handler(self, msg_type: MessageType, handler: MessageHandler) -> None:
        """
        注册消息处理器。

        当 Inbox 中有指定类型的消息被 process() 时，
        调用对应的 handler。一个类型只能注册一个 handler。

        Example:
            inbox.register_handler(MessageType.TASK_ASSIGNMENT, self._handle_task)
        """
        self._handler_map[msg_type] = handler

    def register_default_handler(self, handler: MessageHandler) -> None:
        """
        注册默认处理器（未匹配到特定 handler 时使用）。
        通常用于处理 BROADCAST 等通用消息类型。
        """
        self._handler_map["*"] = handler

    # ── 投递接口（MailboxManager 调用） ──────────────────

    def deliver(self, msg: MailboxMessage) -> bool:
        """
        MailboxManager 将消息投递到此 Inbox。

        Returns:
            True 投递成功，False Inbox 已满。
        """
        if self.total_count >= self.max_size:
            return False

        msg.status = MessageStatus.DELIVERED
        msg.delivered_at = time.time()
        self._messages[msg.priority].append(msg)
        self._stats["received"] += 1

        # 按时间戳排序（同优先级内 FIFO）
        self._messages[msg.priority].sort(key=lambda m: m.created_at)

        # 通知 watcher
        self._new_message_event.set()
        return True

    # ── 消息获取 ──────────────────────────────────────

    def fetch_unread(self) -> list[MailboxMessage]:
        """获取所有未读消息（按优先级排序）。"""
        result = []
        for priority in (MessagePriority.URGENT, MessagePriority.HIGH,
                         MessagePriority.NORMAL, MessagePriority.LOW):
            for msg in self._messages[priority]:
                if msg.status in (MessageStatus.DELIVERED, MessageStatus.SENT):
                    result.append(msg)
        return result

    def fetch_by_correlation(self, correlation_id: str) -> list[MailboxMessage]:
        """获取同一线程的所有消息（用于查看完整对话链）。"""
        result: list[MailboxMessage] = []
        for msgs in self._messages.values():
            for msg in msgs:
                if msg.correlation_id == correlation_id:
                    result.append(msg)
        return result

    async def fetch_next(self, timeout: float = 30.0) -> MailboxMessage | None:
        """
        阻塞等待下一条未读消息（按优先级 + FIFO）。

        如果当前无消息，阻塞等待直到有新消息到达或超时。
        Agent 主循环使用此方法轮询。

        Args:
            timeout: 超时秒数（None = 无限等待，0 = 立即返回）

        Returns:
            下一条待处理的消息，超时返回 None。
        """
        while True:
            # 1. 先检查是否有待处理消息
            for priority in (MessagePriority.URGENT, MessagePriority.HIGH,
                             MessagePriority.NORMAL, MessagePriority.LOW):
                for msg in self._messages[priority]:
                    if msg.status == MessageStatus.DELIVERED:
                        msg.status = MessageStatus.READ
                        msg.read_at = time.time()
                        return msg

            # 2. 无消息 → 等待
            if timeout == 0:
                return None
            try:
                await asyncio.wait_for(
                    self._new_message_event.wait(),
                    timeout=timeout if timeout else None,
                )
                self._new_message_event.clear()
            except asyncio.TimeoutError:
                return None

    # ── 消息处理 ──────────────────────────────────────

    async def process(self, msg: MailboxMessage) -> bool:
        """
        处理一条消息：根据 msg_type 分发到注册的 handler。

        Returns:
            True = 处理成功, False = 处理失败（触发重试）。
        """
        handler = self._handler_map.get(msg.msg_type)
        if handler is None:
            handler = self._handler_map.get("*")

        if handler is None:
            # 未注册处理器 → 标记为失败
            msg.status = MessageStatus.FAILED
            self._stats["failed"] += 1
            return False

        try:
            success = await handler(msg)
            if success:
                msg.status = MessageStatus.PROCESSED
                msg.processed_at = time.time()
                self._stats["processed"] += 1
            else:
                msg.retry_count += 1
                if msg.retry_count >= msg.max_retries:
                    self._move_to_dead_letter(msg)
                else:
                    msg.status = MessageStatus.DELIVERED  # 重置为未读，等待重试
                self._stats["failed"] += 1
            return success
        except Exception:
            msg.retry_count += 1
            if msg.retry_count >= msg.max_retries:
                self._move_to_dead_letter(msg)
            else:
                msg.status = MessageStatus.DELIVERED
            self._stats["failed"] += 1
            return False

    # ── 过期清理 ──────────────────────────────────────

    def clean_expired(self) -> list[MailboxMessage]:
        """清理所有过期消息 → 移到死信队列。返回被清理的消息列表。"""
        expired: list[MailboxMessage] = []
        for priority in MessagePriority:
            kept: list[MailboxMessage] = []
            for msg in self._messages[priority]:
                if msg.is_expired:
                    msg.status = MessageStatus.EXPIRED
                    self._dead_letter.append(msg)
                    expired.append(msg)
                    self._stats["expired"] += 1
                    self._stats["dead_lettered"] += 1
                else:
                    kept.append(msg)
            self._messages[priority] = kept
        return expired

    # ── 诊断 ──────────────────────────────────────────

    @property
    def total_count(self) -> int:
        """当前 Inbox 中的总消息数（含已读/未读）。"""
        return sum(len(msgs) for msgs in self._messages.values())

    @property
    def dead_letter_count(self) -> int:
        """死信队列深度。"""
        return len(self._dead_letter)

    @property
    def dead_letters(self) -> list[MailboxMessage]:
        """返回死信队列的只读副本。"""
        return list(self._dead_letter)

    @property
    def stats(self) -> dict:
        """Inbox 统计快照。"""
        return {
            **self._stats,
            "current_depth": self.total_count,
            "dead_letter_depth": len(self._dead_letter),
            "handlers_registered": [
                t.value if isinstance(t, MessageType) else str(t)
                for t in self._handler_map if t != "*"
            ],
        }

    # ── 内部 ──────────────────────────────────────────

    def _move_to_dead_letter(self, msg: MailboxMessage) -> None:
        """将消息移入死信队列并从活跃队列移除。"""
        msg.status = MessageStatus.FAILED
        self._dead_letter.append(msg)
        self._stats["dead_lettered"] += 1
        # 从活跃队列移除
        if msg in self._messages[msg.priority]:
            self._messages[msg.priority].remove(msg)
