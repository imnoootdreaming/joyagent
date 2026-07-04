"""
Phase 7 Step 1 — AgentOutbox 发件箱

Agent 的发件箱。每个 Agent 拥有独立的 Outbox 实例，用于创建和发送消息。

职责：
  1. Agent 创建消息 → 放入 Outbox
  2. Outbox 自动 flush 到 MailboxManager
  3. 追踪已发送消息的状态

发件箱不是简单的 send-and-forget——
Agent 可以查询自己发出的消息是否已被接收方处理,
从而实现"等待 Coder 完成后 Tester 再开始"的语义。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING

from app.agent.mailbox.message import (
    MailboxMessage,
    MessagePriority,
    MessageStatus,
    MessageType,
)

if TYPE_CHECKING:
    from app.agent.mailbox.manager import MailboxManager


class AgentOutbox:
    """
    Agent 的发件箱。

    职责：
      1. Agent 创建消息 → 放入 Outbox
      2. Outbox 自动 flush 到 MailboxManager
      3. 追踪已发送消息的状态

    发件箱不是简单的 send-and-forget——
    Agent 可以查询自己发出的消息是否已被接收方处理,
    从而实现"等待 Coder 完成后 Tester 再开始"的语义。
    """

    def __init__(self, owner: str):
        self.owner = owner
        self._sent: list[MailboxMessage] = []     # 已发送的历史
        self._pending: list[MailboxMessage] = []   # 等待冲洗
        self._manager: MailboxManager | None = None  # 绑定的路由中枢

    # ── 绑定 ──────────────────────────────────────────

    def bind(self, manager: "MailboxManager") -> None:
        """绑定到 MailboxManager（在 Agent 初始化时由 Manager 注入）。"""
        self._manager = manager

    @property
    def is_bound(self) -> bool:
        """是否已绑定到 MailboxManager。"""
        return self._manager is not None

    # ── 消息创建 ──────────────────────────────────────

    def create_message(
        self,
        recipient: str,
        msg_type: MessageType,
        body: dict | str,
        *,
        subject: str = "",
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: str = "",
        reply_to: str = "",
        ttl_seconds: int = 300,
    ) -> MailboxMessage:
        """
        创建一条新消息（自动填充 sender）。

        消息创建后状态为 DRAFT，需要调用 send() 或 flush() 发送。

        Args:
            recipient: 接收方 Agent 名称（"coder" | "tester" | ...），"*" 为广播
            msg_type: 消息类型
            body: 消息体（结构化 dict 或纯文本）
            subject: 简短标题（用于日志和 Inbox 列表）
            priority: 消息优先级
            correlation_id: 关联线程 ID（为空则自动生成）
            reply_to: 回复的消息 ID
            ttl_seconds: 过期时间（默认 5 分钟）

        Returns:
            创建好的 MailboxMessage（状态为 DRAFT）
        """
        return MailboxMessage(
            sender=self.owner,
            recipient=recipient,
            msg_type=msg_type,
            priority=priority,
            subject=subject,
            body=body,
            correlation_id=correlation_id or f"thread_{uuid.uuid4().hex[:8]}",
            reply_to=reply_to,
            ttl_seconds=ttl_seconds,
        )

    # ── 发送 ──────────────────────────────────────────

    async def send(self, msg: MailboxMessage) -> bool:
        """
        发送单条消息。

        消息进入 MailboxManager → 路由到目标 Agent 的 Inbox。

        Returns:
            True 发送成功，False 发送失败。
        """
        if self._manager is None:
            raise RuntimeError("Outbox not bound to a MailboxManager")

        msg.status = MessageStatus.SENT
        success = await self._manager.route(msg)
        if success:
            self._sent.append(msg)
        return success

    async def flush(self) -> int:
        """发送所有 pending 消息。返回成功发送的数量。"""
        count = 0
        for msg in list(self._pending):
            if await self.send(msg):
                self._pending.remove(msg)
                count += 1
        return count

    async def send_and_wait(
        self,
        msg: MailboxMessage,
        timeout: float = 60.0,
    ) -> MessageStatus:
        """
        发送消息并等待对方处理完毕。

        与普通 send() 的区别：
          send() 是 fire-and-forget
          send_and_wait() 会轮询目标 Inbox，直到消息状态变为 PROCESSED/FAILED

        适合串行依赖场景：
          Coder 发 TASK_RESULT 给 Router → Router 发 TASK_ASSIGNMENT 给 Tester
          Router 需要等 Coder 的 PROCESSED 确认后再分发下一步。

        Args:
            msg: 要发送的消息
            timeout: 等待超时秒数

        Returns:
            消息最终状态（PROCESSED / FAILED / EXPIRED）
        """
        if self._manager is None:
            raise RuntimeError("Outbox not bound to a MailboxManager")

        await self.send(msg)
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = await self._manager.get_message_status(msg.id)
            if status in (MessageStatus.PROCESSED, MessageStatus.FAILED):
                return status
            await asyncio.sleep(0.5)
        return MessageStatus.EXPIRED

    # ── 查询 ──────────────────────────────────────────

    def get_sent_history(self, recipient: str = "") -> list[MailboxMessage]:
        """查询已发送的历史消息（可按收件人过滤）。"""
        if recipient:
            return [m for m in self._sent if m.recipient == recipient]
        return list(self._sent)

    def get_pending(self) -> list[MailboxMessage]:
        """返回当前 pending 队列的只读副本。"""
        return list(self._pending)

    @property
    def sent_count(self) -> int:
        """已发送消息总数。"""
        return len(self._sent)

    @property
    def pending_count(self) -> int:
        """待发送消息数量。"""
        return len(self._pending)
