"""
Phase 7 Step 1 — MailboxManager 消息路由中枢

消息路由中枢 —— 所有 Agent 间消息都必须经过它。

职责：
  1. 管理所有 Agent 的 Inbox 注册
  2. 消息路由（sender → MailboxManager → 目标 Inbox）
  3. 广播消息（BROADCAST → 所有已注册 Inbox）
  4. 消息状态追踪（跨 Agent 查询任意消息的状态）
  5. 死信队列管理（全局视角）
  6. 消息审计日志

它是唯一知道"有哪些 Agent""每个 Agent 的 Inbox 在哪里"的组件。
Agent 之间不直接通信 —— 一切经过 MailboxManager。

面试要点：
  面试官："你的多 Agent 怎么通信？"
  答："我们实现了一个 Mailbox 系统，每个 Agent 有独立的 Inbox/Outbox，
      MailboxManager 作为消息路由中枢。这不是简单的 state['messages'].append()，
      而是完整的消息生命周期管理——每条消息有自己的类型、优先级、TTL、
      correlation_id 用于线程追踪。Agent 之间完全解耦，通过异步消息通信。"
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from app.agent.mailbox.message import (
    MailboxMessage,
    MessageStatus,
)

if TYPE_CHECKING:
    from app.agent.mailbox.inbox import AgentInbox
    from app.agent.mailbox.outbox import AgentOutbox


class MailboxManager:
    """
    消息路由中枢 —— 所有 Agent 间消息都必须经过它。

    职责：
      1. 管理所有 Agent 的 Inbox 注册
      2. 消息路由（sender → MailboxManager → 目标 Inbox）
      3. 广播消息（BROADCAST → 所有已注册 Inbox）
      4. 消息状态追踪（跨 Agent 查询任意消息的状态）
      5. 死信队列管理（全局视角）
      6. 消息审计日志

    它是唯一知道"有哪些 Agent""每个 Agent 的 Inbox 在哪里"的组件。
    Agent 之间不直接通信 —— 一切经过 MailboxManager。
    """

    def __init__(self):
        # Agent Inbox/Outbox 注册表
        self._inboxes: dict[str, AgentInbox] = {}
        self._outboxes: dict[str, AgentOutbox] = {}

        # 全局消息状态索引（msg_id → status）
        self._status_index: dict[str, MessageStatus] = {}

        # 全局死信队列
        self._global_dead_letter: list[MailboxMessage] = []

        # 审计日志（最近 1000 条）
        self._audit_log: list[dict] = []

        # 统计
        self._stats: dict[str, int] = {
            "total_routed": 0,
            "total_broadcast": 0,
            "total_failed": 0,
        }

    # ── Agent 注册 ─────────────────────────────────────

    def register_agent(
        self,
        agent_name: str,
        inbox: "AgentInbox",
        outbox: "AgentOutbox",
    ) -> None:
        """
        注册一个 Agent 的 Inbox + Outbox 到 MailboxManager。

        注册后，该 Agent 就可以接收和发送消息。
        Outbox 会自动绑定到本 Manager。
        """
        self._inboxes[agent_name] = inbox
        self._outboxes[agent_name] = outbox
        outbox.bind(self)
        print(
            f"  [mailbox] Agent '{agent_name}' registered "
            f"(inbox handlers: {inbox.stats['handlers_registered']})"
        )

    def unregister_agent(self, agent_name: str) -> None:
        """注销 Agent（如 Agent 崩溃或热替换时）。"""
        self._inboxes.pop(agent_name, None)
        self._outboxes.pop(agent_name, None)

    def is_registered(self, agent_name: str) -> bool:
        """检查 Agent 是否已注册。"""
        return agent_name in self._inboxes

    @property
    def registered_agents(self) -> list[str]:
        """返回所有已注册 Agent 的名称列表。"""
        return list(self._inboxes.keys())

    # ── 消息路由（核心） ───────────────────────────────

    async def route(self, msg: MailboxMessage) -> bool:
        """
        路由一条消息到目标 Agent 的 Inbox。

        支持特殊 recipient：
          "*"  → 广播到所有已注册 Agent（除 sender 外）
          其他 → 投递到指定 Agent

        Returns:
            True 路由成功, False 目标不存在或 Inbox 已满。
        """
        self._audit(msg, "routing")

        if msg.recipient == "*":
            return await self._broadcast(msg)

        target_inbox = self._inboxes.get(msg.recipient)
        if target_inbox is None:
            self._audit(msg, "recipient_not_found")
            self._stats["total_failed"] += 1
            return False

        success = target_inbox.deliver(msg)
        if success:
            self._status_index[msg.id] = msg.status
            self._stats["total_routed"] += 1
            self._audit(msg, "delivered")
        else:
            self._audit(msg, "inbox_full")
            self._stats["total_failed"] += 1

        return success

    async def _broadcast(self, msg: MailboxMessage) -> bool:
        """广播消息到除 sender 外的所有已注册 Agent。"""
        self._stats["total_broadcast"] += 1
        delivered_count = 0
        for name, inbox in self._inboxes.items():
            if name == msg.sender:
                continue
            # 每个 Agent 收到的是消息副本
            copy = MailboxMessage(
                sender=msg.sender,
                recipient=name,
                msg_type=msg.msg_type,
                priority=msg.priority,
                subject=msg.subject,
                body=msg.body,
                correlation_id=msg.correlation_id,
                reply_to=msg.reply_to,
                ttl_seconds=msg.ttl_seconds,
                tags=list(msg.tags),
            )
            copy.id = f"{msg.id}_to_{name}"
            if inbox.deliver(copy):
                delivered_count += 1
                self._status_index[copy.id] = copy.status
        self._audit(msg, f"broadcast_to_{delivered_count}_agents")
        return delivered_count > 0

    # ── 状态查询 ──────────────────────────────────────

    async def get_message_status(self, msg_id: str) -> MessageStatus | None:
        """查询任意消息的当前状态（跨 Agent）。"""
        return self._status_index.get(msg_id)

    def get_agent_inbox_depth(self, agent_name: str) -> int:
        """查询指定 Agent 的 Inbox 深度（用于负载监控）。"""
        inbox = self._inboxes.get(agent_name)
        return inbox.total_count if inbox else 0

    def get_all_stats(self) -> dict:
        """获取全局 Mailbox 统计 + 每个 Agent 的 Inbox 状态。"""
        agent_stats = {}
        for name, inbox in self._inboxes.items():
            agent_stats[name] = inbox.stats
        return {
            "global": self._stats,
            "agents": agent_stats,
            "dead_letter_count": len(self._global_dead_letter),
        }

    # ── 死信处理 ──────────────────────────────────────

    def sweep_expired(self) -> int:
        """全局过期消息清理。返回清理总数。"""
        total = 0
        for inbox in self._inboxes.values():
            expired = inbox.clean_expired()
            self._global_dead_letter.extend(expired)
            total += len(expired)
        return total

    def reprocess_dead_letter(self, msg_id: str) -> bool:
        """将死信队列中的消息重新投递（手动干预）。"""
        for msg in self._global_dead_letter:
            if msg.id == msg_id:
                msg.retry_count = 0
                msg.status = MessageStatus.DRAFT
                self._global_dead_letter.remove(msg)
                asyncio.create_task(self.route(msg))
                return True
        return False

    # ── 审计 ──────────────────────────────────────────

    def _audit(self, msg: MailboxMessage, event: str) -> None:
        """记录审计日志条目。"""
        entry = {
            "timestamp": time.time(),
            "event": event,
            "msg_id": msg.id,
            "correlation_id": msg.correlation_id,
            "sender": msg.sender,
            "recipient": msg.recipient,
            "msg_type": msg.msg_type.value,
            "priority": msg.priority.value,
        }
        self._audit_log.append(entry)
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-500:]

    def get_audit_trail(
        self,
        correlation_id: str = "",
        agent_name: str = "",
    ) -> list[dict]:
        """
        查询审计日志（可按 correlation_id 或 agent_name 过滤）。

        correlation_id 使用前缀匹配（startswith），因为父级 cid（如
        user_abc）的子步骤会使用 step-level cid（如 user_abc_s0、
        user_abc_plan）。前缀匹配确保查询父级 cid 时能看到完整链路。
        """
        result = self._audit_log
        if correlation_id:
            result = [e for e in result
                      if e["correlation_id"].startswith(correlation_id)]
        if agent_name:
            result = [
                e for e in result
                if e["sender"] == agent_name or e["recipient"] == agent_name
            ]
        return result

    @property
    def audit_log_size(self) -> int:
        """审计日志条目数。"""
        return len(self._audit_log)
