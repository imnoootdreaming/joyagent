"""
Phase 7 Step 1 — Mailbox 通信系统

Mailbox 系统为多 Agent 协作提供松耦合、异步消息通信能力。
每个 Agent 有独立的 Inbox/Outbox，通过 MailboxManager 路由消息。

核心组件：
  - MailboxMessage:  结构化消息数据模型（类型/优先级/TTL/线程追踪）
  - AgentInbox:      收件箱（消息存储 + 处理器注册 + 过期清理）
  - AgentOutbox:     发件箱（消息创建 + 发送 + send_and_wait）
  - MailboxManager:  路由中枢（注册/路由/广播/审计/死信管理）
  - InboxWatcher:    异步监听器（后台持续消费 Inbox）
  - MailboxPersistence: 消息持久化（内存 / 文件 / 未来 Redis）

面试要点：
  Q: "你的多 Agent 怎么通信？"
  A: "我们实现了一个 Mailbox 系统，每个 Agent 有独立的 Inbox/Outbox，
      MailboxManager 作为消息路由中枢。这不是简单的 state['messages'].append()，
      而是完整的消息生命周期管理——每条消息有自己的类型、优先级、TTL、
      correlation_id 用于线程追踪。Agent 之间完全解耦，通过异步消息通信。"
"""

from app.agent.mailbox.message import (
    MailboxMessage,
    MessagePriority,
    MessageStatus,
    MessageType,
)
from app.agent.mailbox.inbox import AgentInbox
from app.agent.mailbox.outbox import AgentOutbox
from app.agent.mailbox.manager import MailboxManager
from app.agent.mailbox.watcher import InboxWatcher
from app.agent.mailbox.persistence import (
    FilePersistence,
    MailboxPersistence,
    MemoryPersistence,
)

__all__ = [
    # ── 消息模型 ──
    "MailboxMessage",
    "MessagePriority",
    "MessageStatus",
    "MessageType",
    # ── 通信组件 ──
    "AgentInbox",
    "AgentOutbox",
    "MailboxManager",
    "InboxWatcher",
    # ── 持久化 ──
    "MailboxPersistence",
    "MemoryPersistence",
    "FilePersistence",
]
