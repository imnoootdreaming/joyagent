"""
Phase 7 Step 1 — Mailbox 消息数据模型

Agent 间通信的唯一数据结构。每条消息包含完整的路由信息（sender → recipient）、
类型标记（MessageType）、优先级、TTL、以及用于线程追踪的 correlation_id 和 reply_to。

设计原则：
  - 不可变（创建后不再修改，状态变更由 MailboxManager 管理）
  - 自描述（不需要外部 context 就能理解消息意图）
  - 可追踪（correlation_id + id 形成 DAG）
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════
# 枚举定义
# ═══════════════════════════════════════════════════════════════════════

class MessageType(Enum):
    """消息类型 —— 决定 Agent 如何处理此消息"""

    TASK_ASSIGNMENT   = "task_assignment"     # Router/Planner → Agent：分配任务
    TASK_RESULT       = "task_result"         # Agent → Router/上下游：汇报完成
    REVIEW_FEEDBACK   = "review_feedback"     # Reviewer → Coder：审查意见
    CONFLICT_ESCALATE = "conflict_escalate"   # 任意 Agent → Planner：冲突升级
    STATUS_QUERY      = "status_query"        # Router → Agent：查询进度
    STATUS_REPLY      = "status_reply"        # Agent → Router：回复进度
    BROADCAST         = "broadcast"           # 任意方 → ALL：系统级通知
    ERROR_REPORT      = "error_report"        # Agent → Router：执行异常


class MessageStatus(Enum):
    """消息生命周期状态"""

    DRAFT     = "draft"        # 创建中，尚未发送
    SENT      = "sent"         # 已发送，等待投递
    DELIVERED = "delivered"    # 已投递到目标 Inbox
    READ      = "read"         # 目标 Agent 已读取
    PROCESSED = "processed"    # 目标 Agent 已处理完毕
    FAILED    = "failed"       # 处理失败
    EXPIRED   = "expired"      # 超过 TTL 未处理，进入死信队列


class MessagePriority(Enum):
    """消息优先级 —— 调度器按优先级排序"""

    LOW    = 1
    NORMAL = 3
    HIGH   = 5
    URGENT = 10                  # 冲突升级用


# ═══════════════════════════════════════════════════════════════════════
# 消息数据模型
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MailboxMessage:
    """
    Agent 间通信的唯一数据结构。

    每条消息包含完整的路由信息（sender → recipient）、
    类型标记（MessageType）、优先级、TTL、以及用于线程追踪的
    correlation_id 和 reply_to。

    设计原则：
      - 不可变（创建后不再修改，状态变更由 MailboxManager 管理）
      - 自描述（不需要外部 context 就能理解消息意图）
      - 可追踪（correlation_id + id 形成 DAG）
    """

    # ── 消息标识 ──
    id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    correlation_id: str = ""          # 关联的 thread ID（同一个任务的多条消息共享）
    reply_to: str = ""                # 回复哪条消息（单向回复链）

    # ── 路由信息 ──
    sender: str = ""                  # 发送方 Agent 名称（"router" | "planner" | ...）
    recipient: str = ""               # 接收方 Agent 名称（"coder" | "tester" | ...）
                                      # "*" 表示广播到所有 Agent

    # ── 消息内容 ──
    msg_type: MessageType = MessageType.TASK_ASSIGNMENT
    priority: MessagePriority = MessagePriority.NORMAL
    subject: str = ""                 # 简短标题（用于日志和 Inbox 列表）
    body: dict | str = ""             # 消息体（结构化 dict 或纯文本）

    # ── 生命周期 ──
    status: MessageStatus = MessageStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int = 300            # 过期时间（默认 5 分钟），0 = 永不过期
    delivered_at: float = 0.0
    read_at: float = 0.0
    processed_at: float = 0.0

    # ── 元数据 ──
    tags: list[str] = field(default_factory=list)  # 如 ["phase:code", "step:3"]
    retry_count: int = 0
    max_retries: int = 3

    # ── 计算属性 ──────────────────────────────────────────

    @property
    def is_expired(self) -> bool:
        """检查消息是否已过期（基于 TTL）。"""
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """消息从创建到现在的存活时间（用于监控/告警）。"""
        return time.time() - self.created_at

    # ── 序列化 ────────────────────────────────────────────

    def to_envelope(self) -> dict:
        """序列化为可持久化的 dict（JSON 安全）。"""
        return {
            "id": self.id,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "sender": self.sender,
            "recipient": self.recipient,
            "msg_type": self.msg_type.value,
            "priority": self.priority.value,
            "subject": self.subject,
            "body": self.body,
            "status": self.status.value,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "delivered_at": self.delivered_at,
            "read_at": self.read_at,
            "processed_at": self.processed_at,
            "tags": self.tags,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_envelope(cls, data: dict) -> MailboxMessage:
        """从持久化的 dict 反序列化（JSON 安全）。"""
        return cls(
            id=data.get("id", ""),
            correlation_id=data.get("correlation_id", ""),
            reply_to=data.get("reply_to", ""),
            sender=data.get("sender", ""),
            recipient=data.get("recipient", ""),
            msg_type=MessageType(data.get("msg_type", "task_assignment")),
            priority=MessagePriority(data.get("priority", 3)),
            subject=data.get("subject", ""),
            body=data.get("body", ""),
            status=MessageStatus(data.get("status", "draft")),
            created_at=data.get("created_at", 0.0),
            ttl_seconds=data.get("ttl_seconds", 300),
            delivered_at=data.get("delivered_at", 0.0),
            read_at=data.get("read_at", 0.0),
            processed_at=data.get("processed_at", 0.0),
            tags=data.get("tags", []),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
        )
