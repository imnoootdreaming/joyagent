# Phase 7：Multi-Agent + Mailbox 通信系统

> **返回主索引：** [0-1 coding.md](0-1%20coding.md)
> **上一阶段：** [Phase 6: Memory System](phase-6-memory-system.md)
> **下一阶段：** [Phase 8: MCP Plugin System](phase-8-mcp.md)

---

## 一、目标与定位

### 目标
将单 Agent 的 Plan → Execute → Reflect 工作流升级为**多 Agent 协同系统**，核心变化：
1. 每个阶段由独立的 Agent 负责，通过 Router 分发任务
2. **用 Mailbox 系统替代共享 State 的紧耦合通信**——每个 Agent 有独立的 Inbox/Outbox，消息异步路由

### 与 Phase 3 的关系（关键澄清）⚠️

```
Phase 3 单 Agent 工作流：
  ┌──────┐    ┌──────┐    ┌──────┐
  │Plan  │───▶│Exec  │───▶│Refl  │    ← 同一个 Agent（同一个 LLM + 同一套工具）
  │Node  │    │Node  │    │Node  │    ← LangGraph StateGraph 的不同节点
  └──────┘    └──────┘    └──────┘

Phase 7 多 Agent + Mailbox 通信：
                  ┌────────────────────────────┐
                  │       MailboxManager        │  ← 消息路由中枢
                  │  ┌──────┐ ┌──────┐ ┌──────┐ │
                  │  │Inbox │ │Inbox │ │Inbox │ │  ← 每个 Agent 一个 Inbox
                  │  └──┬───┘ └──┬───┘ └──┬───┘ │
                  └─────┼───────┼───────┼──────┘
                        ▲       ▲       ▲
          ┌─────────────┤       │       ├─────────────┐
          │(投递任务)    │(投递任务)     │(投递任务)    │(投递任务)
     ┌────┴─────┐  ┌─────┴────┐ ┌─────┴────┐ ┌──────┴──────┐
     │  Router  │  │ Planner  │ │  Coder   │ │  Tester     │
     │          │  │          │ │          │ │             │
     └──────────┘  └──────────┘ └────┬─────┘ └─────────────┘
                                     │ (提交 commit)
                                     ▼
                              ┌──────────────┐
                              │   Reviewer   │
                              └──────────────┘
```

**核心变化（v2 vs v1）：**

| 维度 | v1 共享 State | v2 Mailbox 系统 |
|------|-------------|----------------|
| 通信模型 | `state["messages"].append(...)` | `mailbox.send(to="coder", msg=TaskMessage(...))` |
| Agent 如何读取消息 | 被动接收（state 全量传入） | 主动 pull（`inbox.fetch_unread()`） |
| 消息类型 | 无类型，纯字符串 | 结构化消息（TASK / RESULT / FEEDBACK / STATUS / BROADCAST） |
| 生命周期 | 无 | DRAFT → SENT → DELIVERED → READ → PROCESSED |
| 消息关联 | 无 | `correlation_id` + `reply_to` 线程追踪 |
| 并发投递 | 不支持（串行追加） | 支持（Router 同时投递到 Coder + Tester） |
| 超时/重试 | 无 | TTL + 自动重试 + 死信队列 |
| 持久化 | 依赖 State 序列化 | Inbox/Outbox 可独立持久化 |
| 可观测性 | 差 | 消息审计日志 + 投递状态追踪 |

### 本 Phase 不做什么
- ❌ 不做 Agent 热加载/热插拔（Phase 8 MCP 做）
- ❌ 不做分布式 Agent（多机器协作）——但 Mailbox 架构天然支持未来扩展到 Redis/Kafka

---

## 二、前置依赖

| 依赖 | 用途 |
|------|------|
| Phase 3 完成 | LangGraph 基础 |
| Phase 5 完成 | Sandbox + Testing |
| Phase 6 完成 | Memory System（Agent 间共享记忆） |

---

## 三、目录结构

```text
app/
├── agent/
│   ├── base.py                 # 基类（含 Mailbox 集成）
│   │
│   ├── mailbox/               # ★ 新增：Mailbox 通信系统
│   │   ├── __init__.py
│   │   ├── manager.py         # MailboxManager — 消息路由中枢
│   │   ├── inbox.py           # AgentInbox — 收件箱
│   │   ├── outbox.py          # AgentOutbox — 发件箱
│   │   ├── message.py         # MailboxMessage — 消息数据模型
│   │   ├── watcher.py         # InboxWatcher — 异步消息监听器
│   │   └── persistence.py     # 消息持久化（内存 / Redis）
│   │
│   ├── planner/               # Planner Agent（任务拆解）
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── prompts.py
│   │
│   ├── coder/                 # Coder Agent（代码生成/修改）
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── prompts.py
│   │
│   ├── tester/                # Tester Agent（测试执行/分析）
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── prompts.py
│   │
│   ├── reviewer/              # Reviewer Agent（Code Review）
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── prompts.py
│   │
│   └── router/                # Agent Router（任务分发）
│       ├── __init__.py
│       ├── router.py
│       └── prompts.py
```

---

## 四、Mailbox 通信系统设计（★ 核心新增）

### 4.1 消息数据模型

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time
import uuid


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
```

### 4.2 AgentInbox — 收件箱

```python
from collections import defaultdict
from typing import Callable, Awaitable
import asyncio

# 消息处理回调类型
MessageHandler = Callable[[MailboxMessage], Awaitable[bool]]
# 返回 True = 处理成功, False = 处理失败（触发重试）


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
        self._handler_map: dict[MessageType, MessageHandler] = {}

        # ── 事件通知 ──
        self._new_message_event = asyncio.Event()  # 新消息到达时 set

        # ── 死信队列 ──
        self._dead_letter: list[MailboxMessage] = []

        # ── 统计 ──
        self._stats = {
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
        self._handler_map["*"] = handler  # type: ignore

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
        result = []
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
        expired = []
        for priority in MessagePriority:
            kept = []
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
        return sum(len(msgs) for msgs in self._messages.values())

    @property
    def stats(self) -> dict:
        return {
            **self._stats,
            "current_depth": self.total_count,
            "dead_letter_depth": len(self._dead_letter),
            "handlers_registered": list(
                t.value for t in self._handler_map if t != "*"
            ),
        }

    # ── 内部 ──────────────────────────────────────────

    def _move_to_dead_letter(self, msg: MailboxMessage) -> None:
        msg.status = MessageStatus.FAILED
        self._dead_letter.append(msg)
        self._stats["dead_lettered"] += 1
        # 从活跃队列移除
        if msg in self._messages[msg.priority]:
            self._messages[msg.priority].remove(msg)
```

### 4.3 AgentOutbox — 发件箱

```python
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
        self._sent: list[MailboxMessage] = []    # 已发送的历史
        self._pending: list[MailboxMessage] = []  # 等待冲洗
        self._manager: "MailboxManager | None" = None  # 绑定的路由中枢

    def bind(self, manager: "MailboxManager") -> None:
        """绑定到 MailboxManager（在 Agent 初始化时由 Manager 注入）。"""
        self._manager = manager

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

    async def send(self, msg: MailboxMessage) -> bool:
        """
        发送单条消息。

        消息进入 MailboxManager → 路由到目标 Agent 的 Inbox。
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
        """
        await self.send(msg)
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = await self._manager.get_message_status(msg.id)
            if status in (MessageStatus.PROCESSED, MessageStatus.FAILED):
                return status
            await asyncio.sleep(0.5)
        return MessageStatus.EXPIRED

    def get_sent_history(self, recipient: str = "") -> list[MailboxMessage]:
        """查询已发送的历史消息（可按收件人过滤）。"""
        if recipient:
            return [m for m in self._sent if m.recipient == recipient]
        return list(self._sent)
```

### 4.4 MailboxManager — 消息路由中枢

```python
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

    面试要点：
      面试官："你的多 Agent 怎么通信？"
      答："我们实现了一个 Mailbox 系统，每个 Agent 有独立的 Inbox/Outbox，
          MailboxManager 作为消息路由中枢。这不是简单的 state['messages'].append()，
          而是完整的消息生命周期管理——每条消息有自己的类型、优先级、TTL、
          correlation_id 用于线程追踪。Agent 之间完全解耦，通过异步消息通信。"
    """

    def __init__(self):
        # Agent Inbox 注册表
        self._inboxes: dict[str, AgentInbox] = {}
        self._outboxes: dict[str, AgentOutbox] = {}

        # 全局消息状态索引（msg_id → status）
        self._status_index: dict[str, MessageStatus] = {}

        # 全局死信队列
        self._global_dead_letter: list[MailboxMessage] = []

        # 审计日志（最近 1000 条）
        self._audit_log: list[dict] = []

        # 统计
        self._stats = {
            "total_routed": 0,
            "total_broadcast": 0,
            "total_failed": 0,
        }

    # ── Agent 注册 ─────────────────────────────────────

    def register_agent(
        self,
        agent_name: str,
        inbox: AgentInbox,
        outbox: AgentOutbox,
    ) -> None:
        """
        注册一个 Agent 的 Inbox + Outbox 到 MailboxManager。

        注册后，该 Agent 就可以接收和发送消息。
        Outbox 会自动绑定到本 Manager。
        """
        self._inboxes[agent_name] = inbox
        self._outboxes[agent_name] = outbox
        outbox.bind(self)
        print(f"  [mailbox] Agent '{agent_name}' registered "
              f"(inbox handlers: {inbox.stats['handlers_registered']})")

    def unregister_agent(self, agent_name: str) -> None:
        """注销 Agent（如 Agent 崩溃或热替换时）。"""
        self._inboxes.pop(agent_name, None)
        self._outboxes.pop(agent_name, None)

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
            copy = MailboxMessage(**{k: v for k, v in msg.__dict__.items()
                                     if k != 'id'})
            copy.id = f"{msg.id}_to_{name}"
            copy.recipient = name
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
        """查询审计日志（可过滤）。"""
        result = self._audit_log
        if correlation_id:
            result = [e for e in result if e["correlation_id"] == correlation_id]
        if agent_name:
            result = [e for e in result
                      if e["sender"] == agent_name or e["recipient"] == agent_name]
        return result
```

### 4.5 InboxWatcher — 异步消息监听器

```python
class InboxWatcher:
    """
    异步消息监听器 —— Agent 不需要手动轮询。

    每个 Agent 启动一个 watcher 协程，在后台持续监听 Inbox。
    新消息到达 → 自动 dispatch 到对应 handler。

    这是 Agent 主循环的核心：
      BaseAgent.run() 内部启动 watcher
      → watcher 持续 fetch_next() + process()
      → Agent 收到 TASK_ASSIGNMENT 时自动执行任务
      → 执行完毕后通过 Outbox 发送 TASK_RESULT
    """

    def __init__(self, inbox: AgentInbox, idle_callback=None):
        self.inbox = inbox
        self.idle_callback = idle_callback  # 空闲时回调（如心跳上报）
        self._running = False
        self._task: asyncio.Task | None = None

    async def run(self, poll_interval: float = 0.5) -> None:
        """
        启动监听循环（永不退出，直到外部调用 stop()）。

        流程：
          1. 阻塞等待下一条消息（无消息时由 asyncio.Event 挂起）
          2. 消息到达 → process(msg)
          3. 处理完毕 → 继续等待下一条
          4. 过期消息自动清理
        """
        self._running = True
        while self._running:
            msg = await self.inbox.fetch_next(timeout=5.0)

            if msg is None:
                # 超时——无新消息
                self.inbox.clean_expired()
                if self.idle_callback:
                    await self.idle_callback()
                continue

            await self.inbox.process(msg)

    def start(self) -> asyncio.Task:
        """启动 watcher（非阻塞）。"""
        self._task = asyncio.create_task(self.run())
        return self._task

    def stop(self) -> None:
        """停止 watcher。"""
        self._running = False
        if self._task:
            self._task.cancel()
```

---

## 五、Agent 角色定义（Mailbox 集成版）

### 5.1 AgentRole Schema（不变）

```python
from dataclasses import dataclass

@dataclass
class AgentRole:
    """Agent 角色定义"""
    name: str                  # "planner" | "coder" | "tester" | "reviewer"
    system_prompt: str
    tools: list[str]           # 该 Agent 可用的工具名称列表
    model: str
    temperature: float

    # 权限
    can_modify_files: bool
    can_execute_shell: bool
    needs_user_approval: bool

# 角色配置表（与 v1 相同，略）
```

### 5.2 BaseAgent — 基类（集成 Mailbox）

```python
# app/agent/base.py
class BaseAgent:
    """
    所有 Agent 的基类 —— 集成 Mailbox 通信。

    每个 Agent 实例自动获得：
      - self.inbox:  AgentInbox（收件箱）
      - self.outbox: AgentOutbox（发件箱）
      - self._watcher: InboxWatcher（后台消息监听）
    """

    def __init__(
        self,
        role: AgentRole,
        mailbox_manager: MailboxManager,
        agent_id: str = "",
    ):
        self.role = role
        self.agent_id = agent_id or role.name
        self.client = get_or_create_client(role.model)
        self.tools = tool_registry.get_tool_schemas_for(role.tools)

        # ── Mailbox 系统 ──
        self.inbox = AgentInbox(owner=self.agent_id)
        self.outbox = AgentOutbox(owner=self.agent_id)
        mailbox_manager.register_agent(self.agent_id, self.inbox, self.outbox)

        # ── 注册默认消息处理器 ──
        self.inbox.register_handler(MessageType.TASK_ASSIGNMENT, self._handle_task)
        self.inbox.register_handler(MessageType.STATUS_QUERY, self._handle_status_query)
        self.inbox.register_handler(MessageType.BROADCAST, self._handle_broadcast)

        # ── 启动 watcher ──
        self._watcher = InboxWatcher(self.inbox)

    async def _handle_task(self, msg: MailboxMessage) -> bool:
        """
        处理 TASK_ASSIGNMENT 消息 —— 子类覆盖此方法实现具体逻辑。
        默认实现：调用 run() 然后回复 TASK_RESULT。
        """
        task = msg.body if isinstance(msg.body, str) else msg.body.get("task", "")
        result = await self.run(task)
        reply = self.outbox.create_message(
            recipient=msg.sender,
            msg_type=MessageType.TASK_RESULT,
            body=result,
            subject=f"Task result from {self.agent_id}",
            correlation_id=msg.correlation_id,
            reply_to=msg.id,
        )
        await self.outbox.send(reply)
        return True

    async def _handle_status_query(self, msg: MailboxMessage) -> bool:
        """处理 STATUS_QUERY —— 回复当前状态。"""
        reply = self.outbox.create_message(
            recipient=msg.sender,
            msg_type=MessageType.STATUS_REPLY,
            body={"agent": self.agent_id, "status": "idle", "inbox_depth": self.inbox.total_count},
            correlation_id=msg.correlation_id,
            reply_to=msg.id,
        )
        await self.outbox.send(reply)
        return True

    async def _handle_broadcast(self, msg: MailboxMessage) -> bool:
        """处理 BROADCAST 消息（默认：仅记录日志）。"""
        print(f"  [{self.agent_id}] RECV BROADCAST: {msg.subject}")
        return True

    async def run(self, task: str) -> dict:
        """
        Agent 主逻辑。子类覆盖此方法。

        子类在实现中可以通过 self.outbox 与其他 Agent 通信。
        """
        response = self.client.messages.create(
            model=self.role.model,
            system=self.role.system_prompt,
            messages=[{"role": "user", "content": f"Task: {task}"}],
            tools=self.tools,
            max_tokens=4096,
        )
        return {"output": extract_text(response.content), "agent": self.agent_id}

    async def start(self) -> None:
        """启动 Agent 的消息监听循环。"""
        await self._watcher.run()

    def start_background(self) -> asyncio.Task:
        """在后台启动监听（用于 Agent 池同时运行）。"""
        return self._watcher.start()
```

---

## 六、Mailbox 通信流程示例

### 6.1 完整协作流程（Router → Planner → Coder → Tester → Reviewer）

```
时间线（每个阶段通过 Mailbox 消息触发）:

  ① Router 分析用户需求
     │
     ├→ outbox.send(TASK_ASSIGNMENT → "planner")
     │    subject: "拆解用户需求"
     │    body: {"task": "创建 FastAPI User API, 含 GET/POST"}
     │    correlation_id: "thread_abc001"
     │
  ② Planner Inbox 收到 msg → _handle_task → run()
     │
     ├→ outbox.send(TASK_RESULT → "router")
     │    body: {"steps": [
     │      {"agent": "coder", "task": "生成 user_api.py"},
     │      {"agent": "tester", "task": "测试 user_api"},
     │      {"agent": "reviewer", "task": "审查代码"}
     │    ]}
     │    correlation_id: "thread_abc001"
     │
  ③ Router 收到 Planner 结果 → 并行分发
     │
     ├→ outbox.send(TASK_ASSIGNMENT → "coder")
     │    body: {"task": "生成 user_api.py"}
     │    correlation_id: "thread_abc001_step1"
     │
  ④ Coder 生成代码 → 完成后投递 TASK_RESULT
     │
     ├→ outbox.send(TASK_RESULT → "router")
     │    body: {"files": ["user_api.py"], "diff": "..."}
     │    correlation_id: "thread_abc001_step1"
     │
  ⑤ Router → 并行分发到 Tester + Reviewer
     │
     ├→ outbox.send(TASK_ASSIGNMENT → "tester")
     │    body: {"test": "pytest user_api.py"}
     │    correlation_id: "thread_abc001_step2"
     │
     └→ outbox.send(TASK_ASSIGNMENT → "reviewer")
          body: {"review": "user_diff"}
          correlation_id: "thread_abc001_step3"
     │
  ⑥ Tester 执行 → 如果失败，通过 Mailbox 通知 Coder
     │
     ├→ outbox.send(REVIEW_FEEDBACK → "coder")
     │    body: {"failed_tests": ["test_create_user", ...]}
     │    reply_to: "msg_tester_task_001"
     │
     │  Coder → 修复 → Tester 再测 → 通过
     │
  ⑦ Reviewer 审查 → 如果有问题，通过 Mailbox 通知 Coder
     │
     ├→ outbox.send(REVIEW_FEEDBACK → "coder")
     │    body: {"issues": [...]}
     │
     │  Coder → 修改 → Reviewer 再查 → LGTM
     │
  ⑧ 所有完成 → Router 汇总 → 返回给用户
```

### 6.2 冲突升级流程（Mailbox 版）

```
  Coder 和 Reviewer 产生冲突：
  
  ① Coder → outbox.send(CONFLICT_ESCALATE → "planner")
       body: {
         "issue": "Reviewer 要求移除 try/except，但这是 API 必需的错误处理",
         "reviewer_claim": "避免吞异常",
         "coder_claim": "FastAPI 最佳实践要求返回 HTTPException"
       }
       priority: URGENT
  
  ② Planner Inbox 收到 URGENT 消息（跳过 NORMAL 队列，优先处理）
     │
     ├→ planner.run() → 重评估
     │
     ├→ 方案 A（协商通过）→ outbox.send(TASK_ASSIGNMENT → "coder")
     │    body: {"decision": "保留 try/except，但添加具体异常类型"}
     │
     ├→ 方案 B（无法协商）→ outbox.send(CONFLICT_ESCALATE → "user")
     │    body: {"escalation_reason": "...", "options": [...]}
     │    priority: URGENT
```

---

## 七、详细开发清单（含 HOW）

### Step 1：实现 Mailbox 核心（2 小时）⭐

文件清单：
- `app/agent/mailbox/message.py` — `MailboxMessage` + 枚举（约 100 行）
- `app/agent/mailbox/inbox.py` — `AgentInbox`（约 150 行）
- `app/agent/mailbox/outbox.py` — `AgentOutbox`（约 80 行）
- `app/agent/mailbox/manager.py` — `MailboxManager`（约 120 行）
- `app/agent/mailbox/watcher.py` — `InboxWatcher`（约 40 行）
- `app/agent/mailbox/persistence.py` — 内存/Redis 持久化（约 50 行）
- `app/agent/mailbox/__init__.py` — 统一导出

### Step 2：改造 BaseAgent（30 分钟）

修改 `app/agent/base.py`，将 Mailbox 集成到基类：
- `__init__` 接收 `MailboxManager`，创建 Inbox/Outbox
- 注册默认消息处理器（`TASK_ASSIGNMENT`、`STATUS_QUERY`、`BROADCAST`）
- 每个 Agent 实例化后自动注册到 MailboxManager

### Step 3：实现各 Agent（2 小时）

**Planner Agent：**
- 处理 `TASK_ASSIGNMENT` → 拆解任务 → 通过 Outbox 发 `TASK_RESULT` 回 Router
- 额外注册 `CONFLICT_ESCALATE` handler —— Planner 作为仲裁者

**Coder Agent：**
- 处理 `TASK_ASSIGNMENT` → 编码 → `TASK_RESULT`
- 额外注册 `REVIEW_FEEDBACK` handler —— 处理 Reviewer 的修改意见

**Tester Agent：**
- 处理 `TASK_ASSIGNMENT` → 执行测试 → `TASK_RESULT`
- 如果失败，发 `REVIEW_FEEDBACK` 给 Coder（而非直接 `TASK_RESULT` 给 Router）

**Reviewer Agent：**
- 处理 `TASK_ASSIGNMENT` → Code Review → `REVIEW_FEEDBACK` 给 Coder
- 通过后发 `TASK_RESULT` 给 Router

### Step 4：实现 Agent Router（1 小时）

Router 本身就是第一个 Agent，处理用户的初始请求：
1. 用户请求进入 → Router 分析 → 如果任务复杂，先发 `TASK_ASSIGNMENT` 给 Planner
2. Planner 返回 → Router 根据任务列表，串行/并行分发 `TASK_ASSIGNMENT`
3. 等待各 Agent 的 `TASK_RESULT` → 聚合 → 返回给用户

Router 的策略与 v1 相同（规则路由 + LLM 路由），但分发方式改为 `outbox.send()`。

### Step 5：编排 Multi-Agent Workflow（1 小时）

```python
# 完整的 Multi-Agent 工作流编排
class MultiAgentOrchestrator:
    """
    多 Agent 编排器 —— 管理 Agent 池的生命周期 + 工作流编排。
    """

    def __init__(self):
        self.mailbox = MailboxManager()
        self.agents: dict[str, BaseAgent] = {}

    def register_all(self) -> None:
        """创建并注册所有 Agent。"""
        for role in AGENT_ROLES.values():
            agent_cls = AGENT_CLASS_MAP[role.name]
            agent = agent_cls(role, self.mailbox)
            self.agents[role.name] = agent
        print(f"  [orchestrator] {len(self.agents)} agents registered")

    async def start_all(self) -> None:
        """后台启动所有 Agent 的 InboxWatcher。"""
        tasks = []
        for agent in self.agents.values():
            tasks.append(agent.start_background())
        # 所有 Agent 并行监听各自的 Inbox
        await asyncio.gather(*tasks)

    async def handle_user_request(self, user_message: str) -> str:
        """
        用户请求入口：

        1. Router 分析 → 发送 TASK_ASSIGNMENT 给 Planner
        2. 等待 Planner 返回任务列表
        3. 按依赖顺序分发给 Coder/Tester/Reviewer
        4. 收集所有结果 → 汇总返回
        """
        correlation_id = f"user_{uuid.uuid4().hex[:8]}"

        # Step 1: Router → Planner
        router = self.agents["router"]
        planner_task = router.outbox.create_message(
            recipient="planner",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": user_message},
            correlation_id=correlation_id,
        )
        await router.outbox.send(planner_task)

        # Step 2: 等待 Planner 回复（通过轮询 Inbox 或 reply_to 关联）
        plan_result = await self._wait_for_reply(
            planner_task.id, correlation_id, timeout=30.0
        )
        if not plan_result:
            return "Planner 未在超时时间内响应"

        # Step 3-5: 执行 Coder → Tester → Reviewer 循环
        # ... (根据 plan_result 中的 steps 逐一分发)
```

### Step 6：消息持久化（1 小时，可选）

- MVP：所有 Inbox/Outbox 纯内存（Agent 重启后丢失）
- 进阶：Inbox 消息序列化到本地 JSON 文件（Agent 重启后可恢复未处理消息）
- 生产：接入 Redis Pub/Sub（天然支持消息队列 + 多消费者）

```python
# app/agent/mailbox/persistence.py
class MailboxPersistence(ABC):
    """消息持久化抽象基类"""
    @abstractmethod
    async def save(self, inbox: AgentInbox) -> None: ...
    @abstractmethod
    async def load(self, owner: str) -> AgentInbox | None: ...

class MemoryPersistence(MailboxPersistence):
    """MVP：不做持久化"""
    async def save(self, inbox): pass
    async def load(self, owner): return None

class FilePersistence(MailboxPersistence):
    """消息持久化到 JSON 文件"""
    def __init__(self, path: str = "data/mailbox"): ...
```

---

## 八、完成标志

### 基本完成
- [ ] Mailbox 核心模块通过单元测试（message/inbox/outbox/manager/watcher）
- [ ] Planner/Coder/Tester/Reviewer + Router 五个 Agent 各自能独立处理消息
- [ ] Router → Planner → Coder → Tester → Reviewer 完整流程跑通（消息链路可审计）
- [ ] Agent 间冲突能通过 Mailbox 升级到 Planner
- [ ] 消息审计日志可追溯完整 `correlation_id` 链路

### 自测用例

```bash
# 测试 1：完整 Multi-Agent + Mailbox 流程
curl -X POST /api/chat -d '{
  "message": "创建一个 FastAPI User API，包含 GET /users 和 POST /users，写完整的 CRUD 和测试"
}'
# 查看 Mailbox 审计日志：
#   GET /api/mailbox/audit?correlation_id=user_abc12345
# 期望看到完整链路：
#   router → planner (TASK_ASSIGNMENT)
#   planner → router (TASK_RESULT)
#   router → coder (TASK_ASSIGNMENT)
#   coder → router (TASK_RESULT)
#   router → tester (TASK_ASSIGNMENT) + router → reviewer (TASK_ASSIGNMENT)  ← 并行
#   tester → coder (REVIEW_FEEDBACK)  ← 如果测试失败
#   reviewer → coder (REVIEW_FEEDBACK)

# 测试 2：Mailbox 死信队列
# 故意让一个 Agent 崩溃，TASK_ASSIGNMENT 超时 → 进入死信队列
# 验证 MailboxManager.reprocess_dead_letter() 能重新投递

# 测试 3：冲突升级
# Reviewer 提出 Coder 不同意的修改 → Coder 发 CONFLICT_ESCALATE
# → Planner 收到 URGENT 消息优先处理 → 返回决策
```

---

## 九、面试考点映射

| 面试题 | 答案要点 | 本文档参考 |
|--------|---------|-----------|
| **你的多 Agent 怎么通信？** | Mailbox 系统：每个 Agent 有独立 Inbox/Outbox，MailboxManager 做消息路由。**不是**共享 State 的 `messages.append(...)`。每条消息有类型、优先级、TTL、correlation_id。 | §4 |
| **如何保证消息不丢失？** | 消息生命周期（DRAFT→SENT→DELIVERED→READ→PROCESSED），TTL 过期检测，死信队列兜底。生产环境可接 Redis 持久化。 | §4.1, §4.2 |
| **如何处理 Agent 冲突？** | 三级冲突解决：1) Agent 间通过 Mailbox 直接协商（REVIEW_FEEDBACK 来回），2) Planner 仲裁（CONFLICT_ESCALATE，URGENT 优先级跳队），3) 升级到用户。 | §6.2 |
| **如何支持并行执行？** | Router 可以同时发 TASK_ASSIGNMENT 给多个 Agent（Tester + Reviewer 并行）。Mailbox 天然支持——每个 Agent 独立监听自己的 Inbox，消息投递不阻塞其他 Agent。 | §6.1 |
| **如何追踪一个任务的完整执行链路？** | `correlation_id` 线程追踪。MailboxManager 的审计日志记录了每条消息的 sender/recipient/timestamp，可完整回放。 | §4.1, §4.4 |
| **为什么不用 LangGraph 的 State 传递？** | State 传递是紧耦合——Agent 必须知道 State 的结构。Mailbox 是松耦合——Agent 只关心自己收到了什么消息类型，不关心其他 Agent 的内部状态。未来扩展到分布式时，Mailbox 可以直接换 Redis/Kafka，State 传递做不到。 | §1（对比表） |

---

## 十、与 v1 共享 State 方案的差异一览

| 场景 | v1 共享 State | v2 Mailbox 系统 |
|------|-------------|----------------|
| Coder 完成后通知 Tester | `state["agent_outputs"]["coder"] = ...` | `outbox.send(TASK_RESULT → "router")` |
| Tester 发现 Bug 通知 Coder | `state["messages"].append("[tester→coder]: ...")` | `outbox.send(REVIEW_FEEDBACK → "coder")` |
| 冲突升级到 Planner | 没有机制 | `outbox.send(CONFLICT_ESCALATE, priority=URGENT)` |
| 并行投递 Tester + Reviewer | 不支持 | Router 同时 send 两条 TASK_ASSIGNMENT |
| 消息丢失怎么办 | 丢了就丢了 | TTL + 死信队列 + reprocess |
| 追踪谁给谁发了什么 | 只能 grep messages 列表 | 审计日志 + correlation_id |
| Agent 崩溃后恢复 | 依赖 State 序列化 | Inbox 持久化，重启后未处理消息仍在 |
