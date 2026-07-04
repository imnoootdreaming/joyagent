"""
Phase 7 Step 1 — Mailbox 核心模块单元测试

覆盖：
  - MailboxMessage: 创建、过期检测、序列化/反序列化
  - AgentInbox: 投递、fetch、处理器注册、过期清理、死信
  - AgentOutbox: 消息创建、发送、send_and_wait
  - MailboxManager: 注册、路由、广播、审计、死信
  - InboxWatcher: 启动/停止、消息处理
  - Persistence: 内存/文件后端
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import pytest

from app.agent.mailbox import (
    AgentInbox,
    AgentOutbox,
    FilePersistence,
    InboxWatcher,
    MailboxManager,
    MailboxMessage,
    MailboxPersistence,
    MemoryPersistence,
    MessagePriority,
    MessageStatus,
    MessageType,
)


# ═══════════════════════════════════════════════════════════════════════
# MailboxMessage 测试
# ═══════════════════════════════════════════════════════════════════════

class TestMailboxMessage:
    """消息数据模型测试。"""

    def test_create_default(self):
        """默认创建：状态为 DRAFT，自动生成 ID。"""
        msg = MailboxMessage()
        assert msg.id.startswith("msg_")
        assert msg.status == MessageStatus.DRAFT
        assert msg.priority == MessagePriority.NORMAL
        assert msg.msg_type == MessageType.TASK_ASSIGNMENT

    def test_create_full(self):
        """完整参数创建。"""
        msg = MailboxMessage(
            sender="router",
            recipient="coder",
            msg_type=MessageType.TASK_ASSIGNMENT,
            priority=MessagePriority.HIGH,
            subject="Build user API",
            body={"task": "create FastAPI routes"},
            correlation_id="thread_abc",
            reply_to="msg_prev",
            ttl_seconds=600,
            tags=["phase:code", "step:3"],
            max_retries=5,
        )
        assert msg.sender == "router"
        assert msg.recipient == "coder"
        assert msg.msg_type == MessageType.TASK_ASSIGNMENT
        assert msg.priority == MessagePriority.HIGH
        assert msg.subject == "Build user API"
        assert msg.body == {"task": "create FastAPI routes"}
        assert msg.correlation_id == "thread_abc"
        assert msg.reply_to == "msg_prev"
        assert msg.ttl_seconds == 600
        assert msg.tags == ["phase:code", "step:3"]
        assert msg.max_retries == 5

    def test_is_expired_false_when_ttl_zero(self):
        """TTL 为 0 = 永不过期。"""
        msg = MailboxMessage(ttl_seconds=0)
        assert not msg.is_expired

    def test_is_expired_false_when_fresh(self):
        """刚创建的消息未过期。"""
        msg = MailboxMessage(ttl_seconds=300)
        assert not msg.is_expired

    def test_is_expired_true_when_exceeded(self):
        """创建时间早于 TTL → 过期。"""
        msg = MailboxMessage(
            ttl_seconds=1,
            created_at=time.time() - 2,  # 2 秒前创建
        )
        assert msg.is_expired

    def test_age_seconds(self):
        """age_seconds 计算正确。"""
        msg = MailboxMessage(created_at=time.time() - 10)
        assert 9.5 <= msg.age_seconds <= 10.5

    def test_to_envelope(self):
        """to_envelope 序列化为 JSON 安全的 dict。"""
        msg = MailboxMessage(
            sender="router",
            recipient="coder",
            msg_type=MessageType.TASK_RESULT,
            body={"output": "done"},
            tags=["tag1"],
            created_at=1000.0,
        )
        env = msg.to_envelope()
        assert env["sender"] == "router"
        assert env["msg_type"] == "task_result"
        assert env["priority"] == 3  # NORMAL 枚举值
        assert env["body"] == {"output": "done"}
        assert env["tags"] == ["tag1"]
        assert env["created_at"] == 1000.0
        # 确保 JSON 可序列化
        json.dumps(env)

    def test_from_envelope(self):
        """from_envelope 正确反序列化。"""
        env = {
            "id": "msg_test123",
            "correlation_id": "corr_abc",
            "reply_to": "msg_prev",
            "sender": "planner",
            "recipient": "coder",
            "msg_type": "task_assignment",
            "priority": 5,
            "subject": "Test",
            "body": {"key": "val"},
            "status": "delivered",
            "created_at": 2000.0,
            "ttl_seconds": 600,
            "delivered_at": 2001.0,
            "read_at": 0.0,
            "processed_at": 0.0,
            "tags": ["t1", "t2"],
            "retry_count": 1,
            "max_retries": 5,
        }
        msg = MailboxMessage.from_envelope(env)
        assert msg.id == "msg_test123"
        assert msg.sender == "planner"
        assert msg.msg_type == MessageType.TASK_ASSIGNMENT
        assert msg.priority == MessagePriority.HIGH
        assert msg.status == MessageStatus.DELIVERED
        assert msg.body == {"key": "val"}
        assert msg.created_at == 2000.0
        assert msg.tags == ["t1", "t2"]
        assert msg.retry_count == 1

    def test_to_envelope_round_trip(self):
        """序列化 + 反序列化完整往返。"""
        original = MailboxMessage(
            sender="tester",
            recipient="coder",
            msg_type=MessageType.REVIEW_FEEDBACK,
            priority=MessagePriority.URGENT,
            subject="Bug found",
            body={"failed": ["test_x"]},
            correlation_id="corr_xyz",
            tags=["urgent"],
        )
        restored = MailboxMessage.from_envelope(original.to_envelope())
        assert restored.id == original.id
        assert restored.sender == original.sender
        assert restored.msg_type == original.msg_type
        assert restored.priority == original.priority
        assert restored.body == original.body
        assert restored.tags == original.tags


# ═══════════════════════════════════════════════════════════════════════
# AgentInbox 测试
# ═══════════════════════════════════════════════════════════════════════

class TestAgentInbox:
    """收件箱测试。"""

    def test_deliver_message(self):
        """投递消息到 Inbox → 状态变为 DELIVERED。"""
        inbox = AgentInbox(owner="test")
        msg = MailboxMessage(sender="router", recipient="test")
        assert inbox.deliver(msg) is True
        assert msg.status == MessageStatus.DELIVERED
        assert msg.delivered_at > 0
        assert inbox.total_count == 1

    def test_deliver_inbox_full(self):
        """Inbox 满了返回 False。"""
        inbox = AgentInbox(owner="test", max_size=2)
        inbox.deliver(MailboxMessage())
        inbox.deliver(MailboxMessage())
        assert inbox.deliver(MailboxMessage()) is False
        assert inbox.total_count == 2

    def test_fetch_unread_priority_order(self):
        """fetch_unread 按优先级排序：URGENT > HIGH > NORMAL > LOW。"""
        inbox = AgentInbox(owner="test")
        msg_low = MailboxMessage(priority=MessagePriority.LOW, subject="low")
        msg_urgent = MailboxMessage(priority=MessagePriority.URGENT, subject="urgent")
        msg_normal = MailboxMessage(priority=MessagePriority.NORMAL, subject="normal")
        msg_high = MailboxMessage(priority=MessagePriority.HIGH, subject="high")

        inbox.deliver(msg_low)
        inbox.deliver(msg_urgent)
        inbox.deliver(msg_normal)
        inbox.deliver(msg_high)

        unread = inbox.fetch_unread()
        priorities = [m.priority for m in unread]
        assert priorities == [
            MessagePriority.URGENT,
            MessagePriority.HIGH,
            MessagePriority.NORMAL,
            MessagePriority.LOW,
        ]

    def test_fetch_by_correlation(self):
        """按 correlation_id 过滤消息。"""
        inbox = AgentInbox(owner="test")
        m1 = MailboxMessage(correlation_id="corr_a", subject="a1")
        m2 = MailboxMessage(correlation_id="corr_a", subject="a2")
        m3 = MailboxMessage(correlation_id="corr_b", subject="b1")

        inbox.deliver(m1)
        inbox.deliver(m2)
        inbox.deliver(m3)

        result = inbox.fetch_by_correlation("corr_a")
        assert len(result) == 2
        result_b = inbox.fetch_by_correlation("corr_b")
        assert len(result_b) == 1
        result_empty = inbox.fetch_by_correlation("nonexistent")
        assert len(result_empty) == 0

    @pytest.mark.asyncio
    async def test_fetch_next_immediate_return(self):
        """有消息时 fetch_next 立即返回。"""
        inbox = AgentInbox(owner="test")
        msg = MailboxMessage()
        inbox.deliver(msg)

        result = await inbox.fetch_next(timeout=0)
        assert result is not None
        assert result.status == MessageStatus.READ
        assert result.read_at > 0

    @pytest.mark.asyncio
    async def test_fetch_next_timeout(self):
        """无消息时 fetch_next 超时返回 None。"""
        inbox = AgentInbox(owner="test")
        result = await inbox.fetch_next(timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_next_priority_order(self):
        """fetch_next 按优先级返回：URGENT 优先。"""
        inbox = AgentInbox(owner="test")
        inbox.deliver(MailboxMessage(priority=MessagePriority.LOW, subject="low"))
        inbox.deliver(MailboxMessage(priority=MessagePriority.URGENT, subject="urgent"))

        first = await inbox.fetch_next(timeout=0)
        assert first.subject == "urgent"

        second = await inbox.fetch_next(timeout=0)
        assert second.subject == "low"

    @pytest.mark.asyncio
    async def test_register_and_process_handler(self):
        """注册 handler 并处理消息。"""
        inbox = AgentInbox(owner="test")
        handled = []

        async def my_handler(msg: MailboxMessage) -> bool:
            handled.append(msg.subject)
            return True

        inbox.register_handler(MessageType.TASK_ASSIGNMENT, my_handler)

        msg = MailboxMessage(
            sender="router",
            recipient="test",
            msg_type=MessageType.TASK_ASSIGNMENT,
            subject="handle_me",
        )
        inbox.deliver(msg)

        success = await inbox.process(msg)
        assert success is True
        assert handled == ["handle_me"]
        assert msg.status == MessageStatus.PROCESSED
        assert inbox.stats["processed"] == 1

    @pytest.mark.asyncio
    async def test_process_no_handler(self):
        """没有注册 handler → 处理失败。"""
        inbox = AgentInbox(owner="test")
        msg = MailboxMessage(
            sender="router",
            msg_type=MessageType.STATUS_QUERY,
        )
        inbox.deliver(msg)

        success = await inbox.process(msg)
        assert success is False
        assert msg.status == MessageStatus.FAILED

    @pytest.mark.asyncio
    async def test_process_default_handler(self):
        """未匹配特定 handler 时使用默认 handler。"""
        inbox = AgentInbox(owner="test")
        handled = []

        async def default_handler(msg: MailboxMessage) -> bool:
            handled.append(msg.msg_type)
            return True

        inbox.register_default_handler(default_handler)

        msg = MailboxMessage(
            sender="router",
            msg_type=MessageType.BROADCAST,
        )
        inbox.deliver(msg)

        success = await inbox.process(msg)
        assert success is True
        assert handled == [MessageType.BROADCAST]

    @pytest.mark.asyncio
    async def test_handler_returns_false_triggers_retry(self):
        """handler 返回 False → 重试。"""
        inbox = AgentInbox(owner="test")
        call_count = 0

        async def flaky_handler(msg: MailboxMessage) -> bool:
            nonlocal call_count
            call_count += 1
            return call_count >= 2  # 第二次才成功

        inbox.register_handler(MessageType.TASK_ASSIGNMENT, flaky_handler)

        msg = MailboxMessage(
            sender="router",
            msg_type=MessageType.TASK_ASSIGNMENT,
            subject="flaky",
        )
        inbox.deliver(msg)

        # 第一次处理
        success1 = await inbox.process(msg)
        assert success1 is False
        assert msg.retry_count == 1
        assert msg.status == MessageStatus.DELIVERED  # 重置为未读

        # 第二次处理
        success2 = await inbox.process(msg)
        assert success2 is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_to_dead_letter(self):
        """超过最大重试次数 → 进入死信队列。"""
        inbox = AgentInbox(owner="test")

        async def always_fail(msg: MailboxMessage) -> bool:
            return False

        inbox.register_handler(MessageType.TASK_ASSIGNMENT, always_fail)

        msg = MailboxMessage(
            sender="router",
            msg_type=MessageType.TASK_ASSIGNMENT,
            max_retries=2,
        )
        inbox.deliver(msg)

        # 第一次处理失败 → retry_count=1，重置为 DELIVERED（未超过 max_retries=2）
        await inbox.process(msg)
        assert msg.retry_count == 1
        assert msg.status == MessageStatus.DELIVERED

        # 第二次处理失败 → retry_count=2 >= max_retries=2 → 死信
        await inbox.process(msg)
        assert msg.retry_count == 2
        assert msg.status == MessageStatus.FAILED
        assert inbox.dead_letter_count == 1
        assert inbox.total_count == 0  # 从活跃队列移除

    @pytest.mark.asyncio
    async def test_handler_exception_triggers_retry(self):
        """handler 抛异常 → 触发重试。"""
        inbox = AgentInbox(owner="test")
        call_count = 0

        async def crash_then_ok(msg: MailboxMessage) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")
            return True

        inbox.register_handler(MessageType.TASK_ASSIGNMENT, crash_then_ok)

        msg = MailboxMessage(
            sender="router",
            msg_type=MessageType.TASK_ASSIGNMENT,
            max_retries=3,
        )
        inbox.deliver(msg)

        # 第一次抛异常
        success1 = await inbox.process(msg)
        assert success1 is False
        assert msg.retry_count == 1

        # 第二次成功
        success2 = await inbox.process(msg)
        assert success2 is True
        assert call_count == 2

    def test_clean_expired(self):
        """过期消息清理到死信。"""
        inbox = AgentInbox(owner="test")
        fresh = MailboxMessage(ttl_seconds=300, subject="fresh")
        expired = MailboxMessage(
            ttl_seconds=1,
            created_at=time.time() - 10,
            subject="expired",
        )

        inbox.deliver(fresh)
        inbox.deliver(expired)

        cleaned = inbox.clean_expired()
        assert len(cleaned) == 1
        assert cleaned[0].subject == "expired"
        assert cleaned[0].status == MessageStatus.EXPIRED
        assert inbox.dead_letter_count == 1
        assert inbox.total_count == 1  # 只剩 fresh

    def test_stats(self):
        """stats 属性返回正确的统计信息。"""
        inbox = AgentInbox(owner="test")

        async def dummy(msg: MailboxMessage) -> bool:
            return True

        inbox.register_handler(MessageType.TASK_ASSIGNMENT, dummy)

        s = inbox.stats
        assert s["received"] == 0
        assert s["current_depth"] == 0
        assert "task_assignment" in s["handlers_registered"]


# ═══════════════════════════════════════════════════════════════════════
# AgentOutbox 测试
# ═══════════════════════════════════════════════════════════════════════

class TestAgentOutbox:
    """发件箱测试。"""

    def test_create_message(self):
        """create_message 自动填充 sender。"""
        outbox = AgentOutbox(owner="planner")
        msg = outbox.create_message(
            recipient="coder",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "do X"},
            subject="New task",
            priority=MessagePriority.HIGH,
        )
        assert msg.sender == "planner"
        assert msg.recipient == "coder"
        assert msg.msg_type == MessageType.TASK_ASSIGNMENT
        assert msg.body == {"task": "do X"}
        assert msg.subject == "New task"
        assert msg.priority == MessagePriority.HIGH
        assert msg.status == MessageStatus.DRAFT
        assert msg.correlation_id != ""

    def test_create_message_auto_correlation_id(self):
        """不指定 correlation_id → 自动生成。"""
        outbox = AgentOutbox(owner="test")
        msg = outbox.create_message(
            recipient="coder",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body="hello",
        )
        assert msg.correlation_id.startswith("thread_")

    def test_send_unbound_raises(self):
        """未绑定的 Outbox 调用 send 抛出异常。"""
        outbox = AgentOutbox(owner="test")
        msg = MailboxMessage()

        with pytest.raises(RuntimeError, match="not bound"):
            asyncio.run(outbox.send(msg))

    @pytest.mark.asyncio
    async def test_send_and_route(self):
        """send 通过 Manager 投递到目标 Inbox。"""
        manager = MailboxManager()
        inbox = AgentInbox(owner="coder")
        outbox = AgentOutbox(owner="router")
        manager.register_agent("coder", inbox, outbox)

        msg = outbox.create_message(
            recipient="coder",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body="test",
        )
        success = await outbox.send(msg)
        assert success is True
        # send() 先设 SENT → route() 成功后 inbox.deliver() 设为 DELIVERED
        assert msg.status == MessageStatus.DELIVERED
        assert inbox.total_count == 1
        assert len(outbox.get_sent_history()) == 1

    @pytest.mark.asyncio
    async def test_send_and_wait(self):
        """send_and_wait 等待消息被处理。"""
        manager = MailboxManager()
        inbox = AgentInbox(owner="coder")
        outbox = AgentOutbox(owner="router")
        manager.register_agent("coder", inbox, outbox)

        msg = outbox.create_message(
            recipient="coder",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body="test",
        )

        # 在后台模拟 coder 处理消息
        async def coder_sim():
            received = await inbox.fetch_next(timeout=5.0)
            assert received is not None
            # 标记为已处理
            received.status = MessageStatus.PROCESSED
            # 更新 manager 的状态索引
            manager._status_index[received.id] = MessageStatus.PROCESSED

        asyncio.create_task(coder_sim())

        final_status = await outbox.send_and_wait(msg, timeout=5.0)
        assert final_status == MessageStatus.PROCESSED

    @pytest.mark.asyncio
    async def test_flush(self):
        """flush 发送所有 pending 消息。"""
        manager = MailboxManager()
        inbox = AgentInbox(owner="coder")
        outbox = AgentOutbox(owner="router")
        manager.register_agent("coder", inbox, outbox)

        m1 = outbox.create_message("coder", MessageType.TASK_ASSIGNMENT, "t1")
        m2 = outbox.create_message("coder", MessageType.TASK_ASSIGNMENT, "t2")
        outbox._pending = [m1, m2]

        count = await outbox.flush()
        assert count == 2
        assert inbox.total_count == 2

    def test_get_sent_history_filter(self):
        """按收件人过滤已发送历史。"""
        outbox = AgentOutbox(owner="router")
        outbox._sent = [
            MailboxMessage(recipient="coder", subject="to_coder"),
            MailboxMessage(recipient="tester", subject="to_tester"),
            MailboxMessage(recipient="coder", subject="to_coder_2"),
        ]
        coder_msgs = outbox.get_sent_history(recipient="coder")
        assert len(coder_msgs) == 2


# ═══════════════════════════════════════════════════════════════════════
# MailboxManager 测试
# ═══════════════════════════════════════════════════════════════════════

class TestMailboxManager:
    """消息路由中枢测试。"""

    def test_register_agent(self):
        """注册 Agent → 自动绑定 Outbox。"""
        manager = MailboxManager()
        inbox = AgentInbox(owner="planner")
        outbox = AgentOutbox(owner="planner")

        manager.register_agent("planner", inbox, outbox)
        assert "planner" in manager.registered_agents
        assert outbox.is_bound is True

    def test_unregister_agent(self):
        """注销 Agent 后路由失败。"""
        manager = MailboxManager()
        inbox = AgentInbox(owner="temp")
        outbox = AgentOutbox(owner="temp")
        manager.register_agent("temp", inbox, outbox)

        manager.unregister_agent("temp")
        assert "temp" not in manager.registered_agents

    @pytest.mark.asyncio
    async def test_route_success(self):
        """路由消息到目标 Agent。"""
        manager = MailboxManager()
        inbox = AgentInbox(owner="coder")
        outbox = AgentOutbox(owner="router")
        manager.register_agent("coder", inbox, outbox)

        msg = MailboxMessage(sender="router", recipient="coder", subject="task1")
        success = await manager.route(msg)

        assert success is True
        assert inbox.total_count == 1
        assert msg.status == MessageStatus.DELIVERED

    @pytest.mark.asyncio
    async def test_route_recipient_not_found(self):
        """目标 Agent 不存在 → 路由失败。"""
        manager = MailboxManager()
        msg = MailboxMessage(sender="router", recipient="nonexistent")

        success = await manager.route(msg)
        assert success is False
        assert manager._stats["total_failed"] == 1

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """广播消息到所有已注册 Agent（除 sender）。"""
        manager = MailboxManager()
        coder_inbox = AgentInbox(owner="coder")
        tester_inbox = AgentInbox(owner="tester")
        reviewer_inbox = AgentInbox(owner="reviewer")

        manager.register_agent("coder", coder_inbox, AgentOutbox(owner="coder"))
        manager.register_agent("tester", tester_inbox, AgentOutbox(owner="tester"))
        manager.register_agent("reviewer", reviewer_inbox, AgentOutbox(owner="reviewer"))

        msg = MailboxMessage(
            sender="router",
            recipient="*",  # 广播
            msg_type=MessageType.BROADCAST,
            subject="system shutdown",
        )
        success = await manager.route(msg)
        assert success is True
        # 所有非 sender 的 Agent 都应收到
        assert coder_inbox.total_count == 1
        assert tester_inbox.total_count == 1
        assert reviewer_inbox.total_count == 1

    @pytest.mark.asyncio
    async def test_broadcast_excludes_sender(self):
        """广播不发给发送者。"""
        manager = MailboxManager()
        router_inbox = AgentInbox(owner="router")
        manager.register_agent("router", router_inbox, AgentOutbox(owner="router"))

        msg = MailboxMessage(
            sender="router",
            recipient="*",
            msg_type=MessageType.BROADCAST,
            subject="test",
        )
        await manager.route(msg)
        # router 是 sender，不应收到
        assert router_inbox.total_count == 0

    @pytest.mark.asyncio
    async def test_get_message_status(self):
        """查询消息状态。"""
        manager = MailboxManager()
        inbox = AgentInbox(owner="coder")
        outbox = AgentOutbox(owner="router")
        manager.register_agent("coder", inbox, outbox)

        msg = MailboxMessage(sender="router", recipient="coder")
        await manager.route(msg)

        status = await manager.get_message_status(msg.id)
        assert status == MessageStatus.DELIVERED

    def test_get_agent_inbox_depth(self):
        """查询 Agent Inbox 深度。"""
        manager = MailboxManager()
        inbox = AgentInbox(owner="coder")
        manager.register_agent("coder", inbox, AgentOutbox(owner="coder"))

        inbox.deliver(MailboxMessage())
        inbox.deliver(MailboxMessage())

        assert manager.get_agent_inbox_depth("coder") == 2
        assert manager.get_agent_inbox_depth("nonexistent") == 0

    def test_get_all_stats(self):
        """get_all_stats 包含全局和每个 Agent 的统计。"""
        manager = MailboxManager()
        inbox = AgentInbox(owner="coder")
        manager.register_agent("coder", inbox, AgentOutbox(owner="coder"))

        stats = manager.get_all_stats()
        assert "global" in stats
        assert "agents" in stats
        assert "coder" in stats["agents"]

    def test_sweep_expired(self):
        """全局过期清理。"""
        manager = MailboxManager()
        inbox = AgentInbox(owner="coder")
        manager.register_agent("coder", inbox, AgentOutbox(owner="coder"))

        inbox.deliver(MailboxMessage(ttl_seconds=1, created_at=time.time() - 10, subject="old"))
        inbox.deliver(MailboxMessage(ttl_seconds=300, subject="fresh"))

        cleaned = manager.sweep_expired()
        assert cleaned == 1
        assert inbox.total_count == 1
        assert inbox.dead_letter_count == 1

    def test_audit_trail(self):
        """审计日志可追踪完整链路。"""
        manager = MailboxManager()
        inbox = AgentInbox(owner="coder")
        manager.register_agent("coder", inbox, AgentOutbox(owner="router"))

        async def _route():
            msg = MailboxMessage(
                sender="router",
                recipient="coder",
                correlation_id="corr_test",
                subject="audit test",
            )
            await manager.route(msg)

        asyncio.run(_route())

        trail = manager.get_audit_trail(correlation_id="corr_test")
        assert len(trail) >= 2  # routing + delivered
        assert trail[0]["correlation_id"] == "corr_test"

    def test_audit_trail_filter_by_agent(self):
        """审计日志按 Agent 名称过滤。"""
        manager = MailboxManager()
        coder_inbox = AgentInbox(owner="coder")
        manager.register_agent("coder", coder_inbox, AgentOutbox(owner="router"))

        async def _route():
            msg = MailboxMessage(sender="router", recipient="coder", subject="test")
            await manager.route(msg)

        asyncio.run(_route())

        router_trail = manager.get_audit_trail(agent_name="router")
        coder_trail = manager.get_audit_trail(agent_name="coder")
        assert len(router_trail) > 0
        assert len(coder_trail) > 0


# ═══════════════════════════════════════════════════════════════════════
# InboxWatcher 测试
# ═══════════════════════════════════════════════════════════════════════

class TestInboxWatcher:
    """异步消息监听器测试。"""

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """启动和停止 watcher。"""
        inbox = AgentInbox(owner="test")
        watcher = InboxWatcher(inbox)

        task = watcher.start()
        await asyncio.sleep(0.05)  # 给 task 时间启动

        assert watcher.is_running

        watcher.stop()
        await asyncio.sleep(0.1)

        # task 应该已经完成或被取消
        assert task.done()

    @pytest.mark.asyncio
    async def test_processes_incoming_message(self):
        """watcher 自动处理到达的消息。"""
        inbox = AgentInbox(owner="test")
        processed_msgs = []

        async def handler(msg: MailboxMessage) -> bool:
            processed_msgs.append(msg.subject)
            return True

        inbox.register_handler(MessageType.TASK_ASSIGNMENT, handler)

        watcher = InboxWatcher(inbox)
        watcher.start()

        # 模拟一条消息到达
        msg = MailboxMessage(
            sender="router",
            recipient="test",
            msg_type=MessageType.TASK_ASSIGNMENT,
            subject="auto_processed",
        )
        inbox.deliver(msg)

        # 等待 watcher 处理
        await asyncio.sleep(0.5)

        assert "auto_processed" in processed_msgs
        assert msg.status == MessageStatus.PROCESSED

        watcher.stop()
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_idle_callback(self):
        """空闲时调用 idle_callback。"""
        inbox = AgentInbox(owner="test")
        idle_calls = []

        async def on_idle():
            idle_calls.append(True)

        watcher = InboxWatcher(inbox, idle_callback=on_idle)
        watcher.start()

        # 没有消息 → watcher 超时后调用 idle_callback
        await asyncio.sleep(6.0)

        assert len(idle_calls) > 0

        watcher.stop()
        await asyncio.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════════
# Persistence 测试
# ═══════════════════════════════════════════════════════════════════════

class TestMemoryPersistence:
    """内存持久化（无操作）测试。"""

    @pytest.mark.asyncio
    async def test_save_load_delete(self):
        """MemoryPersistence 的 save/load/delete 都是空操作。"""
        persistence = MemoryPersistence()
        inbox = AgentInbox(owner="test")
        inbox.deliver(MailboxMessage(subject="will_be_lost"))

        await persistence.save(inbox)
        result = await persistence.load("test")
        assert result is None

        await persistence.delete("test")
        # 不抛异常即可


class TestFilePersistence:
    """文件持久化测试。"""

    @pytest.mark.asyncio
    async def test_save_and_load(self, tmp_path):
        """保存到文件 → 加载回来。"""
        path = str(tmp_path / "mailbox")
        persistence = FilePersistence(path=path)

        inbox = AgentInbox(owner="test_agent")
        msg1 = MailboxMessage(sender="router", recipient="test_agent", subject="task1")
        msg2 = MailboxMessage(sender="planner", recipient="test_agent", subject="task2")
        inbox.deliver(msg1)
        inbox.deliver(msg2)

        await persistence.save(inbox)

        # 验证文件存在
        file_path = Path(path) / "test_agent.json"
        assert file_path.exists()

        # 加载
        envelopes = await persistence.load("test_agent")
        assert envelopes is not None
        assert len(envelopes) == 2

        # 从 envelope 恢复
        msgs = [MailboxMessage.from_envelope(e) for e in envelopes]
        subjects = [m.subject for m in msgs]
        assert "task1" in subjects
        assert "task2" in subjects

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, tmp_path):
        """加载不存在的文件返回 None。"""
        path = str(tmp_path / "mailbox")
        persistence = FilePersistence(path=path)

        result = await persistence.load("nonexistent_agent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, tmp_path):
        """删除持久化文件。"""
        path = str(tmp_path / "mailbox")
        persistence = FilePersistence(path=path)

        inbox = AgentInbox(owner="temp")
        inbox.deliver(MailboxMessage(subject="x"))
        await persistence.save(inbox)

        file_path = Path(path) / "temp.json"
        assert file_path.exists()

        await persistence.delete("temp")
        assert not file_path.exists()

    @pytest.mark.asyncio
    async def test_save_empty_inbox(self, tmp_path):
        """空 Inbox 不创建文件。"""
        path = str(tmp_path / "mailbox")
        persistence = FilePersistence(path=path)

        inbox = AgentInbox(owner="empty")
        await persistence.save(inbox)

        file_path = Path(path) / "empty.json"
        assert not file_path.exists()


# ═══════════════════════════════════════════════════════════════════════
# 集成测试：完整消息生命周期
# ═══════════════════════════════════════════════════════════════════════

class TestMailboxIntegration:
    """端到端集成测试。"""

    @pytest.mark.asyncio
    async def test_full_message_lifecycle(self):
        """完整的消息生命周期：创建 → 发送 → 投递 → 处理。"""
        manager = MailboxManager()

        # 创建 Coder Agent 的 Inbox/Outbox
        coder_inbox = AgentInbox(owner="coder")
        coder_outbox = AgentOutbox(owner="coder")
        manager.register_agent("coder", coder_inbox, coder_outbox)

        # Coder 注册任务处理器
        completed_tasks = []

        async def handle_task(msg: MailboxMessage) -> bool:
            completed_tasks.append(msg.body)
            return True

        coder_inbox.register_handler(MessageType.TASK_ASSIGNMENT, handle_task)

        # Router 发送任务
        router_outbox = AgentOutbox(owner="router")
        router_inbox = AgentInbox(owner="router")
        manager.register_agent("router", router_inbox, router_outbox)

        task_msg = router_outbox.create_message(
            recipient="coder",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "implement login"},
            subject="Login feature",
            priority=MessagePriority.HIGH,
        )
        await router_outbox.send(task_msg)

        # 验证 Coder 收到消息
        received = await coder_inbox.fetch_next(timeout=1.0)
        assert received is not None
        assert received.body == {"task": "implement login"}

        # Coder 处理任务
        await coder_inbox.process(received)
        assert len(completed_tasks) == 1

        # 审计日志完整
        trail = manager.get_audit_trail(correlation_id=task_msg.correlation_id)
        assert len(trail) >= 2  # routing + delivered

    @pytest.mark.asyncio
    async def test_coder_tester_reviewer_flow(self):
        """模拟 Coder → Tester → Reviewer 协作流程。"""
        manager = MailboxManager()

        # 创建所有 Agent
        agents = {}
        for name in ["router", "planner", "coder", "tester", "reviewer"]:
            inbox = AgentInbox(owner=name)
            outbox = AgentOutbox(owner=name)
            manager.register_agent(name, inbox, outbox)
            agents[name] = (inbox, outbox)

        coder_inbox, coder_outbox = agents["coder"]
        tester_inbox, tester_outbox = agents["tester"]
        reviewer_inbox, reviewer_outbox = agents["reviewer"]
        router_inbox, router_outbox = agents["router"]

        # Router 注册 TASK_RESULT 处理器（收集结果）
        router_results = []

        async def collect_result(msg: MailboxMessage) -> bool:
            router_results.append({"sender": msg.sender, "body": msg.body})
            return True

        router_inbox.register_handler(MessageType.TASK_RESULT, collect_result)

        # Coder 注册任务处理器
        async def coder_task(msg: MailboxMessage) -> bool:
            # 模拟编码完成 → 发送结果给 Router
            reply = coder_outbox.create_message(
                recipient="router",
                msg_type=MessageType.TASK_RESULT,
                body={"status": "code_done", "files": ["api.py"]},
                correlation_id=msg.correlation_id,
            )
            await coder_outbox.send(reply)
            return True

        coder_inbox.register_handler(MessageType.TASK_ASSIGNMENT, coder_task)

        # 同时注册 REVIEW_FEEDBACK 处理器
        async def coder_review_feedback(msg: MailboxMessage) -> bool:
            # 模拟根据反馈修改代码
            reply = coder_outbox.create_message(
                recipient="reviewer",
                msg_type=MessageType.TASK_RESULT,
                body={"status": "fixes_applied"},
                correlation_id=msg.correlation_id,
            )
            await coder_outbox.send(reply)
            return True

        coder_inbox.register_handler(MessageType.REVIEW_FEEDBACK, coder_review_feedback)

        # Router 发任务给 Coder
        task_msg = router_outbox.create_message(
            recipient="coder",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "create api"},
            correlation_id="corr_integration",
        )
        await router_outbox.send(task_msg)

        # 让 Coder 处理
        received = await coder_inbox.fetch_next(timeout=1.0)
        assert received is not None
        await coder_inbox.process(received)

        # Router 应该收到 Coder 的结果
        await asyncio.sleep(0.1)
        router_received = await router_inbox.fetch_next(timeout=0)
        if router_received:
            await router_inbox.process(router_received)

        assert len(router_results) >= 1
        assert router_results[0]["sender"] == "coder"

        # 验证审计日志可追踪完整链路
        trail = manager.get_audit_trail(correlation_id="corr_integration")
        senders = {e["sender"] for e in trail}
        assert "router" in senders
        assert "coder" in senders
