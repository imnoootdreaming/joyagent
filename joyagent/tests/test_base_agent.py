"""
Phase 7 Step 2 — BaseAgent + AgentRole 单元测试

覆盖：
  - AgentRole: 角色创建、预定义角色验证
  - BaseAgent: 构造、Mailbox 集成、消息处理、生命周期
  - BaseAgent: _handle_task / _handle_status_query / _handle_broadcast
  - BaseAgent: start_background / stop
  - extract_text: 从 Anthropic response content 提取文本
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent.base import BaseAgent, extract_text
from app.agent.mailbox import (
    MailboxManager,
    MailboxMessage,
    MessageStatus,
    MessageType,
)
from app.agent.roles import (
    AGENT_ROLES,
    ROLE_CODER,
    ROLE_PLANNER,
    ROLE_REVIEWER,
    ROLE_ROUTER,
    ROLE_TESTER,
    AgentRole,
)


# ═══════════════════════════════════════════════════════════════════════
# AgentRole 测试
# ═══════════════════════════════════════════════════════════════════════

class TestAgentRole:
    """AgentRole 数据类测试。"""

    def test_create_role(self):
        """创建自定义角色。"""
        role = AgentRole(
            name="custom",
            system_prompt="You are a custom agent.",
            tools=["file_read"],
            model="claude-sonnet-4-20250514",
            temperature=0.3,
            can_modify_files=True,
            can_execute_shell=False,
            needs_user_approval=True,
        )
        assert role.name == "custom"
        assert role.tools == ["file_read"]
        assert role.temperature == 0.3
        assert role.can_modify_files is True
        assert role.can_execute_shell is False
        assert role.needs_user_approval is True

    def test_role_defaults(self):
        """默认值：空工具列表、默认模型、不允许修改/执行。"""
        role = AgentRole(name="minimal", system_prompt="minimal")
        assert role.tools == []
        assert role.model == ""
        assert role.temperature == 0.0
        assert role.can_modify_files is False
        assert role.can_execute_shell is False
        assert role.needs_user_approval is False

    def test_all_predefined_roles_exist(self):
        """所有 5 个预定义角色都在 AGENT_ROLES 注册表中。"""
        assert set(AGENT_ROLES.keys()) == {
            "router", "planner", "coder", "tester", "reviewer"
        }

    def test_router_permissions(self):
        """Router: 只读，不写文件，不执行 Shell。"""
        assert ROLE_ROUTER.can_modify_files is False
        assert ROLE_ROUTER.can_execute_shell is False
        assert ROLE_ROUTER.needs_user_approval is False

    def test_planner_permissions(self):
        """Planner: 只读，不写文件，不执行 Shell。"""
        assert ROLE_PLANNER.can_modify_files is False
        assert ROLE_PLANNER.can_execute_shell is False

    def test_coder_permissions(self):
        """Coder: 可以写文件和执行 Shell，需要用户确认。"""
        assert ROLE_CODER.can_modify_files is True
        assert ROLE_CODER.can_execute_shell is True
        assert ROLE_CODER.needs_user_approval is True

    def test_tester_permissions(self):
        """Tester: 不写文件，可以执行 Shell（运行测试），需要确认。"""
        assert ROLE_TESTER.can_modify_files is False
        assert ROLE_TESTER.can_execute_shell is True
        assert ROLE_TESTER.needs_user_approval is True

    def test_reviewer_permissions(self):
        """Reviewer: 只读，不写文件，不执行 Shell。"""
        assert ROLE_REVIEWER.can_modify_files is False
        assert ROLE_REVIEWER.can_execute_shell is False


# ═══════════════════════════════════════════════════════════════════════
# extract_text 测试
# ═══════════════════════════════════════════════════════════════════════

class TestExtractText:
    """extract_text 工具函数测试。"""

    def test_extract_from_object_blocks(self):
        """从 Anthropic SDK 对象 blocks 中提取文本。"""

        class FakeTextBlock:
            type = "text"
            text = "Hello world"

        class FakeToolBlock:
            type = "tool_use"
            name = "read_file"

        content = [FakeTextBlock(), FakeToolBlock(), FakeTextBlock()]
        content[2].text = "Goodbye"

        result = extract_text(content)
        assert result == "Hello world\nGoodbye"

    def test_extract_from_dict_blocks(self):
        """从纯 dict blocks 中提取文本。"""
        content = [
            {"type": "text", "text": "Line 1"},
            {"type": "tool_use", "name": "write_file"},
            {"type": "text", "text": "Line 2"},
        ]
        result = extract_text(content)
        assert result == "Line 1\nLine 2"

    def test_extract_empty(self):
        """空 content → 空字符串。"""
        assert extract_text([]) == ""

    def test_extract_no_text_blocks(self):
        """全部是 tool_use blocks → 空字符串。"""

        class FakeToolBlock:
            type = "tool_use"
            name = "test"

        assert extract_text([FakeToolBlock(), FakeToolBlock()]) == ""


# ═══════════════════════════════════════════════════════════════════════
# BaseAgent 测试
# ═══════════════════════════════════════════════════════════════════════

class TestBaseAgent:
    """BaseAgent 基类测试。"""

    def test_create_agent(self):
        """BaseAgent 构造 + Mailbox 自动注册。"""
        manager = MailboxManager()
        agent = BaseAgent(ROLE_CODER, manager, agent_id="coder-1")

        assert agent.agent_id == "coder-1"
        assert agent.role == ROLE_CODER
        assert agent.inbox.owner == "coder-1"
        assert agent.outbox.owner == "coder-1"
        assert agent.is_watching is False
        assert "coder-1" in manager.registered_agents

    def test_create_agent_default_id(self):
        """不指定 agent_id → 使用 role.name。"""
        manager = MailboxManager()
        agent = BaseAgent(ROLE_PLANNER, manager)
        assert agent.agent_id == "planner"

    def test_agent_stats(self):
        """stats 属性返回正确的状态快照。"""
        manager = MailboxManager()
        agent = BaseAgent(ROLE_CODER, manager)

        stats = agent.stats
        assert stats["agent_id"] == "coder"
        assert stats["role"] == "coder"
        assert "inbox" in stats
        assert "outbox_sent" in stats

    def test_agent_repr(self):
        """__repr__ 输出可读。"""
        manager = MailboxManager()
        agent = BaseAgent(ROLE_TESTER, manager)

        rep = repr(agent)
        assert "BaseAgent" in rep
        assert "tester" in rep

    def test_default_handlers_registered(self):
        """构造时自动注册 3 个默认 handler。"""
        manager = MailboxManager()
        agent = BaseAgent(ROLE_CODER, manager)

        handlers = agent.inbox.stats["handlers_registered"]
        assert "task_assignment" in handlers
        assert "status_query" in handlers
        assert "broadcast" in handlers

    def test_start_background_and_stop(self):
        """后台启动 + 停止（在 async 上下文中）。"""

        async def _test():
            manager = MailboxManager()
            agent = BaseAgent(ROLE_CODER, manager, agent_id="bg-test")
            task = agent.start_background()
            await asyncio.sleep(0.05)
            assert agent.is_watching is True
            agent.stop()
            await asyncio.sleep(0.05)
            assert agent.is_watching is False

        asyncio.run(_test())

    @pytest.mark.asyncio
    async def test_handle_status_query(self):
        """STATUS_QUERY → 自动回复 STATUS_REPLY。"""
        manager = MailboxManager()
        router_inbox = manager._inboxes.get("router", None)

        # 创建 Router + Coder
        router = BaseAgent(ROLE_ROUTER, manager, agent_id="router")
        coder = BaseAgent(ROLE_CODER, manager, agent_id="coder")

        # Router 发 STATUS_QUERY 给 Coder
        msg = router.outbox.create_message(
            recipient="coder",
            msg_type=MessageType.STATUS_QUERY,
            body="status check",
            subject="How are you?",
        )
        await router.outbox.send(msg)

        # Coder 收到并处理（_handle_status_query 自动回复）
        received = await coder.inbox.fetch_next(timeout=1.0)
        assert received is not None
        assert received.msg_type == MessageType.STATUS_QUERY
        await coder.inbox.process(received)

        # Router 应该收到 STATUS_REPLY
        reply = await router.inbox.fetch_next(timeout=1.0)
        assert reply is not None
        assert reply.msg_type == MessageType.STATUS_REPLY
        assert reply.body["agent"] == "coder"
        assert reply.body["status"] == "idle"

    @pytest.mark.asyncio
    async def test_handle_task_sends_result(self):
        """TASK_ASSIGNMENT → 调用 run() → 自动回复 TASK_RESULT."""
        manager = MailboxManager()

        # 创建一个带有自定义 run() 的 Coder
        class TestCoder(BaseAgent):
            async def run(self, task: str) -> dict:
                return {"output": f"completed: {task}", "success": True}

        router = BaseAgent(ROLE_ROUTER, manager, agent_id="router")
        coder = TestCoder(ROLE_CODER, manager, agent_id="coder")

        # Router 发任务给 Coder
        msg = router.outbox.create_message(
            recipient="coder",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "build login page"},
            subject="Login task",
        )
        await router.outbox.send(msg)

        # Coder 处理任务
        received = await coder.inbox.fetch_next(timeout=1.0)
        assert received is not None
        await coder.inbox.process(received)

        # Router 应该收到 TASK_RESULT
        reply = await router.inbox.fetch_next(timeout=1.0)
        assert reply is not None
        assert reply.msg_type == MessageType.TASK_RESULT
        assert "completed" in str(reply.body)
        assert reply.correlation_id == msg.correlation_id

    @pytest.mark.asyncio
    async def test_handle_broadcast(self):
        """BROADCAST → 记录日志，返回 True。"""
        manager = MailboxManager()

        router = BaseAgent(ROLE_ROUTER, manager, agent_id="router")
        coder = BaseAgent(ROLE_CODER, manager, agent_id="coder")

        # Router 广播
        msg = router.outbox.create_message(
            recipient="*",
            msg_type=MessageType.BROADCAST,
            body="system maintenance",
            subject="System notice",
        )
        await router.outbox.send(msg)

        # Coder 收到广播
        received = await coder.inbox.fetch_next(timeout=1.0)
        assert received is not None
        assert received.msg_type == MessageType.BROADCAST
        success = await coder.inbox.process(received)
        assert success is True
        assert received.status == MessageStatus.PROCESSED


# ═══════════════════════════════════════════════════════════════════════
# 集成测试：多 Agent 通过 Mailbox 协作
# ═══════════════════════════════════════════════════════════════════════

class TestMultiAgentIntegration:
    """Step 2 集成测试：多 Agent 通过 Mailbox 通信。"""

    @pytest.mark.asyncio
    async def test_router_to_planner_flow(self):
        """
        Router → Planner → Router 完整流程：
        1. Router 发 TASK_ASSIGNMENT 给 Planner
        2. Planner 拆解任务 → TASK_RESULT 回 Router
        3. Router 收到 Planner 的计划
        """
        manager = MailboxManager()

        # 自定义 Planner：返回拆解后的计划
        class PlannerAgent(BaseAgent):
            async def run(self, task: str) -> dict:
                return {
                    "plan": [
                        {"step": 1, "agent": "coder", "task": "create api.py"},
                        {"step": 2, "agent": "tester", "task": "run tests"},
                        {"step": 3, "agent": "reviewer", "task": "review code"},
                    ],
                    "estimated_time": "5 min",
                }

        router = BaseAgent(ROLE_ROUTER, manager, agent_id="router")
        planner = PlannerAgent(ROLE_PLANNER, manager, agent_id="planner")

        # Step 1: Router → Planner
        task_msg = router.outbox.create_message(
            recipient="planner",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "create FastAPI User API with CRUD"},
            subject="New feature request",
            correlation_id="corr_flow_1",
        )
        await router.outbox.send(task_msg)

        # Step 2: Planner 处理
        received = await planner.inbox.fetch_next(timeout=1.0)
        assert received is not None
        assert received.body["task"] == "create FastAPI User API with CRUD"
        await planner.inbox.process(received)

        # Step 3: Router 收到 Planner 的计划
        reply = await router.inbox.fetch_next(timeout=1.0)
        assert reply is not None
        assert reply.msg_type == MessageType.TASK_RESULT
        assert "plan" in reply.body
        assert len(reply.body["plan"]) == 3
        assert reply.body["plan"][0]["agent"] == "coder"

        # 审计日志完整
        trail = manager.get_audit_trail(correlation_id="corr_flow_1")
        assert len(trail) >= 4  # router→planner routing+delivered, planner→router routing+delivered

    @pytest.mark.asyncio
    async def test_agent_conflict_escalation(self):
        """
        Coder 遇到冲突 → CONFLICT_ESCALATE → Planner 仲裁。

        1. Coder 检测到与 Reviewer 的冲突
        2. Coder 发 CONFLICT_ESCALATE (URGENT) 给 Planner
        3. Planner 收到 URGENT 消息（优先级最高，跳过其他消息）
        """
        manager = MailboxManager()
        decisions = []

        # Planner 注册 CONFLICT_ESCALATE handler
        class ArbiterPlanner(BaseAgent):
            async def _handle_conflict(self, msg: MailboxMessage) -> bool:
                decisions.append({
                    "issue": msg.body.get("issue", ""),
                    "decision": "keep_try_except_with_specific_types",
                })
                # 回复决策
                reply = self.outbox.create_message(
                    recipient=msg.sender,
                    msg_type=MessageType.TASK_ASSIGNMENT,
                    body={"decision": "keep_try_except_with_specific_types"},
                    correlation_id=msg.correlation_id,
                    reply_to=msg.id,
                )
                await self.outbox.send(reply)
                return True

        planner = ArbiterPlanner(ROLE_PLANNER, manager, agent_id="planner")
        planner.inbox.register_handler(
            MessageType.CONFLICT_ESCALATE, planner._handle_conflict
        )

        coder = BaseAgent(ROLE_CODER, manager, agent_id="coder")

        # Coder 发冲突升级
        conflict_msg = coder.outbox.create_message(
            recipient="planner",
            msg_type=MessageType.CONFLICT_ESCALATE,
            body={
                "issue": "Reviewer wants to remove try/except, but it's required for FastAPI error handling",
                "reviewer_claim": "avoid swallowing exceptions",
                "coder_claim": "FastAPI best practice requires HTTPException",
            },
            priority="URGENT",  # type: ignore — 会被正确传递
            correlation_id="corr_conflict_1",
        )
        # Force URGENT priority
        from app.agent.mailbox import MessagePriority
        conflict_msg.priority = MessagePriority.URGENT

        await coder.outbox.send(conflict_msg)

        # Planner 收到冲突（应该是 URGENT 优先）
        received = await planner.inbox.fetch_next(timeout=1.0)
        assert received is not None
        assert received.msg_type == MessageType.CONFLICT_ESCALATE
        assert received.priority == MessagePriority.URGENT

        await planner.inbox.process(received)

        # Planner 做出了决策
        assert len(decisions) == 1
        assert "keep_try_except" in decisions[0]["decision"]
