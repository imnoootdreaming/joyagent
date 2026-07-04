"""
Phase 7 Step 3 — Specialized Agent 单元测试

覆盖：
  - PlannerAgent: 构造、CONFLICT_ESCALATE handler、run() 计划解析
  - CoderAgent: 构造、REVIEW_FEEDBACK handler、run() 编码
  - TesterAgent: 构造、测试通过发 TASK_RESULT、测试失败发 REVIEW_FEEDBACK
  - ReviewerAgent: 构造、LGTM 发 TASK_RESULT、NEEDS_WORK 发 REVIEW_FEEDBACK
  - 集成测试: Planner → Coder → Reviewer 完整邮路
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent.base import BaseAgent
from app.agent.mailbox import (
    MailboxManager,
    MailboxMessage,
    MessageStatus,
    MessageType,
)
from app.agent.planner import PlannerAgent
from app.agent.coder import CoderAgent
from app.agent.tester import TesterAgent
from app.agent.reviewer import ReviewerAgent
from app.agent.roles import (
    ROLE_CODER,
    ROLE_PLANNER,
    ROLE_REVIEWER,
    ROLE_ROUTER,
    ROLE_TESTER,
)


# ═══════════════════════════════════════════════════════════════════════
# PlannerAgent 测试
# ═══════════════════════════════════════════════════════════════════════

class TestPlannerAgent:
    """PlannerAgent —— 任务拆解 + 冲突仲裁。"""

    def test_create_planner(self):
        """Planner 构造 + Mailbox 注册 + CONFLICT_ESCALATE handler。"""
        manager = MailboxManager()
        planner = PlannerAgent(ROLE_PLANNER, manager, agent_id="planner-1")

        assert planner.agent_id == "planner-1"
        assert planner.role == ROLE_PLANNER
        assert "planner-1" in manager.registered_agents

        # 验证 CONFLICT_ESCALATE handler 已注册
        handlers = planner.inbox.stats["handlers_registered"]
        assert "conflict_escalate" in handlers

    def test_create_planner_default_id(self):
        """不指定 agent_id → 使用 role.name。"""
        manager = MailboxManager()
        planner = PlannerAgent(ROLE_PLANNER, manager)
        assert planner.agent_id == "planner"

    @pytest.mark.asyncio
    async def test_handle_conflict_escalation(self):
        """
        Planner 收到 CONFLICT_ESCALATE → 仲裁 → 回复决策。

        验证：
          1. Planner 收到 URGENT 冲突消息
          2. 处理完成后发送 TASK_RESULT（含仲裁决策）回发送方
        """
        manager = MailboxManager()

        # 自定义 Planner：覆盖 _handle_conflict_escalation 避免真实 LLM 调用
        class MockArbiter(PlannerAgent):
            async def _handle_conflict_escalation(self, msg: MailboxMessage) -> bool:
                body = msg.body if isinstance(msg.body, dict) else {}
                reply = self.outbox.create_message(
                    recipient=msg.sender,
                    msg_type=MessageType.TASK_RESULT,
                    body={
                        "decision": "Keep try/except with specific exception types",
                        "reasoning": "FastAPI best practice requires HTTPException",
                        "action": "coder keeps try/except, adds specific types",
                        "compromise": None,
                        "arbitrated_by": self.agent_id,
                    },
                    subject=f"Arbitration: {body.get('issue', '')[:60]}",
                    correlation_id=msg.correlation_id,
                    reply_to=msg.id,
                )
                await self.outbox.send(reply)
                return True

        coder = BaseAgent(ROLE_CODER, manager, agent_id="coder")
        planner = MockArbiter(ROLE_PLANNER, manager, agent_id="planner")

        # Coder 发送冲突升级
        conflict_msg = coder.outbox.create_message(
            recipient="planner",
            msg_type=MessageType.CONFLICT_ESCALATE,
            body={
                "issue": "Reviewer wants to remove try/except",
                "coder_claim": "HTTPException is required by FastAPI best practices",
                "reviewer_claim": "Avoid swallowing exceptions",
            },
            subject="Conflict: error handling pattern",
            correlation_id="corr_conflict_test",
        )
        from app.agent.mailbox import MessagePriority
        conflict_msg.priority = MessagePriority.URGENT

        await coder.outbox.send(conflict_msg)

        # Planner 收到冲突
        received = await planner.inbox.fetch_next(timeout=1.0)
        assert received is not None
        assert received.msg_type == MessageType.CONFLICT_ESCALATE
        assert received.priority == MessagePriority.URGENT

        # Planner 处理
        await planner.inbox.process(received)

        # Coder 应该收到仲裁决策
        reply = await coder.inbox.fetch_next(timeout=1.0)
        assert reply is not None
        assert reply.msg_type == MessageType.TASK_RESULT
        assert "decision" in (reply.body if isinstance(reply.body, dict) else {})

    @pytest.mark.asyncio
    async def test_planner_receives_task_assignment(self):
        """
        Planner 收到 TASK_ASSIGNMENT → 拆解 → TASK_RESULT。

        使用 mock Planner 避免真实 LLM 调用，验证消息流转。
        """
        manager = MailboxManager()

        # 自定义 Planner: override run() 避免真实 LLM 调用
        class MockPlanner(PlannerAgent):
            async def run(self, task: str) -> dict:
                return {
                    "summary": "Build REST API",
                    "steps": [
                        {"step": 1, "agent": "coder", "task": "Create api.py"},
                        {"step": 2, "agent": "tester", "task": "Test api.py"},
                    ],
                    "estimated_time": "3 min",
                    "agent": self.agent_id,
                }

        router = BaseAgent(ROLE_ROUTER, manager, agent_id="router")
        planner = MockPlanner(ROLE_PLANNER, manager, agent_id="planner")

        # Router → Planner
        task_msg = router.outbox.create_message(
            recipient="planner",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "Build a REST API for user management"},
            subject="New feature",
            correlation_id="corr_plan_1",
        )
        await router.outbox.send(task_msg)

        # Planner 收到
        received = await planner.inbox.fetch_next(timeout=1.0)
        assert received is not None
        assert received.msg_type == MessageType.TASK_ASSIGNMENT

        # 处理（默认 handler → 调用 run() → 发 TASK_RESULT）
        await planner.inbox.process(received)

        # Router 收到结果
        reply = await router.inbox.fetch_next(timeout=1.0)
        assert reply is not None
        assert reply.msg_type == MessageType.TASK_RESULT
        assert reply.correlation_id == "corr_plan_1"


# ═══════════════════════════════════════════════════════════════════════
# CoderAgent 测试
# ═══════════════════════════════════════════════════════════════════════

class TestCoderAgent:
    """CoderAgent —— 编码 + 反馈处理。"""

    def test_create_coder(self):
        """Coder 构造 + REVIEW_FEEDBACK handler 已注册。"""
        manager = MailboxManager()
        coder = CoderAgent(ROLE_CODER, manager, agent_id="coder-1")

        assert coder.agent_id == "coder-1"
        handlers = coder.inbox.stats["handlers_registered"]
        assert "review_feedback" in handlers

    @pytest.mark.asyncio
    async def test_coder_receives_task(self):
        """Coder 收到 TASK_ASSIGNMENT → 编码 → TASK_RESULT。"""
        manager = MailboxManager()

        # 自定义 Coder: override run() 避免真实 LLM 调用
        class MockCoder(CoderAgent):
            async def run(self, task: str) -> dict:
                return {
                    "files_created": ["health.py"],
                    "files_modified": [],
                    "summary": "Created /health endpoint",
                    "agent": self.agent_id,
                }

        router = BaseAgent(ROLE_ROUTER, manager, agent_id="router")
        coder = MockCoder(ROLE_CODER, manager, agent_id="coder")

        # Router → Coder
        task_msg = router.outbox.create_message(
            recipient="coder",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "Create a FastAPI health check endpoint"},
            subject="Coding task",
            correlation_id="corr_code_1",
        )
        await router.outbox.send(task_msg)

        # Coder 收到
        received = await coder.inbox.fetch_next(timeout=1.0)
        assert received is not None
        assert received.msg_type == MessageType.TASK_ASSIGNMENT

        # 处理
        await coder.inbox.process(received)

        # Router 收到 TASK_RESULT
        reply = await router.inbox.fetch_next(timeout=1.0)
        assert reply is not None
        assert reply.msg_type == MessageType.TASK_RESULT
        assert reply.correlation_id == "corr_code_1"

        # 验证 Coder 保存了任务上下文
        assert coder._last_task == "Create a FastAPI health check endpoint"

    @pytest.mark.asyncio
    async def test_coder_handles_review_feedback(self):
        """
        Coder 收到 REVIEW_FEEDBACK → 应用反馈 → TASK_RESULT 回 Reviewer。

        使用 mock Coder 覆盖 _handle_review_feedback 避免真实 LLM 调用。
        """
        manager = MailboxManager()

        # 自定义 Coder: override _handle_review_feedback 避免真实 LLM 调用
        class MockCoder(CoderAgent):
            async def _handle_review_feedback(self, msg: MailboxMessage) -> bool:
                reply = self.outbox.create_message(
                    recipient=msg.sender,
                    msg_type=MessageType.TASK_RESULT,
                    body={
                        "changes_made": ["Added input validation", "Added docstring"],
                        "issues_declined": [],
                        "summary": "Applied all feedback",
                    },
                    subject=f"Feedback applied: {msg.subject[:60]}",
                    correlation_id=msg.correlation_id or self._last_correlation_id,
                    reply_to=msg.id,
                )
                await self.outbox.send(reply)
                return True

        reviewer = BaseAgent(ROLE_REVIEWER, manager, agent_id="reviewer")
        coder = MockCoder(ROLE_CODER, manager, agent_id="coder")

        # 设置 Coder 的上次任务上下文
        coder._last_task = "Create FastAPI endpoint"
        coder._last_correlation_id = "corr_fb_1"

        # Reviewer → Coder (REVIEW_FEEDBACK)
        feedback_msg = reviewer.outbox.create_message(
            recipient="coder",
            msg_type=MessageType.REVIEW_FEEDBACK,
            body={
                "feedback": [
                    {"severity": "major", "description": "Missing input validation"},
                    {"severity": "minor", "description": "Add docstring"},
                ],
                "original_task": "Create FastAPI endpoint",
            },
            subject="Review: 2 issues found",
            correlation_id="corr_fb_1",
        )
        await reviewer.outbox.send(feedback_msg)

        # Coder 收到反馈
        received = await coder.inbox.fetch_next(timeout=1.0)
        assert received is not None
        assert received.msg_type == MessageType.REVIEW_FEEDBACK

        # 处理反馈
        await coder.inbox.process(received)

        # Reviewer 收到 Coder 的修改结果
        reply = await reviewer.inbox.fetch_next(timeout=1.0)
        assert reply is not None
        assert reply.msg_type == MessageType.TASK_RESULT


# ═══════════════════════════════════════════════════════════════════════
# TesterAgent 测试
# ═══════════════════════════════════════════════════════════════════════

class TestTesterAgent:
    """TesterAgent —— 测试执行 + 失败反馈。"""

    def test_create_tester(self):
        """Tester 构造。"""
        manager = MailboxManager()
        tester = TesterAgent(ROLE_TESTER, manager, agent_id="tester-1")

        assert tester.agent_id == "tester-1"
        assert tester.role == ROLE_TESTER

    @pytest.mark.asyncio
    async def test_tester_handles_task(self):
        """
        Tester 收到 TASK_ASSIGNMENT → 测试 → 发消息。

        使用 mock Tester 避免真实 LLM 调用。
        验证 Tester 的 _handle_task 覆盖：
          - 测试失败时向 sender 发送 TASK_RESULT
          - 额外向 coder 发送 REVIEW_FEEDBACK
        """
        manager = MailboxManager()

        # 自定义 Tester: override run() 返回失败结果
        class MockTester(TesterAgent):
            async def run(self, task: str) -> dict:
                return {
                    "passed": False,
                    "total_tests": 2,
                    "passed_count": 0,
                    "failed_count": 2,
                    "failures": [
                        {"test": "test_a", "error": "Error A",
                         "root_cause": "bug", "fix_suggestion": "fix A"},
                    ],
                    "summary": "Tests failed",
                    "agent": self.agent_id,
                }

        router = BaseAgent(ROLE_ROUTER, manager, agent_id="router")
        coder = BaseAgent(ROLE_CODER, manager, agent_id="coder")
        tester = MockTester(ROLE_TESTER, manager, agent_id="tester")

        # Router → Tester
        task_msg = router.outbox.create_message(
            recipient="tester",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "Run tests for user_api.py"},
            subject="Test task",
            correlation_id="corr_test_1",
        )
        await router.outbox.send(task_msg)

        # Tester 收到
        received = await tester.inbox.fetch_next(timeout=1.0)
        assert received is not None
        assert received.msg_type == MessageType.TASK_ASSIGNMENT

        # Tester 处理
        await tester.inbox.process(received)

        # Router 应该收到 TASK_RESULT（无论通过或失败）
        router_reply = await router.inbox.fetch_next(timeout=1.0)
        assert router_reply is not None
        assert router_reply.msg_type == MessageType.TASK_RESULT

    @pytest.mark.asyncio
    async def test_tester_sends_feedback_on_failure(self):
        """
        Tester 测试失败 → 发送 REVIEW_FEEDBACK 给 Coder。

        使用自定义 Tester 子类模拟测试失败场景。
        """
        manager = MailboxManager()

        coder = BaseAgent(ROLE_CODER, manager, agent_id="coder")
        router = BaseAgent(ROLE_ROUTER, manager, agent_id="router")

        # 自定义 Tester：总是返回失败结果
        class FailingTester(TesterAgent):
            async def run(self, task: str) -> dict:
                return {
                    "passed": False,
                    "total_tests": 3,
                    "passed_count": 1,
                    "failed_count": 2,
                    "failures": [
                        {
                            "test": "test_create_user",
                            "error": "AssertionError: expected 201 got 500",
                            "root_cause": "Missing database migration",
                            "fix_suggestion": "Run alembic upgrade head before test",
                        },
                        {
                            "test": "test_get_users",
                            "error": "KeyError: 'results'",
                            "root_cause": "Wrong response key name",
                            "fix_suggestion": "Change 'results' to 'data' in response",
                        },
                    ],
                    "summary": "2 out of 3 tests failed",
                }

        tester = FailingTester(ROLE_TESTER, manager, agent_id="tester")

        # Router → Tester
        task_msg = router.outbox.create_message(
            recipient="tester",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "Test user_api.py"},
            subject="Test after code change",
            correlation_id="corr_test_fail",
        )
        await router.outbox.send(task_msg)

        # Tester 处理
        received = await tester.inbox.fetch_next(timeout=1.0)
        await tester.inbox.process(received)

        # Coder 应该收到 REVIEW_FEEDBACK（含失败详情）
        feedback = await coder.inbox.fetch_next(timeout=1.0)
        assert feedback is not None
        assert feedback.msg_type == MessageType.REVIEW_FEEDBACK
        assert feedback.sender == "tester"
        assert feedback.recipient == "coder"

        fb_body = feedback.body if isinstance(feedback.body, dict) else {}
        assert fb_body.get("from_tester") is True
        fb_data = fb_body.get("feedback", {})
        assert fb_data.get("test_passed") is False
        assert len(fb_data.get("failures", [])) == 2

    @pytest.mark.asyncio
    async def test_tester_sends_only_result_on_pass(self):
        """
        Tester 测试全部通过 → 只发 TASK_RESULT，不发 REVIEW_FEEDBACK。

        使用自定义 Tester 子类模拟全部通过场景。
        """
        manager = MailboxManager()

        coder = BaseAgent(ROLE_CODER, manager, agent_id="coder")
        router = BaseAgent(ROLE_ROUTER, manager, agent_id="router")

        # 自定义 Tester：总是返回成功
        class PassingTester(TesterAgent):
            async def run(self, task: str) -> dict:
                return {
                    "passed": True,
                    "total_tests": 3,
                    "passed_count": 3,
                    "failed_count": 0,
                    "failures": [],
                    "summary": "All tests passed",
                }

        tester = PassingTester(ROLE_TESTER, manager, agent_id="tester")

        # Router → Tester
        task_msg = router.outbox.create_message(
            recipient="tester",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "Test user_api.py"},
            subject="Verify tests pass",
            correlation_id="corr_test_pass",
        )
        await router.outbox.send(task_msg)

        # Tester 处理
        received = await tester.inbox.fetch_next(timeout=1.0)
        await tester.inbox.process(received)

        # Router 收到 TASK_RESULT
        router_reply = await router.inbox.fetch_next(timeout=1.0)
        assert router_reply is not None
        assert router_reply.msg_type == MessageType.TASK_RESULT
        assert router_reply.body.get("passed") is True

        # Coder 不应收到任何消息（因为测试全通过）
        coder_msg = await coder.inbox.fetch_next(timeout=0.3)
        assert coder_msg is None


# ═══════════════════════════════════════════════════════════════════════
# ReviewerAgent 测试
# ═══════════════════════════════════════════════════════════════════════

class TestReviewerAgent:
    """ReviewerAgent —— 代码审查 + 反馈。"""

    def test_create_reviewer(self):
        """Reviewer 构造。"""
        manager = MailboxManager()
        reviewer = ReviewerAgent(ROLE_REVIEWER, manager, agent_id="reviewer-1")

        assert reviewer.agent_id == "reviewer-1"
        assert reviewer.role == ROLE_REVIEWER
        assert not reviewer.role.can_modify_files
        assert not reviewer.role.can_execute_shell

    @pytest.mark.asyncio
    async def test_reviewer_lgtm_sends_result(self):
        """
        Reviewer LGTM → 只发 TASK_RESULT 给 sender。

        使用自定义 Reviewer 子类模拟 LGTM 场景。
        """
        manager = MailboxManager()

        router = BaseAgent(ROLE_ROUTER, manager, agent_id="router")
        coder = BaseAgent(ROLE_CODER, manager, agent_id="coder")

        class LGTMReviewer(ReviewerAgent):
            async def run(self, task: str) -> dict:
                return {
                    "verdict": "LGTM",
                    "scores": {"correctness": 5, "security": 5, "performance": 4,
                               "maintainability": 5, "style_consistency": 5},
                    "issues": [],
                    "praise": ["Clean implementation", "Good test coverage"],
                    "summary": "Excellent work, no issues found.",
                }

        reviewer = LGTMReviewer(ROLE_REVIEWER, manager, agent_id="reviewer")

        # Router → Reviewer
        task_msg = router.outbox.create_message(
            recipient="reviewer",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "Review user_api.py diff"},
            subject="Code review request",
            correlation_id="corr_review_lgtm",
        )
        await router.outbox.send(task_msg)

        # Reviewer 处理
        received = await reviewer.inbox.fetch_next(timeout=1.0)
        await reviewer.inbox.process(received)

        # Router 收到 LGTM
        router_reply = await router.inbox.fetch_next(timeout=1.0)
        assert router_reply is not None
        assert router_reply.msg_type == MessageType.TASK_RESULT
        assert router_reply.body.get("verdict") == "LGTM"

        # Coder 不应收到 REVIEW_FEEDBACK（因为 LGTM）
        coder_msg = await coder.inbox.fetch_next(timeout=0.3)
        assert coder_msg is None

    @pytest.mark.asyncio
    async def test_reviewer_needs_work_sends_feedback(self):
        """
        Reviewer NEEDS_WORK → 发 REVIEW_FEEDBACK 给 Coder + TASK_RESULT 给 Router。

        使用自定义 Reviewer 子类模拟发现问题场景。
        """
        manager = MailboxManager()

        router = BaseAgent(ROLE_ROUTER, manager, agent_id="router")
        coder = BaseAgent(ROLE_CODER, manager, agent_id="coder")

        class StrictReviewer(ReviewerAgent):
            async def run(self, task: str) -> dict:
                return {
                    "verdict": "NEEDS_WORK",
                    "scores": {"correctness": 3, "security": 2, "performance": 4,
                               "maintainability": 3, "style_consistency": 4},
                    "issues": [
                        {
                            "severity": "critical",
                            "dimension": "security",
                            "file": "user_api.py",
                            "line": "42",
                            "description": "SQL injection risk — using f-strings in query",
                            "suggestion": "Use parameterized queries",
                            "must_fix": True,
                        },
                        {
                            "severity": "major",
                            "dimension": "correctness",
                            "file": "user_api.py",
                            "line": "18",
                            "description": "No input validation on POST /users",
                            "suggestion": "Add Pydantic model with validators",
                            "must_fix": True,
                        },
                        {
                            "severity": "minor",
                            "dimension": "style_consistency",
                            "file": "user_api.py",
                            "line": "5",
                            "description": "Import order doesn't follow isort convention",
                            "suggestion": "Run isort on the file",
                            "must_fix": False,
                        },
                    ],
                    "praise": ["Good API structure"],
                    "summary": "2 must-fix issues (1 critical, 1 major), 1 minor suggestion.",
                }

        reviewer = StrictReviewer(ROLE_REVIEWER, manager, agent_id="reviewer")

        # Router → Reviewer
        task_msg = router.outbox.create_message(
            recipient="reviewer",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "Review user_api.py"},
            subject="Code review",
            correlation_id="corr_review_nw",
        )
        await router.outbox.send(task_msg)

        # Reviewer 处理
        received = await reviewer.inbox.fetch_next(timeout=1.0)
        await reviewer.inbox.process(received)

        # Coder 应该收到 REVIEW_FEEDBACK
        feedback = await coder.inbox.fetch_next(timeout=1.0)
        assert feedback is not None
        assert feedback.msg_type == MessageType.REVIEW_FEEDBACK
        assert feedback.sender == "reviewer"
        assert feedback.recipient == "coder"

        fb_body = feedback.body if isinstance(feedback.body, dict) else {}
        fb_data = fb_body.get("feedback", {})
        assert fb_data.get("verdict") == "NEEDS_WORK"
        must_fix = fb_data.get("must_fix", [])
        assert len(must_fix) == 2

        # Router 也收到状态通知
        router_reply = await router.inbox.fetch_next(timeout=1.0)
        assert router_reply is not None
        assert router_reply.msg_type == MessageType.TASK_RESULT
        assert router_reply.body.get("verdict") == "NEEDS_WORK"


# ═══════════════════════════════════════════════════════════════════════
# 集成测试：完整 Multi-Agent 协作流程
# ═══════════════════════════════════════════════════════════════════════

class TestMultiAgentWorkflow:
    """
    Step 3 集成测试：多个 specialized Agent 通过 Mailbox 协作。

    完整流程：
      Router → Planner (TASK_ASSIGNMENT)
      Planner → Router (TASK_RESULT with plan)
      Router → Coder (TASK_ASSIGNMENT)
      Coder → Router (TASK_RESULT)
      Router → Reviewer (TASK_ASSIGNMENT)
      Reviewer → Coder (REVIEW_FEEDBACK) if issues found
      Coder → Reviewer (TASK_RESULT with fixes)
    """

    @pytest.mark.asyncio
    async def test_full_plan_code_review_cycle(self):
        """
        完整 Plan → Code → Review 循环。

        1. Router → Planner: 用户请求
        2. Planner → Router: 计划
        3. Router → Coder: 编码任务
        4. Coder → Router: 代码结果
        5. Router → Reviewer: 审查
        6. Reviewer → Coder: 反馈（如果有问题）
        """
        manager = MailboxManager()

        # 自定义 Planner: 返回固定计划
        class MockPlanner(PlannerAgent):
            async def run(self, task: str) -> dict:
                return {
                    "summary": "Build health check endpoint",
                    "steps": [
                        {"step": 1, "agent": "coder", "task": "Create health.py"},
                        {"step": 2, "agent": "reviewer", "task": "Review health.py"},
                    ],
                    "estimated_time": "2 min",
                }

        # 自定义 Coder: 返回固定代码
        class MockCoder(CoderAgent):
            async def run(self, task: str) -> dict:
                return {
                    "files_created": ["health.py"],
                    "files_modified": [],
                    "summary": "Created /health endpoint",
                }

        # 自定义 Reviewer: 返回 LGTM
        class MockReviewer(ReviewerAgent):
            async def run(self, task: str) -> dict:
                return {
                    "verdict": "LGTM",
                    "scores": {"correctness": 5, "security": 5, "performance": 5,
                               "maintainability": 5, "style_consistency": 5},
                    "issues": [],
                    "praise": ["Clean and simple"],
                    "summary": "Looks great!",
                }

        router = BaseAgent(ROLE_ROUTER, manager, agent_id="router")
        planner = MockPlanner(ROLE_PLANNER, manager, agent_id="planner")
        coder = MockCoder(ROLE_CODER, manager, agent_id="coder")
        reviewer = MockReviewer(ROLE_REVIEWER, manager, agent_id="reviewer")

        corr_id = "corr_e2e_1"

        # Step 1: Router → Planner
        plan_msg = router.outbox.create_message(
            recipient="planner",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "Add health check to the API"},
            subject="New feature",
            correlation_id=corr_id,
        )
        await router.outbox.send(plan_msg)

        # Step 2: Planner 处理 → Router 收到计划
        received = await planner.inbox.fetch_next(timeout=1.0)
        await planner.inbox.process(received)

        plan_reply = await router.inbox.fetch_next(timeout=1.0)
        assert plan_reply is not None
        assert plan_reply.msg_type == MessageType.TASK_RESULT
        assert len(plan_reply.body.get("steps", [])) == 2

        # Step 3: Router → Coder（分发第一个步骤）
        code_msg = router.outbox.create_message(
            recipient="coder",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "Create health.py"},
            subject="Coding step 1",
            correlation_id=f"{corr_id}_step1",
        )
        await router.outbox.send(code_msg)

        # Step 4: Coder 处理 → Router 收到代码结果
        received = await coder.inbox.fetch_next(timeout=1.0)
        await coder.inbox.process(received)

        code_reply = await router.inbox.fetch_next(timeout=1.0)
        assert code_reply is not None
        assert code_reply.msg_type == MessageType.TASK_RESULT
        assert "health.py" in str(code_reply.body.get("files_created", []))

        # Step 5: Router → Reviewer
        review_msg = router.outbox.create_message(
            recipient="reviewer",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "Review health.py"},
            subject="Review step 2",
            correlation_id=f"{corr_id}_step2",
        )
        await router.outbox.send(review_msg)

        # Step 6: Reviewer 处理 → Router 收到 LGTM
        received = await reviewer.inbox.fetch_next(timeout=1.0)
        await reviewer.inbox.process(received)

        review_reply = await router.inbox.fetch_next(timeout=1.0)
        assert review_reply is not None
        assert review_reply.msg_type == MessageType.TASK_RESULT
        assert review_reply.body.get("verdict") == "LGTM"

        # 审计日志完整性
        trail = manager.get_audit_trail(correlation_id=corr_id)
        assert len(trail) >= 2  # at minimum: plan message routed+delivered

    @pytest.mark.asyncio
    async def test_review_feedback_loop(self):
        """
        Reviewer 发现 → Coder 修复 → Reviewer 复查 循环。

        1. Reviewer 发现 must-fix 问题 → REVIEW_FEEDBACK → Coder
        2. Coder 修复 → TASK_RESULT → Reviewer
        3. Reviewer 复查 → LGTM
        """
        manager = MailboxManager()

        # 第一轮 Reviewer：发现问题
        class FirstPassReviewer(ReviewerAgent):
            async def run(self, task: str) -> dict:
                return {
                    "verdict": "NEEDS_WORK",
                    "scores": {"correctness": 3, "security": 5, "performance": 4,
                               "maintainability": 4, "style_consistency": 4},
                    "issues": [
                        {
                            "severity": "major",
                            "dimension": "correctness",
                            "description": "Missing error handling",
                            "suggestion": "Add try/except around DB call",
                            "must_fix": True,
                        },
                    ],
                    "praise": [],
                    "summary": "One must-fix issue.",
                }

        # Coder 修复: override both run() and _handle_review_feedback()
        class FixCoder(CoderAgent):
            async def run(self, task: str) -> dict:
                return {
                    "changes_made": ["Added try/except around DB call"],
                    "issues_declined": [],
                    "summary": "Fixed error handling as suggested",
                }

            async def _handle_review_feedback(self, msg: MailboxMessage) -> bool:
                reply = self.outbox.create_message(
                    recipient=msg.sender,
                    msg_type=MessageType.TASK_RESULT,
                    body={
                        "changes_made": ["Added try/except around DB call"],
                        "issues_declined": [],
                        "summary": "Fixed error handling as suggested",
                    },
                    subject=f"Fix applied: {msg.subject[:60]}",
                    correlation_id=msg.correlation_id or self._last_correlation_id,
                    reply_to=msg.id,
                )
                await self.outbox.send(reply)
                return True

        router = BaseAgent(ROLE_ROUTER, manager, agent_id="router")
        coder = FixCoder(ROLE_CODER, manager, agent_id="coder")
        reviewer = FirstPassReviewer(ROLE_REVIEWER, manager, agent_id="reviewer")

        corr_id = "corr_feedback_loop"

        # Round 1: Router → Reviewer
        review_msg = router.outbox.create_message(
            recipient="reviewer",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "Review user_api.py"},
            subject="Code review round 1",
            correlation_id=corr_id,
        )
        await router.outbox.send(review_msg)

        received = await reviewer.inbox.fetch_next(timeout=1.0)
        await reviewer.inbox.process(received)

        # Coder 收到 REVIEW_FEEDBACK
        feedback = await coder.inbox.fetch_next(timeout=1.0)
        assert feedback is not None
        assert feedback.msg_type == MessageType.REVIEW_FEEDBACK

        # Coder 处理反馈 → 回复 Reviewer
        await coder.inbox.process(feedback)

        # Reviewer 收到 Coder 的修复结果
        fix_reply = await reviewer.inbox.fetch_next(timeout=1.0)
        assert fix_reply is not None
        assert fix_reply.msg_type == MessageType.TASK_RESULT
        assert "try/except" in str(fix_reply.body.get("changes_made", []))

    @pytest.mark.asyncio
    async def test_all_five_agents_registered(self):
        """验证所有 5 个 Agent 类型都能成功注册到同一个 MailboxManager。"""
        manager = MailboxManager()

        router = BaseAgent(ROLE_ROUTER, manager, agent_id="router")
        planner = PlannerAgent(ROLE_PLANNER, manager, agent_id="planner")
        coder = CoderAgent(ROLE_CODER, manager, agent_id="coder")
        tester = TesterAgent(ROLE_TESTER, manager, agent_id="tester")
        reviewer = ReviewerAgent(ROLE_REVIEWER, manager, agent_id="reviewer")

        registered = manager.registered_agents
        assert set(registered) == {"router", "planner", "coder", "tester", "reviewer"}

        # 每个 Agent 都有正确的 handlers
        assert "conflict_escalate" in planner.inbox.stats["handlers_registered"]
        assert "review_feedback" in coder.inbox.stats["handlers_registered"]
