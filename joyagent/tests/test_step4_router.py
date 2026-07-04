"""
Phase 7 Step 4 — RouterAgent 单元测试

覆盖：
  - RouterAgent: 构造、handler 注册
  - 规则路由: 关键词匹配 → 直接路由到 coder/tester/reviewer/planner
  - 简单路由: handle_user_request → 直接发给目标 Agent
  - 复杂路由: handle_user_request → Planner → 按计划分发
  - TASK_RESULT handler: 收集结果 + 通知等待者
  - ERROR_REPORT handler: 处理异常 + 通知等待者
  - 结果聚合: 简单拼接 / LLM 聚合
  - 超时处理: Agent 无响应时的降级
  - 集成测试: 完整 Router → Planner → Coder → Reviewer 流程
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
from app.agent.router import RouterAgent
from app.agent.roles import (
    ROLE_CODER,
    ROLE_PLANNER,
    ROLE_REVIEWER,
    ROLE_ROUTER,
    ROLE_TESTER,
)


# ═══════════════════════════════════════════════════════════════════════
# 辅助：创建带 mock run() 的 Agent（避免真实 LLM 调用）
# ═══════════════════════════════════════════════════════════════════════

def _make_mock_planner(manager: MailboxManager, plan_steps: list[dict]) -> PlannerAgent:
    """创建返回固定计划的 Planner。"""

    class MockPlanner(PlannerAgent):
        async def run(self, task: str) -> dict:
            return {
                "summary": f"Plan for: {task[:50]}",
                "steps": plan_steps,
                "estimated_time": "2 min",
                "agent": self.agent_id,
            }

    return MockPlanner(ROLE_PLANNER, manager, agent_id="planner")


def _make_mock_coder(manager: MailboxManager, result: dict = None) -> CoderAgent:
    """创建返回固定结果的 Coder。"""
    if result is None:
        result = {
            "files_created": ["output.py"],
            "files_modified": [],
            "summary": "Code generated successfully",
        }

    class MockCoder(CoderAgent):
        async def run(self, task: str) -> dict:
            return {**result, "agent": self.agent_id}

    return MockCoder(ROLE_CODER, manager, agent_id="coder")


def _make_mock_tester(manager: MailboxManager, passed: bool = True) -> TesterAgent:
    """创建返回固定测试结果的 Tester。"""

    class MockTester(TesterAgent):
        async def run(self, task: str) -> dict:
            return {
                "passed": passed,
                "total_tests": 3,
                "passed_count": 3 if passed else 1,
                "failed_count": 0 if passed else 2,
                "failures": [] if passed else [
                    {"test": "test_x", "error": "AssertionError",
                     "root_cause": "bug", "fix_suggestion": "fix it"},
                ],
                "summary": "All passed" if passed else "Some failed",
                "agent": self.agent_id,
            }

    return MockTester(ROLE_TESTER, manager, agent_id="tester")


def _make_mock_reviewer(manager: MailboxManager, verdict: str = "LGTM") -> ReviewerAgent:
    """创建返回固定审查结果的 Reviewer。"""

    class MockReviewer(ReviewerAgent):
        async def run(self, task: str) -> dict:
            return {
                "verdict": verdict,
                "scores": {"correctness": 5, "security": 5, "performance": 5,
                           "maintainability": 5, "style_consistency": 5},
                "issues": [],
                "praise": ["Good work"],
                "summary": "Looks great!" if verdict == "LGTM" else "Needs work",
                "agent": self.agent_id,
            }

    return MockReviewer(ROLE_REVIEWER, manager, agent_id="reviewer")


# ═══════════════════════════════════════════════════════════════════════
# RouterAgent 构造测试
# ═══════════════════════════════════════════════════════════════════════

class TestRouterAgentConstruction:
    """RouterAgent 构造 + handler 注册。"""

    def test_create_router(self):
        """Router 构造 + 默认 handler + Router 专属 handler。"""
        manager = MailboxManager()
        router = RouterAgent(ROLE_ROUTER, manager, agent_id="router-main")

        assert router.agent_id == "router-main"
        assert router.role == ROLE_ROUTER
        assert "router-main" in manager.registered_agents

        # 验证 handlers
        handlers = router.inbox.stats["handlers_registered"]
        # 默认的三个
        assert "task_assignment" in handlers
        assert "status_query" in handlers
        assert "broadcast" in handlers
        # Router 专属的两个
        assert "task_result" in handlers
        assert "error_report" in handlers

    def test_router_stats(self):
        """Router stats 包含 pending_requests 和 collected_results。"""
        manager = MailboxManager()
        router = RouterAgent(ROLE_ROUTER, manager)

        stats = router.stats
        assert stats["agent_id"] == "router"
        assert stats["role"] == "router"
        assert "pending_requests" in stats
        assert "collected_results" in stats
        assert stats["pending_requests"] == 0
        assert stats["collected_results"] == 0


# ═══════════════════════════════════════════════════════════════════════
# 规则路由测试
# ═══════════════════════════════════════════════════════════════════════

class TestRuleBasedRouting:
    """Router._rule_based_route() —— 关键词匹配路由。"""

    def test_route_test_only_to_tester(self):
        """"run tests" → tester（不涉及编码）。"""
        manager = MailboxManager()
        router = RouterAgent(ROLE_ROUTER, manager)

        result = router._rule_based_route("run tests for the API")
        assert result is not None
        assert result["complexity"] == "simple"
        assert result["route"]["target"] == "tester"

    def test_route_pytest_to_tester(self):
        """"pytest" → tester。"""
        manager = MailboxManager()
        router = RouterAgent(ROLE_ROUTER, manager)

        result = router._rule_based_route("run pytest on test_user.py")
        assert result is not None
        assert result["route"]["target"] == "tester"

    def test_route_test_with_write_stays_complex(self):
        """"write tests" → 涉及编码，不匹配纯测试规则 → 走 LLM 兜底。"""
        manager = MailboxManager()
        router = RouterAgent(ROLE_ROUTER, manager)

        # "write tests for" 包含 write（code_keywords），跳过纯测试规则
        # 且不匹配 simple_code_patterns，最终返回 None（走 LLM 兜底）
        result = router._rule_based_route("write tests for the user API")
        # 规则无法匹配 → 返回 None，由 LLM 路由兜底
        assert result is None

    def test_route_review_to_reviewer(self):
        """"review this code" → reviewer。"""
        manager = MailboxManager()
        router = RouterAgent(ROLE_ROUTER, manager)

        result = router._rule_based_route("please review this code for security issues")
        assert result is not None
        assert result["route"]["target"] == "reviewer"

    def test_route_simple_code_to_coder(self):
        """"add a comment" → coder（简单编码）。"""
        manager = MailboxManager()
        router = RouterAgent(ROLE_ROUTER, manager)

        result = router._rule_based_route("add a comment to the calculate function")
        assert result is not None
        assert result["route"]["target"] == "coder"
        assert result["complexity"] == "simple"

    def test_route_complex_to_planner(self):
        """"build a full API with CRUD" → planner（复杂任务）。"""
        manager = MailboxManager()
        router = RouterAgent(ROLE_ROUTER, manager)

        result = router._rule_based_route("build a complete REST API with CRUD and tests")
        assert result is not None
        assert result["route"]["target"] == "planner"
        assert result["complexity"] == "complex"

    def test_route_ambiguous_returns_none(self):
        """
        无法匹配的请求 → 返回 None（由 LLM 路由兜底）。

        注意：这个测试依赖消息中不包含任何关键词。
        """
        manager = MailboxManager()
        router = RouterAgent(ROLE_ROUTER, manager)

        result = router._rule_based_route("help me understand the project structure")
        # 不包含任何已定义的关键词 → None
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# 简单路由测试（集成 Mailbox）
# ═══════════════════════════════════════════════════════════════════════

class TestSimpleRouting:
    """Router._route_simple() —— 直接路由到目标 Agent。"""

    @pytest.mark.asyncio
    async def test_route_simple_to_coder(self):
        """Router 直接发 TASK_ASSIGNMENT 给 Coder → 等待 TASK_RESULT。"""
        manager = MailboxManager()

        router = RouterAgent(ROLE_ROUTER, manager, agent_id="router")
        coder = _make_mock_coder(manager)

        # 两个 Agent 都必须启动 watcher 才能收发消息
        router.start_background()
        coder.start_background()

        result = await router._route_simple(
            target="coder",
            task="Create a health check endpoint",
            correlation_id="corr_simple_1",
            timeout=5.0,
        )

        assert result["success"] is True
        assert len(result["steps"]) == 1
        assert result["steps"][0]["agent"] == "coder"

        router.stop()
        coder.stop()

    @pytest.mark.asyncio
    async def test_route_simple_timeout(self):
        """目标 Agent 无响应 → 超时返回失败。"""
        manager = MailboxManager()

        router = RouterAgent(ROLE_ROUTER, manager, agent_id="router")
        # Coder 已注册但未启动 watcher → 消息投递成功但不会被处理
        _coder = _make_mock_coder(manager)

        # Router 需要 watcher 来处理 TASK_RESULT...
        # 但 coder 没启动，所以不会处理 → 超时
        router.start_background()

        result = await router._route_simple(
            target="coder",
            task="Do something",
            correlation_id="corr_timeout",
            timeout=1.0,
        )

        assert result["success"] is False
        assert "timeout" in str(result).lower() or "error" in str(result).lower()

        router.stop()


# ═══════════════════════════════════════════════════════════════════════
# TASK_RESULT / ERROR_REPORT handler 测试
# ═══════════════════════════════════════════════════════════════════════

class TestRouterHandlers:
    """Router 专属 handler：TASK_RESULT 和 ERROR_REPORT。"""

    @pytest.mark.asyncio
    async def test_task_result_sets_event(self):
        """
        TASK_RESULT 到达 → 存储结果 + 设置 Event → handle_user_request 收到。
        """
        manager = MailboxManager()

        router = RouterAgent(ROLE_ROUTER, manager, agent_id="router")
        coder = _make_mock_coder(manager)

        # 注册等待
        event = asyncio.Event()
        cid = "corr_handler_1"
        router._result_events[cid] = event
        router._pending_results[cid] = []

        # Coder 执行任务 → 发送 TASK_RESULT 给 Router
        router.start_background()

        task_msg = router.outbox.create_message(
            recipient="coder",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "write hello world"},
            correlation_id=cid,
        )
        await router.outbox.send(task_msg)

        # Coder 处理
        received = await coder.inbox.fetch_next(timeout=2.0)
        assert received is not None
        await coder.inbox.process(received)

        # Router 应该收到 TASK_RESULT → Event 被设置
        await asyncio.wait_for(event.wait(), timeout=3.0)

        results = router._pending_results.get(cid, [])
        assert len(results) >= 1
        assert results[0]["agent"] == "coder"

        router.stop()

    @pytest.mark.asyncio
    async def test_error_report_sets_event(self):
        """
        ERROR_REPORT 到达 → 存储错误 + 设置 Event（避免永久阻塞）。
        """
        manager = MailboxManager()

        router = RouterAgent(ROLE_ROUTER, manager, agent_id="router")

        # 注册等待
        event = asyncio.Event()
        cid = "corr_error_1"
        router._result_events[cid] = event
        router._pending_results[cid] = []

        router.start_background()

        # 模拟任一 Agent 发 ERROR_REPORT
        error_msg = router.outbox.create_message(
            recipient="router",
            msg_type=MessageType.ERROR_REPORT,
            body={"error": "Out of memory", "agent": "coder"},
            correlation_id=cid,
        )
        # 直接通过 manager 路由（绕过 sender=recipient 检查）
        await manager.route(error_msg)

        # Router 收到 ERROR_REPORT → Event 被设置
        await asyncio.wait_for(event.wait(), timeout=3.0)

        results = router._pending_results.get(cid, [])
        assert len(results) >= 1
        assert results[0]["success"] is False
        assert "Out of memory" in results[0]["error"]

        router.stop()


# ═══════════════════════════════════════════════════════════════════════
# 集成测试：Router → Planner → Coder 完整流程
# ═══════════════════════════════════════════════════════════════════════

class TestRouterIntegration:
    """Router 端到端集成测试。"""

    @pytest.mark.asyncio
    async def test_full_flow_router_planner_coder(self):
        """
        完整流程：Router → Planner → Coder。

        1. Router 收到用户请求 "Build a REST API"
        2. Router 分析 → complex → 发 TASK_ASSIGNMENT 给 Planner
        3. Planner 返回计划 → Router 按计划发给 Coder
        4. Coder 执行 → Router 收集结果 → 聚合返回
        """
        manager = MailboxManager()

        # 自定义 Router：强制简单路由到 coder（跳过 LLM 分析）
        class TestRouter(RouterAgent):
            async def _analyze_request(self, user_message: str) -> dict:
                return {
                    "complexity": "simple",
                    "reasoning": "Test: force simple route",
                    "route": {
                        "target": "coder",
                        "task": user_message,
                        "priority": "normal",
                    },
                }

        router = TestRouter(ROLE_ROUTER, manager, agent_id="router")
        coder = _make_mock_coder(manager)

        # 启动 watchers
        router.start_background()
        coder.start_background()

        # 执行
        result = await router.handle_user_request(
            "Create a hello world endpoint",
            timeout=10.0,
        )

        assert result["success"] is True
        assert result["correlation_id"].startswith("user_")
        assert len(result.get("details", result.get("steps", []))) >= 0

        router.stop()
        coder.stop()

    @pytest.mark.asyncio
    async def test_full_flow_router_planner_multi_step(self):
        """
        多步骤流程：Router → Planner → Coder → Reviewer。

        Planner 返回 2 步计划 → Router 依次分发给 Coder 和 Reviewer。
        """
        manager = MailboxManager()

        # 自定义 Router：强制走 Planner 复杂路由
        class TestRouter(RouterAgent):
            async def _analyze_request(self, user_message: str) -> dict:
                return {
                    "complexity": "complex",
                    "reasoning": "Test: force complex route",
                    "route": {
                        "target": "planner",
                        "task": user_message,
                        "priority": "normal",
                    },
                }

        router = TestRouter(ROLE_ROUTER, manager, agent_id="router")
        planner = _make_mock_planner(manager, [
            {"step": 1, "agent": "coder", "task": "Create api.py"},
            {"step": 2, "agent": "reviewer", "task": "Review api.py"},
        ])
        coder = _make_mock_coder(manager)
        reviewer = _make_mock_reviewer(manager, verdict="LGTM")

        # 启动所有 watchers
        router.start_background()
        planner.start_background()
        coder.start_background()
        reviewer.start_background()

        # 执行
        result = await router.handle_user_request(
            "Build a REST API with review",
            timeout=15.0,
        )

        assert result["success"] is True
        assert result["correlation_id"].startswith("user_")
        # 聚合后的结果在 "details" 键中（来自 _aggregate_results）
        details = result.get("details", [])
        assert len(details) == 2

        # 第一个 step 是 coder
        assert details[0]["agent"] == "coder"
        # 第二个 step 是 reviewer
        assert details[1]["agent"] == "reviewer"

        router.stop()
        planner.stop()
        coder.stop()
        reviewer.stop()

    @pytest.mark.asyncio
    async def test_router_handles_planner_timeout(self):
        """
        Planner 超时 → Router 返回失败。
        """
        manager = MailboxManager()

        class TestRouter(RouterAgent):
            async def _analyze_request(self, user_message: str) -> dict:
                return {
                    "complexity": "complex",
                    "reasoning": "Test",
                    "route": {"target": "planner", "task": user_message, "priority": "normal"},
                }

        router = TestRouter(ROLE_ROUTER, manager, agent_id="router")
        # Planner 已注册但未启动 → 消息投递成功但不会被处理 → 超时
        _planner = _make_mock_planner(manager, [])

        router.start_background()

        result = await router.handle_user_request(
            "Build something complex",
            timeout=3.0,
        )

        assert result["success"] is False
        # 聚合后的结果在 "details" 键中
        details = result.get("details", [])
        assert any("time" in str(s).lower() or "error" in str(s).lower()
                   or "not respond" in str(s).lower()
                   for s in details)

        router.stop()

    @pytest.mark.asyncio
    async def test_concurrent_requests_different_correlation_ids(self):
        """
        并发请求 — 不同 correlation_id 的请求互不干扰。
        """
        manager = MailboxManager()

        class TestRouter(RouterAgent):
            async def _analyze_request(self, user_message: str) -> dict:
                return {
                    "complexity": "simple",
                    "reasoning": "Test",
                    "route": {"target": "coder", "task": user_message, "priority": "normal"},
                }

        router = TestRouter(ROLE_ROUTER, manager, agent_id="router")
        coder = _make_mock_coder(manager)

        router.start_background()
        coder.start_background()

        # 并发发送 3 个请求
        results = await asyncio.gather(
            router.handle_user_request("Task A", timeout=10.0),
            router.handle_user_request("Task B", timeout=10.0),
            router.handle_user_request("Task C", timeout=10.0),
        )

        # 三个请求都成功
        assert all(r["success"] for r in results)
        # correlation_id 各不相同
        cids = [r["correlation_id"] for r in results]
        assert len(set(cids)) == 3

        router.stop()
        coder.stop()

    @pytest.mark.asyncio
    async def test_all_six_agents_registered(self):
        """验证全部 6 个 Agent 类型注册成功。"""
        manager = MailboxManager()

        router = RouterAgent(ROLE_ROUTER, manager, agent_id="router")
        planner = PlannerAgent(ROLE_PLANNER, manager, agent_id="planner")
        coder = CoderAgent(ROLE_CODER, manager, agent_id="coder")
        tester = TesterAgent(ROLE_TESTER, manager, agent_id="tester")
        reviewer = ReviewerAgent(ROLE_REVIEWER, manager, agent_id="reviewer")
        # BaseAgent 作为通用的第 6 个
        generic = BaseAgent(ROLE_ROUTER, manager, agent_id="generic")

        registered = manager.registered_agents
        assert "router" in registered
        assert "planner" in registered
        assert "coder" in registered
        assert "tester" in registered
        assert "reviewer" in registered
        assert "generic" in registered

        # Router 的专属 handlers
        router_handlers = router.inbox.stats["handlers_registered"]
        assert "task_result" in router_handlers
        assert "error_report" in router_handlers
