"""
Phase 7 Step 5 — MultiAgentOrchestrator 单元测试

覆盖：
  - OrchestratorConfig: 默认值、自定义配置
  - MultiAgentOrchestrator: 构造、注册、启动/停止、shutdown
  - handle_user_request: 简单请求、复杂请求、错误处理
  - 监控运维: get_stats, get_audit_trail, sweep_expired
  - 回调: on_request_start, on_request_end, on_error
  - 上下文管理器: async with
  - 集成测试: 完整端到端流程，审计日志可追溯
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent.base import BaseAgent
from app.agent.mailbox import (
    MailboxManager,
    MailboxMessage,
    MessageType,
)
from app.agent.orchestrator import (
    MultiAgentOrchestrator,
    OrchestratorConfig,
    _get_agent_class_map,
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
# 辅助：注册 mock agent（替换 mailbox 中原有注册）
# ═══════════════════════════════════════════════════════════════════════

def _register_mock_agents(orch: MultiAgentOrchestrator) -> None:
    """
    将全套 mock agent 注册到 orchestrator。

    替换 mailbox 中已有的 agent 注册（避免 duplicate），
    并设置 orch.agents。

    所有 mock agent 都覆盖了 run() / _analyze_request()，
    不会调用真实的 LLM。
    """
    manager = orch.mailbox

    class MockRouter(RouterAgent):
        async def _analyze_request(self, user_message: str) -> dict:
            return {
                "complexity": "simple",
                "reasoning": "Mock: force simple route",
                "route": {"target": "coder", "task": user_message, "priority": "normal"},
            }

    class MockPlanner(PlannerAgent):
        async def run(self, task: str) -> dict:
            return {
                "summary": f"Plan for: {task[:50]}",
                "steps": [{"step": 1, "agent": "coder", "task": task}],
                "estimated_time": "1 min",
                "agent": self.agent_id,
            }

    class MockCoder(CoderAgent):
        async def run(self, task: str) -> dict:
            return {
                "files_created": ["output.py"],
                "summary": "Code generated",
                "agent": self.agent_id,
            }

    class MockTester(TesterAgent):
        async def run(self, task: str) -> dict:
            return {
                "passed": True,
                "total_tests": 1, "passed_count": 1, "failed_count": 0,
                "failures": [],
                "summary": "All passed",
                "agent": self.agent_id,
            }

    class MockReviewer(ReviewerAgent):
        async def run(self, task: str) -> dict:
            return {
                "verdict": "LGTM",
                "scores": {"correctness": 5, "security": 5, "performance": 5,
                           "maintainability": 5, "style_consistency": 5},
                "issues": [],
                "praise": ["Good"],
                "summary": "Looks great",
                "agent": self.agent_id,
            }

    orch.agents = {
        "router": MockRouter(ROLE_ROUTER, manager, agent_id="router"),
        "planner": MockPlanner(ROLE_PLANNER, manager, agent_id="planner"),
        "coder": MockCoder(ROLE_CODER, manager, agent_id="coder"),
        "tester": MockTester(ROLE_TESTER, manager, agent_id="tester"),
        "reviewer": MockReviewer(ROLE_REVIEWER, manager, agent_id="reviewer"),
    }


# ═══════════════════════════════════════════════════════════════════════
# OrchestratorConfig 测试
# ═══════════════════════════════════════════════════════════════════════

class TestOrchestratorConfig:
    """OrchestratorConfig 数据类测试。"""

    def test_default_config(self):
        cfg = OrchestratorConfig()
        assert cfg.request_timeout == 300.0
        assert cfg.expire_sweep_interval == 60.0
        assert cfg.verbose is True

    def test_custom_config(self):
        cfg = OrchestratorConfig(
            request_timeout=60.0,
            expire_sweep_interval=0,
            verbose=False,
        )
        assert cfg.request_timeout == 60.0
        assert cfg.expire_sweep_interval == 0
        assert cfg.verbose is False


# ═══════════════════════════════════════════════════════════════════════
# Agent 工厂映射测试
# ═══════════════════════════════════════════════════════════════════════

class TestAgentClassMap:
    """_get_agent_class_map() 测试。"""

    def test_all_five_classes_present(self):
        class_map = _get_agent_class_map()
        assert set(class_map.keys()) == {"router", "planner", "coder", "tester", "reviewer"}
        assert class_map["router"] is RouterAgent
        assert class_map["planner"] is PlannerAgent


# ═══════════════════════════════════════════════════════════════════════
# MultiAgentOrchestrator 构造测试
# ═══════════════════════════════════════════════════════════════════════

class TestOrchestratorConstruction:
    """Orchestrator 构造 + 默认状态。"""

    def test_create_orchestrator(self):
        orch = MultiAgentOrchestrator()
        assert orch.mailbox is not None
        assert isinstance(orch.mailbox, MailboxManager)
        assert len(orch.agents) == 0
        assert orch.is_running is False
        assert orch._request_count == 0

    def test_create_with_config(self):
        cfg = OrchestratorConfig(request_timeout=90.0, verbose=False)
        orch = MultiAgentOrchestrator(config=cfg)
        assert orch.config.request_timeout == 90.0

    def test_register_all(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(verbose=False))
        orch.register_all()

        assert len(orch.agents) == 5
        assert set(orch.agents.keys()) == {"router", "planner", "coder", "tester", "reviewer"}
        registered = orch.mailbox.registered_agents
        assert len(registered) == 5
        for name, agent in orch.agents.items():
            assert agent.inbox is not None
            assert agent.outbox is not None
            assert agent.inbox.owner == name

    def test_register_manual_agent(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(verbose=False))
        custom = BaseAgent(ROLE_CODER, orch.mailbox, agent_id="custom-agent")
        orch.register_agent("custom-agent", custom)
        assert "custom-agent" in orch.agents
        assert "custom-agent" in orch.mailbox.registered_agents


# ═══════════════════════════════════════════════════════════════════════
# 生命周期测试
# ═══════════════════════════════════════════════════════════════════════

class TestOrchestratorLifecycle:
    """Orchestrator 生命周期：启动 / 停止 / 关闭。"""

    @pytest.mark.asyncio
    async def test_start_all_and_stop_all(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0,
        ))
        orch.register_all()

        await orch.start_all()
        # 给 task 一点时间让事件循环调度
        await asyncio.sleep(0.01)

        assert orch.is_running is True
        assert orch.uptime_seconds >= 0
        for agent in orch.agents.values():
            assert agent.is_watching is True

        await orch.stop_all()
        assert orch.is_running is False
        for agent in orch.agents.values():
            assert agent.is_watching is False

    @pytest.mark.asyncio
    async def test_start_all_is_idempotent(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0,
        ))
        orch.register_all()

        await orch.start_all()
        await asyncio.sleep(0.01)
        task_count_1 = len(orch._watcher_tasks)
        assert task_count_1 == 5

        await orch.start_all()
        task_count_2 = len(orch._watcher_tasks)
        assert task_count_2 == 5

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_shutdown_clears_all(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0,
        ))
        orch.register_all()
        await orch.start_all()

        await orch.shutdown()

        assert orch.is_running is False
        assert len(orch.mailbox.registered_agents) == 0
        assert len(orch.agents) == 0

    @pytest.mark.asyncio
    async def test_auto_start_on_request(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0, request_timeout=10.0,
        ))
        # 不调用 register_all()，直接用 mock
        _register_mock_agents(orch)
        # 也不调用 start_all()

        result = await orch.handle_user_request("Hello world")
        assert "success" in result
        assert orch._request_count == 1

        await orch.stop_all()


# ═══════════════════════════════════════════════════════════════════════
# handle_user_request 测试
# ═══════════════════════════════════════════════════════════════════════

class TestHandleUserRequest:
    """Orchestrator.handle_user_request() 端到端测试。"""

    @pytest.mark.asyncio
    async def test_simple_request(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0, request_timeout=10.0,
        ))
        _register_mock_agents(orch)
        await orch.start_all()

        result = await orch.handle_user_request("Create a hello world function")
        assert result["success"] is True
        assert "correlation_id" in result

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_request_increments_counter(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0, request_timeout=10.0,
        ))
        _register_mock_agents(orch)
        await orch.start_all()

        for i in range(3):
            result = await orch.handle_user_request(f"Task {i}")
            assert result["success"] is True
        assert orch._request_count == 3

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_request_without_router_returns_error(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(verbose=False))
        result = await orch.handle_user_request("Do something")
        assert result["success"] is False
        assert "no_router" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_multi_step_via_planner(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0, request_timeout=15.0,
        ))

        manager = orch.mailbox

        class ComplexRouter(RouterAgent):
            async def _analyze_request(self, user_message: str) -> dict:
                return {
                    "complexity": "complex",
                    "reasoning": "Test",
                    "route": {"target": "planner", "task": user_message, "priority": "normal"},
                }

        class MockPlanner(PlannerAgent):
            async def run(self, task: str) -> dict:
                return {
                    "summary": "Multi-step plan",
                    "steps": [
                        {"step": 1, "agent": "coder", "task": "Write code"},
                        {"step": 2, "agent": "reviewer", "task": "Review code"},
                    ],
                    "estimated_time": "2 min",
                    "agent": self.agent_id,
                }

        class MockCoder(CoderAgent):
            async def run(self, task: str) -> dict:
                return {"files_created": ["api.py"], "summary": "Done", "agent": self.agent_id}

        class MockReviewer(ReviewerAgent):
            async def run(self, task: str) -> dict:
                return {"verdict": "LGTM", "scores": {}, "issues": [],
                        "praise": [], "summary": "Approved", "agent": self.agent_id}

        orch.agents = {
            "router": ComplexRouter(ROLE_ROUTER, manager, agent_id="router"),
            "planner": MockPlanner(ROLE_PLANNER, manager, agent_id="planner"),
            "coder": MockCoder(ROLE_CODER, manager, agent_id="coder"),
            "reviewer": MockReviewer(ROLE_REVIEWER, manager, agent_id="reviewer"),
            "tester": TesterAgent(ROLE_TESTER, manager, agent_id="tester"),
        }
        await orch.start_all()

        result = await orch.handle_user_request("Build a full REST API with review")
        assert result["success"] is True
        details = result.get("details", [])
        assert len(details) == 2
        assert details[0]["agent"] == "coder"
        assert details[1]["agent"] == "reviewer"

        await orch.stop_all()


# ═══════════════════════════════════════════════════════════════════════
# 监控与运维测试
# ═══════════════════════════════════════════════════════════════════════

class TestOrchestratorMonitoring:
    """Orchestrator 监控运维：stats、审计日志、过期清理。"""

    @pytest.mark.asyncio
    async def test_get_stats(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(verbose=False))
        orch.register_all()

        stats = orch.get_stats()
        assert "system" in stats
        assert "mailbox" in stats
        assert "agents" in stats
        assert stats["system"]["request_count"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_after_request(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0, request_timeout=10.0,
        ))
        _register_mock_agents(orch)
        await orch.start_all()

        result = await orch.handle_user_request("Task X")
        assert result["success"] is True

        stats = orch.get_stats()
        assert stats["system"]["request_count"] == 1

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_audit_trail_traceable(self):
        """
        完整请求链路可通过审计日志追溯 —— 按 agent_name 过滤。
        """
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0, request_timeout=10.0,
        ))
        _register_mock_agents(orch)
        await orch.start_all()

        result = await orch.handle_user_request("Create a function")
        cid = result.get("correlation_id", "")
        assert cid.startswith("user_")

        # 按 agent_name 追溯 router 和 coder 的活动
        router_trail = orch.get_audit_trail(agent_name="router")
        assert len(router_trail) >= 2  # routing + delivered for task + routing + delivered for reply

        coder_trail = orch.get_audit_trail(agent_name="coder")
        assert len(coder_trail) >= 2

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_sweep_expired(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(verbose=False))
        orch.register_all()
        swept = orch.sweep_expired()
        assert swept == 0

    @pytest.mark.asyncio
    async def test_get_dead_letter_count(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(verbose=False))
        orch.register_all()
        count = orch.get_dead_letter_count()
        assert count == 0


# ═══════════════════════════════════════════════════════════════════════
# 回调测试
# ═══════════════════════════════════════════════════════════════════════

class TestOrchestratorCallbacks:
    """Orchestrator 事件回调。"""

    @pytest.mark.asyncio
    async def test_on_request_start_callback(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0, request_timeout=10.0,
        ))
        _register_mock_agents(orch)
        await orch.start_all()

        started_msgs = []
        orch.on_request_start(lambda msg: started_msgs.append(msg))

        await orch.handle_user_request("Callback test")
        assert len(started_msgs) == 1
        assert started_msgs[0] == "Callback test"

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_on_request_end_callback(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0, request_timeout=10.0,
        ))
        _register_mock_agents(orch)
        await orch.start_all()

        ended = []
        orch.on_request_end(lambda msg, res: ended.append((msg, res["success"])))

        await orch.handle_user_request("End callback test")
        assert len(ended) == 1
        assert ended[0][0] == "End callback test"
        assert ended[0][1] is True

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_on_error_callback(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(verbose=False))

        errors = []
        orch.on_error(lambda msg, err: errors.append((msg, type(err).__name__)))

        # 手动触发错误回调
        try:
            raise RuntimeError("test error")
        except RuntimeError as e:
            for cb in orch._on_error:
                cb("test msg", e)
        assert len(errors) == 1
        assert errors[0][0] == "test msg"
        assert errors[0][1] == "RuntimeError"


# ═══════════════════════════════════════════════════════════════════════
# 上下文管理器测试
# ═══════════════════════════════════════════════════════════════════════

class TestOrchestratorContextManager:
    """async with MultiAgentOrchestrator() 测试。"""

    @pytest.mark.asyncio
    async def test_context_manager_auto_register_and_start(self):
        async with MultiAgentOrchestrator(
            config=OrchestratorConfig(verbose=False, expire_sweep_interval=0)
        ) as orch:
            assert len(orch.agents) == 5
            assert orch.is_running is True
            await asyncio.sleep(0.01)
            for agent in orch.agents.values():
                assert agent.is_watching is True

        # 退出后已停止
        assert orch.is_running is False
        assert len(orch.agents) == 0


# ═══════════════════════════════════════════════════════════════════════
# 集成测试：完整端到端流程
# ═══════════════════════════════════════════════════════════════════════

class TestOrchestratorIntegration:
    """Orchestrator 端到端集成测试。"""

    @pytest.mark.asyncio
    async def test_full_e2e_with_mock_agents(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0, request_timeout=15.0,
        ))
        _register_mock_agents(orch)
        await orch.start_all()

        tasks = [
            orch.handle_user_request("Create a login endpoint"),
            orch.handle_user_request("Add logging to the API"),
            orch.handle_user_request("Write unit tests for utils.py"),
        ]
        results = await asyncio.gather(*tasks)

        for r in results:
            assert r["success"] is True
            assert "correlation_id" in r

        cids = [r["correlation_id"] for r in results]
        assert len(set(cids)) == 3

        stats = orch.get_stats()
        assert stats["system"]["request_count"] == 3

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_audit_log_chain_complete(self):
        """
        审计日志可追溯完整链路。

        验证一个简单的 router → coder 链路中：
          router 和 coder 都有审计条目
        """
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0, request_timeout=10.0,
        ))
        _register_mock_agents(orch)
        await orch.start_all()

        result = await orch.handle_user_request("Write a quick sort function")
        assert result["success"] is True

        # 按 agent_name 追溯 router
        router_trail = orch.get_audit_trail(agent_name="router")
        assert len(router_trail) >= 2

        # 按 agent_name 追溯 coder
        coder_trail = orch.get_audit_trail(agent_name="coder")
        assert len(coder_trail) >= 2

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_concurrent_requests_isolation(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0, request_timeout=15.0,
        ))
        _register_mock_agents(orch)
        await orch.start_all()

        results = await asyncio.gather(*[
            orch.handle_user_request(f"Task {i}") for i in range(5)
        ])
        assert all(r["success"] for r in results)
        cids = [r["correlation_id"] for r in results]
        assert len(set(cids)) == 5
        assert orch._request_count == 5

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_expire_sweeper_runs(self):
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=1.0,
        ))
        orch.register_all()
        await orch.start_all()
        assert orch._sweeper_task is not None

        await asyncio.sleep(2.1)

        assert not orch._sweeper_task.done()

        await orch.stop_all()
        assert orch._sweeper_task is None
