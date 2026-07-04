"""
Phase 7 — Multi-Agent API 集成测试

覆盖：
  - POST /api/multi-agent: 请求处理、响应结构、correlation_id
  - GET  /api/multi-agent/status: 状态监控
  - GET  /api/multi-agent/audit: 审计日志查询
  - 端到端：请求 → 审计链路可追溯
"""

from __future__ import annotations

import uuid
from typing import Optional

import pytest
from fastapi import APIRouter, Body, FastAPI, Query
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════
# Pydantic 模型（模块级定义，避免 Pydantic v2 ForwardRef 问题）
# ═══════════════════════════════════════════════════════════════════════

class MultiAgentRequest(BaseModel):
    message: str = Field(..., description="用户输入的任务描述")
    session_id: Optional[str] = Field(default=None)


class MultiAgentResponse(BaseModel):
    summary: str = ""
    success: bool = True
    correlation_id: str = ""
    backend: str = "multi-agent"
    steps: list = []
    session_id: str = ""


class MultiAgentStatusResponse(BaseModel):
    is_running: bool = False
    uptime_seconds: float = 0.0
    request_count: int = 0
    error_count: int = 0
    persistence_backend: str = "memory"
    agents: dict = {}
    mailbox: dict = {}
    dead_letter_count: int = 0
    audit_log_size: int = 0


# ═══════════════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════════════

# 模块级变量，在 _build_test_app 中设置，供端点闭包捕获
_test_orchestrator = None


def _make_mock_orch():
    """创建内含全部 mock agent 的 orchestrator（不调用真实 LLM）。"""
    from app.agent.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
    from app.agent.router import RouterAgent
    from app.agent.planner import PlannerAgent
    from app.agent.coder import CoderAgent
    from app.agent.tester import TesterAgent
    from app.agent.reviewer import ReviewerAgent
    from app.agent.roles import (
        ROLE_CODER, ROLE_PLANNER, ROLE_REVIEWER, ROLE_ROUTER, ROLE_TESTER,
    )

    class MockRouter(RouterAgent):
        async def _analyze_request(self, user_message: str) -> dict:
            return {
                "complexity": "simple", "reasoning": "Mock",
                "route": {"target": "coder", "task": user_message, "priority": "normal"},
            }

    class MockCoder(CoderAgent):
        async def run(self, task: str) -> dict:
            return {"files_created": ["out.py"], "summary": f"Done: {task[:40]}",
                    "agent": self.agent_id}

    class MockPlanner(PlannerAgent):
        async def run(self, task: str) -> dict:
            return {"summary": "Plan",
                    "steps": [{"step": 1, "agent": "coder", "task": task}],
                    "estimated_time": "1 min", "agent": self.agent_id}

    class MockTester(TesterAgent):
        async def run(self, task: str) -> dict:
            return {"passed": True, "total_tests": 1, "passed_count": 1,
                    "failed_count": 0, "failures": [], "summary": "OK",
                    "agent": self.agent_id}

    class MockReviewer(ReviewerAgent):
        async def run(self, task: str) -> dict:
            return {"verdict": "LGTM", "scores": {}, "issues": [],
                    "praise": [], "summary": "OK", "agent": self.agent_id}

    orch = MultiAgentOrchestrator(config=OrchestratorConfig(
        verbose=False, expire_sweep_interval=0, request_timeout=10.0,
    ))
    mgr = orch.mailbox
    orch.agents = {
        "router": MockRouter(ROLE_ROUTER, mgr, agent_id="router"),
        "planner": MockPlanner(ROLE_PLANNER, mgr, agent_id="planner"),
        "coder": MockCoder(ROLE_CODER, mgr, agent_id="coder"),
        "tester": MockTester(ROLE_TESTER, mgr, agent_id="tester"),
        "reviewer": MockReviewer(ROLE_REVIEWER, mgr, agent_id="reviewer"),
    }
    return orch


# ═══════════════════════════════════════════════════════════════════════
# 端点函数（模块级，避免 Pydantic 局部类问题）
# ═══════════════════════════════════════════════════════════════════════

def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


async def _endpoint_multi_agent(body: MultiAgentRequest = Body(...)):
    global _test_orchestrator
    session_id = body.session_id or _new_session_id()
    result = await _test_orchestrator.handle_user_request(body.message)
    return MultiAgentResponse(
        summary=result.get("summary", ""),
        success=result.get("success", True),
        correlation_id=result.get("correlation_id", ""),
        backend="multi-agent",
        steps=result.get("details", result.get("steps", [])),
        session_id=session_id,
    )


async def _endpoint_multi_agent_status():
    global _test_orchestrator
    stats = _test_orchestrator.get_stats()
    return MultiAgentStatusResponse(
        is_running=stats["system"]["is_running"],
        uptime_seconds=stats["system"]["uptime_seconds"],
        request_count=stats["system"]["request_count"],
        error_count=stats["system"]["error_count"],
        persistence_backend=_test_orchestrator.persistence_backend_name,
        agents=stats["agents"],
        mailbox=stats["mailbox"],
        dead_letter_count=stats["mailbox"].get("dead_letter_count", 0),
        audit_log_size=stats["audit_log_size"],
    )


async def _endpoint_multi_agent_audit(
    correlation_id: str = Query(default=""),
    agent_name: str = Query(default=""),
):
    global _test_orchestrator
    trail = _test_orchestrator.get_audit_trail(
        correlation_id=correlation_id,
        agent_name=agent_name,
    )
    return {
        "count": len(trail),
        "filters": {"correlation_id": correlation_id, "agent_name": agent_name},
        "trail": trail,
    }


def _build_test_app(orchestrator):
    """创建独立的 FastAPI app，只注册 Phase 7 multi-agent 端点。"""
    global _test_orchestrator
    _test_orchestrator = orchestrator

    router = APIRouter(prefix="/api", tags=["multi-agent"])
    router.post("/multi-agent", response_model=MultiAgentResponse)(_endpoint_multi_agent)
    router.get("/multi-agent/status", response_model=MultiAgentStatusResponse)(_endpoint_multi_agent_status)
    router.get("/multi-agent/audit")(_endpoint_multi_agent_audit)

    app = FastAPI(title="JoyAgent-Test", version="0.7.0")
    app.include_router(router)
    return app


# ═══════════════════════════════════════════════════════════════════════
# 测试类
# ═══════════════════════════════════════════════════════════════════════

class TestMultiAgentEndpoint:

    @pytest.mark.asyncio
    async def test_simple_request(self):
        orch = _make_mock_orch()
        await orch.start_all()
        app = _build_test_app(orch)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/multi-agent", json={
                "message": "Create a health check endpoint",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["backend"] == "multi-agent"
            assert data["correlation_id"].startswith("user_")

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_response_structure(self):
        orch = _make_mock_orch()
        await orch.start_all()
        app = _build_test_app(orch)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/multi-agent", json={
                "message": "Write a function",
            })
            data = resp.json()
            for key in ("summary", "success", "correlation_id", "backend",
                       "steps", "session_id"):
                assert key in data, f"Missing key: {key}"
            assert data["backend"] == "multi-agent"

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_multiple_requests_different_cids(self):
        orch = _make_mock_orch()
        await orch.start_all()
        app = _build_test_app(orch)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            results = []
            for msg in ("Task A", "Task B", "Task C"):
                resp = await client.post("/api/multi-agent", json={"message": msg})
                results.append(resp.json())

            cids = [r["correlation_id"] for r in results]
            assert len(set(cids)) == 3
            assert all(r["success"] for r in results)

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_session_id_passthrough(self):
        orch = _make_mock_orch()
        await orch.start_all()
        app = _build_test_app(orch)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/multi-agent", json={
                "message": "Test", "session_id": "my-custom-session",
            })
            assert resp.json()["session_id"] == "my-custom-session"

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_steps_contain_agent_info(self):
        orch = _make_mock_orch()
        await orch.start_all()
        app = _build_test_app(orch)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/multi-agent", json={
                "message": "Create a login page",
            })
            data = resp.json()
            if data["steps"]:
                assert "agent" in data["steps"][0]

        await orch.stop_all()


class TestMultiAgentStatusEndpoint:

    @pytest.mark.asyncio
    async def test_status_structure(self):
        orch = _make_mock_orch()
        await orch.start_all()
        app = _build_test_app(orch)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/multi-agent/status")
            assert resp.status_code == 200
            data = resp.json()
            for key in ("is_running", "uptime_seconds", "request_count",
                        "persistence_backend", "agents", "mailbox",
                        "dead_letter_count", "audit_log_size"):
                assert key in data, f"Missing key: {key}"

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_status_reflects_requests(self):
        orch = _make_mock_orch()
        await orch.start_all()
        app = _build_test_app(orch)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/api/multi-agent", json={"message": "Task 1"})
            resp = await client.get("/api/multi-agent/status")
            assert resp.json()["request_count"] >= 1

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_status_agents_have_stats(self):
        orch = _make_mock_orch()
        await orch.start_all()
        app = _build_test_app(orch)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/api/multi-agent", json={"message": "Some task"})
            resp = await client.get("/api/multi-agent/status")
            agents = resp.json()["agents"]
            for name in ("router", "coder", "planner"):
                assert name in agents
                assert "inbox" in agents[name]

        await orch.stop_all()


class TestMultiAgentAuditEndpoint:

    @pytest.mark.asyncio
    async def test_audit_by_correlation_id(self):
        orch = _make_mock_orch()
        await orch.start_all()
        app = _build_test_app(orch)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/multi-agent", json={
                "message": "Write hello world",
            })
            cid = resp.json()["correlation_id"]
            resp2 = await client.get(f"/api/multi-agent/audit?correlation_id={cid}")
            assert resp2.status_code == 200
            data = resp2.json()
            assert data["count"] >= 2  # routing + delivered

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_audit_by_agent_name(self):
        orch = _make_mock_orch()
        await orch.start_all()
        app = _build_test_app(orch)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/api/multi-agent", json={"message": "Audit task"})
            resp = await client.get("/api/multi-agent/audit?agent_name=router")
            assert resp.status_code == 200
            assert resp.json()["count"] >= 1

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_audit_empty_filters(self):
        orch = _make_mock_orch()
        await orch.start_all()
        app = _build_test_app(orch)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/multi-agent/audit")
            data = resp.json()
            assert data["filters"]["correlation_id"] == ""
            assert data["filters"]["agent_name"] == ""

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_audit_trail_entry_structure(self):
        orch = _make_mock_orch()
        await orch.start_all()
        app = _build_test_app(orch)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/api/multi-agent", json={"message": "Audit test"})
            resp = await client.get("/api/multi-agent/audit")
            data = resp.json()
            if data["trail"]:
                entry = data["trail"][0]
                for key in ("timestamp", "event", "msg_id", "sender",
                           "recipient", "msg_type", "priority"):
                    assert key in entry, f"Missing key: {key}"

        await orch.stop_all()


class TestApiIntegration:

    @pytest.mark.asyncio
    async def test_full_chain_request_to_audit(self):
        orch = _make_mock_orch()
        await orch.start_all()
        app = _build_test_app(orch)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/multi-agent", json={
                "message": "Create a REST API endpoint",
            })
            assert resp.status_code == 200
            result = resp.json()
            assert result["success"] is True
            cid = result["correlation_id"]

            resp2 = await client.get(f"/api/multi-agent/audit?correlation_id={cid}")
            data = resp2.json()
            assert data["count"] >= 2
            senders = {e["sender"] for e in data["trail"]}
            assert "router" in senders

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_status_after_multiple_requests(self):
        orch = _make_mock_orch()
        await orch.start_all()
        app = _build_test_app(orch)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp0 = await client.get("/api/multi-agent/status")
            initial = resp0.json()["request_count"]

            for i in range(3):
                await client.post("/api/multi-agent", json={"message": f"Req {i}"})

            resp1 = await client.get("/api/multi-agent/status")
            assert resp1.json()["request_count"] == initial + 3

        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_backend_is_multi_agent(self):
        orch = _make_mock_orch()
        await orch.start_all()
        app = _build_test_app(orch)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/multi-agent", json={
                "message": "Add logging",
            })
            assert resp.json()["backend"] == "multi-agent"

        await orch.stop_all()
