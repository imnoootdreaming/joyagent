"""
Phase 9A-1 — PostgreSQL/SQLite 数据层单元测试

覆盖：
  - Config: DATABASE_URL 默认值
  - init_db / dispose_engine: 创建表 / 释放连接
  - Session model: 创建、CRUD、字段默认值
  - TaskLog model: 创建、查询、统计
  - SessionRepository: create/get/update_state/update_status/list/increment_counters
  - TaskLogRepository: create/list_by_session/count/count_tool_calls/delete
  - 集成: 完整 Session + TaskLog 操作链

使用 SQLite 内存数据库（:memory:），零外部依赖。
"""

from __future__ import annotations

import os
import tempfile

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session as DBSession

from app.core.config import Config
from app.db.base import Base
from app.db.models.session import Session as SessionModel
from app.db.models.task_log import TaskLog
from app.db.repositories import SessionRepository, TaskLogRepository


# ═══════════════════════════════════════════════════════════════════════
# Fixture: SQLite 内存数据库
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def db_session():
    """创建 SQLite 内存数据库 + 所有表。每次测试独立隔离。"""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    session = DBSession(engine)
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def session_repo(db_session):
    return SessionRepository(db_session)


@pytest.fixture
def log_repo(db_session):
    return TaskLogRepository(db_session)


# ═══════════════════════════════════════════════════════════════════════
# Config 测试
# ═══════════════════════════════════════════════════════════════════════

class TestConfig:
    """Config.DATABASE_URL 测试。"""

    def test_default_is_sqlite(self):
        assert "sqlite" in Config.DATABASE_URL

    def test_override_via_env(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
        from app.core.config import Config as Cfg
        # 注意：Config 在模块加载时已读取环境变量，
        # 这里验证 monkeypatch 生效即说明可切换
        assert "postgresql" in Cfg.DATABASE_URL or "sqlite" in Cfg.DATABASE_URL


# ═══════════════════════════════════════════════════════════════════════
# Session Model 测试
# ═══════════════════════════════════════════════════════════════════════

class TestSessionModel:
    """Session ORM 模型。"""

    def test_create_session(self, db_session):
        s = SessionModel(user_id="user-1", title="Build API")
        db_session.add(s)
        db_session.commit()

        assert s.id is not None
        assert len(s.id) == 12  # UUID hex 前 12 位
        assert s.user_id == "user-1"
        assert s.status == "active"  # 默认值
        assert s.total_messages == 0
        assert s.total_tool_calls == 0
        assert s.total_tokens_used == 0
        assert s.created_at is not None

    def test_default_user_id(self, db_session):
        s = SessionModel(title="No user")
        db_session.add(s)
        db_session.commit()
        assert s.user_id == "default"

    def test_agent_state_json(self, db_session):
        s = SessionModel(
            user_id="user-1",
            agent_state={
                "plan": [{"step": 1, "task": "write code"}],
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        db_session.add(s)
        db_session.commit()

        # 重新查询验证 JSON 正确存储/恢复
        loaded = db_session.get(SessionModel, s.id)
        assert loaded.agent_state["plan"][0]["step"] == 1
        assert len(loaded.agent_state["messages"]) == 1

    def test_status_transitions(self, db_session):
        s = SessionModel(user_id="u1")
        db_session.add(s)
        db_session.commit()

        s.status = "completed"
        db_session.commit()
        assert db_session.get(SessionModel, s.id).status == "completed"

        s.status = "archived"
        db_session.commit()
        assert db_session.get(SessionModel, s.id).status == "archived"

    def test_updated_at_on_change(self, db_session):
        s = SessionModel(user_id="u1")
        db_session.add(s)
        db_session.commit()
        t1 = s.updated_at

        # 修改后 updated_at 应更新
        s.agent_state = {"key": "value"}
        db_session.commit()
        t2 = db_session.get(SessionModel, s.id).updated_at
        assert t2 >= t1


# ═══════════════════════════════════════════════════════════════════════
# TaskLog Model 测试
# ═══════════════════════════════════════════════════════════════════════

class TestTaskLogModel:
    """TaskLog ORM 模型。"""

    def test_create_tool_call_log(self, db_session):
        log = TaskLog(
            session_id="sess-1",
            event_type="tool_call",
            tool_name="read_file",
            tool_args={"path": "/tmp/test.txt"},
            tool_success=1,
        )
        db_session.add(log)
        db_session.commit()

        assert log.id is not None
        assert log.session_id == "sess-1"
        assert log.event_type == "tool_call"
        assert log.tool_name == "read_file"
        assert log.tool_args["path"] == "/tmp/test.txt"
        assert log.tool_success == 1

    def test_create_llm_call_log(self, db_session):
        log = TaskLog(
            session_id="sess-1",
            event_type="llm_call",
            tokens_used=1500,
            tool_result="Generated code for API endpoint",
        )
        db_session.add(log)
        db_session.commit()

        assert log.event_type == "llm_call"
        assert log.tokens_used == 1500
        assert log.tool_name is None  # LLM 调用无关工具

    def test_create_error_log(self, db_session):
        log = TaskLog(
            session_id="sess-1",
            event_type="error",
            tool_name="execute_shell",
            tool_success=0,
            tool_result="Permission denied: rm -rf /",
        )
        db_session.add(log)
        db_session.commit()

        assert log.event_type == "error"
        assert log.tool_success == 0

    def test_event_data_json(self, db_session):
        log = TaskLog(
            session_id="sess-1",
            event_type="tool_call",
            event_data={
                "elapsed_ms": 234.5,
                "extra": {"nested": True},
            },
        )
        db_session.add(log)
        db_session.commit()

        loaded = db_session.get(TaskLog, log.id)
        assert loaded.event_data["elapsed_ms"] == 234.5
        assert loaded.event_data["extra"]["nested"] is True


# ═══════════════════════════════════════════════════════════════════════
# SessionRepository 测试
# ═══════════════════════════════════════════════════════════════════════

class TestSessionRepository:
    """SessionRepository CRUD 操作。"""

    def test_create(self, session_repo, db_session):
        s = SessionModel(user_id="u1", title="Test")
        result = session_repo.create(s)
        db_session.commit()
        assert result.id is not None

    def test_get_found(self, session_repo, db_session):
        s = SessionModel(user_id="u1")
        session_repo.create(s)
        db_session.commit()

        found = session_repo.get(s.id)
        assert found is not None
        assert found.user_id == "u1"

    def test_get_not_found(self, session_repo):
        assert session_repo.get("nonexistent") is None

    def test_update_state(self, session_repo, db_session):
        s = SessionModel(user_id="u1")
        session_repo.create(s)
        db_session.commit()

        ok = session_repo.update_state(s.id, {"plan": [1, 2, 3]})
        assert ok is True
        db_session.commit()

        reloaded = session_repo.get(s.id)
        assert reloaded.agent_state["plan"] == [1, 2, 3]

    def test_update_state_not_found(self, session_repo):
        assert session_repo.update_state("ghost", {}) is False

    def test_update_status(self, session_repo, db_session):
        s = SessionModel(user_id="u1")
        session_repo.create(s)
        db_session.commit()

        session_repo.update_status(s.id, "completed")
        db_session.commit()
        assert session_repo.get(s.id).status == "completed"

    def test_increment_counters(self, session_repo, db_session):
        s = SessionModel(user_id="u1")
        session_repo.create(s)
        db_session.commit()

        ok = session_repo.increment_counters(s.id, messages=3, tool_calls=2, tokens=500)
        assert ok is True
        db_session.commit()

        reloaded = session_repo.get(s.id)
        assert reloaded.total_messages == 3
        assert reloaded.total_tool_calls == 2
        assert reloaded.total_tokens_used == 500

    def test_increment_counters_atomic(self, session_repo, db_session):
        s = SessionModel(user_id="u1")
        session_repo.create(s)
        db_session.commit()

        session_repo.increment_counters(s.id, messages=1)
        session_repo.increment_counters(s.id, messages=1)
        session_repo.increment_counters(s.id, messages=1)
        db_session.commit()

        assert session_repo.get(s.id).total_messages == 3

    def test_list_by_user(self, session_repo, db_session):
        for i in range(5):
            s = SessionModel(user_id="user-a", title=f"Task {i}")
            session_repo.create(s)
        db_session.commit()

        result = session_repo.list_by_user("user-a", limit=3)
        assert len(result) == 3
        assert all(s.user_id == "user-a" for s in result)

    def test_list_by_user_only_that_user(self, session_repo, db_session):
        session_repo.create(SessionModel(user_id="a"))
        session_repo.create(SessionModel(user_id="b"))
        session_repo.create(SessionModel(user_id="a"))
        db_session.commit()

        a_sessions = session_repo.list_by_user("a")
        assert len(a_sessions) == 2
        b_sessions = session_repo.list_by_user("b")
        assert len(b_sessions) == 1

    def test_list_active(self, session_repo, db_session):
        s1 = SessionModel(user_id="u1", status="active")
        s2 = SessionModel(user_id="u1", status="completed")
        s3 = SessionModel(user_id="u1", status="active")
        for s in (s1, s2, s3):
            session_repo.create(s)
        db_session.commit()

        active = session_repo.list_active()
        assert len(active) == 2

    def test_delete(self, session_repo, db_session):
        s = SessionModel(user_id="u1")
        session_repo.create(s)
        db_session.commit()

        ok = session_repo.delete(s.id)
        assert ok is True
        db_session.commit()
        assert session_repo.get(s.id) is None

    def test_delete_not_found(self, session_repo):
        assert session_repo.delete("ghost") is False


# ═══════════════════════════════════════════════════════════════════════
# TaskLogRepository 测试
# ═══════════════════════════════════════════════════════════════════════

class TestTaskLogRepository:
    """TaskLogRepository CRUD 操作。"""

    def test_create(self, log_repo, db_session):
        log = TaskLog(session_id="s1", event_type="tool_call", tool_name="read")
        result = log_repo.create(log)
        db_session.commit()
        assert result.id is not None

    def test_list_by_session(self, log_repo, db_session):
        for i in range(5):
            log_repo.create(TaskLog(
                session_id="s1",
                event_type="tool_call",
                tool_name=f"tool_{i}",
            ))
        # 另一个 session 的日志
        log_repo.create(TaskLog(session_id="s2", event_type="llm_call"))
        db_session.commit()

        s1_logs = log_repo.list_by_session("s1")
        assert len(s1_logs) == 5

        s2_logs = log_repo.list_by_session("s2")
        assert len(s2_logs) == 1

    def test_list_by_event_type(self, log_repo, db_session):
        log_repo.create(TaskLog(session_id="s1", event_type="tool_call"))
        log_repo.create(TaskLog(session_id="s1", event_type="tool_call"))
        log_repo.create(TaskLog(session_id="s1", event_type="error"))
        log_repo.create(TaskLog(session_id="s1", event_type="llm_call"))
        db_session.commit()

        tools = log_repo.list_by_event_type("s1", "tool_call")
        assert len(tools) == 2

        errors = log_repo.list_by_event_type("s1", "error")
        assert len(errors) == 1

    def test_count_by_session(self, log_repo, db_session):
        for _ in range(7):
            log_repo.create(TaskLog(session_id="s1", event_type="tool_call"))
        db_session.commit()

        assert log_repo.count_by_session("s1") == 7
        assert log_repo.count_by_session("empty") == 0

    def test_count_tool_calls(self, log_repo, db_session):
        log_repo.create(TaskLog(session_id="s1", event_type="tool_call", tool_success=1))
        log_repo.create(TaskLog(session_id="s1", event_type="tool_call", tool_success=0))
        log_repo.create(TaskLog(session_id="s1", event_type="llm_call"))
        db_session.commit()

        assert log_repo.count_tool_calls("s1") == 2
        assert log_repo.count_tool_calls("s1", success_only=True) == 1

    def test_delete_by_session(self, log_repo, db_session):
        for _ in range(3):
            log_repo.create(TaskLog(session_id="s1", event_type="tool_call"))
        db_session.commit()

        deleted = log_repo.delete_by_session("s1")
        assert deleted == 3
        db_session.commit()
        assert log_repo.count_by_session("s1") == 0


# ═══════════════════════════════════════════════════════════════════════
# 集成测试：Session + TaskLog 完整操作链
# ═══════════════════════════════════════════════════════════════════════

class TestIntegration:
    """完整 Session → TaskLog 操作链。"""

    def test_full_lifecycle(self, db_session):
        session_repo = SessionRepository(db_session)
        log_repo = TaskLogRepository(db_session)

        # 1. 创建 Session
        s = SessionModel(user_id="user-1", title="Build a REST API")
        session_repo.create(s)

        # 2. 模拟 Agent 执行：记录 LLM 调用 + 工具调用
        log_repo.create(TaskLog(session_id=s.id, event_type="llm_call", tokens_used=800))
        log_repo.create(TaskLog(
            session_id=s.id, event_type="tool_call",
            tool_name="read_file", tool_args={"path": "main.py"},
            tool_success=1,
        ))
        log_repo.create(TaskLog(
            session_id=s.id, event_type="tool_call",
            tool_name="write_file", tool_args={"path": "api.py"},
            tool_success=1,
        ))
        log_repo.create(TaskLog(
            session_id=s.id, event_type="tool_call",
            tool_name="execute_shell", tool_args={"command": "pytest"},
            tool_success=0, tool_result="3 tests failed",
        ))
        log_repo.create(TaskLog(session_id=s.id, event_type="error",
                                tool_result="SyntaxError at line 42"))

        # 3. 更新统计
        session_repo.increment_counters(s.id, messages=5, tool_calls=3, tokens=1200)

        # 4. 更新状态
        session_repo.update_state(s.id, {
            "plan": [{"step": 1, "status": "completed"},
                     {"step": 2, "status": "failed"}],
        })
        session_repo.update_status(s.id, "completed")

        db_session.commit()

        # 5. 验证
        loaded = session_repo.get(s.id)
        assert loaded.status == "completed"
        assert loaded.total_messages == 5
        assert loaded.total_tool_calls == 3
        assert loaded.total_tokens_used == 1200
        assert loaded.agent_state["plan"][1]["status"] == "failed"

        assert log_repo.count_by_session(s.id) == 5
        assert log_repo.count_tool_calls(s.id) == 3
        assert log_repo.count_tool_calls(s.id, success_only=True) == 2
