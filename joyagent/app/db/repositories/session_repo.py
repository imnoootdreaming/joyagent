"""
Phase 9A-1 — SessionRepository

Session 的 CRUD 操作封装。调用方只关心业务逻辑，
不关心底层是 SQLite 还是 PostgreSQL。
"""
from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.orm import Session as DBSession

from app.db.models.session import Session


class SessionRepository:
    """
    Session 的数据访问层。

    使用方式：
      repo = SessionRepository(db_session)
      session = await repo.create(Session(user_id="user-1", title="Build API"))
      session = await repo.get("abc123")
      await repo.update_state("abc123", {"plan": [...]})
    """

    def __init__(self, db: DBSession):
        self.db = db

    def create(self, session: Session) -> Session:
        """创建新 Session。"""
        self.db.add(session)
        self.db.flush()  # 立即生成 id（不等待 commit）
        return session

    def get(self, session_id: str) -> Session | None:
        """按 ID 查询 Session。"""
        return self.db.get(Session, session_id)

    def update_state(self, session_id: str, agent_state: dict) -> bool:
        """
        更新 Agent 状态快照（同时更新 updated_at）。

        Returns:
            True 更新成功，False Session 不存在。
        """
        session = self.get(session_id)
        if session is None:
            return False
        session.agent_state = agent_state
        self.db.flush()
        return True

    def update_status(self, session_id: str, status: str) -> bool:
        """更新 Session 状态（active/completed/failed/archived）。"""
        session = self.get(session_id)
        if session is None:
            return False
        session.status = status
        self.db.flush()
        return True

    def increment_counters(
        self,
        session_id: str,
        messages: int = 0,
        tool_calls: int = 0,
        tokens: int = 0,
    ) -> bool:
        """
        原子更新统计计数（避免 get + set 的竞态条件）。

        SQLAlchemy 的 update 语句直接翻译为 UPDATE ... SET x = x + N，
        保证原子性。
        """
        result = self.db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(
                total_messages=Session.total_messages + messages,
                total_tool_calls=Session.total_tool_calls + tool_calls,
                total_tokens_used=Session.total_tokens_used + tokens,
            )
        )
        self.db.flush()
        return result.rowcount > 0

    def list_by_user(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Session]:
        """按用户查询 Session 列表（按更新时间降序）。"""
        stmt = (
            select(Session)
            .where(Session.user_id == user_id)
            .order_by(Session.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.db.execute(stmt).scalars().all())

    def list_active(self, limit: int = 50) -> list[Session]:
        """查询所有活跃 Session。"""
        stmt = (
            select(Session)
            .where(Session.status == "active")
            .order_by(Session.updated_at.desc())
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())

    def delete(self, session_id: str) -> bool:
        """删除 Session（物理删除，生产应改为软删除）。"""
        session = self.get(session_id)
        if session is None:
            return False
        self.db.delete(session)
        self.db.flush()
        return True
