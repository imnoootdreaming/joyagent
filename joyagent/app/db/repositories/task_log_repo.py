"""
Phase 9A-1 — TaskLogRepository

TaskLog 的 CRUD 操作封装。用于记录 Agent 的每一次操作。
"""
from __future__ import annotations

from sqlalchemy import select, func
from sqlalchemy.orm import Session as DBSession

from app.db.models.task_log import TaskLog


class TaskLogRepository:
    """
    TaskLog 的数据访问层。

    使用方式：
      repo = TaskLogRepository(db_session)
      await repo.create(TaskLog(session_id="...", event_type="tool_call", ...))
      count = await repo.count_by_session("abc123")
    """

    def __init__(self, db: DBSession):
        self.db = db

    def create(self, log: TaskLog) -> TaskLog:
        """创建新 TaskLog。"""
        self.db.add(log)
        self.db.flush()
        return log

    def get(self, log_id: str) -> TaskLog | None:
        """按 ID 查询。"""
        return self.db.get(TaskLog, log_id)

    def list_by_session(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TaskLog]:
        """按 Session 查询所有操作日志（按时间升序）。"""
        stmt = (
            select(TaskLog)
            .where(TaskLog.session_id == session_id)
            .order_by(TaskLog.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.db.execute(stmt).scalars().all())

    def list_by_event_type(
        self,
        session_id: str,
        event_type: str,
        limit: int = 50,
    ) -> list[TaskLog]:
        """按事件类型过滤（如只看 tool_call 或 error）。"""
        stmt = (
            select(TaskLog)
            .where(
                TaskLog.session_id == session_id,
                TaskLog.event_type == event_type,
            )
            .order_by(TaskLog.created_at.asc())
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())

    def count_by_session(self, session_id: str) -> int:
        """统计某个 Session 的操作日志总数。"""
        stmt = (
            select(func.count())
            .select_from(TaskLog)
            .where(TaskLog.session_id == session_id)
        )
        return self.db.execute(stmt).scalar() or 0

    def count_tool_calls(self, session_id: str, success_only: bool = False) -> int:
        """统计工具调用次数（可选只看成功的）。"""
        stmt = (
            select(func.count())
            .select_from(TaskLog)
            .where(
                TaskLog.session_id == session_id,
                TaskLog.event_type == "tool_call",
            )
        )
        if success_only:
            stmt = stmt.where(TaskLog.tool_success == 1)
        return self.db.execute(stmt).scalar() or 0

    def delete_by_session(self, session_id: str) -> int:
        """删除某个 Session 的所有操作日志。返回删除数。"""
        from sqlalchemy import delete
        result = self.db.execute(
            delete(TaskLog).where(TaskLog.session_id == session_id)
        )
        self.db.flush()
        return result.rowcount
