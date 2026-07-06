"""
Phase 9A-1 — TaskLog 模型

Agent 每次操作的审计日志：
  - LLM 调用：记录模型、tokens、耗时
  - 工具调用：记录工具名、参数、结果、成功/失败
  - 错误：记录异常类型和消息
  - 用户审批：记录审批决策

设计意图：
  面试时可以说"所有 Agent 操作都可审计"——一条 SQL 查询就能
  回溯任意 Session 的完整操作历史。

与 Phase 7 Mailbox 审计日志的关系：
  Mailbox 的 get_audit_trail() 记录 Agent 间的消息路由（实时通信层），
  TaskLog 记录 Agent 内部的操作执行（业务层）。
  两层审计互补——面试时可以展示完整的可观测性设计。
"""
from __future__ import annotations

import datetime
import uuid

from sqlalchemy import Column, String, Text, DateTime, JSON, Integer

from app.db.base import Base


class TaskLog(Base):
    """任务日志 —— Agent 操作审计。"""

    __tablename__ = "task_logs"

    id = Column(String(36), primary_key=True,
                default=lambda: uuid.uuid4().hex[:12])
    session_id = Column(String(36), index=True)

    event_type = Column(String(50), default="")
    event_data = Column(JSON, nullable=True)

    tool_name = Column(String(100), nullable=True)
    tool_args = Column(JSON, nullable=True)
    tool_result = Column(Text, nullable=True)
    tool_success = Column(Integer, nullable=True)

    tokens_used = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<TaskLog id='{self.id}' session='{self.session_id}' "
            f"event='{self.event_type}' tool='{self.tool_name}'>"
        )
