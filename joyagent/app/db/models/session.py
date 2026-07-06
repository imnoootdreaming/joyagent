"""
Phase 9A-1 — Session 模型

每个用户的每个对话是一个 Session：
  - user_id:  用户标识（多用户隔离）
  - agent_state: Agent 完整状态（JSON），重启后可恢复
  - 统计信息：消息数、工具调用数、token 消耗

设计意图：
  前 Phase 6/7 的 MemoryManager 和 Mailbox 都在内存中，
  服务重启后状态丢失。此模型将关键状态持久化到数据库，
  支持 Session 恢复和多用户并发。

与 Phase 7 Mailbox 的关系：
  Mailbox 负责 Agent 间实时通信（内存级，低延迟），
  Session 负责跨重启持久化（数据库级，可恢复）。
  两者互补，不冲突。
"""
from __future__ import annotations

import datetime
import uuid

from sqlalchemy import Column, String, DateTime, JSON, Integer

from app.db.base import Base


class Session(Base):
    """
    用户会话 —— Agent 对话的持久化容器。
    """

    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True,
                default=lambda: uuid.uuid4().hex[:12])
    user_id = Column(String(36), index=True, default="default")
    title = Column(String(255), default="")
    status = Column(String(20), default="active")

    agent_state = Column(JSON, nullable=True)

    total_messages = Column(Integer, default=0)
    total_tool_calls = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow,
                        onupdate=datetime.datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<Session id='{self.id}' user='{self.user_id}' "
            f"status='{self.status}' msgs={self.total_messages}>"
        )
