"""
Phase 9A-1 — Repository Pattern 实现

Repository 模式：将数据访问逻辑从业务代码中抽离，提供类型安全的
CRUD 接口。Agent 不直接依赖 SQLAlchemy Session，而是通过 Repository 访问。

面试加分点：
  - Repository 模式让单元测试时可 mock 数据层
  - 未来从 SQLite 切到 PostgreSQL 只需换 engine，Repository 接口不变
"""

from app.db.repositories.session_repo import SessionRepository
from app.db.repositories.task_log_repo import TaskLogRepository

__all__ = ["SessionRepository", "TaskLogRepository"]
