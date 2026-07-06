"""
Phase 9A-1 — PostgreSQL / SQLite 数据层

SQLite（MVP，零配置）→ 生产 PostgreSQL（通过 DATABASE_URL 切换）
"""

from app.db.base import Base

__all__ = ["Base"]
