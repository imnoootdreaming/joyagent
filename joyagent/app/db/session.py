"""
Phase 9A-1 — 数据库连接管理

支持 SQLite（开发）和 PostgreSQL（生产），通过 DATABASE_URL 环境变量切换。

两种模式：
  SQLite:
    DATABASE_URL=sqlite:///./data/joyagent.db  （默认）
    零外部依赖，适合开发/单用户部署

  PostgreSQL:
    DATABASE_URL=postgresql://user:pass@localhost:5432/joyagent
    需要 psycopg2 或 asyncpg，适合生产/多用户

连接管理：
  - `get_engine()`:    创建 SQLAlchemy Engine（单例缓存）
  - `get_session()`:   context manager，自动 commit/rollback/close
  - `init_db()`:       启动时创建所有表
  - `dispose_engine()`: 关闭时释放连接池

使用方式：
  from app.db.session import get_session, init_db

  # startup 时
  init_db()

  # 业务代码中
  with get_session() as session:
      session.add(Session(...))
      session.commit()
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session

from app.core.config import Config

# ═══════════════════════════════════════════════════════════════════════
# 引擎管理（单例缓存）
# ═══════════════════════════════════════════════════════════════════════

_engine: Engine | None = None


def get_engine() -> Engine:
    """
    获取 SQLAlchemy Engine（单例缓存）。

    根据 Config.DATABASE_URL 决定后端：
      - sqlite:// → SQLite
      - postgresql:// → PostgreSQL
      - 未设置 → SQLite（默认，零配置）

    SQLite 额外参数：
      - check_same_thread=False: FastAPI 多线程环境下允许跨线程访问
      - connect_args: 仅 SQLite 需要
    """
    global _engine
    if _engine is not None:
        return _engine

    db_url = Config.DATABASE_URL

    if db_url.startswith("sqlite"):
        _engine = create_engine(
            db_url,
            echo=False,
            connect_args={"check_same_thread": False},
            # SQLite 不支持 pool_size，但 create_engine 会忽略
        )
    else:
        # PostgreSQL 或其他
        _engine = create_engine(
            db_url,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,     # 连接前验证（防止断线）
        )

    return _engine


def dispose_engine() -> None:
    """释放引擎连接池（shutdown 时调用）。"""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None


# ═══════════════════════════════════════════════════════════════════════
# Session 管理
# ═══════════════════════════════════════════════════════════════════════

@contextmanager
def get_session() -> Iterator[Session]:
    """
    SQLAlchemy Session context manager。

    自动 commit / rollback / close。用法：:

        with get_session() as session:
            result = session.execute(select(Session).where(...))
            return result.scalars().all()
    """
    engine = get_engine()
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ═══════════════════════════════════════════════════════════════════════
# 初始化
# ═══════════════════════════════════════════════════════════════════════

def init_db() -> None:
    """
    创建所有 SQLAlchemy 表（幂等——已存在的表不会重复创建）。

    在 FastAPI startup 时调用一次。
    """
    # 导入 Base + 所有模型以确保它们注册到 Base.metadata
    import app.db.base  # noqa: F401
    import app.db.models.session  # noqa: F401
    import app.db.models.task_log  # noqa: F401
    engine = get_engine()
    app.db.base.Base.metadata.create_all(engine)

    if Config.DATABASE_URL.startswith("sqlite"):
        db_path = Config.DATABASE_URL.replace("sqlite:///", "")
        print(f"  [db] SQLite initialized: {db_path}", flush=True)
    else:
        print(f"  [db] Database initialized", flush=True)
