import os
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)


class Config:
    """全局配置 —— Anthropic 原生 SDK 风格"""
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "DeepSeek-v4-pro[1m]")
    FALLBACK_MODEL = os.getenv("FALLBACK_MODEL")
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "30"))

    # ── Phase 9: 数据库 ──
    # SQLite（默认，零配置）: sqlite:///./data/joyagent.db
    # PostgreSQL（生产）:     postgresql://user:pass@localhost:5432/joyagent
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "sqlite:///./data/joyagent.db",
    )

    # Redis（Phase 9A-2 任务队列）
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
