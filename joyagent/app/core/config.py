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

    # ── RAG 知识库 ──
    # 与 Agent 的长期记忆（code/conversation/task）物理隔离，
    # 独立使用 knowledge 集合，避免外部文档污染记忆语义空间。

    # 总开关：关闭后 /api/chat 的 use_rag 参数被忽略（运维降级用）
    RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"

    # 默认知识库集合名（预留多知识库能力，UI 首期只暴露布尔开关）
    RAG_COLLECTION = os.getenv("RAG_COLLECTION", "knowledge")

    # 切分参数（字符数，中文按字符计）
    RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))

    # 检索参数
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
    RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.3"))

    # 注入 prompt 的上下文总长度上限（保护 LLM 上下文窗口）
    RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))

    # 目录导入白名单（分号分隔的绝对路径前缀）。留空 = 不限制。
    # 例：RAG_ALLOWED_DIRS=/workspace/docs;/mnt/hgfs/joyagent
    RAG_ALLOWED_DIRS = [
        p.strip() for p in os.getenv("RAG_ALLOWED_DIRS", "").split(";") if p.strip()
    ]
