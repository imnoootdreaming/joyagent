FROM python:3.11-slim

WORKDIR /app

# ── Node.js 20.x LTS（MCP Server 依赖 npx） ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml .

RUN uv pip install --system \
    "anthropic>=0.40.0" \
    "chromadb>=1.5.9" \
    "docker>=7.1.0" \
    "fastapi>=0.138.0" \
    "gitpython>=3.1.50" \
    "langgraph>=1.2.6" \
    "langgraph-checkpoint>=4.1.1" \
    "pathspec>=1.1.1" \
    "pytest>=9.1.1" \
    "python-dotenv>=1.2.2" \
    "python-multipart>=0.0.9" \
    "redis>=8.0.1" \
    "numpy>=1.26,<2.0" \
    "scikit-learn>=1.3,<1.6" \
    "sentence-transformers>=4.0,<5.0" \
    "tiktoken>=0.13.0" \
    "uvicorn>=0.49.0"

# 模型文件打入镜像，离线可用
COPY models/ /models/

# 告诉 sentence-transformers 用本地路径，不走 HuggingFace 网络
ENV EMBEDDING_MODEL=/models/all-MiniLM-L6-v2
ENV HF_HUB_OFFLINE=1

# Sandbox 宿主机挂载路径：JoyAgent 容器内 /app = 宿主机 /mnt/hgfs/joyagent
# DockerRunner 通过 docker.sock 在宿主机创建 sandbox 容器时，
# 需要用宿主机的真实路径做 bind mount
ENV SANDBOX_HOST_MOUNT_PATH=/mnt/hgfs/joyagent

COPY . .
# 清除构建时的 __pycache__ + 禁止 Python 写入新 .pyc（避免代码更新后被旧缓存覆盖）
RUN find /app -name '*.pyc' -delete && find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["sh", "-c", "find /app -name '*.pyc' -delete 2>/dev/null; find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; exec python -B main.py"]
