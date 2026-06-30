FROM python:3.11-slim

WORKDIR /app

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
    "redis>=8.0.1" \
    "numpy>=1.26,<2.0" \
    "scikit-learn>=1.3,<1.6" \
    "sentence-transformers>=4.0,<5.0" \
    "tiktoken>=0.13.0" \
    "uvicorn>=0.49.0"

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
