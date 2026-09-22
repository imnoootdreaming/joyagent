"""
RAG 知识库 REST 接口 —— 挂载在 /api/rag/* 下。

端点一览：
  POST   /api/rag/ingest          文件上传入库（multipart，支持多选）
  POST   /api/rag/ingest/dir      本地目录批量导入
  POST   /api/rag/search          检索测试（不进 Agent 链路，直接返回命中）
  GET    /api/rag/collections     列出所有知识库集合及其规模
  GET    /api/rag/documents       列出知识库内的文档来源
  DELETE /api/rag/documents       按来源删除文档
  GET    /api/rag/health          健康检查 + 统计

设计约定：
  1. 所有响应都带 ok / error 字段，失败时 HTTP 4xx/5xx 且 error 有可读信息
  2. Pydantic 模型风格与 app/api/agent.py 保持一致（BaseModel + Field）
  3. 不抛裸异常 —— 入库/检索失败统一转成 HTTPException，前端能直接展示
"""

# ── 标准库 ──
from typing import Optional

# ── FastAPI ──
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

# ── Pydantic ──
from pydantic import BaseModel, Field

# ── 项目内导入 ──
from app.core.config import Config
from app.rag.loaders import UnsupportedDocumentError, load_bytes, load_dir
from app.rag.models import RagDocument
from app.rag.pipeline import ingest_documents
from app.rag.retriever import retrieve, to_sources
from app.rag.store import get_knowledge_store


router = APIRouter(prefix="/api/rag", tags=["rag"])


# ═══════════════════════════════════════════════════════════════════
# 请求 / 响应模型
# ═══════════════════════════════════════════════════════════════════

class IngestResponse(BaseModel):
    """入库结果"""
    ok: bool = Field(default=True, description="是否全部成功")
    documents: int = Field(default=0, description="成功入库的文档数")
    chunks: int = Field(default=0, description="写入的切片数")
    collection: str = Field(default="", description="目标知识库")
    elapsed_ms: int = Field(default=0, description="耗时（毫秒）")
    skipped: list = Field(default_factory=list, description="被跳过的文件及原因")
    error: str = Field(default="", description="失败原因（ok=False 时有值）")


class SearchRequest(BaseModel):
    """检索测试请求"""
    query: str = Field(..., description="检索Query")
    collection: Optional[str] = Field(default=None, description="知识库名，默认 Config.RAG_COLLECTION")
    top_k: int = Field(default=None, ge=1, le=50, description="召回条数")
    score_threshold: float = Field(
        default=None, ge=0.0, le=1.0,
        description="相关性阈值，低于此值的命中被丢弃",
    )


class SearchResponse(BaseModel):
    """检索测试响应"""
    ok: bool = Field(default=True)
    query: str = Field(default="")
    collection: str = Field(default="")
    count: int = Field(default=0, description="命中条数")
    hits: list = Field(default_factory=list, description="命中结果（含 source/score/content）")
    elapsed_ms: int = Field(default=0)


class DirIngestRequest(BaseModel):
    """目录导入请求"""
    path: str = Field(..., description="本地目录绝对路径")
    collection: Optional[str] = Field(default=None, description="目标知识库")
    max_files: int = Field(default=500, ge=1, le=5000, description="最多导入的文件数")


class CollectionInfo(BaseModel):
    """单个知识库集合的概览"""
    name: str = Field(default="")
    documents: int = Field(default=0, description="文档数")
    chunks: int = Field(default=0, description="切片数")


class CollectionsResponse(BaseModel):
    """集合列表"""
    ok: bool = Field(default=True)
    default_collection: str = Field(default="")
    count: int = Field(default=0)
    collections: list = Field(default_factory=list)


class DocumentsResponse(BaseModel):
    """知识库内文档列表"""
    ok: bool = Field(default=True)
    collection: str = Field(default="")
    count: int = Field(default=0)
    documents: list = Field(default_factory=list)


class DeleteDocumentsResponse(BaseModel):
    """按来源删除的结果"""
    ok: bool = Field(default=True)
    collection: str = Field(default="")
    source: str = Field(default="")
    deleted_chunks: int = Field(default=0)


class HealthResponse(BaseModel):
    """健康检查"""
    ok: bool = Field(default=True)
    enabled: bool = Field(default=True, description="RAG 总开关状态")
    collection: str = Field(default="")
    chunks: int = Field(default=0)
    documents: int = Field(default=0)
    sources: list = Field(default_factory=list)
    persist_dir: str = Field(default="")
    embedding_model: str = Field(default="")
    error: str = Field(default="")


# ═══════════════════════════════════════════════════════════════════
# 内部工具
# ═══════════════════════════════════════════════════════════════════

def _resolve_collection(collection: Optional[str]) -> str:
    """空值 → 默认集合名。"""
    return collection or Config.RAG_COLLECTION


def _require_enabled() -> None:
    """RAG 总开关关闭时拒绝写操作（读操作仍允许，便于排障）。"""
    if not Config.RAG_ENABLED:
        raise HTTPException(
            status_code=403,
            detail="RAG 已被禁用（RAG_ENABLED=false）",
        )


# ═══════════════════════════════════════════════════════════════════
# 端点：入库
# ═══════════════════════════════════════════════════════════════════

@router.post("/ingest", response_model=IngestResponse)
async def ingest_files(
    files: list[UploadFile] = File(..., description="待入库的文件（可多选）"),
    collection: str = Form(default=None, description="目标知识库"),
):
    """
    POST /api/rag/ingest —— 上传文件到知识库（multipart/form-data）。

    流程：读取字节 → 解析（文本/PDF/DOCX）→ 切分 → embedding → 写入 ChromaDB。
    同名文件重复上传会**覆盖**旧切片（先删后增），不会翻倍。

    用法：
      curl -X POST /api/rag/ingest \
        -F "files=@docs/arch.md" -F "files=@README.md" \
        -F "collection=knowledge"
    """
    _require_enabled()
    col = _resolve_collection(collection)

    documents: list[RagDocument] = []
    skipped: list[str] = []

    for upload in files:
        try:
            data = await upload.read()
            documents.append(load_bytes(data, upload.filename or "uploaded"))
        except UnsupportedDocumentError as e:
            skipped.append(f"{upload.filename} —— {e}")
        except Exception as e:
            skipped.append(f"{upload.filename} —— {type(e).__name__}: {e}")
        finally:
            await upload.close()

    if not documents:
        return IngestResponse(
            ok=False, collection=col, skipped=skipped,
            error="没有可入库的文件（全部被跳过）",
        )

    try:
        result = await ingest_documents(documents, collection=col, skipped=skipped)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"入库失败: {e}")

    return IngestResponse(
        ok=True,
        documents=result.documents,
        chunks=result.chunks,
        collection=result.collection,
        elapsed_ms=result.elapsed_ms,
        skipped=result.skipped,
    )


@router.post("/ingest/dir", response_model=IngestResponse)
async def ingest_directory(request: DirIngestRequest):
    """
    POST /api/rag/ingest/dir —— 把本地目录下的可解析文件批量导入知识库。

    安全：若配置了 RAG_ALLOWED_DIRS，目标目录必须位于白名单之内。

    用法：
      curl -X POST /api/rag/ingest/dir \\
        -H "Content-Type: application/json" \\
        -d '{"path": "/workspace/docs", "max_files": 200}'
    """
    _require_enabled()
    col = _resolve_collection(request.collection)

    try:
        documents, skipped = load_dir(
            request.path,
            allowed_dirs=Config.RAG_ALLOWED_DIRS or None,
            max_files=request.max_files,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not documents:
        return IngestResponse(
            ok=False, collection=col, skipped=skipped,
            error="目录下没有可解析的文件",
        )

    try:
        result = await ingest_documents(documents, collection=col, skipped=skipped)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"入库失败: {e}")

    return IngestResponse(
        ok=True,
        documents=result.documents,
        chunks=result.chunks,
        collection=result.collection,
        elapsed_ms=result.elapsed_ms,
        skipped=result.skipped,
    )


# ═══════════════════════════════════════════════════════════════════
# 端点：检索
# ═══════════════════════════════════════════════════════════════════

@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    POST /api/rag/search —— 检索测试（不经过 Agent，纯检索）。

    用于入库后验证召回效果：看看某个问题能召回哪些片段、相似度多少。

    用法：
      curl -X POST /api/rag/search -H "Content-Type: application/json" \\
        -d '{"query": "项目有哪些模块", "top_k": 5}'
    """
    import time
    col = _resolve_collection(request.collection)
    start = time.time()

    try:
        hits = await retrieve(
            request.query,
            collection=col,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {e}")

    return SearchResponse(
        ok=True,
        query=request.query,
        collection=col,
        count=len(hits),
        hits=to_sources(hits),
        elapsed_ms=int((time.time() - start) * 1000),
    )


# ═══════════════════════════════════════════════════════════════════
# 端点：知识库管理
# ═══════════════════════════════════════════════════════════════════

@router.get("/collections", response_model=CollectionsResponse)
async def list_collections():
    """
    GET /api/rag/collections —— 列出所有知识库集合及其文档/切片数。

    用于前端展示"当前有哪些知识库、各有多大"。
    """
    try:
        store = get_knowledge_store()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"知识库不可用: {e}")

    names = store.collection_names()
    # 默认集合始终出现在列表里（即使还没创建）
    if Config.RAG_COLLECTION not in names:
        names.append(Config.RAG_COLLECTION)

    infos = []
    for name in names:
        sources = store.list_sources(name)
        infos.append({
            "name": name,
            "documents": len(sources),
            "chunks": store.count(name),
        })

    return CollectionsResponse(
        ok=True,
        default_collection=Config.RAG_COLLECTION,
        count=len(infos),
        collections=infos,
    )


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(
    collection: str = Query(default=None, description="知识库名"),
):
    """
    GET /api/rag/documents —— 列出知识库内已入库的文档来源。

    返回每个来源的切片数、哈希、扩展名，供前端渲染知识库清单。
    """
    col = _resolve_collection(collection)
    try:
        store = get_knowledge_store()
        docs = store.list_sources(col)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"知识库不可用: {e}")

    return DocumentsResponse(ok=True, collection=col, count=len(docs), documents=docs)


@router.delete("/documents", response_model=DeleteDocumentsResponse)
async def delete_documents(
    source: str = Query(..., description="要删除的来源标识（文件名 / 相对路径）"),
    collection: str = Query(default=None, description="知识库名"),
):
    """
    DELETE /api/rag/documents —— 按来源删除文档的所有切片。

    用法：
      curl -X DELETE "/api/rag/documents?source=docs/arch.md"
    """
    _require_enabled()
    col = _resolve_collection(collection)

    try:
        store = get_knowledge_store()
        removed = store.delete_source(source, collection=col)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")

    if removed == 0:
        raise HTTPException(
            status_code=404,
            detail=f"未找到来源 '{source}'（知识库 '{col}'）",
        )

    return DeleteDocumentsResponse(
        ok=True, collection=col, source=source, deleted_chunks=removed,
    )


@router.get("/health", response_model=HealthResponse)
async def health(
    collection: str = Query(default=None, description="知识库名"),
):
    """
    GET /api/rag/health —— 健康检查 + 统计。

    返回持久化目录、embedding 模型、切片数、文档数，用于前端状态徽标与排障。
    """
    col = _resolve_collection(collection)

    try:
        store = get_knowledge_store()
        info = store.health_check(col)
    except Exception as e:
        return HealthResponse(ok=False, enabled=Config.RAG_ENABLED, collection=col, error=str(e))

    if not info.get("ok"):
        return HealthResponse(
            ok=False, enabled=Config.RAG_ENABLED, collection=col,
            error=str(info.get("error", "unknown")),
        )

    return HealthResponse(
        ok=True,
        enabled=Config.RAG_ENABLED,
        collection=col,
        chunks=info.get("chunks", 0),
        documents=info.get("documents", 0),
        sources=info.get("sources", []),
        persist_dir=info.get("persist_dir", ""),
        embedding_model=info.get("embedding_model", ""),
    )
