"""
JoyAgent v0.7.0 — Multi-Agent Coding System

启动入口。管理所有 Agent 的生命周期、API 路由注册、
以及 Multi-Agent + Mailbox 系统的初始化和优雅关闭。

Phase 7: 新增 MultiAgentOrchestrator 全局单例，在 startup 时
自动注册并启动 Router / Planner / Coder / Tester / Reviewer 六 Agent，
在 shutdown 时持久化未处理消息并优雅关闭。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.agent import router as agent_router
from app.agent.orchestrator import MultiAgentOrchestrator, OrchestratorConfig


# ═══════════════════════════════════════════════════════════════════════
# Phase 7: Multi-Agent Orchestrator 全局单例
# ═══════════════════════════════════════════════════════════════════════

multi_agent_orch = MultiAgentOrchestrator(
    config=OrchestratorConfig(
        # 持久化后端："memory"（默认）| "file" | "redis"
        # 生产环境推荐 "file" 或 "redis"
        persistence_backend="file",
        persistence_path="data/mailbox",

        # 超时与清理
        request_timeout=300.0,
        expire_sweep_interval=60.0,

        verbose=True,
    )
)


# ═══════════════════════════════════════════════════════════════════════
# 生命周期管理（asynccontextmanager 替代 on_event）
# ═══════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan —— 替代 @app.on_event("startup"/"shutdown")。

    Startup:
      1. 注册所有工具到 ToolRegistry
      2. 创建并注册所有 Agent 到 MailboxManager
      3. 从持久化恢复未处理消息
      4. 后台启动所有 Agent 的 InboxWatcher

    Shutdown:
      1. 停止所有 Agent 的 InboxWatcher
      2. 持久化未处理消息
      3. 清理过期消息 + 注销 Agent
    """
    # ── Startup ──
    from app.tools import register_all_tools
    register_all_tools()
    print("  [OK] ToolRegistry initialized.\n")

    # Phase 7: 初始化并启动多 Agent 系统
    multi_agent_orch.register_all()
    await multi_agent_orch.start_all()
    print(f"  [OK] Multi-Agent system online "
          f"(backend={multi_agent_orch.persistence_backend_name})\n")

    yield  # ← 应用运行中

    # ── Shutdown ──
    print("\n  [shutdown] Stopping Multi-Agent system...")
    await multi_agent_orch.shutdown()
    print("  [OK] Multi-Agent system shut down.\n")


# ═══════════════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="JoyAgent",
    version="0.7.0",
    lifespan=lifespan,
)

app.include_router(agent_router)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
