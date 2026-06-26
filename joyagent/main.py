from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.agent import router as agent_router

app = FastAPI(title="JoyAgent", version="0.1.0")
app.include_router(agent_router)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup():
    """Phase 2: 启动时注册所有工具到 ToolRegistry。"""
    from app.tools import register_all_tools
    register_all_tools()
    print("  [OK] JoyAgent startup complete -- ToolRegistry initialized.\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
