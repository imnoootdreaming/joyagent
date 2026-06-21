from fastapi import FastAPI
from app.api.agent import router as agent_router

app = FastAPI(title="JoyAgent", version="0.1.0")
app.include_router(agent_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
