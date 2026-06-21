from fastapi import APIRouter
from pydantic import BaseModel
from app.agent.agent import Agent

router = APIRouter(prefix="/api", tags=["agent"])

# 全局单例 Agent（所有请求共享同一个 Agent 实例）
agent = Agent()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    stop_reason: str = ""
    tool_calls: list = []
    iterations: int = 0


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """POST /api/chat  —  把 HTTP 请求转交给 Agent.agent_loop()"""
    result = await agent.agent_loop(request.message)
    return ChatResponse(**result)
