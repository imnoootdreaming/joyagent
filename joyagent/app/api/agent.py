"""
Phase 3 Step 6: FastAPI 接入 — 智能路由 Simple Agent / LangGraph Workflow。

在 Phase 1-2 的基础上新增：
  1. POST /api/chat     — 保留，新增 force_workflow 参数支持手动选择
  2. POST /api/workflow  — 新增，强制使用 LangGraph Plan→Execute→Reflect 工作流
  3. WS /ws/workflow/{id} — 新增，实时推送工作流各 Node 执行进度

路由策略：
  - 简单任务（"读取 xxx 文件"） → Phase 1-2 Simple Agent（更快）
  - 复杂任务（"构建一个 xxx 项目"） → Phase 3 LangGraph Workflow（更可靠）
  - 用户可通过 force_workflow=true 强制使用 LangGraph
"""

# ── Python 标准库 ──
import uuid                          # 生成唯一的 session_id / request_id
from typing import Optional          # Pydantic 可选字段类型

# ── FastAPI ──
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
# APIRouter: 模块化路由定义（最终注册到 main.py 的 app 上）
# WebSocket: 双向实时通信端点（推送工作流进度）

# ── Pydantic ──
from pydantic import BaseModel, Field
# BaseModel: 请求/响应数据校验和序列化

# ── 项目内导入 ──
from app.agent.agent import Agent
# Phase 1-2 Simple Agent: 适合单步骤任务的 ReAct Loop

from app.agent.graph.workflow import agent_workflow
# Phase 3 LangGraph Workflow: Plan→Execute→Reflect 三阶段工作流


# ═══════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════

router = APIRouter(prefix="/api", tags=["agent"])
# 所有端点都在 /api 前缀下

# 全局单例 Simple Agent（简单任务仍用 Phase 1-2 的 ReAct Loop）
simple_agent = Agent()


# ═══════════════════════════════════════════════════════════════════
# 请求/响应模型
# ═══════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """POST /api/chat 请求体（Phase 1-2 兼容 + Phase 3 扩展）"""
    message: str = Field(..., description="用户输入的自然语言任务描述")
    force_workflow: bool = Field(
        default=False,
        description="强制使用 LangGraph Workflow（Phase 3）。False 时自动判断复杂度。"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="会话 ID。用于 WebSocket 关联和工作流状态追踪。不传则自动生成。"
    )


class ChatResponse(BaseModel):
    """POST /api/chat & /api/workflow 共用响应体"""
    response: str = Field(..., description="Agent 的最终文本回复")
    stop_reason: str = Field(default="", description="LLM 的 stop_reason")
    tool_calls: list = Field(default_factory=list, description="工具调用记录")
    iterations: int = Field(default=0, description="迭代次数（Simple Agent 用）")
    session_id: str = Field(default="", description="会话 ID")
    backend: str = Field(
        default="simple",
        description="实际使用的后端：'simple' (Phase 1-2 Agent) 或 'workflow' (Phase 3 LangGraph)"
    )
    plan: list = Field(
        default_factory=list,
        description="LangGraph Planner 生成的计划（仅 workflow 后端有值）"
    )


class WorkflowRequest(BaseModel):
    """POST /api/workflow 请求体（强制 LangGraph）"""
    message: str = Field(..., description="用户输入的自然语言任务描述")
    session_id: Optional[str] = Field(
        default=None,
        description="会话 ID。用于 WebSocket 关联。不传则自动生成。"
    )
    max_reflections: int = Field(
        default=3,
        ge=1, le=10,
        description="最大反思轮次（1-10，默认 3）"
    )


# ═══════════════════════════════════════════════════════════════════
# 任务复杂度判断
# ═══════════════════════════════════════════════════════════════════

# 复杂任务关键词 —— 命中任一词即认为任务需要 LangGraph Workflow
COMPLEX_KEYWORDS = [
    # 中文关键词
    "构建", "搭建", "创建项目", "重构", "实现一个", "开发一个",
    "部署", "配置环境", "批量", "整个", "完整",
    # 英文关键词
    "build", "implement", "create a", "refactor",
    "setup", "scaffold", "generate a", "bootstrap",
    "multi-step", "workflow", "pipeline",
]


def _should_use_workflow(message: str, force: bool = False) -> bool:
    """
    判断任务是否应使用 LangGraph Workflow（Phase 3）。

    判断策略（由快到慢）：
      1. force=True → 用户显式要求，直接返回 True
      2. 关键词匹配 → 命中 COMPLEX_KEYWORDS 中的任意词
      3. 消息长度 → 超过 200 字符通常是多步骤任务
      4. 默认 → False（使用简单 Agent）

    简单 Agent 更快（无 Planner/Reflector 开销），适合单步骤任务。
    LangGraph Workflow 更可靠，适合需要多步协调的复杂任务。

    Args:
        message: 用户输入文本
        force: 是否强制使用 Workflow

    Returns:
        True → 使用 LangGraph Workflow；False → 使用 Simple Agent
    """
    if force:
        return True

    message_lower = message.lower()

    # ── 策略 2：关键词匹配 ──
    for kw in COMPLEX_KEYWORDS:
        if kw.lower() in message_lower:
            return True

    # ── 策略 3：消息长度（长消息通常描述复杂需求） ──
    if len(message) > 200:
        return True                         # 长消息大概率是多步骤任务

    # ── 默认：简单 Agent ──
    return False


# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════

def _new_session_id() -> str:
    """生成唯一会话 ID（短 UUID）。"""
    return uuid.uuid4().hex[:12]            # 12 位 hex 足够区分


def _build_initial_state(message: str, max_reflections: int = 3) -> dict:
    """
    构建 LangGraph Workflow 的初始 AgentState。

    将用户的自然语言消息包装为符合 AgentState TypedDict 的初始 dict。
    """
    return {
        "messages": [
            {"role": "user", "content": message}
        ],
        "plan": [],                          # Planner 会填充
        "current_step_index": 0,             # 从第 0 步开始
        "reflection_count": 0,               # 尚未反思
        "max_reflections": max_reflections,  # 允许的最大反思轮次
        "reflection_notes": "",              # 暂无反思笔记
        "task_completed": False,             # 尚未完成
        "need_replan": False,                # 尚未触发 replan
        "error_message": None,               # 暂无错误
        "tool_call_history": [],             # 暂无工具调用记录
    }


def _extract_final_response(final_state: dict) -> str:
    """
    从 LangGraph Workflow 的最终 state 提取用户可读的响应文本。

    提取顺序：
      1. messages 中最后一条 assistant 消息的文本
      2. Planner 生成的计划摘要（无 assistant 消息时）
      3. reflection_notes（无计划时）
    """
    messages = final_state.get("messages", [])
    # 从后向前找最后一条 assistant 消息
    for msg in reversed(messages):
        # LangChain HumanMessage 对象
        if hasattr(msg, 'type') and msg.type == "ai":
            content = msg.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif hasattr(block, 'type') and getattr(block, 'type') == "text":
                        text_parts.append(getattr(block, 'text', ''))
                return " ".join(text_parts).strip() or "Task completed."

        # 原生 dict 格式
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                return " ".join(text_parts).strip() or "Task completed."

    # Fallback: 无 assistant 消息 → 从 plan/reflection 中提取
    plan = final_state.get("plan", [])
    if plan:
        done = sum(1 for s in plan if s.get("status") == "completed")
        return f"Plan executed: {done}/{len(plan)} steps completed."

    return final_state.get("reflection_notes", "Task completed.")


# ═══════════════════════════════════════════════════════════════════
# REST 端点
# ═══════════════════════════════════════════════════════════════════

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    POST /api/chat — 智能路由的 Agent 对话端点。

    自动判断任务复杂度：
      - 简单任务 → Phase 1-2 Simple Agent（ReAct Loop，零开销）
      - 复杂任务 → Phase 3 LangGraph Workflow（Plan→Execute→Reflect）
      - force_workflow=true → 强制使用 LangGraph

    请求示例：
      # 简单任务（自动走 Simple Agent）
      curl -X POST /api/chat -d '{"message": "读取 main.py"}'

      # 复杂任务（自动走 LangGraph）
      curl -X POST /api/chat -d '{"message": "构建一个 Flask API 项目"}'

      # 强制走 LangGraph
      curl -X POST /api/chat -d '{"message": "读取 main.py", "force_workflow": true}'
    """
    session_id = request.session_id or _new_session_id()

    if _should_use_workflow(request.message, request.force_workflow):
        # ── Phase 3: LangGraph Workflow ──────────────────────────
        initial_state = _build_initial_state(request.message)
        final_state = await agent_workflow.ainvoke(initial_state)

        response_text = _extract_final_response(final_state)
        plan = final_state.get("plan", [])

        return ChatResponse(
            response=response_text,
            stop_reason="end_turn",
            tool_calls=final_state.get("tool_call_history", []),
            iterations=final_state.get("reflection_count", 0),
            session_id=session_id,
            backend="workflow",
            plan=[{
                "step_id": s.get("step_id"),
                "description": s.get("description", ""),
                "tool_name": s.get("tool_name"),
                "status": s.get("status"),
            } for s in plan],
        )
    else:
        # ── Phase 1-2: Simple Agent ──────────────────────────────
        result = await simple_agent.agent_loop(request.message)

        return ChatResponse(
            response=result.get("response", ""),
            stop_reason=result.get("stop_reason", ""),
            tool_calls=result.get("tool_calls", []),
            iterations=result.get("iterations", 0),
            session_id=session_id,
            backend="simple",
            plan=[],
        )


@router.post("/workflow", response_model=ChatResponse)
async def workflow(request: WorkflowRequest):
    """
    POST /api/workflow — 强制使用 LangGraph Plan→Execute→Reflect 工作流。

    与 /api/chat 的区别：
      - /api/chat: 智能路由（简单→Simple Agent, 复杂→Workflow）
      - /api/workflow: 始终使用 LangGraph Workflow，支持自定义 max_reflections

    请求示例：
      curl -X POST /api/workflow -d '{
        "message": "创建 calculator.py 并写测试",
        "max_reflections": 5
      }'
    """
    session_id = request.session_id or _new_session_id()
    initial_state = _build_initial_state(
        request.message,
        max_reflections=request.max_reflections,
    )

    final_state = await agent_workflow.ainvoke(initial_state)

    response_text = _extract_final_response(final_state)
    plan = final_state.get("plan", [])

    return ChatResponse(
        response=response_text,
        stop_reason="end_turn",
        tool_calls=final_state.get("tool_call_history", []),
        iterations=final_state.get("reflection_count", 0),
        session_id=session_id,
        backend="workflow",
        plan=[{
            "step_id": s.get("step_id"),
            "description": s.get("description", ""),
            "tool_name": s.get("tool_name"),
            "status": s.get("status"),
        } for s in plan],
    )


# ═══════════════════════════════════════════════════════════════════
# WebSocket 端点 — 实时推送工作流 Node 执行进度
# ═══════════════════════════════════════════════════════════════════

# 内存中 WebSocket 连接注册表：{session_id: [WebSocket, ...]}
_active_connections: dict[str, list[WebSocket]] = {}


async def _broadcast_event(session_id: str, event: dict):
    """
    向指定 session 的所有 WebSocket 连接广播事件。

    自动清理已断开的连接（客户端关闭页面或网络中断）。
    """
    if session_id not in _active_connections:
        return

    dead: list[WebSocket] = []               # 收集已断开的连接
    for ws in _active_connections[session_id]:
        try:
            await ws.send_json(event)        # 推送 JSON 事件
        except Exception:
            dead.append(ws)                  # 推送失败 → 标记清理

    for ws in dead:
        _active_connections[session_id].remove(ws)


@router.websocket("/ws/workflow/{session_id}")
async def websocket_workflow(websocket: WebSocket, session_id: str):
    """
    WS /ws/workflow/{session_id} — 实时推送 LangGraph Workflow 执行进度。

    客户端连接后，每次后端执行 workflow 时会推送以下事件类型：
      - node_start:   Node 开始执行（planner / executor / reflector）
      - node_done:    Node 执行完成（含 planning / tool_calls / reflection 数据）
      - route:        条件路由决策（下一步走向）
      - workflow_end: 工作流终止
      - error:        执行错误

    使用方式（前端 JavaScript）：
      const ws = new WebSocket("ws://localhost:8000/api/ws/workflow/abc123");
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(data.type, data.node);
      };

    心跳机制：客户端发 "ping"，服务端回 "pong"。
    """
    await websocket.accept()                 # 接受 WebSocket 握手

    # 注册连接
    _active_connections.setdefault(session_id, []).append(websocket)
    print(f"  [ws] client connected to session {session_id} "
          f"(total: {len(_active_connections[session_id])})")

    try:
        # 保持连接，处理客户端心跳
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        # 客户端主动断开
        pass

    except Exception:
        # 其他异常（网络中断等）
        pass

    finally:
        # 清理：从注册表中移除该连接
        if session_id in _active_connections:
            _active_connections[session_id].remove(websocket)
            remaining = len(_active_connections[session_id])
            if remaining == 0:
                del _active_connections[session_id]  # 无连接 → 清理 key
            else:
                print(
                    f"  [ws] client disconnected from session {session_id} "
                    f"(remaining: {remaining})"
                )
