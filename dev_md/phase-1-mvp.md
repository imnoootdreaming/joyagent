# Phase 1：Agent MVP

> **返回主索引：** [0-1 coding.md](0-1%20coding.md)
> **下一阶段：** [Phase 2: Tool Calling Framework](phase-2-tool-calling.md)

---

## 一、目标与定位

### 目标
实现最简单的 Claude Code Demo：一个能理解用户输入、调用工具、多轮对话的 Agent。

### 在整体架构中的位置
本 Phase 实现的是 ReAct Loop 的**最简版本**——只有 `文件读取` + `文件写入` 两个工具，所有状态存内存，无持久化。

```
User Input → FastAPI endpoint → LLM (with tools) → Tool Execution → Response
                                    ↑                                  │
                                    └──────── Message History ←────────┘
```

### 本 Phase 不做什么
- ❌ 不做 Shell 执行（Phase 2）
- ❌ 不做 LangGraph（Phase 3）
- ❌ 不做持久化（Phase 6/9）
- ❌ 不做安全沙箱（Phase 5）

---

## 二、前置依赖

| 依赖 | 版本建议 | 用途 |
|------|---------|------|
| Python | ≥ 3.11 | 基础运行环境 |
| FastAPI | ≥ 0.110 | Web 框架 |
| uvicorn | ≥ 0.27 | ASGI 服务器 |
| langchain | ≥ 0.3.0 | ChatModel 抽象 + Message 类型 |
| langchain-openai | ≥ 0.2.0 | OpenAI 模型适配 |
| langchain-anthropic | ≥ 0.3.0 | Claude 模型适配 |
| python-dotenv | ≥ 1.0 | 环境变量管理 |
| uv 或 Poetry | 最新 | 包管理器 |

```bash
# 推荐使用 uv（更快）
uv init joyagent
cd joyagent
uv add fastapi uvicorn langchain langchain-openai langchain-anthropic python-dotenv
```

---

## 三、目录结构

```text
app/
│
├── api/
│   └── agent.py          # POST /chat 端点
│
├── agent/
│   ├── agent.py          # Agent 核心：ReAct Loop
│   └── prompts.py        # System Prompt 管理
│
├── tools/
│   ├── file_read.py      # 文件读取工具
│   └── file_write.py     # 文件写入工具
│
├── services/
│   └── llm_service.py    # LLM 模型工厂
│
├── core/
│   └── config.py         # 配置（API Key, 模型选择等）
│
└── main.py               # FastAPI 应用入口
```

---

## 四、核心数据模型 / Schema 定义

### 4.1 消息模型（使用 LangChain 标准类型）

```python
from langchain_core.messages import (
    SystemMessage,    # 系统提示词
    HumanMessage,     # 用户输入
    AIMessage,        # LLM 回复（可能包含 tool_calls）
    ToolMessage,      # 工具执行结果
)
```

### 4.2 Agent 状态（本 Phase 仅存内存）

```python
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage

@dataclass
class AgentState:
    """Phase 1 的 Agent 状态——极简版"""
    messages: list[BaseMessage] = field(default_factory=list)
    max_iterations: int = 15           # 最大工具调用轮次
    current_iteration: int = 0
```

### 4.3 工具定义格式

```python
# 每个工具遵循 OpenAI Function Calling 格式
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file at the given path",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file"
                }
            },
            "required": ["path"]
        }
    }
}
```

### 4.4 API 契约

```python
# POST /chat
# Request:
{
    "message": "创建 hello.py，内容是 print('hello world')",
    "session_id": "optional-session-id"   # Phase 1 可选
}

# Response:
{
    "response": "已创建 hello.py，内容为 print('hello world')",
    "tool_calls": [                        # 本轮的工具有哪些被调用了
        {"tool": "write_file", "path": "hello.py", "success": true}
    ],
    "messages": [...]                      # 完整的消息历史
}
```

---

## 五、详细开发清单（含 HOW）

### Step 1：项目初始化（30 分钟）

**具体操作：**
```bash
uv init joyagent
cd joyagent
uv add fastapi uvicorn langchain langchain-openai langchain-anthropic python-dotenv
```

**创建 `.env`：**
```env
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
DEFAULT_MODEL=claude-sonnet-4-6       # 或 gpt-4o
```

**创建 `app/core/config.py`：**
```python
import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-6")
    MAX_ITERATIONS = 15
```

### Step 2：LLM Service（30 分钟）

**`app/services/llm_service.py`：** 核心是构建一个支持 Tool Calling 的 LLM 实例。

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

def get_llm(model_name: str = None, tools: list[dict] = None):
    """LLM 工厂函数。绑定工具定义，返回可调用的 ChatModel。"""
    model_name = model_name or Config.DEFAULT_MODEL
    
    if "claude" in model_name.lower():
        llm = ChatAnthropic(
            model=model_name,
            api_key=Config.ANTHROPIC_API_KEY,
            temperature=0.3,
            max_tokens=4096,
        )
    else:
        llm = ChatOpenAI(
            model=model_name,
            api_key=Config.OPENAI_API_KEY,
            temperature=0.3,
        )
    
    if tools:
        llm = llm.bind_tools(tools)  # ← 关键：绑定工具定义
    return llm
```

### Step 3：Tool 实现（30 分钟）

**`app/tools/file_read.py`：**
```python
import os

def read_file(path: str) -> str:
    """读取文件内容。注意：本 Phase 不做路径安全校验（Phase 5 补充）。"""
    if not os.path.exists(path):
        return f"Error: File '{path}' not found."
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content[:10000]  # 截断过长文件
```

**`app/tools/file_write.py`：**
```python
def write_file(path: str, content: str) -> str:
    """写文件。自动创建父目录。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Successfully wrote {len(content)} bytes to {path}"
```

**工具 Schema 定义（`app/tools/schemas.py`）：**
```python
READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file at the given path",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"}
            },
            "required": ["path"]
        }
    }
}

WRITE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a file. Creates parent directories if needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write to"},
                "content": {"type": "string", "description": "Content to write"}
            },
            "required": ["path", "content"]
        }
    }
}

TOOLS = [READ_FILE_SCHEMA, WRITE_FILE_SCHEMA]

# 工具执行映射
TOOL_EXECUTORS = {
    "read_file": read_file,
    "write_file": write_file,
}
```

### Step 4：System Prompt 设计（30 分钟）⭐ 重要

**`app/agent/prompts.py`：**
```python
SYSTEM_PROMPT = """You are an autonomous coding agent, similar to Claude Code.
You help users write, read, and modify code files.

## Your capabilities
- Read files using the read_file tool
- Write/create files using the write_file tool

## Rules
1. When asked to create a file, ALWAYS use the write_file tool — do not just output code in your response.
2. When you need to understand existing code, use read_file first.
3. Think step by step: understand the request → gather context → act → verify.
4. After completing a task, briefly explain what you did.

## Output style
- Be concise and direct.
- When you use a tool, wait for its result before responding.
- If you encounter an error, explain it and try to fix it.
"""
```

**System Prompt 设计要点（面试考点）：**
- 必须明确告诉 LLM"你必须使用工具"，否则它可能直接输出代码文本而不调用 write_file
- "wait for its result before responding"很重要——否则 LLM 可能在工具结果返回前就生成回复
- 后续 Phase 中这个 Prompt 会变得更复杂（加入 ReAct 指令、安全规则等）

### Step 5：Agent Runtime 核心（1 小时）

**`app/agent/agent.py`：** 这是 Phase 1 最关键的文件。

```python
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from app.agent.prompts import SYSTEM_PROMPT
from app.services.llm_service import get_llm
from app.tools.schemas import TOOLS, TOOL_EXECUTORS
from app.core.config import Config

class Agent:
    def __init__(self, model_name: str = None):
        self.llm = get_llm(model_name, tools=TOOLS)
        self.max_iterations = Config.MAX_ITERATIONS
    
    async def chat(self, user_message: str, history: list = None) -> dict:
        """
        单轮对话入口。
        Phase 1 简化版：每次 chat 独立处理，不做跨轮状态管理。
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        if history:
            messages.extend(history)
        messages.append(HumanMessage(content=user_message))
        
        iterations = 0
        tool_calls_made = []
        
        while iterations < self.max_iterations:
            iterations += 1
            
            # 1. 调用 LLM
            response = await self.llm.ainvoke(messages)
            messages.append(response)
            
            # 2. 检查是否有工具调用
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tc in response.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]
                    
                    # 执行工具
                    executor = TOOL_EXECUTORS.get(tool_name)
                    if executor:
                        result = executor(**tool_args)
                    else:
                        result = f"Error: Unknown tool '{tool_name}'"
                    
                    tool_calls_made.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result[:500],  # 截断过长结果
                    })
                    
                    # 追加工具结果到消息历史
                    messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tc["id"],
                    ))
            else:
                # 没有工具调用 → 任务完成
                return {
                    "response": response.content,
                    "tool_calls": tool_calls_made,
                    "iterations": iterations,
                }
        
        # 超出最大迭代次数
        return {
            "response": "Task exceeded maximum iterations. Last state:\n" + messages[-1].content,
            "tool_calls": tool_calls_made,
            "iterations": iterations,
        }
```

### Step 6：FastAPI 端点 + 入口（30 分钟）

**`app/api/agent.py`：**
```python
from fastapi import APIRouter
from pydantic import BaseModel
from app.agent.agent import Agent

router = APIRouter(prefix="/api", tags=["agent"])
agent = Agent()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    tool_calls: list = []
    iterations: int = 0

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = await agent.chat(request.message)
    return ChatResponse(**result)
```

**`app/main.py`：**
```python
from fastapi import FastAPI
from app.api.agent import router as agent_router

app = FastAPI(title="JoyAgent", version="0.1.0")
app.include_router(agent_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 7：测试验证（30 分钟）

```bash
# 启动服务
uv run uvicorn app.main:app --reload

# 测试请求
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "创建 hello.py，内容是 print(\"hello world\")"}'
```

---

## 六、关键代码模式与伪代码

### ReAct Loop（本 Phase 简化版）

```python
# 核心模式：LLM ↔ Tool 的交互循环
messages = [system_prompt, user_input]

while not done and iterations < max:
    response = llm.invoke(messages)     # LLM 决定：说话 or 调工具
    
    if response.has_tool_calls:
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call)
            messages.append(ToolMessage(result))
        # 循环继续：让 LLM 看到工具结果后再决定
    else:
        done = True                      # LLM 认为任务完成
        return response.content
```

---

## 七、完成标志

### 基本完成
- [ ] Agent 能响应 `"读取 hello.py"` → 输出文件内容
- [ ] Agent 能响应 `"创建 hello.py 内容是 xxx"` → 文件被创建
- [ ] Agent 能响应 `"修改 hello.py 把 xxx 改成 yyy"` → 先 read_file 再 write_file（多轮工具调用）

### 自测用例

```python
# 测试 1：单工具调用
await agent.chat("创建 test.py，内容是 print(1+1)")

# 测试 2：多工具调用（先读后写）
await agent.chat("读取 test.py，在末尾加一行 print('done')")

# 测试 3：纯对话（不需要工具）
await agent.chat("Python 的 list 和 tuple 有什么区别？")
```

---

## 八、文档差距分析

### 原文档缺失的内容及补充

| # | 原文档问题 | 实际开发需要 | 本文档补充位置 |
|---|-----------|-------------|-------------|
| 1 | 只说 "Prompt 管理"，没说 System Prompt 具体写什么 | 必须有明确的 System Prompt 模板，告诉 LLM"你必须调用工具" | §5 Step 4 |
| 2 | 没说 Message History 存哪里 | 本 Phase 存内存 list；需定义消息追加规则（System → Human → AI → Tool → AI → ...） | §4.1, §5 Step 5 |
| 3 | 没说 LLM 怎么绑定工具（bind_tools） | LangChain ChatModel 的 `.bind_tools()` 是关键一步 | §5 Step 2 |
| 4 | 完成标志 "创建 hello.py" 太弱 | 应该要求 **多轮工具调用**（先读后写）才算 Agent 真正工作了 | §7 |
| 5 | 没有 API 契约定义 | FastAPI 端点需要明确的 Request/Response Schema | §4.4 |
| 6 | 没有提到 max_iterations | 没有上限的话 LLM 可能无限循环调工具 | §4.2, §5 Step 5 |
| 7 | 没说工具返回结果截断 | LLM Context Window 有限，长文件需要截断 | §5 Step 3 |

### 本 Phase 已知债务（将在后续 Phase 偿还）

| 债务 | 偿还 Phase |
|------|-----------|
| 工具定义分散在各文件，无统一抽象 | Phase 2 |
| 无安全检查（路径穿越、任意文件读写） | Phase 5 |
| 状态仅存内存，重启丢失 | Phase 6 |
| 无流式输出（Streaming） | Phase 9 |
| 只支持单轮对话（history 参数预留但未实现跨轮） | Phase 6 |

---

## 九、面试考点映射

| 面试题 | 答案要点 | 本文档参考 |
|--------|---------|-----------|
| **什么是 ReAct？** | Reasoning + Acting：LLM 交替进行"推理"和"行动"，推理决定下一步做什么，行动通过工具执行。循环直到任务完成。 | §6 伪代码 |
| **ReAct vs Chain-of-Thought？** | CoT 只推理不行动；ReAct 推理+行动+观察循环，能获取外部信息。CoT 适合数学推理，ReAct 适合需要与环境交互的任务。 | §6 |
| **Tool Calling 实现原理？** | LLM 输出特殊 token 标识函数调用（含函数名+参数 JSON），应用层解析后执行函数，结果作为 ToolMessage 追加到上下文，LLM 继续推理。 | §5 Step 5 |
| **为什么 Agent 需要 max_iterations？** | 防止无限循环（LLM 可能反复调同一个工具），控制 Token 消耗，设置止损边界。 | §4.2 |
| **System Prompt 设计要点？** | 1) 明确角色和能力边界 2) 强制使用工具而非输出代码文本 3) 要求等待工具结果 4) 定义输出风格 5) 安全规则。 | §5 Step 4 |
