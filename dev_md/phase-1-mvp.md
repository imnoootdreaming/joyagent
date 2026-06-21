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
| anthropic | ≥ 0.40.0 | Anthropic Messages API（原生 SDK） |
| python-dotenv | ≥ 1.0 | 环境变量管理 |
| uv 或 Poetry | 最新 | 包管理器 |

```bash
# 推荐使用 uv（更快）
uv init joyagent
cd joyagent
uv add fastapi uvicorn anthropic python-dotenv
```

> ⚠️ **重要：本项目不依赖 langchain、langchain-openai、langchain-anthropic。** 直接使用 Anthropic Python SDK 原生调用 Messages API。DeepSeek 通过其 Anthropic 兼容端点接入，同样使用 `anthropic` 包。

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

### 4.1 消息模型（Anthropic Messages API 原生格式）

```python
# Anthropic Messages API 使用纯 Python dict，不需要 LangChain Message 对象
# 消息只有两种 role：user 和 assistant

# 用户消息
user_message = {"role": "user", "content": "创建 hello.py"}

# 用户消息（含工具结果——Anthropic 格式）
user_message_with_tool_results = {
    "role": "user",
    "content": [
        {"type": "tool_result", "tool_use_id": "tool_001", "content": "Wrote 20 bytes to hello.py"}
    ]
}

# assistant 消息（Anthropic 返回的内容块列表，含 text + tool_use）
assistant_message = {
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Let me write that file for you."},
        {"type": "tool_use", "id": "tool_001", "name": "write_file", "input": {"path": "hello.py", "content": "print('hello')"}}
    ]
}
```

> **关键差异：** Anthropic API 的 system prompt 是独立参数传入 `client.messages.create(system=...)`，不作为消息列表的一部分。

### 4.2 Agent 状态（本 Phase 仅存内存）

```python
from dataclasses import dataclass, field

@dataclass
class AgentState:
    """Phase 1 的 Agent 状态——极简版"""
    messages: list[dict] = field(default_factory=list)  # dict 格式的消息列表
    max_iterations: int = 15           # 最大工具调用轮次
    current_iteration: int = 0
```

### 4.3 工具定义格式（Anthropic 原生 `input_schema`）

```python
# Anthropic Messages API 工具格式：无需 "type": "function" 包装层
READ_FILE_TOOL = {
    "name": "read_file",
    "description": "Read the contents of a file at the given path",
    "input_schema": {
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

WRITE_FILE_TOOL = {
    "name": "write_file",
    "description": "Write content to a file. Creates parent directories if needed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write to"},
            "content": {"type": "string", "description": "Content to write"}
        },
        "required": ["path", "content"]
    }
}
```

> **与 OpenAI 格式的对比：** OpenAI 需要 `{"type": "function", "function": {"name": ..., "parameters": ...}}`，而 Anthropic 直接 `{"name": ..., "input_schema": ...}`，少一层嵌套。

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
    "stop_reason": "end_turn",             # Anthropic stop_reason
    "iterations": 2                        # 多少轮 LLM 调用
}
```

---

## 五、详细开发清单（含 HOW）

### Step 1：项目初始化（30 分钟）

**具体操作：**
```bash
uv init joyagent
cd joyagent
uv add fastapi uvicorn anthropic python-dotenv
```

**创建 `.env`：**
```env
# === Anthropic API（Claude + DeepSeek 共用 anthropic 包） ===
ANTHROPIC_API_KEY=sk-ant-xxx       # Claude 原生 API Key
ANTHROPIC_BASE_URL=                 # DeepSeek 兼容端点：https://api.deepseek.com/anthropic

# === 默认模型 ===
DEFAULT_MODEL=DeepSeek-v4-pro[1m]   # 或 claude-sonnet-4-6
```

**创建 `app/core/config.py`：**
```python
import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")  # DeepSeek 兼容端点
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "DeepSeek-v4-pro[1m]")
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "15"))
```

### Step 2：LLM Service — 原生 Anthropic 客户端（30 分钟）

**`app/services/llm_service.py`：** 按协议族创建 Anthropic 客户端。DeepSeek 与 Claude 共用 `anthropic` 包，仅 `base_url` 不同。

```python
import os
from anthropic import Anthropic
from joyagent.app.core.config import Config


def get_client(model_name: str = None) -> Anthropic:
    """创建 Anthropic 客户端 —— 按协议族分流

    设计思路：
    - Claude（原生 Anthropic API）和 DeepSeek（Anthropic 兼容端点）都使用
      同一个 `anthropic` 包，仅通过 base_url 区分目标地址
    - 一行注册：Anthropic(base_url=...) 即可，无需工厂模式
    - API Key 同样通过 ANTHROPIC_API_KEY 环境变量；DeepSeek 兼容端点
      可能用不同的 key，按需设置 ANTHROPIC_BASE_URL 即可
    """
    base_url = os.getenv("ANTHROPIC_BASE_URL")  # DeepSeek 兼容端点需要设置此值
    return Anthropic(base_url=base_url)


# 全局客户端实例（按需创建，支持不同的 base_url）
_client_cache: dict[str, Anthropic] = {}


def get_or_create_client(model_name: str = None) -> Anthropic:
    """获取或创建客户端（带缓存）"""
    model_name = model_name or Config.DEFAULT_MODEL
    name_lower = model_name.lower()

    # DeepSeek 使用其 Anthropic 兼容端点
    if "deepseek" in name_lower:
        cache_key = "deepseek"
        if cache_key not in _client_cache:
            _client_cache[cache_key] = Anthropic(
                base_url="https://api.deepseek.com/anthropic"
            )
        return _client_cache[cache_key]

    # Claude 原生 API（或未设置的 base_url）
    cache_key = "default"
    if cache_key not in _client_cache:
        _client_cache[cache_key] = get_client(model_name)
    return _client_cache[cache_key]
```

> **为什么不去 LangChain 包装？** `ChatAnthropic` 隐藏了 `stop_reason` 和 `block.type` 等关键控制点；`bind_tools()` 内部做了工具格式转换，调试困难。原生 SDK 的一行 `client.messages.create()` 足够简洁，无需额外封装层。

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
# Anthropic 原生工具格式：{"name": ..., "description": ..., "input_schema": {...}}
# 无需 OpenAI Function Calling 的 "type": "function" 包装层

READ_FILE_TOOL = {
    "name": "read_file",
    "description": "Read the contents of a file at the given path",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read"}
        },
        "required": ["path"]
    }
}

WRITE_FILE_TOOL = {
    "name": "write_file",
    "description": "Write content to a file. Creates parent directories if needed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write to"},
            "content": {"type": "string", "description": "Content to write"}
        },
        "required": ["path", "content"]
    }
}

TOOLS = [READ_FILE_TOOL, WRITE_FILE_TOOL]

# 工具执行映射：tool_name -> handler function
TOOL_HANDLERS = {
    "read_file": read_file,
    "write_file": write_file,
}
```

### Step 4：System Prompt 设计（30 分钟）⭐ 重要

**`app/agent/prompts.py`：**

```python
# Anthropic API 的 system prompt 是独立字符串参数，不是消息列表的一部分
# 注意：Anthropic 建议 system prompt 用纯文本而非 dict/JSON

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

# 后续 Phase 会扩展为从多个 section 组装：
# PROMPT_SECTIONS = {
#     "identity": "You are a coding agent...",
#     "tools": "Available tools: read_file, write_file, ...",
#     "workspace": f"Working directory: {WORKDIR}",
#     "memory": "Relevant memories are injected below when available.",
# }
```

**System Prompt 设计要点（面试考点）：**
- Anthropic API 中 system prompt 是**独立参数** (`client.messages.create(system=...)`)，不在 messages 列表里
- 必须明确告诉 LLM"你必须使用工具"，否则它可能直接输出代码文本而不调用 write_file
- "wait for its result before responding"很重要——否则 LLM 可能在工具结果返回前就生成回复
- 后续 Phase 中这个 Prompt 会变得更复杂（加入 ReAct 指令、安全规则等）
- 当启用记忆和 skill 系统时，system prompt 需要支持动态组装

### Step 5：Agent Runtime 核心（1 小时）

**`app/agent/agent.py`：** 这是 Phase 1 最关键的文件——使用 Anthropic 原生 SDK 的 ReAct Loop。

```python
import json
from app.agent.prompts import SYSTEM_PROMPT
from app.services.llm_service import get_or_create_client
from app.tools.schemas import TOOLS, TOOL_HANDLERS
from app.core.config import Config

# ── Agent 核心循环常量 ──
DEFAULT_MAX_TOKENS = 4096
ESCALATED_MAX_TOKENS = 32000  # max_tokens 升级上限
MAX_RECOVERY_RETRIES = 3      # 续写最多尝试次数
CONTINUATION_PROMPT = (
    "Output token limit hit. Resume directly — "
    "no apology, no recap. Pick up mid-thought."
)


class RecoveryState:
    """追踪单次 Agent Loop 中的恢复状态"""
    def __init__(self):
        self.has_escalated = False        # 是否已升级过 max_tokens
        self.recovery_count = 0           # 续写次数
        self.has_attempted_compact = False # 是否已尝试紧急压缩


class Agent:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.DEFAULT_MODEL
        self.client = get_or_create_client(self.model_name)
        self.max_iterations = Config.MAX_ITERATIONS

    async def chat(self, user_message: str, history: list = None) -> dict:
        """
        单轮对话入口 — Anthropic 原生 Agent Loop。

        核心模式（Anthropic 风格）：
            while stop_reason == "tool_use":
                response = client.messages.create(messages, tools)
                tool_results = execute_tools(response.content)
                messages.append({"role": "user", "content": tool_results})
        """
        # 构建消息历史（纯 dict，非 LangChain 对象）
        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        iterations = 0
        tool_calls_made = []
        state = RecoveryState()
        max_tokens = DEFAULT_MAX_TOKENS

        while iterations < self.max_iterations:
            iterations += 1

            # ── 1. 调用 LLM（Anthropic Messages API 原生调用）──
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    system=SYSTEM_PROMPT,       # system 是独立参数
                    messages=messages,           # 纯 dict 列表
                    tools=TOOLS,                 # Anthropic 原生工具格式
                    max_tokens=max_tokens,
                )
            except Exception as e:
                # 简单错误处理（Phase 9 会有完整 error recovery）
                return {
                    "response": f"[Error] {type(e).__name__}: {e}",
                    "tool_calls": tool_calls_made,
                    "iterations": iterations,
                }

            # ── 2. 追加 assistant 回复到消息历史 ──
            messages.append({"role": "assistant", "content": response.content})

            # ── 3. max_tokens 恢复 ──
            if response.stop_reason == "max_tokens":
                if not state.has_escalated:
                    # 第一次：升级 max_tokens 后重试同一请求
                    max_tokens = ESCALATED_MAX_TOKENS
                    state.has_escalated = True
                    messages.pop()  # 移除截断的 assistant 消息
                    print(f"  \033[33m[max_tokens] escalating to {max_tokens}\033[0m")
                    continue
                # 已升级仍截断：追加续写提示
                if state.recovery_count < MAX_RECOVERY_RETRIES:
                    messages.append({"role": "user", "content": CONTINUATION_PROMPT})
                    state.recovery_count += 1
                    print(f"  \033[33m[max_tokens] continuation {state.recovery_count}/{MAX_RECOVERY_RETRIES}\033[0m")
                    continue
                return {
                    "response": "Task output exceeded max token limits.",
                    "tool_calls": tool_calls_made,
                    "iterations": iterations,
                }

            # ── 4. 检查是否有工具调用（Anthropic 原生方式）──
            if response.stop_reason != "tool_use":
                # 没有工具调用 → 模型认为任务完成
                text_output = ""
                for block in response.content:
                    if block.type == "text":
                        text_output += block.text
                return {
                    "response": text_output or "Task completed.",
                    "stop_reason": response.stop_reason,
                    "tool_calls": tool_calls_made,
                    "iterations": iterations,
                }

            # ── 5. 执行工具调用 ──
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input  # Anthropic 用 .input，不是 .args

                # 查找并执行 handler
                handler = TOOL_HANDLERS.get(tool_name)
                if handler:
                    result = handler(**tool_input)
                else:
                    result = f"Error: Unknown tool '{tool_name}'"

                tool_calls_made.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "result": str(result)[:500],
                })

                # 构造 Anthropic 格式的 tool_result
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })

            # ── 6. 工具结果作为 user 消息追加，循环继续 ──
            messages.append({"role": "user", "content": tool_results})

        # 超出最大迭代次数
        return {
            "response": "Task exceeded maximum iterations.",
            "tool_calls": tool_calls_made,
            "iterations": iterations,
        }
```

**Anthropic 原生模式要点（面试必问）：**

| 概念 | OpenAI/LangChain 方式 | Anthropic 原生方式 |
|------|----------------------|-------------------|
| 客户端创建 | `ChatOpenAI(...)` / `ChatAnthropic(...)` | `Anthropic(base_url=...)` — 一行注册 |
| API 调用 | `llm.ainvoke(messages)` | `client.messages.create(model=..., system=..., messages=..., tools=..., max_tokens=...)` |
| 工具绑定 | `llm.bind_tools(tools)` | 直接传入 `tools=TOOLS` 参数 |
| 判断有工具调用 | `hasattr(response, "tool_calls")` | `response.stop_reason == "tool_use"` |
| 遍历工具调用 | 遍历 `response.tool_calls` 列表 | 遍历 `response.content` 检查 `block.type == "tool_use"` |
| 工具参数 | `tc["args"]` | `block.input`（属性访问，非字典键） |
| 工具结果 | `ToolMessage(content=..., tool_call_id=...)` | `{"role": "user", "content": [{"type": "tool_result", "tool_use_id": ..., "content": ...}]}` |
| System Prompt | `SystemMessage(content=...)` 放到消息列表 | 独立参数 `system=SYSTEM_PROMPT` 传入 |

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
    stop_reason: str = ""
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

### ReAct Loop（本 Phase 简化版 — Anthropic 原生模式）

```python
# 核心模式：LLM ↔ Tool 的交互循环
# Anthropic 原生风格 — stop_reason + block.type 判断

messages = [{"role": "user", "content": user_input}]

while not done and iterations < max:
    # LLM 决定：说话 or 调工具
    response = client.messages.create(
        model=MODEL, system=SYSTEM_PROMPT,
        messages=messages, tools=TOOLS, max_tokens=4096,
    )

    messages.append({"role": "assistant", "content": response.content})

    # Anthropic 原生判断：通过 stop_reason
    if response.stop_reason != "tool_use":
        done = True  # 没有工具调用，LLM 认为任务完成
        return extract_text(response.content)

    # 有工具调用：遍历 content 中的 tool_use 块
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = execute_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

    # 工具结果回传给 LLM
    messages.append({"role": "user", "content": tool_results})
```

### 错误恢复模式（Error Recovery — 从参考项目 learn-claude-code 引入）⭐ 新增

```python
# 三层错误恢复（Anthropic 原生 Agent Loop 的标准防护）
class RecoveryState:
    def __init__(self):
        self.has_escalated = False          # max_tokens 升级标记
        self.recovery_count = 0             # 续写次数
        self.has_attempted_compact = False  # 紧急压缩标记
        self.consecutive_529 = 0            # 529 过载计数

# Path 1: max_tokens → 升级上限 or 续写
if response.stop_reason == "max_tokens":
    if not state.has_escalated:
        max_tokens = ESCALATED_MAX_TOKENS  # 4K→32K
        state.has_escalated = True
        continue  # 不 append，用更大 max_tokens 重试
    if state.recovery_count < MAX_RECOVERY_RETRIES:
        messages.append({"role": "user", "content": CONTINUATION_PROMPT})
        state.recovery_count += 1
        continue

# Path 2: prompt_too_long → 紧急压缩（只做一次）
if is_prompt_too_long_error(e):
    if not state.has_attempted_compact:
        messages[:] = reactive_compact(messages)  # 保留最近 N 条
        state.has_attempted_compact = True
        continue

# Path 3: 429/529 → 指数退避 + 备用模型
# with_retry() 包装：429 等指数退避重试；529 连续 3 次切换 fallback 模型
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
| 1 | 只说 "Prompt 管理"，没说 System Prompt 具体写什么 | 必须有明确的 System Prompt 模板，告诉 LLM"你必须调用工具"；Anthropic API 中 system 是独立参数 | §5 Step 4 |
| 2 | 没说 Message History 存哪里 | 本 Phase 存内存 list[dict]；Anthropic 消息只有 user/assistant 两种 role | §4.1, §5 Step 5 |
| 3 | ~~没说 LLM 怎么绑定工具（bind_tools）~~ | **已移除**：Anthropic SDK 直接在 `client.messages.create(tools=TOOLS)` 传入，无需 `bind_tools()` | §5 Step 2 |
| 4 | 完成标志 "创建 hello.py" 太弱 | 应该要求 **多轮工具调用**（先读后写）才算 Agent 真正工作了 | §7 |
| 5 | 没有 API 契约定义 | FastAPI 端点需要明确的 Request/Response Schema；返回 stop_reason 字段 | §4.4 |
| 6 | 没有提到 max_iterations | 没有上限的话 LLM 可能无限循环调工具 | §4.2, §5 Step 5 |
| 7 | 没说工具返回结果截断 | LLM Context Window 有限，长文件需要截断 | §5 Step 3 |
| 8 | **缺少错误恢复 (Error Recovery) 设计** | max_tokens 升级、reactive compact、指数退避重试 —— 生产级 Agent 必备 | §6 错误恢复模式 |
| 9 | **使用 OpenAI Function Calling 工具格式** | **已改为** Anthropic 原生格式：`{"name": ..., "input_schema": {...}}` | §4.3 |

### 本 Phase 已知债务（将在后续 Phase 偿还）

| 债务 | 偿还 Phase |
|------|-----------|
| 工具定义分散在各文件，无统一抽象 | Phase 2 |
| 无安全检查（路径穿越、任意文件读写） | Phase 5 |
| 状态仅存内存，重启丢失 | Phase 6 |
| 无流式输出（Streaming） | Phase 9 |
| 只支持单轮对话（history 参数预留但未实现跨轮） | Phase 6 |
| 多模型客户端创建无缓存（每次 chat 创建新 client） | Phase 1 Step 2 已解决（_client_cache） |
| 错误恢复仅基础实现（需补充 with_retry + reactive_compact） | Phase 9 / 参考 s11 |

---

## 九、面试考点映射

| 面试题 | 答案要点 | 本文档参考 |
|--------|---------|-----------|
| **什么是 ReAct？** | Reasoning + Acting：LLM 交替进行"推理"和"行动"，推理决定下一步做什么，行动通过工具执行。循环直到任务完成。 | §6 伪代码 |
| **ReAct vs Chain-of-Thought？** | CoT 只推理不行动；ReAct 推理+行动+观察循环，能获取外部信息。CoT 适合数学推理，ReAct 适合需要与环境交互的任务。 | §6 |
| **Anthropic Tool Use 实现原理？** | `client.messages.create(tools=TOOLS)` 传入工具定义 → 模型返回 `stop_reason == "tool_use"` → 遍历 `response.content` 检查 `block.type == "tool_use"` → 执行 handler → tool_result 以 user 消息追加 → 循环继续。与 OpenAI 的核心区别：使用 `stop_reason` + `block.type` 而非 `hasattr(response, "tool_calls")`。 | §5 Step 5, §6 |
| **为什么 Agent 需要 max_iterations？** | 防止无限循环（LLM 可能反复调同一个工具），控制 Token 消耗，设置止损边界。 | §4.2 |
| **System Prompt 设计要点？** | 1) 明确角色和能力边界 2) 强制使用工具而非输出代码文本 3) 要求等待工具结果 4) 定义输出风格 5) 安全规则。Anthropic API 中 system 是独立参数，不在消息列表里。 | §5 Step 4 |
| **Anthropic 原生 SDK vs LangChain 封装？** | LangChain `ChatAnthropic` + `bind_tools()` 隐藏了 `stop_reason` 和 `block.type` 关键控制点，调试困难。原生 SDK 一行 `Anthropic(base_url=...)` 注册，`client.messages.create()` 直接调用，所有细节可见可控。 | §5 Step 2, §5 Step 5 |
| **Agent 的错误恢复机制有哪些？** | 三层：1) max_tokens → 升级上限(4K→32K) + 续写提示 2) prompt_too_long → 紧急压缩 3) 429/529 → 指数退避 + 备用模型。 | §6 错误恢复模式 |
