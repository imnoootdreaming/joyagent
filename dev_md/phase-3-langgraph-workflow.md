# Phase 3：LangGraph Workflow

> **返回主索引：** [0-1 coding.md](0-1%20coding.md)
> **上一阶段：** [Phase 2: Tool Calling Framework](phase-2-tool-calling.md)
> **下一阶段：** [Phase 4: Coding Agent](phase-4-coding-agent.md)

---

## 一、目标与定位

### 目标
用 LangGraph StateGraph 替代 Phase 1-2 的手写 while 循环，构建 Plan → Execute → Reflect 三阶段工作流，引入条件路由（Conditional Edge）和 Checkpoint。

### 在整体架构中的位置

**关键澄清：Phase 3 ≠ Phase 7 的 Multi-Agent**

```
Phase 1-2：单 Agent + 手写 while 循环 (ReAct Loop)
                ↓
Phase 3：  单 Agent + LangGraph StateGraph (Plan → Execute → Reflect)
           — 这是单 Agent 的内部工作流结构化
           — planner / executor / reflector 是 StateGraph 的 Node，不是独立 Agent
                ↓
Phase 7：  多 Agent 协作 (Planner Agent / Coder Agent / Tester Agent / Reviewer Agent)
           — 每个 Node 升级为独立 Agent 进程，有自己的 LLM 实例和工具集
           — 引入 Agent Router 在多 Agent 之间分发任务
```

**Phase 3 的本质：** 把 Phase 1-2 的"平铺直叙"的 ReAct Loop 升级为**结构化的三阶段工作流**：
- **Plan 阶段：** LLM 先分析任务，输出结构化计划（步骤列表）
- **Execute 阶段：** 按计划逐步执行，使用工具
- **Reflect 阶段：** 反思执行结果，决定是否需要重新规划

### 本 Phase 不做什么
- ❌ 不做 Multi-Agent 协同（Phase 7）
- ❌ 不做代码 AST 分析（Phase 4）
- ❌ 不做记忆持久化（Phase 6）

---

## 二、前置依赖

| 依赖 | 用途 |
|------|------|
| Phase 1-2 完成 | Agent 基础 + Tool Calling |
| langgraph | StateGraph + Node + Edge |
| langgraph-checkpoint | Checkpoint 持久化（可选） |

```bash
uv add langgraph langgraph-checkpoint
```

---

## 三、目录结构

```text
app/
├── agent/
│   ├── agent.py              # 旧的 Agent 类保留（简单任务仍可用）
│   │
│   ├── runtime/              # 工作流节点实现（Plan/Execute/Reflect 的逻辑）
│   │   ├── __init__.py
│   │   ├── planner.py        # Plan Node：任务拆解 + 生成计划
│   │   ├── executor.py       # Execute Node：按计划执行工具调用
│   │   └── reflector.py      # Reflect Node：反思结果 + 决定下一步
│   │
│   ├── graph/                # LangGraph StateGraph 定义
│   │   ├── __init__.py
│   │   ├── workflow.py       # StateGraph 构建 + 编译
│   │   ├── nodes.py          # 节点包装函数（调用 runtime/ 的逻辑）
│   │   └── edges.py          # 条件边逻辑（路由决策）
│   │
│   ├── prompts/              # 各阶段专用 Prompt
│   │   ├── planner.txt       # 规划阶段 System Prompt
│   │   ├── executor.txt      # 执行阶段 System Prompt
│   │   └── reflector.txt     # 反思阶段 System Prompt
│   │
│   └── schemas/
│       └── state.py          # AgentState TypedDict 定义
```

**目录关系说明：**
- `runtime/` — 节点逻辑（what each node does），可独立测试
- `graph/` — StateGraph 构建（how nodes connect），LangGraph 的"配置层"
- `prompts/` — 每个节点的 Prompt 模板
- `schemas/` — 共享的状态类型定义

---

## 四、核心数据模型 / Schema 定义

### 4.1 AgentState（LangGraph State）

```python
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
# 消息使用纯 dict 格式（Anthropic 原生），不再依赖 langchain_core.messages

class TaskStep(TypedDict):
    """计划中的单个步骤"""
    step_id: int
    description: str        # 步骤描述（给 LLM 看）
    tool_name: str | None   # 需要调用的工具
    status: str             # "pending" | "in_progress" | "completed" | "failed"

class AgentState(TypedDict):
    """LangGraph StateGraph 的核心状态"""
    
    # 消息历史（自动追加——add_messages reducer）
    # Anthropic 原生格式：list[dict]，每个 dict 为 {"role": "user/assistant", "content": ...}
    messages: Annotated[Sequence[dict], add_messages]
    
    # 任务计划
    plan: list[TaskStep]
    current_step_index: int        # 当前执行到第几步
    
    # 反思
    reflection_count: int          # 已经反思了几次
    max_reflections: int           # 最多反思几轮（默认 3）
    reflection_notes: str          # 反思笔记（LLM 输出）
    
    # 控制
    task_completed: bool           # 任务是否完成
    error_message: str | None      # 最近一次错误信息
    
    # 工具调用记录（用于反思）
    tool_call_history: list[dict]  # [{tool_name, args, result, success}]
```

### 4.2 工作流状态流转

```text
                    ┌─────────────┐
                    │   __start__  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Planner    │  ← 分析任务，生成 plan[]
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Executor    │  ← 执行 plan[current_step_index]
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Reflector   │  ← 检查结果，评估是否完成
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         task_completed  need_replan  继续执行
              │            │            │
              ▼            ▼            │
          __end__      Planner    Executor (current_step_index++)
```

### 4.3 Conditional Edge 路由逻辑

```python
def should_continue(state: AgentState) -> str:
    """Reflector 之后的 Conditional Edge"""
    
    # 条件 1：任务完成
    if state["task_completed"]:
        return "__end__"
    
    # 条件 2：需要重新规划
    if state.get("need_replan"):
        return "planner"
    
    # 条件 3：反思次数超限 → 强制结束
    if state["reflection_count"] >= state["max_reflections"]:
        return "__end__"
    
    # 条件 4：还有步骤未完成 → 继续执行
    if state["current_step_index"] < len(state["plan"]):
        return "executor"
    
    # 默认：规划完成但任务未完成 → 反思（可能触发 replan）
    return "reflector"
```

---

## 五、详细开发清单（含 HOW）

### Step 1：定义 AgentState（30 分钟）
- 按 §4.1 实现 `schemas/state.py`
- 关键：`messages` 字段用 `add_messages` reducer（LangGraph 自动追加消息）
- `plan` 是核心——Phase 1-2 的 Agent 没有显式计划，本 Phase 加入结构化计划

### Step 2：实现 Planner Node（1 小时）

**`agent/runtime/planner.py` 核心逻辑：**
```python
async def plan(state: AgentState) -> dict:
    """
    Planner Node 的职责：
    1. 分析用户输入（从 messages 中提取）
    2. 拆解为可执行的步骤（TaskStep 列表）
    3. 为每个步骤推荐工具
    """
    from app.services.llm_service import get_or_create_client
    client = get_or_create_client()

    plan_prompt = f"""
    You are a task planner. Given the user's request, break it down into
    concrete, executable steps. For each step, specify which tool to use.

    Available tools: {tool_registry.list_tool_names()}

    User request: {extract_user_request(state["messages"])}

    Output format (JSON array):
    [
      {{"step_id": 1, "description": "...", "tool_name": "read_file"}},
      ...
    ]
    """

    # Anthropic 原生调用（不带 tools——Planner 只输出计划，不执行）
    response = client.messages.create(
        model=MODEL,
        system="You are a task planner. Output only valid JSON.",
        messages=[{"role": "user", "content": plan_prompt}],
        max_tokens=4096,
    )
    plan_text = extract_text(response.content)
    plan = parse_plan_json(plan_text)  # 解析为 list[TaskStep]

    return {"plan": plan, "current_step_index": 0}
```

### Step 3：实现 Executor Node（1 小时）

**`agent/runtime/executor.py` 核心逻辑：**
```python
async def execute_step(state: AgentState) -> dict:
    """
    Executor Node 的职责：
    1. 取 plan[current_step_index]
    2. 调用 LLM + 工具执行当前步骤（Anthropic 原生 Tool Use）
    3. 记录执行结果到 tool_call_history
    """
    from app.services.llm_service import get_or_create_client
    client = get_or_create_client()
    current_step = state["plan"][state["current_step_index"]]

    step_prompt = f"""
    Execute this step: {current_step['description']}
    Use the {current_step['tool_name']} tool if needed.
    After completion, report the result.
    """

    # 使用 Anthropic 原生 Tool Use loop 执行单个步骤
    messages = [{"role": "user", "content": step_prompt}]
    tool_log = []

    for _ in range(5):  # 单步最多 5 轮工具调用
        response = client.messages.create(
            model=MODEL,
            system="You are a task executor. Use tools to complete each step.",
            messages=messages,
            tools=tool_registry.get_tool_schemas(),
            max_tokens=4096,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = await tool_registry.execute(block.name, **block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result.output,
            })
            tool_log.append({
                "tool_name": block.name, "input": block.input,
                "result": result.output, "success": result.success,
            })

        messages.append({"role": "user", "content": tool_results})

    return {
        "messages": messages,
        "current_step_index": state["current_step_index"] + 1,
        "tool_call_history": state["tool_call_history"] + tool_log,
    }
```

### Step 4：实现 Reflector Node（1 小时）

**`agent/runtime/reflector.py` 核心逻辑：**
```python
async def reflect(state: AgentState) -> dict:
    """
    Reflector Node 的职责：
    1. 检查 plan 执行情况
    2. 判断任务是否完成
    3. 如果未完成，分析原因（工具失败？计划不合理？）
    4. 决定：继续 / 重新规划 / 结束
    """
    from app.services.llm_service import get_or_create_client
    client = get_or_create_client()

    reflection_prompt = f"""
    You are a quality inspector. Review the execution results:

    Original plan: {state['plan']}
    Steps completed: {state['current_step_index']}
    Tool call history: {state['tool_call_history']}
    Errors: {state.get('error_message', 'None')}

    Answer these questions:
    1. Is the task completed? (yes/no)
    2. If not, what went wrong?
    3. Should we replan? (yes/no)
    4. What should we do differently?

    Output format: JSON with keys: task_completed, analysis, need_replan, suggestion
    """

    response = client.messages.create(
        model=MODEL,
        system="You are a quality inspector. Output only valid JSON.",
        messages=[{"role": "user", "content": reflection_prompt}],
        max_tokens=4096,
    )
    reflection = parse_reflection_json(extract_text(response.content))

    return {
        "task_completed": reflection["task_completed"],
        "need_replan": reflection.get("need_replan", False),
        "reflection_notes": reflection.get("analysis", ""),
        "reflection_count": state["reflection_count"] + 1,
    }
```

### Step 5：构建 StateGraph（1 小时）

**`agent/graph/workflow.py`：**
```python
from langgraph.graph import StateGraph, END
from app.agent.schemas.state import AgentState
from app.agent.graph.nodes import planner_node, executor_node, reflector_node
from app.agent.graph.edges import should_continue

def build_workflow() -> StateGraph:
    """构建 Plan → Execute → Reflect 工作流"""
    
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reflector", reflector_node)
    
    # 添加边
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "reflector")
    
    # Conditional Edge：Reflector 之后去哪里？
    workflow.add_conditional_edges(
        "reflector",
        should_continue,
        {
            "executor": "executor",    # 继续执行下一步
            "planner": "planner",      # 重新规划
            "__end__": END,            # 任务结束
        }
    )
    
    return workflow.compile()

# 全局 Agent Workflow 实例
agent_workflow = build_workflow()
```

### Step 6：接入 FastAPI（30 分钟）
- 修改 `app/api/agent.py` 的 `/chat` 端点
- 简单任务仍用 Phase 1-2 的直接 Agent
- 复杂任务（用户指定或自动判断）使用 LangGraph workflow

---

## 六、关键代码模式与伪代码

### 6.1 工作流调用模式

```python
# Phase 1-2 的调用方式（简单任务保留）
result = await agent.chat("创建 hello.py")

# Phase 3 的调用方式（复杂任务）
initial_state = {
    "messages": [{"role": "user", "content": "构建一个 Flask API 服务..."}],
    "plan": [],
    "current_step_index": 0,
    "reflection_count": 0,
    "max_reflections": 3,
    "task_completed": False,
    "error_message": None,
    "tool_call_history": [],
}
final_state = await agent_workflow.ainvoke(initial_state)
```

### 6.2 任务复杂度判断（何时用 LangGraph）

```python
def should_use_workflow(user_message: str) -> bool:
    """简单启发式：包含多步骤关键词的任务使用 LangGraph workflow"""
    complex_keywords = [
        "构建", "实现", "搭建", "创建项目", "重构",
        "build", "implement", "create a", "refactor",
    ]
    return any(kw in user_message.lower() for kw in complex_keywords)
```

---

## 七、完成标志

### 基本完成
- [ ] StateGraph 正确编译和运行
- [ ] Planner 能生成结构化步骤列表
- [ ] Executor 能按计划逐步执行，使用工具
- [ ] Reflector 能判断任务是否完成，触发 replan 或结束
- [ ] Conditional Edge 正确路由（完→结束、不成→replan、继续→executor）
- [ ] 简单任务（"创建 hello.py"）仍可通过旧 Agent 快速处理

### 自测用例

```bash
# 测试 1：多步骤任务（触发 LangGraph）
curl -X POST /api/chat -d '{"message": "创建 Python Flask 项目结构，包含 app.py 和 requirements.txt"}'
# 期望：Planner 生成计划 → Executor 逐步创建文件

# 测试 2：需要反思的任务
curl -X POST /api/chat -d '{"message": "修复 test.py 中的语法错误"}'
# 期望：Exec 尝试修复 → Reflect 发现仍有问题 → Replan → 再 Exec

# 测试 3：简单任务（不走 LangGraph）
curl -X POST /api/chat -d '{"message": "读取 README.md"}'
# 期望：直接用 Phase 1-2 的简单 Agent 处理
```

---

## 八、文档差距分析

### 原文档缺失的内容及补充

| # | 原文档问题 | 实际开发需要 | 本文档补充位置 |
|---|-----------|-------------|-------------|
| 1 | **Phase 3 与 Phase 1-2 的关系未说明** | 是替换旧的 Agent 还是包装？→ **包装**：简单任务仍用 Phase 1-2 的 while 循环，复杂任务用 LangGraph | §1, §6.2 |
| 2 | **Phase 3 与 Phase 7 的关系未说明** | Phase 3 = 单 Agent 内部结构化；Phase 7 = 多 Agent 拆分 | §1 (关系图) |
| 3 | `runtime/` 和 `graph/` 目录关系未说明 | runtime = 节点逻辑实现；graph = StateGraph 配置+编译 | §3 |
| 4 | 没有 AgentState 定义 | StateGraph 的核心是 State，必须明确字段和类型 | §4.1 |
| 5 | 没有 Conditional Edge 的具体场景 | 具体路由逻辑：完成→结束、不成→replan、继续→executor | §4.3, §5 Step 5 |
| 6 | 没说 Checkpoint 怎么用 | `langgraph-checkpoint` 可做状态持久化（暂停/恢复），本 Phase 先了解 API，Phase 6 接入 Redis | §2 |
| 7 | Prompt 没有分阶段设计 | Planner/Executor/Reflector 各需要不同的 System Prompt | §5 Step 2-4 |

### 设计决策说明：为什么不是完全替换 Phase 1-2？

Phase 1-2 的简单 ReAct Agent 对于单步骤任务（如"读取某文件"）更高效——无需走 Planner→Executor→Reflector 的完整流程。LangGraph Workflow 是为**需要多步协调的复杂任务**配置的。两者共存是合理的设计。

---

## 九、面试考点映射

| 面试题 | 答案要点 | 本文档参考 |
|--------|---------|-----------|
| **为什么选 LangGraph 而不是 AgentExecutor？** | AgentExecutor 是黑盒，内部循环不可控（只能设 max_iterations）；LangGraph 让我们显式定义每一步，精准控制状态流转，支持 Conditional Edge 和 Checkpoint。 | §1 |
| **StateGraph 是什么？** | LangGraph 的核心抽象——用有向图定义 Agent 工作流。Node = 处理逻辑，Edge = 流转方向。State 在所有 Node 间共享，每次 Node 执行返回部分 State 更新。 | §4.1, §5 Step 5 |
| **Node 和 Edge 如何设计？** | Node 按职责拆分（Planner/Executor/Reflector），每个 Node 是纯函数 (State → Partial<State>)；Edge 分普通边和条件边，条件边根据 State 动态决定下一步。 | §5 |
| **Conditional Edge 的应用场景？** | 错误重试、任务 replan、动态工具选择、权限检查后的路由。本质是根据当前 State 做 if/else 路由。 | §4.3 |
| **Checkpoint 有什么作用？** | 保存每个 Node 执行后的 State 快照。支持：1) 暂停/恢复长时间任务 2) 回退到历史状态 3) 调试和审计。 | §5 Step 5 |
