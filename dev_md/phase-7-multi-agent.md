# Phase 7：Multi-Agent

> **返回主索引：** [0-1 coding.md](0-1%20coding.md)
> **上一阶段：** [Phase 6: Memory System](phase-6-memory-system.md)
> **下一阶段：** [Phase 8: MCP Plugin System](phase-8-mcp.md)

---

## 一、目标与定位

### 目标
将单 Agent 的 Plan → Execute → Reflect 工作流升级为**多 Agent 协同系统**：每个阶段由独立的 Agent 负责，通过 Router 分发任务。

### 与 Phase 3 的关系（关键澄清）⚠️

```
Phase 3 单 Agent 工作流：
  ┌──────┐    ┌──────┐    ┌──────┐
  │Plan  │───▶│Exec  │───▶│Refl  │    ← 同一个 Agent（同一个 LLM + 同一套工具）
  │Node  │    │Node  │    │Node  │    ← LangGraph StateGraph 的不同节点
  └──────┘    └──────┘    └──────┘

Phase 7 多 Agent 协作：
  ┌─────────────────────────────────────────┐
  │              Agent Router                │
  └──┬────────┬──────────┬──────────┬───────┘
     │        │          │          │
  ┌──▼───┐ ┌──▼───┐ ┌───▼──┐ ┌───▼──────┐
  │Plan  │ │Code  │ │Test  │ │Review    │     ← 独立 Agent 各有
  │Agent │ │Agent │ │Agent │ │Agent     │     ← 自己的 LLM + 工具 + Prompt
  └──────┘ └──────┘ └──────┘ └──────────┘
       │        │         │          │
       └────────┴─────────┴──────────┘
                     │
              ┌──────▼──────┐
              │ Shared State │           ← 所有 Agent 共享同一个 AgentState
              │ (LangGraph)  │
              └─────────────┘
```

**核心差异：**
| 维度 | Phase 3 | Phase 7 |
|------|---------|---------|
| 粒度 | 单 Agent 内部节点 | 多个独立 Agent 进程 |
| LLM | 所有节点共用一个 LLM 实例 | 每个 Agent 可有不同 LLM（强 Agent 用 Opus/4o，弱 Agent 用 Haiku/4o-mini） |
| 工具 | 所有节点共享全部工具 | 每个 Agent 有专属工具子集（如 Tester 有 pytest，Coder 没有） |
| Prompt | 3 个节点 Prompt | 每个 Agent 有独立 System Prompt + 角色定位 |
| 通信 | StateGraph 自动传递 State | 共享 State + 可选的 Message Queue |

### 本 Phase 不做什么
- ❌ 不做 Agent 热加载/热插拔（Phase 8 MCP 做）
- ❌ 不做分布式 Agent（多机器协作）

---

## 二、前置依赖

| 依赖 | 用途 |
|------|------|
| Phase 3 完成 | LangGraph 基础 |
| Phase 5 完成 | Sandbox + Testing |
| Phase 6 完成 | Memory System（Agent 间共享记忆） |

---

## 三、目录结构

```text
app/
├── agent/
│   ├── planner/              # Planner Agent（任务拆解）
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── prompts.py
│   │
│   ├── coder/                # Coder Agent（代码生成/修改）
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── prompts.py
│   │
│   ├── tester/               # Tester Agent（测试执行/分析）
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── prompts.py
│   │
│   ├── reviewer/             # Reviewer Agent（Code Review）
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── prompts.py
│   │
│   └── router/               # Agent Router（任务分发）
│       ├── __init__.py
│       ├── router.py
│       └── prompts.py
```

---

## 四、核心数据模型 / Schema 定义

### 4.1 各 Agent 角色定义

```python
from dataclasses import dataclass

@dataclass
class AgentRole:
    """Agent 角色定义"""
    name: str                  # "planner" | "coder" | "tester" | "reviewer"
    system_prompt: str
    tools: list[str]           # 该 Agent 可用的工具名称列表
    model: str                 # 该 Agent 使用的模型（可为不同 Agent 选不同模型）
    temperature: float
    
    # 权限
    can_modify_files: bool     # Coder=True, Tester=False
    can_execute_shell: bool    # Coder=True, Tester=True
    needs_user_approval: bool  # Tester=False（测试可自动），Reviewer=True

# Agent 角色配置表
AGENT_ROLES = {
    "planner": AgentRole(
        name="planner",
        system_prompt="You are a task planner...",
        tools=["code_search", "read_file"],
        model="claude-sonnet-4-6",      # 规划需要强推理
        temperature=0.2,
        can_modify_files=False,
        can_execute_shell=False,
        needs_user_approval=False,
    ),
    "coder": AgentRole(
        name="coder",
        system_prompt="You are a code generator...",
        tools=["read_file", "write_file", "execute_shell", "git_diff"],
        model="claude-sonnet-4-6",      # 代码生成需要强模型
        temperature=0.1,
        can_modify_files=True,
        can_execute_shell=True,
        needs_user_approval=True,        # Coder 写文件需要审批
    ),
    "tester": AgentRole(
        name="tester",
        system_prompt="You are a test runner...",
        tools=["execute_shell", "read_file", "pytest_run"],
        model="claude-haiku-4-5",        # 测试执行可用轻量模型
        temperature=0.0,
        can_modify_files=False,
        can_execute_shell=True,
        needs_user_approval=False,       # 测试可自动执行
    ),
    "reviewer": AgentRole(
        name="reviewer",
        system_prompt="You are a code reviewer...",
        tools=["read_file", "code_search", "git_diff", "ast_analyze"],
        model="claude-sonnet-4-6",       # Code Review 需要强推理
        temperature=0.3,
        can_modify_files=False,
        can_execute_shell=False,
        needs_user_approval=False,
    ),
}
```

### 4.2 Router 决策模型

```python
@dataclass
class RouterDecision:
    """Router 的任务分发决策"""
    target_agent: str          # "planner" | "coder" | "tester" | "reviewer"
    reason: str                # 为什么选择这个 Agent
    task_description: str      # 给目标 Agent 的任务描述
    priority: int              # 1-5，越高越紧急
    expected_output: str       # 期望的输出格式
```

### 4.3 多 Agent 共享 State

```python
class MultiAgentState(TypedDict):
    """多 Agent 共享的全局状态（扩展现有 AgentState）"""
    
    # 继承 Phase 3 AgentState 的所有字段
    messages: Annotated[Sequence[BaseMessage], add_messages]
    plan: list[TaskStep]
    # ... 其他 Phase 3 字段
    
    # Phase 7 新增
    active_agent: str                    # 当前活跃的 Agent
    agent_outputs: dict[str, str]        # {agent_name: output}
    review_feedback: list[dict]          # Reviewer 的反馈 [{file, issue, suggestion}]
    task_assignments: list[RouterDecision]  # 任务分发记录
```

---

## 五、详细开发清单（含 HOW）

### Step 1：重构目录，建立 Agent 基类（30 分钟）

```python
# app/agent/base.py
class BaseAgent:
    """所有 Agent 的基类"""

    def __init__(self, role: AgentRole):
        self.role = role
        self.client = get_or_create_client(role.model)
        self.tools = tool_registry.get_tool_schemas_for(role.tools)

    async def run(self, task: str, state: MultiAgentState) -> dict:
        """Agent 主入口。子类可覆盖。"""
        # Anthropic 原生调用——system 是独立参数
        response = self.client.messages.create(
            model=self.role.model,
            system=self.role.system_prompt,
            messages=[{"role": "user", "content": f"Task: {task}\nCurrent state: {state}"}],
            tools=self.tools,
            max_tokens=4096,
        )
        return {"output": extract_text(response.content), "agent": self.role.name}
```

### Step 2：实现各 Agent（2 小时）

**Planner Agent：**
- 职责：接收用户需求 → 分析 → 拆解为子任务 → 分配给 Coder/Tester/Reviewer
- 输入：用户需求 + 项目上下文
- 输出：任务分解列表 + 分配建议

**Coder Agent：**
- 职责：接收编码任务 → 搜索代码 → 生成/修改代码 → 生成 Diff → 应用 Patch
- 工具：read_file, write_file, execute_shell, git_diff, code_search, ast_analyze
- 复用 Phase 4 的 Coding Agent 能力

**Tester Agent：**
- 职责：接收测试任务 → 在 Docker Sandbox 中执行测试 → 解析结果 → 报告
- 工具：execute_shell, read_file, pytest_run
- 复用 Phase 5 的 Sandbox + Pytest Runner

**Reviewer Agent：**
- 职责：审阅 Coder 生成的代码 → 检查 bugs、性能、风格、安全
- 工具：read_file, code_search, git_diff, ast_analyze
- 输出：结构化的 Review Report（问题列表 + 严重程度 + 建议）
- 关键 Prompt：要求 Reviewer 输出结构化 JSON

### Step 3：实现 Agent Router（1 小时）

**`agent/router/router.py`：**

Router 有两种策略：
1. **规则路由**（MVP）：基于关键词匹配
2. **LLM 路由**（进阶）：Router 本身也是一个轻量 Agent，用 LLM 判断任务应该发给谁

```python
class AgentRouter:
    """多 Agent 路由器"""
    
    # 规则路由（MVP）
    RULES = [
        (["设计", "规划", "拆解", "plan", "design"], "planner"),
        (["写代码", "实现", "编码", "修改", "创建", "code", "implement", "fix"], "coder"),
        (["测试", "运行", "验证", "test", "run", "verify", "pytest"], "tester"),
        (["审查", "检查", "review", "check", "audit"], "reviewer"),
    ]
    
    def __init__(self, strategy: str = "rule"):
        self.strategy = strategy
        if strategy == "llm":
            # 轻量 Anthropic client 做路由
            self.router_client = get_or_create_client("claude-haiku-4-5")

    async def route(self, task: str, state: MultiAgentState) -> RouterDecision:
        if self.strategy == "rule":
            return self._rule_route(task)
        else:
            return await self._llm_route(task, state)

    def _rule_route(self, task: str) -> RouterDecision:
        task_lower = task.lower()
        for keywords, agent_name in self.RULES:
            if any(kw in task_lower for kw in keywords):
                return RouterDecision(
                    target_agent=agent_name,
                    reason=f"Keyword match: {keywords}",
                    task_description=task,
                    priority=3,
                    expected_output="",
                )
        # 默认 → Planner（让 Planner 决定）
        return RouterDecision(target_agent="planner", reason="Default routing", ...)

    async def _llm_route(self, task: str, state: MultiAgentState) -> RouterDecision:
        """让轻量 LLM 判断任务应该发到哪个 Agent"""
        prompt = f"""
        Given this task: "{task}"
        And current state: {summarize(state)}

        Which agent should handle this?
        Options: planner, coder, tester, reviewer

        Respond with JSON: {{"agent": "...", "reason": "..."}}
        """
        response = self.router_client.messages.create(
            model="claude-haiku-4-5",
            system="You are a task router. Output only valid JSON.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )
        return parse_router_json(extract_text(response.content))
```

### Step 4：Agent 间通信（1 小时）

```python
class AgentCommunication:
    """Agent 间通信机制"""
    
    def __init__(self, state: MultiAgentState):
        self.state = state
    
    async def send_to_agent(self, from_agent: str, to_agent: str,
                            message: str) -> None:
        """发送消息给另一个 Agent（Anthropic dict 格式）"""
        self.state["messages"].append(
            {"role": "user", "content": f"[{from_agent} → {to_agent}]: {message}"}
        )

    async def broadcast(self, from_agent: str, message: str) -> None:
        """广播消息给所有 Agent"""
        self.state["messages"].append(
            {"role": "user", "content": f"[{from_agent} → ALL]: {message}"}
        )
    
    async def request_review(self, code_diff: str) -> dict:
        """Coder 请求 Reviewer 审查"""
        return await self.send_to_agent("coder", "reviewer", 
            f"Please review this diff:\n{code_diff}")
```

### Step 5：解决 Agent 冲突（1 小时）

```python
class ConflictResolver:
    """
    多 Agent 冲突解决策略。
    
    常见冲突：
    1. Coder 改了文件，Reviewer 要求改回去
    2. Planner 制定了 A 计划，Coder 在执行时发现 A 不可行，执行了 B
    3. Tester 发现了 Coder 不认可的"Bug"
    """
    
    async def resolve(self, conflict: dict, state: MultiAgentState) -> dict:
        """
        冲突升级策略：
        1. 两个 Agent 协商（Coder ↔ Reviewer 直接对话）
        2. 协商失败 → Planner 重新评估
        3. 仍无法解决 → 升级到用户
        """
        
        # Level 1: Agent 间协商
        negotiation_prompt = f"""
        Conflict: Agent {conflict['agent_a']} says "{conflict['claim_a']}"
        Agent {conflict['agent_b']} says "{conflict['claim_b']}"
        
        Can you find a compromise? Respond with: {{"resolved": true/false, "solution": "..."}}
        """
        
        response = self.client.messages.create(
            model=MODEL, system="You are a conflict resolver.",
            messages=[{"role": "user", "content": negotiation_prompt}],
            max_tokens=1024,
        )
        result = parse_json(extract_text(response.content))
        
        if result["resolved"]:
            return {"status": "resolved", "solution": result["solution"]}
        
        # Level 2: 升级到 Planner
        # Level 3: 升级到用户（Human-in-the-Loop, Phase 9）
        return {"status": "escalated_to_user", "conflict": conflict}
```

### Step 6：编排 Multi-Agent Workflow（1 小时）
- 在 LangGraph 中新增 Router Node
- 修改 Executor Node → 根据 Router 决策分发到对应 Agent
- Reviewer Node 的输出反馈到 Coder Node（修改建议）

---

## 六、关键代码模式与伪代码

### 6.1 Multi-Agent 协作流程

```python
async def multi_agent_pipeline(user_request: str) -> str:
    state = MultiAgentState(messages=[{"role": "user", "content": user_request}])
    router = AgentRouter(strategy="llm")
    
    # Phase 1: Planner 拆解任务
    plan_decision = await router.route(user_request, state)
    planner = PlannerAgent(AGENT_ROLES["planner"])
    plan_result = await planner.run(plan_decision.task_description, state)
    state["plan"] = plan_result["steps"]
    
    # Phase 2: 循环执行（Coder → Tester → Reviewer）
    for step in state["plan"]:
        # 2a. 路由到 Coder
        coder = CoderAgent(AGENT_ROLES["coder"])
        code_result = await coder.run(step["description"], state)
        
        # 2b. 路由到 Tester
        tester = TesterAgent(AGENT_ROLES["tester"])
        test_result = await tester.run(f"Test: {code_result['modified_files']}", state)
        
        if not test_result["all_passed"]:
            # 2c. Coder 修复 → 再测试（Fix Loop，Phase 5）
            for _ in range(3):
                code_result = await coder.run(f"Fix: {test_result['failures']}", state)
                test_result = await tester.run(f"Retest", state)
                if test_result["all_passed"]:
                    break
        
        # 2d. 路由到 Reviewer（仅在测试通过后）
        reviewer = ReviewerAgent(AGENT_ROLES["reviewer"])
        review_result = await reviewer.run(f"Review: {code_result['diff']}", state)
        
        if review_result["issues"]:
            # 反馈给 Coder
            await coder.run(f"Fix review issues: {review_result['issues']}", state)
    
    return state["agent_outputs"]
```

---

## 七、完成标志

### 基本完成
- [ ] Planner/Coder/Tester/Reviewer 四个 Agent 各自能独立工作
- [ ] Router 能正确将任务分发到合适的 Agent
- [ ] Coder → Tester → Reviewer 的协作流程跑通
- [ ] Agent 间能通过共享 State 通信
- [ ] 冲突能升级到用户（Phase 9 完善）

### 自测用例

```bash
# 测试 1：完整 Multi-Agent 流程
curl -X POST /api/chat -d '{
  "message": "创建一个 FastAPI User API，包含 GET /users 和 POST /users，写完整的 CRUD 和测试"
}'
# 期望：
# Router → Planner（拆解任务）→ Coder（生成代码）
# → Tester（运行测试）→ Coder（修复）→ Tester（通过）
# → Reviewer（Code Review）→ Coder（修改）→ 完成

# 测试 2：Reviewer 发现问题的流程
# 故意让 Coder 生成有问题的代码，验证 Reviewer 能发现并反馈
```

---

## 八、文档差距分析

### 原文档缺失的内容及补充

| # | 原文档问题 | 实际开发需要 | 本文档补充位置 |
|---|-----------|-------------|-------------|
| 1 | **Phase 3 和 Phase 7 的关系模糊** | Phase 3 = 单 Agent 内部工作流，Phase 7 = 多 Agent 拆分。这是两个不同的抽象层。 | §1 |
| 2 | 没说 Agent Router 怎么实现 | 可以规则路由（MVP）或 LLM 路由（进阶）。规则路由简单可靠，LLM 路由灵活但可能分错。 | §5 Step 3 |
| 3 | 没说多 Agent 如何通信 | 方案：共享 StateGraph State（紧耦合，简单）+ Message Queue（松耦合，复杂）。本项目用共享 State。 | §5 Step 4 |
| 4 | **没说如何避免 Agent 冲突** | 分级解决：Agent 协商 → Planner 仲裁 → 用户决策。这是面试重点。 | §5 Step 5 |
| 5 | 没有每个 Agent 的角色定义 | 需要明确：Prompt、可用工具、权限边界、推荐的 LLM 模型。 | §4.1 |
| 6 | 没说不同 Agent 是否用不同模型 | 是——Planner/Coder/Reviewer 用强模型（Sonnet/4o），Tester/Router 可用轻量模型（Haiku/4o-mini）降成本 | §4.1 |

### 为什么需要 Multi-Agent？

**面试答案要素：**
1. **专业分工：** 规划和编码是不同的认知任务，一个 Prompt 难以同时做好
2. **不同模型：** 规划用强推理模型，测试执行用轻量模型 —— 性能/成本最优
3. **安全隔离：** Coder 能写文件，Reviewer 只能读——权限边界清晰
4. **并行执行：** Tester 和 Reviewer 可同时工作（Coder 提交后）

---

## 九、面试考点映射

| 面试题 | 答案要点 | 本文档参考 |
|--------|---------|-----------|
| **为什么需要 Multi-Agent？** | 专业分工、模型分层（强/弱）、安全隔离、并行执行。单 Agent 的 Prompt 太复杂时性能下降。 | §1, §8 |
| **Planner Agent 的职责？** | 接收用户需求 → 理解 → 拆解为子任务 → 确定每个子任务需要的 Agent → 制定执行顺序和依赖。输出结构化任务列表。 | §5 Step 2 |
| **Agent Router 如何实现？** | 两种方案：1) 规则路由（关键词匹配，快且可靠）2) LLM 路由（轻量模型判断意图，灵活但可能误判）。MVP 用规则 + LLM fallback。 | §5 Step 3 |
| **多 Agent 如何通信？** | 1) 共享 StateGraph（紧耦合，本项目用）2) Message Queue（松耦合，适合分布式）3) 黑板模式（Agent 读/写公共知识库）。 | §5 Step 4 |
| **如何避免多个 Agent 冲突？** | 三级冲突解决：Agent 间直接协商 → Planner 仲裁 → 用户决策。关键：定义清晰的 Agent 权限边界（谁能改什么）。 | §5 Step 5 |
