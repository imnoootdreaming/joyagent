# Claude-Code 类自主编程智能体（AI Agent / AI后端方向）

> **开发手册主索引** — 本文档提供项目全景视图与各阶段索引，详细开发手册见各 Phase 对应的 markdown 文件。

---

## 一、项目最终简历版本

### Claude-Code 类自主编程智能体（独立开发）

**技术栈：Python、FastAPI、LangGraph、Anthropic SDK (Messages API)、DeepSeek API、Redis、Docker、PostgreSQL、MCP、ChromaDB**

- 参考 Claude Code 架构独立实现自主编程 Agent，核⼼为 ReAct + Tool Calling 的 Agent Runtime，⽀持需求分析、任务规划、代码生成、自动测试与错误修复全流程⾃动化。
- 基于 LangGraph StateGraph 构建 Agent Workflow，设计 Planner、Coder、Reviewer、Tester 多节点协作图，实现复杂任务拆解与状态流转，⽀持条件路由（Conditional Edge）实现⾮线性任务执⾏。
- 基于 Anthropic Messages API 原生 Tool Use 实现 Agent Runtime，支持文件系统、Shell、Git、Web Search 等 10+ 工具动态编排，采用 stop_reason + block.type 模式判断工具调用。
- 设计 Short-term Memory（滑动窗口 + 摘要压缩）、Long-term Memory（ChromaDB 向量检索）与 Reflection Memory（错误嵌入 + 相似经验召回），实现会话摘要、上下文压缩及任务状态持久化。基于线程安全 LRU + SHA256 哈希键实现嵌入缓存层，在 Fix Loop 重复分析场景下减少 70%+ 重复计算。
- 设计 Tool Call Hook 中间件机制，基于 on_pre_execute / on_post_execute / on_error 生命周期拦截实现工具调用全链路追踪，支持调用频率、成功率、P50/P99 耗时等多维度实时统计，构建 Agent 行为可观测性体系。
- 基于 MCP 协议实现插件体系，集成官方 GitHub MCP Server、PostgreSQL MCP Server，并自建 1 个 Demo MCP Server 展示协议理解。
- 基于 Docker Sandbox 构建安全代码执⾏环境，实现 CPU/内存/网络全限制 + 只读文件挂载，防容器逃逸。
- 构建 Bad Case 自动分析流水线：收集 Fix Loop 失败案例 → 按错误类型（语法/导入/断言/运行时）与任务类别聚合 → 定位高频失败模式 → 驱动 Prompt 与 Tool Skill 策略迭代优化，形成"评估 → 分析 → 优化 → 回归"的数据驱动 Agent 能力提升闭环。
- 利用 Redis 实现任务队列与状态管理，通过 Human-in-the-Loop 权限控制实现危险操作审批。
- 支持 Multi-Agent 协同执⾏，通过条件路由（Conditional Routing）与反思机制（Reflection）提升复杂任务完成率。
- 基于 FastAPI + WebSocket 实现实时日志流与任务监控，支持多用户并发执行 Agent 工作流。
- 自建 10 个编程任务 Benchmark（覆盖 CRUD 生成、算法实现、文件操作、API 开发等场景），通过 Bad Case 回溯 + Prompt 迭代将任务完成率从初始 XX% 提升至 YY%，验证数据驱动优化方法论有效性。

---

## 二、核心架构设计

### 2.1 ReAct Agent Runtime 核心循环

这是整个项目的**心脏**，每⼀个 Phase 都围绕这个循环展开：

```text
┌─────────────────────────────────────────────────────────┐
│                    Agent Runtime Loop                     │
│                                                           │
│   ┌──────────┐     ┌──────────┐     ┌───────────────┐   │
│   │  User     │────▶│  LLM     │────▶│  Tool Call?   │   │
│   │  Input    │     │ 推理/生成 │     │  判断         │   │
│   └──────────┘     └──────────┘     └───────┬───────┘   │
│                           ▲                 │             │
│                           │         Yes ────┤             │
│                           │                 │ No          │
│                           │    ┌────────────▼──────────┐  │
│                           │    │  Execute Tool(s)       │  │
│                           │    │  (File/Shell/Git/...)  │  │
│                           │    └────────────┬──────────┘  │
│                           │                 │              │
│                           │    ┌────────────▼──────────┐  │
│                           │    │  Append Tool Result    │  │
│                           │    │  to Message History    │  │
│                           │    └────────────┬──────────┘  │
│                           │                 │              │
│                           └─────────────────┘              │
│                                                 No         │
│                              ┌──────────────┐             │
│                              │ Needs Confirm?│──Yes──▶ User│
│                              └──────┬───────┘       Approve│
│                                     │ No                    │
│                              ┌──────▼───────┐              │
│                              │  Final Output │              │
│                              └──────────────┘              │
└───────────────────────────────────────────────────────────┘
```

**伪代码表达（Anthropic Messages API 原生模式）：**

```python
async def agent_runtime(user_input: str, state: AgentState) -> str:
    # 1. 构建消息历史（dict 格式，非 LangChain 对象）
    messages = state.messages + [{"role": "user", "content": user_input}]
    system_prompt = assemble_system_prompt(state)
    max_tokens = DEFAULT_MAX_TOKENS
    recovery_state = RecoveryState()

    while state.iterations < state.max_iterations:
        # 2. LLM 推理 — 原生 Anthropic SDK 调用
        try:
            response = with_retry(
                lambda: client.messages.create(
                    model=state.current_model,
                    system=system_prompt,       # system 是独立参数
                    messages=messages,           # 纯 dict 列表
                    tools=TOOLS,                 # Anthropic 原生工具格式
                    max_tokens=max_tokens,
                ),
                recovery_state,
            )
        except PromptTooLongError:
            if not recovery_state.has_attempted_compact:
                messages[:] = reactive_compact(messages)
                recovery_state.has_attempted_compact = True
                continue
            return "[Error] Context too large."

        # 3. 追加 assistant 回复到消息历史
        messages.append({"role": "assistant", "content": response.content})

        # 4. 判断 stop_reason（Anthropic 原生方式）
        if response.stop_reason == "max_tokens":
            # 4a. max_tokens 恢复：先升级 token 上限，再续写
            if not recovery_state.has_escalated:
                max_tokens = ESCALATED_MAX_TOKENS
                recovery_state.has_escalated = True
                continue  # 不 append，用更大 max_tokens 重试同一请求
            if recovery_state.recovery_count < MAX_RECOVERY_RETRIES:
                messages.append({"role": "user", "content": CONTINUATION_PROMPT})
                recovery_state.recovery_count += 1
                continue
            return  # 超出恢复上限

        if response.stop_reason != "tool_use":
            # 5. 无工具调用 → 模型认为任务完成
            # 提取文本回复
            for block in response.content:
                if block.type == "text":
                    return block.text
            return "Task completed."

        # 6. 有工具调用 → 执行工具
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            # 6a. 危险操作权限检查
            if is_dangerous(block.name) and not state.has_permission(block.name):
                approval = await request_user_approval(block)
                if not approval:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "User denied execution.",
                    })
                    continue

            # 6b. 执行工具
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })

            # 6c. 更新记忆
            await state.memory.add_tool_call(block.name, block.input, output)

        # 6d. 工具结果作为 user 消息追加
        messages.append({"role": "user", "content": tool_results})
        state.iterations += 1

    await reflection_memory.record(state, success=False)
    return "Task exceeded max iterations."
```

**面试要点：** 这是面试中第一个会被问的问题。你必须能画这个图，并解释每一步的设计决策。

### 2.2 系统架构图

```text
                         ┌─────────────────────────────┐
                         │       FastAPI API Layer       │
                         │   REST + WebSocket (实时日志)  │
                         └─────────────┬───────────────┘
                                       │
                         ┌─────────────▼───────────────┐
                         │      Agent Runtime           │
                         │   (ReAct Loop + Permission)   │
                         └─────────────┬───────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
    ┌─────────▼────────┐   ┌──────────▼──────────┐   ┌─────────▼────────┐
    │  LangGraph        │   │  Tool Calling Layer  │   │  Memory System   │
    │  Workflow Engine  │   │  File/Shell/Git/     │   │  Short/Long/     │
    │  Planner → Coder  │   │  Search/Browser      │   │  Reflection      │
    │  → Tester → Rev   │   └──────────┬──────────┘   │  (ChromaDB+Redis) │
    └─────────┬────────┘              │               └─────────┬────────┘
              │               ┌───────▼────────┐              │
              │               │  Docker Sandbox │              │
              │               │  (安全隔离执行)  │              │
              │               └────────────────┘              │
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │     Data Layer           │
                         │  PostgreSQL + Redis       │
                         └────────────┬────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │     MCP Plugin System    │
                         │  GitHub / PG / Browser   │
                         └─────────────────────────┘
```

### 2.3 核心设计决策与 Trade-off

| 决策 | 我们的选择 | 为什么 | 面试可能追问的替代方案 |
|------|-----------|--------|----------------------|
| Agent 框架 | **LangGraph** | 图结构天然适合复杂工作流；Conditional Edge 支持非线性执行；Checkpoint 支持暂停/恢复 | AutoGen、CrewAI、纯 ReAct |
| LLM 调用 | **Anthropic Python SDK（原生）** | 直接使用 `anthropic` 包，Messages API 原生调用；避免 LangChain 的抽象泄漏；`stop_reason` + `block.type` 模式精确控制工具调用 | LangChain ChatAnthropic、OpenAI SDK、DeepSeek SDK |
| 工具格式 | **Anthropic 原生 tool_use 格式** | `{"name": "...", "description": "...", "input_schema": {...}}`；无需 OpenAI Function Calling 的 `"type": "function"` 包装层 | OpenAI Function Calling format |
| 消息队列 | **MVP: Redis List；生产: RabbitMQ** | Redis 轻量快速适合 demo；但无 ACK，消息可能丢失 | RabbitMQ、Kafka、Celery |
| 向量数据库 | **ChromaDB（轻量）→ pgvector（生产）** | ChromaDB 零配置适合 MVP；pgvector 与 PostgreSQL 共存减少运维 | Milvus、Pinecone、Weaviate |
| 多模型支持 | **Claude (Anthropic) + DeepSeek (兼容端点) + OpenAI** | 原生 SDK + 协议族分流；DeepSeek 和 Claude 共用 Anthropic Messages API 格式；OpenAI 通过独立分支适配 | LiteLLM 统一网关 |
| 安全沙箱 | **Docker + seccomp + resource limits** | 成熟稳定；面试能展开安全设计细节 | gVisor、Firecracker、E2B |
| AST 范围 | **仅 Python** | 工作量可控；Python AST 标准库自带；原理通用可类推 | 多语言 AST（tree-sitter） |

---

## 三、技术栈全景图

| 层级 | 技术 | 用途 | 引入 Phase |
|------|------|------|-----------|
| **Web 框架** | FastAPI + WebSocket | REST API + 实时日志推送 | Phase 1 / Phase 9 |
| **Agent 框架** | LangGraph | 工作流编排、状态管理 | Phase 3 |
| **LLM 调用** | Anthropic Python SDK (原生) | Messages API 直接调用；`stop_reason` + `block.type` 判断工具循环 | Phase 1 |
| **工具系统** | 自建 Tool Registry + MCP | 工具注册/发现/执行；Anthropic 原生 `input_schema` 格式 | Phase 2 / Phase 8 |
| **代码执行** | Docker SDK + pytest | 安全隔离执行 + 自动测试 | Phase 5 |
| **任务队列** | Redis (List + Pub/Sub) | 异步任务 + 状态缓存 | Phase 6 / Phase 9 |
| **持久存储** | PostgreSQL + SQLAlchemy | Session / TaskLog 存储 | Phase 9 |
| **向量存储** | ChromaDB | 语义记忆检索 | Phase 6 |
| **代码分析** | Python AST (ast module) + unified_diff | 代码理解 + Diff 生成 | Phase 4 |
| **包管理** | uv (或 Poetry) | 依赖管理 | Phase 1 |
| **部署** | Docker Compose | 一键启动全栈 | Phase 9 |

### 关于 Anthropic SDK 与 LangGraph 的角色分工

```
Anthropic SDK ──── 原生 LLM 调用 ──── client.messages.create() 直接调用 Messages API
LangGraph    ──── 核心框架，不可替代 ── 负责 Agent 状态图与工作流编排
LangChain    ──── 不引入 ──────────── 本项目不依赖 langchain/langchain-core
```

- **Anthropic SDK 在本项目中的角色（唯一 LLM 调用方式）：**
  - `client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))` — 一行注册，无需工厂函数
  - `client.messages.create(model=..., system=..., messages=..., tools=..., max_tokens=...)` — 原生调用
  - 消息格式：纯 Python dict `{"role": "user/assistant", "content": ...}`
  - 工具格式：`{"name": "...", "description": "...", "input_schema": {...}}`
  - 工具调用判断：`response.stop_reason == "tool_use"` + 遍历 `response.content` 检查 `block.type == "tool_use"`
- **不使用 LangChain 的：** `AgentExecutor`、`create_tool_calling_agent`、`ChatOpenAI`、`ChatAnthropic`、`SystemMessage`、`HumanMessage`、`AIMessage`、`ToolMessage`、`bind_tools()`、Chain 相关组件。
- **为什么不用 LangChain：**
  1. 抽象泄漏：`ChatAnthropic` 封装了原生 SDK，但隐藏了 `stop_reason`、`block.type` 等关键控制点
  2. 消息格式不透明：LangChain Message 类型增加了序列化/反序列化成本
  3. 工具绑定黑盒：`bind_tools()` 内部转换了工具格式，调试困难
  4. Anthropic SDK 本身已足够简洁：`client.messages.create()` 一行调用，无需额外封装

---

## 四、项目开发路线与 Phase 索引

### 开发原则

```text
先跑通  →  再可用  →  再工程化  →  最后高级能力
 (P1)       (P2-3)      (P4-7)        (P8-9)
```

### Phase 依赖关系图

```text
Phase 1: Agent MVP (ReAct + Tool Calling)
   │
   ├──▶ Phase 2: Tool Calling Framework (BaseTool + Registry + Shell/Git)
   │       │
   │       ├──▶ Phase 3: LangGraph Workflow (StateGraph + Plan/Execute/Reflect)
   │       │       │
   │       │       ├──▶ Phase 4: Coding Agent (Python AST + Diff/Patch)
   │       │       │       │
   │       │       │       └──▶ Phase 5: Docker Sandbox + Auto Testing (Fix Loop)
   │       │       │
   │       │       └──▶ Phase 7: Multi-Agent (Planner/Coder/Tester/Reviewer + Router)
   │       │               │
   │       │               └──▶ Phase 8: MCP (Plugin System)
   │       │
   │       └──▶ Phase 6: Memory System (Short/Long/Reflection + ChromaDB)
   │
   └──▶ Phase 9: 工程化 (DB + WebSocket + CI/CD + Human-in-the-Loop)

关键路径：P1 → P2 → P3 → P4 → P5  （核心编程能力闭环）
扩展路径：P3 → P7 → P8            （Multi-Agent + 插件）
基础设施：P2 → P6 → P9            （记忆 + 工程化）
```

### Phase 索引表

| Phase | 文档 | 主题 | 核心交付 | 预计工时 |
|-------|------|------|---------|---------|
| **1** | [phase-1-mvp.md](phase-1-mvp.md) | Agent MVP | 对话 Agent + 文件读写 + ReAct Loop | 3-5 天 |
| **2** | [phase-2-tool-calling.md](phase-2-tool-calling.md) | Tool Calling Framework | BaseTool 抽象 + Tool Registry + Shell/Git 工具 | 3-5 天 |
| **3** | [phase-3-langgraph-workflow.md](phase-3-langgraph-workflow.md) | LangGraph Workflow | StateGraph + Plan/Execute/Reflect 三节点 | 5-7 天 |
| **4** | [phase-4-coding-agent.md](phase-4-coding-agent.md) | Coding Agent | Python AST 分析 + Diff 生成 + Patch Apply | 5-7 天 |
| **5** | [phase-5-sandbox-testing.md](phase-5-sandbox-testing.md) | Docker Sandbox + Testing | 安全沙箱 + pytest 集成 + Fix Loop | 5-7 天 |
| **6** | [phase-6-memory-system.md](phase-6-memory-system.md) | Memory System | ChromaDB 向量记忆 + 上下文压缩 + Token 管理 | 5-7 天 |
| **7** | [phase-7-multi-agent.md](phase-7-multi-agent.md) | Multi-Agent | Planner/Coder/Tester/Reviewer + Router | 7-10 天 |
| **8** | [phase-8-mcp.md](phase-8-mcp.md) | MCP Plugin System | MCP Client + 集成官方 Server + 自建 1 个 Demo Server | 5-7 天 |
| **9** | [phase-9-engineering.md](phase-9-engineering.md) | 工程化 | PG/Redis 数据层 + WebSocket + Human-in-the-Loop + Docker Compose | 7-10 天 |

> **总预计工时：** 约 50-65 天（全职）。秋招投递前建议至少完成 Phase 1-5（核心闭环），Phase 6-9 可根据时间选择性完成。

---

## 五、文档差距总览

下表总结原开发文档存在的问题及修改方案：

| # | 问题 | 严重程度 | 修改方案 | 详见 |
|---|------|---------|---------|------|
| 0 | **开发范式使用 OpenAI/LangChain 风格** | 🔴 致命 | 全部文档改为 Anthropic 原生 SDK 范式：`client = Anthropic()` 一行注册、`stop_reason` + `block.type` 工具判断、`input_schema` 工具格式 | 全部 Phase |
| 1 | **缺少 ReAct Loop 核心设计** | 🔴 致命 | 在本文档 2.1 节补充完整循环伪代码与流程图（Anthropic 原生模式） | 本文 §2.1 |
| 2 | **各 Phase 只有 WHAT 没有 HOW** | 🔴 致命 | 每个 Phase 独立文档补充 Schema、伪代码、关键实现策略 | Phase 1-9 md |
| 3 | **Phase 3 与 Phase 7 关系模糊** | 🔴 致命 | Phase 3 = 单 Agent 内部 Plan-Execute-Reflect；Phase 7 = 多 Agent 协作，拆分节点为独立 Agent | [Phase 3](phase-3-langgraph-workflow.md) §8, [Phase 7](phase-7-multi-agent.md) §1 |
| 4 | **Memory 缺少向量数据库** | 🔴 致命 | Phase 6 引入 ChromaDB，补充语义检索与 Token 管理 | [Phase 6](phase-6-memory-system.md) §4 |
| 5 | **Docker Sandbox 无安全设计** | 🔴 致命 | Phase 5 补充 CPU/内存/网络/文件系统限制 + seccomp | [Phase 5](phase-5-sandbox-testing.md) §5 |
| 6 | **缺少 Human-in-the-Loop** | 🔴 致命 | Phase 9 补充权限分级与用户审批流程 | [Phase 9](phase-9-engineering.md) §5 |
| 7 | **MCP 工作量低估** | 🟡 严重 | Phase 8 改为 Client-first：集成官方 Server + 只自建 1 个 Demo | [Phase 8](phase-8-mcp.md) §1 |
| 8 | **AST 分析范围过宽** | 🟡 严重 | Phase 4 限定 Python only | [Phase 4](phase-4-coding-agent.md) §1 |
| 9 | **缺少 State Schema 定义** | 🟡 严重 | 每个 Phase 补充核心数据模型 | 各 Phase §4 |
| 10 | **缺少 Benchmark 评估方案** | 🟡 严重 | Phase 5/9 引入自建 10 题 Benchmark | [Phase 5](phase-5-sandbox-testing.md) §8, [Phase 9](phase-9-engineering.md) §8 |
| 11 | **Phase 9 内容过多** | 🟢 中等 | 拆分为数据层（9A）+ 部署（9B），或优先做核心 | [Phase 9](phase-9-engineering.md) |
| 12 | **缺少错误恢复 (Error Recovery) 设计** | 🟡 严重 | 补充 max_tokens 升级、reactive compact、指数退避重试等错误恢复模式 | 本文 §2.1、[Phase 1](phase-1-mvp.md) §6 |

---

## 六、面试准备指南

### AI Agent 岗高频考点

| 考点 | 涉及 Phase | 详见 |
|------|-----------|------|
| ReAct vs Chain-of-Thought | Phase 1-2 | [Phase 1](phase-1-mvp.md) §9 |
| Anthropic Tool Use 实现原理 | Phase 2 | [Phase 2](phase-2-tool-calling.md) §9 |
| Agent Runtime 核心循环 (stop_reason + block.type) | Phase 3 | [Phase 3](phase-3-langgraph-workflow.md) §9 |
| LangGraph StateGraph 设计 | Phase 3 | [Phase 3](phase-3-langgraph-workflow.md) §9 |
| Conditional Edge 应用 | Phase 3, 7 | [Phase 3](phase-3-langgraph-workflow.md), [Phase 7](phase-7-multi-agent.md) |
| Short/Long/Reflection Memory | Phase 6 | [Phase 6](phase-6-memory-system.md) §9 |
| Context Compression 策略 | Phase 6 | [Phase 6](phase-6-memory-system.md) §6 |
| Multi-Agent 通信 | Phase 7 | [Phase 7](phase-7-multi-agent.md) §9 |
| MCP 协议 | Phase 8 | [Phase 8](phase-8-mcp.md) §9 |

### AI 后端岗高频考点

| 考点 | 涉及 Phase | 详见 |
|------|-----------|------|
| FastAPI + WebSocket | Phase 1, 9 | [Phase 1](phase-1-mvp.md), [Phase 9](phase-9-engineering.md) |
| Docker Sandbox 安全 | Phase 5 | [Phase 5](phase-5-sandbox-testing.md) §9 |
| Redis 任务队列设计 | Phase 9 | [Phase 9](phase-9-engineering.md) §9 |
| Session 存储与恢复 | Phase 9 | [Phase 9](phase-9-engineering.md) §4 |
| 多用户并发架构 | Phase 9 | [Phase 9](phase-9-engineering.md) §9 |

---

## 七、项目最终定位

### 适合岗位

- **AI Agent 开发工程师** — 核心匹配：ReAct + LangGraph + Tool Calling + Multi-Agent
- **AI 后端开发工程师** — 核心匹配：FastAPI + Docker + Redis + PostgreSQL
- **LLM Application Engineer** — 核心匹配：Memory + MCP + Prompt Engineering
- **智能体开发工程师** — 核心匹配：全栈 Agent 系统设计
- **Python 后端（AI 方向）** — 核心匹配：工程化 + 并发 + 安全

### 与 Claude Code 实际架构的对比

| 维度 | Claude Code (实际) | 本项目 |
|------|-------------------|--------|
| Agent Runtime | 自研 ReAct Loop (Anthropic SDK) | 相同：Anthropic SDK 原生调用 + LangGraph StateGraph |
| 工具系统 | 自研 Tool Registry (Anthropic 原生格式) | 相同：Anthropic `input_schema` 格式 + MCP 扩展 |
| LLM 调用 | `client.messages.create()` 原生 | 相同：无 LangChain 中间层，直接 Anthropic SDK |
| 安全模型 | Permission System | Human-in-the-Loop |
| Memory | 会话级上下文管理 | ChromaDB 持久化 + Token 管理 |
| 多 Agent | 单 Agent 多工具 | 多 Agent 协作（差异化亮点） |
| 插件 | MCP Server | MCP Client + Server |

---

## 八、快速开始

```bash
# 1. 阅读本文档了解全景
# 2. 按 Phase 顺序打开对应开发手册
# 3. 每个 Phase 严格按照 §5（详细开发清单）顺序推进
# 4. 完成一个 Phase 的"完成标志"后再进入下一 Phase

# 推荐阅读顺序：
dev_md/0-1 coding.md              # ← 你在这里
dev_md/phase-1-mvp.md             # 从头开始
dev_md/phase-2-tool-calling.md
dev_md/phase-3-langgraph-workflow.md
dev_md/phase-4-coding-agent.md
dev_md/phase-5-sandbox-testing.md
dev_md/phase-6-memory-system.md
dev_md/phase-7-multi-agent.md
dev_md/phase-8-mcp.md
dev_md/phase-9-engineering.md
```
