# Claude-Code 类自主编程智能体（AI Agent / AI后端方向）

> **开发手册主索引** — 本文档提供项目全景视图与各阶段索引，详细开发手册见各 Phase 对应的 markdown 文件。

---

## 一、项目最终简历版本

### Claude-Code 类自主编程智能体（独立开发）

**技术栈：Python、FastAPI、LangGraph、OpenAI/Claude API、Redis、Docker、PostgreSQL、MCP、ChromaDB**

- 参考 Claude Code 架构独立实现自主编程 Agent，核⼼为 ReAct + Tool Calling 的 Agent Runtime，⽀持需求分析、任务规划、代码生成、自动测试与错误修复全流程⾃动化。
- 基于 LangGraph StateGraph 构建 Agent Workflow，设计 Planner、Coder、Reviewer、Tester 多节点协作图，实现复杂任务拆解与状态流转，⽀持条件路由（Conditional Edge）实现⾮线性任务执⾏。
- 基于 ReAct 与 Tool Calling 实现 Agent Runtime，支持文件系统、Shell、Git、Web Search、Browser 等 10+ 工具动态编排。
- 设计 Short-term Memory（滑动窗口 + 摘要压缩）、Long-term Memory（ChromaDB 向量检索）与 Reflection Memory（错误嵌入 + 相似经验召回），实现会话摘要、上下文压缩及任务状态持久化。
- 基于 MCP 协议实现插件体系，集成官方 GitHub MCP Server、PostgreSQL MCP Server，并自建 1 个 Demo MCP Server 展示协议理解。
- 基于 Docker Sandbox 构建安全代码执⾏环境，实现 CPU/内存/网络全限制 + 只读文件挂载，防容器逃逸。
- 利用 Redis 实现任务队列与状态管理，通过 Human-in-the-Loop 权限控制实现危险操作审批。
- 支持 Multi-Agent 协同执⾏，通过条件路由（Conditional Routing）与反思机制（Reflection）提升复杂任务完成率。
- 基于 FastAPI + WebSocket 实现实时日志流与任务监控，支持多用户并发执行 Agent 工作流。
- 在自建 10 个编程任务 Benchmark 上任务完成率达 XX%。

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

**伪代码表达：**

```python
async def agent_runtime(user_input: str, state: AgentState) -> str:
    messages = state.messages + [HumanMessage(content=user_input)]
    
    while state.iterations < state.max_iterations:
        # 1. 构建上下文（含裁剪后的历史 + 可用工具定义）
        context = build_context(messages, state.available_tools, state.memory)
        
        # 2. LLM 推理
        response = await llm.ainvoke(context)
        messages.append(response)
        
        # 3. 判断是否需要工具调用
        if response.tool_calls:
            for tc in response.tool_calls:
                # 3a. 危险操作权限检查
                if is_dangerous(tc) and not state.has_permission(tc):
                    approval = await request_user_approval(tc)
                    if not approval:
                        messages.append(ToolMessage(content="User denied"))
                        continue
                
                # 3b. 执行工具
                result = await tool_registry.execute(tc)
                messages.append(ToolMessage(content=str(result), tool_call_id=tc.id))
            
            # 3c. 更新记忆
            await state.memory.add_tool_call(tc, result)
        
        # 4. 判断是否需要用户输入（如 LLM 提出问题）
        elif response.needs_user_clarification:
            user_response = await ask_user(response.content)
            messages.append(HumanMessage(content=user_response))
        
        # 5. 无工具调用 → 任务完成
        else:
            # 触发反思（Reflection）
            await reflection_memory.record(state, success=True)
            return response.content
        
        state.iterations += 1
    
    # 达到最大迭代次数 → 记录反思
    await reflection_memory.record(state, success=False)
    return "Task exceeded max iterations, final state saved."
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
| LangChain 的角色 | **仅用于 ChatModel 抽象 + Prompt 模板**，不⽤ AgentExecutor | AgentExecutor 是黑盒，不可控；LangGraph 给出更细粒度控制 | 直接用 OpenAI/Anthropic SDK |
| 消息队列 | **MVP: Redis List；生产: RabbitMQ** | Redis 轻量快速适合 demo；但无 ACK，消息可能丢失 | RabbitMQ、Kafka、Celery |
| 向量数据库 | **ChromaDB（轻量）→ pgvector（生产）** | ChromaDB 零配置适合 MVP；pgvector 与 PostgreSQL 共存减少运维 | Milvus、Pinecone、Weaviate |
| 多模型支持 | **OpenAI + Claude 双模型** | 覆盖主流 API；展示多 provider 适配能力 | 单一模型、LiteLLM 统一网关 |
| 安全沙箱 | **Docker + seccomp + resource limits** | 成熟稳定；面试能展开安全设计细节 | gVisor、Firecracker、E2B |
| AST 范围 | **仅 Python** | 工作量可控；Python AST 标准库自带；原理通用可类推 | 多语言 AST（tree-sitter） |

---

## 三、技术栈全景图

| 层级 | 技术 | 用途 | 引入 Phase |
|------|------|------|-----------|
| **Web 框架** | FastAPI + WebSocket | REST API + 实时日志推送 | Phase 1 / Phase 9 |
| **Agent 框架** | LangGraph | 工作流编排、状态管理 | Phase 3 |
| **LLM 调用** | LangChain ChatModel + OpenAI/Anthropic SDK | 统一模型调用接口 | Phase 1 |
| **工具系统** | 自建 Tool Registry + MCP | 工具注册/发现/执行 | Phase 2 / Phase 8 |
| **代码执行** | Docker SDK + pytest | 安全隔离执行 + 自动测试 | Phase 5 |
| **任务队列** | Redis (List + Pub/Sub) | 异步任务 + 状态缓存 | Phase 6 / Phase 9 |
| **持久存储** | PostgreSQL + SQLAlchemy | Session / TaskLog 存储 | Phase 9 |
| **向量存储** | ChromaDB | 语义记忆检索 | Phase 6 |
| **代码分析** | Python AST (ast module) + unified_diff | 代码理解 + Diff 生成 | Phase 4 |
| **包管理** | uv (或 Poetry) | 依赖管理 | Phase 1 |
| **部署** | Docker Compose | 一键启动全栈 | Phase 9 |

### 关于 LangChain vs LangGraph 的说明

```
LangChain ──── 仅用于 ChatModel 抽象层 ──── 可选依赖，可用原生 SDK 替代
LangGraph ──── 核心框架，不可替代 ──── 负责 Agent 状态图与工作流编排
```

- **LangChain 在本项目中的角色（最小化）：** `ChatOpenAI` / `ChatAnthropic` 封装、`SystemMessage` / `HumanMessage` / `AIMessage` / `ToolMessage` 类型、`PromptTemplate` 模板。
- **不使用 LangChain 的：** `AgentExecutor`、`create_tool_calling_agent`、Chain 相关组件。
- **为什么：** AgentExecutor 是黑盒，内部 ReAct 循环不可控；LangGraph 让我们自己写循环，精细控制每一步。

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
| 1 | **缺少 ReAct Loop 核心设计** | 🔴 致命 | 在本文档 2.1 节补充完整循环伪代码与流程图 | 本文 §2.1 |
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
| 12 | **LangChain 用途不明确** | 🟢 中等 | 在本文档 3.1 节明确：仅 ChatModel + Message 类型 | 本文 §3 |

---

## 六、面试准备指南

### AI Agent 岗高频考点

| 考点 | 涉及 Phase | 详见 |
|------|-----------|------|
| ReAct vs Chain-of-Thought | Phase 1-2 | [Phase 1](phase-1-mvp.md) §9 |
| Tool Calling 实现原理 | Phase 2 | [Phase 2](phase-2-tool-calling.md) §9 |
| Agent Runtime 核心循环 | Phase 3 | [Phase 3](phase-3-langgraph-workflow.md) §9 |
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
| Agent Runtime | 自研 ReAct Loop | LangGraph StateGraph + 自研 Loop |
| 工具系统 | 自研 Tool Registry | 自研 Tool Registry + MCP |
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
