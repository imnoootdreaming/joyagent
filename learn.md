# JoyAgent — AI Agent 开发面试通关手册

> 本文档对应项目 `joyagent` 的全部 9 个 Phase，按面试叙述逻辑组织，
> 每一章回答一个面试高频问题。附带真实面试真题及来源链接。

---

## 目录

1. [整体架构：你的 Agent 系统长什么样？](#ch1)
2. [ReAct 核心循环：Agent 怎么实现自主决策的？](#ch2)
3. [Tool Calling：工具调用怎么做的？](#ch3)
4. [LangGraph 工作流：Plan→Execute→Reflect 怎么编排的？](#ch4)
5. [Docker Sandbox：怎么防止 Agent 执行 rm -rf /？](#ch5)
6. [Memory System：四层压缩 + 三级记忆怎么设计的？](#ch6)
7. [Multi-Agent + Mailbox：多 Agent 怎么通信的？](#ch7)
8. [MCP Plugin System：MCP 协议怎么接入的？](#ch8)
9. [Human-in-the-Loop + 工程化：生产级 Agent 还需要什么？](#ch9)
10. [大厂 Agent 面试真题 28 题（含来源链接）](#ch10)

---

## <a id="ch1"></a>一、整体架构：你的 Agent 系统长什么样？

### 一句话回答

**六层架构：API 层 → Agent Runtime → LangGraph 工作流 → Tool Calling 层 → Docker Sandbox → 数据持久化层，外加 MCP 插件系统和 Human-in-the-Loop 权限。**

### 架构图（面试可画）

```
┌──────────────────────────────────────────────────────────┐
│                    FastAPI API Layer                     │
│              POST /api/chat  /api/multi-agent            │
│              GET  /api/multi-agent/status                │
│              GET  /api/multi-agent/audit                 │
├──────────────────────────────────────────────────────────┤
│                   Agent Runtime                          │
│         ReAct Loop (Anthropic Messages API 原生)          │
│         stop_reason + block.type 判断工具调用             │
├──────────────────────┬───────────────────────────────────┤
│  LangGraph Workflow  │  Multi-Agent + Mailbox (Phase 7)  │
│  Plan→Execute→Reflect│  Router→Planner→Coder→Tester→Rev  │
├──────────────────────┴───────────────────────────────────┤
│               Tool Calling Layer                         │
│  File/Shell/Git (Phase 2) + MCP Adapter (Phase 8)        │
│  Hook 中间件: SafetyCheck → PermissionManager → Stats    │
├──────────────────────────────────────────────────────────┤
│              Docker Sandbox (Phase 5)                    │
│  进程隔离 + CPU/内存限制 + 网络隔离 + 只读根 + 非root     │
├──────────────────────────────────────────────────────────┤
│              Data Layer                                  │
│  SQLite/PG Sessions (Phase 9) + Redis Queue (Phase 9)    │
│  ChromaDB Vector Memory (Phase 6) + FilePersistence (P7) │
└──────────────────────────────────────────────────────────┘
```

### 关键设计决策

| 问题 | 选择 | 为什么 |
|------|------|--------|
| LLM 调用方式 | Anthropic SDK 原生，不用 LangChain | `stop_reason` + `block.type` 能精确控制工具调用，LangChain 把这层抽象掉了 |
| 工作流编排 | LangGraph StateGraph | 图结构天然适合 Plan→Execute→Reflect 非线性流程，Conditional Edge 支持条件路由 |
| 单 Agent vs 多 Agent | 两者都有 | Phase 3 用单 Agent + LangGraph 做 Plan/Execute/Reflect；Phase 7 用 6 个独立 Agent + Mailbox 通信 |
| 向量数据库 | ChromaDB | 零配置，轻量级。生产可切 pgvector |
| 消息队列 | MVP 用内存 FilePersistence；生产 Redis | Redis List 做任务队列；Redis 不可用自动降级 Mock |

---

## <a id="ch2"></a>二、ReAct 核心循环：Agent 怎么实现自主决策的？

### 一句话回答

**调用 LLM → 检查 stop_reason → 如果有 tool_use 就执行工具 → 把工具结果追加到 messages → 再次调用 LLM，直到 stop_reason 不再是 tool_use。**

### 核心代码逻辑（面试时要能讲清楚）

```python
while iterations < max_iterations:
    # 1. 调用 LLM（Anthropic 原生 SDK）
    response = client.messages.create(
        model=model,
        system=system_prompt,
        messages=messages,        # 纯 Python dict 列表，不用 LangChain Message
        tools=tool_schemas,        # Anthropic 原生 format: {"name","description","input_schema"}
        max_tokens=4096,
    )

    # 2. 追加 assistant 回复
    messages.append({"role": "assistant", "content": response.content})

    # 3. Anthropic 原生 stop_reason 判断（不用 tool_calls 字段）
    if response.stop_reason == "max_tokens":
        # 输出被截断 → 升级 max_tokens 重试 / 追加续写 prompt
        max_tokens = 8192; continue

    if response.stop_reason != "tool_use":
        # 没有工具调用 → LLM 认为任务完成 → 返回文本
        return extract_text(response.content)

    # 4. 执行工具调用
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            # Hook 链: SafetyCheck → PermissionManager → 执行
            result = await tool_registry.execute(block.name, **block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result.message,
            })

    # 5. 工具结果作为 user 消息追加 → 循环继续
    messages.append({"role": "user", "content": tool_results})
```

### 面试追问应对

> "为什么不用 `response.tool_calls` 而是用 `stop_reason + block.type`？"

因为 Anthropic Messages API 的响应结构是 `content: list[ContentBlock]`，每个 block 有 `type: "text" | "tool_use"`。`stop_reason == "tool_use"` 告诉你有工具要调，遍历 blocks 找 `type=="tool_use"` 拿到具体调用。这是 Anthropic 原生 API 的设计方式，不经过 LangChain 的任何转换。

---

## <a id="ch3"></a>三、Tool Calling：工具调用怎么做的？

### 一句话回答

**BaseTool 抽象 + ToolRegistry 单例 + Anthropic 原生 `input_schema` 格式 + Hook 链（SafetyCheck → PermissionManager → StatsCollector）。**

### 核心设计

```
BaseTool (ABC)
  ├── name: str              # "read_file"
  ├── description: str        # "读取文件内容"
  ├── input_schema: dict      # {"type":"object","properties":{...}}
  ├── execute(**kwargs) -> ToolResult
  └── is_dangerous: bool      # True = 需要安全审查

ToolRegistry (单例)
  ├── 注册: register_tool(tool)
  ├── 发现: get_tool_schemas() → 传给 LLM 的 tools 参数
  ├── 执行: execute(name, **kwargs) → 走 Hook 链
  └── Hook: SafetyCheckHook → PermissionManager → ToolStatsCollector
```

### 14 个内置工具

| 类别 | 工具 |
|------|------|
| 文件 | read_file, write_file |
| Shell | execute_shell（危险） |
| Git | git_status, git_diff, git_log, git_branch, git_commit（危险） |
| 代码 | load_repo, search_code, analyze_code, generate_diff, apply_patch（危险） |
| 记忆 | remember |

### 面试追问应对

> "MCP 工具和本地工具有什么区别？"

通过 MCPToolAdapter 实现 BaseTool 的 duck typing 接口。Agent 调用时无感知——`github__search_repositories` 和 `read_file` 都是同一个 `tool_registry.execute(name, **kwargs)`。

---

## <a id="ch4"></a>四、LangGraph 工作流：Plan→Execute→Reflect 怎么编排的？

### 一句话回答

**用 LangGraph StateGraph 构建 Plan→Execute→Reflect 三节点图，Conditional Edge 实现非线性任务路由，Checkpoint 支持暂停/恢复。**

### 工作流

```
          ┌─────────┐
          │  START  │
          └────┬────┘
               │
          ┌────▼────┐      ┌──────────┐
          │ Planner │─────▶│ Executor │
          │ (拆解任务)│      │ (执行步骤)│
          └─────────┘      └────┬─────┘
                               │
                     condition: step_index >= len(plan)?
                         │              │
                        Yes            No
                         │              │
                    ┌────▼────┐    ┌───▼──────┐
                    │Reflector│    │ Executor │
                    │(反思评估) │    │ (下一工具) │
                    └────┬────┘    └──────────┘
                         │
              condition: task_completed?
                  │           │
                 Yes         No
                  │           │
              ┌───▼──┐  ┌───▼────────┐
              │ END  │  │ need_replan?│
              └──────┘  └──┬────┬─────┘
                           Yes  No
                            │    │
                       ┌────▼─┐ ┌▼────────┐
                       │Planner│ │Executor │
                       └──────┘ └─────────┘
```

### 与 Phase 7 Multi-Agent 的关系（面试关键区分）

| 维度 | Phase 3 LangGraph | Phase 7 Multi-Agent |
|------|------------------|---------------------|
| 执行者 | 同一个 LLM 扮演所有角色 | 6 个独立 Agent 实例 |
| 通信 | StateGraph 内部 state 流转 | Mailbox 异步消息（Inbox/Outbox） |
| 适用场景 | 单个用户请求的完整执行 | 多 Agent 长期协作 |

---

## <a id="ch5"></a>五、Docker Sandbox：怎么防止 Agent 执行 rm -rf /？

### 一句话回答

**六层防线：进程隔离 + 资源限制 + 网络隔离 + 只读根文件系统 + 权限限制（非root + cap_drop=ALL） + 超时销毁 + Human-in-the-Loop 审批。**

### 六层防线详解

| 层级 | 措施 | 防什么 |
|------|------|--------|
| 1. 进程隔离 | 每次命令新建 Docker 容器 | Agent 的 `rm -rf /` 只能删容器内的文件 |
| 2. 资源限制 | CPU 1 核 + 内存 512M + 禁 swap | 死循环/内存泄漏不拖垮宿主机 |
| 3. 网络隔离 | `network_mode=none` | 防止下载恶意脚本、泄露数据 |
| 4. 文件系统 | 根文件只读 + 仅挂载 /workspace | 无法修改系统文件 |
| 5. 权限限制 | `no_new_privileges` + `cap_drop=ALL` + `user=sandbox` | 禁止提权 |
| 6. 超时控制 | 每次 60s 超时 + 容器用完即销毁 | 僵尸进程不堆积 |

### 附加：Human-in-the-Loop

```
SafetyCheckHook (关键词黑名单: rm -rf, sudo, mkfs → 直接拒绝)
     │
PermissionManager (三级权限: AUTO / CONFIRM / DENY)
     │
ToolStatsCollector (统计记录)
```

---

## <a id="ch6"></a>六、Memory System：Agent 的记忆怎么设计的？

### 一句话回答

**三级记忆 × 四层压缩管线。记忆 = Short-term（四层压缩 + 递进式摘要）+ Long-term（ChromaDB 向量检索）+ Reflection（错误嵌入 + 相似经验召回）。四层压缩参考 Claude Code 标准："便宜的先跑，贵的后跑"。**

### 四层上下文压缩管线（核心亮点 ⭐）

Claude Code 源码 `compact.ts` / `autoCompact.ts` 揭示的设计原则：**每轮 LLM 调用前跑三层 0 API 预处理器，还不够才调 LLM 做摘要。**

```
每轮 LLM 调用前:
  ┌─────────────────────────────────────────────────────────┐
  │ L3: tool_result_budget  → 大结果(>200KB)落盘,留预览     │ 0 API
  │ L1: snip_compact        → 裁掉中间无关对话(头3+尾47)     │ 0 API
  │ L2: micro_compact       → 旧工具结果换为占位符(保留最近3) │ 0 API
  │                         ↓                              │
  │             token 仍超阈值?                              │
  │              Yes → L4: compress (1 API 递进式摘要)       │
  │              API报413? → reactive_truncate (应急截断)    │
  └─────────────────────────────────────────────────────────┘
```

| 层 | 名称 | API 成本 | 做什么 | 面试表达 |
|---|------|---------|--------|---------|
| **L3** | tool_result_budget | 0 | 单条 user 消息的 tool_result 总大小 > 200KB → 按大小排序从最大的开始落盘到 `data/tool_outputs/`，上下文只留 `<persisted-output>` 标记 + 前 2000 字符预览 | "防止一个 `cat` 大文件就打满上下文" |
| **L1** | snip_compact | 0 | 消息数 > 50 → 保留头 3 条（初始上下文）+ 尾 47 条（当前工作），中间裁掉。切口保护：不拆开 `assistant(tool_use)` + `user(tool_result)` 对 | "80 轮对话里前 30 轮和当前工作无关，裁掉" |
| **L2** | micro_compact | 0 | 只保留最近 3 条 `tool_result` 完整内容，更旧的替换为 `[Earlier tool result compacted. Re-run if needed.]` | "Agent 连续读了 10 个文件的完整内容，前 7 次早就不需要了" |
| **L4** | compress (递进式摘要) | **1 API** | 消息二分，前半用 LLM 生成递进式摘要（合并已有摘要），丢弃前半保留后半。O(1) 而非 O(n) | "不是重写全书，是在笔记本后面追加新的一页" |
| **应急** | reactive_truncate | 0 | API 报 413 时暴力截断到最后 10 条 | "上下文增长快于压缩速度时的最后防线" |

**执行顺序不能换**：L3(budget) 必须在 L2(micro) 前面——因为 L2 会把旧的大 tool_result 替换成一行占位符，L3 必须在被替换前把完整内容落盘。这也是 Claude Code 源码 `query.ts` 的真实顺序。

### 三级记忆架构

```
┌─────────────────────────────────────────────────────┐
│              MemoryManager (统一入口)                │
├─────────────────┬─────────────────┬─────────────────┤
│  ShortTermMemory│ LongTermMemory  │ ReflectionMemory│
│  (STM)           │  (LTM)          │                 │
│                  │                 │                 │
│  四层压缩管线    │  ChromaDB       │  错误嵌入       │
│  + Token管理     │  + 向量检索      │  + 相似召回     │
│  + 递进式摘要    │  + 三类Collection│  + 经验复用     │
│                  │  (code/conv/task)│                 │
└─────────────────┴─────────────────┴─────────────────┘
```

### 关键机制

| 机制 | 何时触发 | 做什么 |
|------|---------|--------|
| L3 budget | 每轮 LLM 调用前 | 大工具结果落盘到 data/tool_outputs/ |
| L1 snip | 每轮 LLM 调用前 | 消息 > 50 条时裁掉中间无关对话 |
| L2 micro | 每轮 LLM 调用前 | 旧 tool_result 替换为占位符 |
| L4 递进式摘要 | L1-L3 跑完 token 仍超 80% 阈值 | LLM 合并摘要 + 丢弃旧消息 |
| Token 计数 | 每次 LLM 调用前 | 检查是否接近模型 Context Window 上限 |
| 向量检索 | begin_session 时 | 用 embedding 检索最相似的历史经验 |
| 自动记存 | 每次工具调用后 | 工具调用记录异步写入 ChromaDB |

---

## <a id="ch7"></a>七、Multi-Agent + Mailbox：多 Agent 怎么通信的？

### 一句话回答

**不是共享 State 的 `messages.append()`——每个 Agent 有独立 Inbox/Outbox，通过 MailboxManager 做消息路由。每条消息有 type、priority、TTL、correlation_id。**

### Mailbox 通信系统

```
                   ┌────────────────────────────┐
                   │      MailboxManager         │  ← 消息路由中枢
                   │  ┌──────┐ ┌──────┐ ┌──────┐ │
                   │  │Inbox │ │Inbox │ │Inbox │ │  ← 每 Agent 独立收件箱
                   │  └──┬───┘ └──┬───┘ └──┬───┘ │
                   └─────┼───────┼───────┼──────┘
                         ▲       ▲       ▲
            ┌────────────┤       │       ├────────────┐
       ┌────┴─────┐  ┌────┴────┐ ┌────┴────┐  ┌──────┴──────┐
       │  Router  │  │ Planner │ │  Coder  │  │  Reviewer   │
       │  (入口)   │  │ (拆解)  │ │ (编码)  │  │  (审查)     │
       └──────────┘  └─────────┘ └─────────┘  └─────────────┘
```

### 消息数据模型

| 字段 | 作用 |
|------|------|
| `msg_type` | TASK_ASSIGNMENT / TASK_RESULT / REVIEW_FEEDBACK / CONFLICT_ESCALATE / ERROR_REPORT |
| `priority` | URGENT(10) > HIGH(5) > NORMAL(3) > LOW(1) |
| `correlation_id` | 同任务的所有消息共享此 ID，可完整追链 |
| `ttl_seconds` | 过期自动进死信队列 |
| `status` | DRAFT → SENT → DELIVERED → READ → PROCESSED |

### 与 v1 共享 State 方案的对比

| 场景 | 共享 State | Mailbox |
|------|-----------|---------|
| Coder→Tester 通知 | `state["messages"].append(...)` | `outbox.send(TASK_RESULT → router)` |
| 并行分发 | ❌ 不支持 | ✅ Router 同时 send 给 Coder+Tester |
| 冲突升级 | ❌ 无机制 | CONFLICT_ESCALATE + URGENT 优先级跳队 |
| 审计追踪 | grep messages | audit_log + correlation_id |
| 消息丢失 | 丢了就丢了 | TTL + 死信队列 + retry |

---

## <a id="ch8"></a>八、MCP Plugin System：MCP 协议怎么接入的？

### 一句话回答

**纯 Python 实现 JSON-RPC 2.0 over stdio，不用第三方 mcp SDK。MCPClient 通过子进程 stdin/stdout 与 MCP Server 通信。MCPToolAdapter 将 MCP 工具适配为 Phase 2 的 BaseTool。**

### MCP 协议流程

```
Client (JoyAgent)                      Server (外部进程)
     │                                      │
     │── {"method":"initialize"} ──────────▶│  握手 + 获取 capabilities
     │◀── {"result":{"serverInfo":...}} ────│
     │── {"method":"tools/list"} ──────────▶│  发现工具
     │◀── {"result":{"tools":[...]}} ───────│
     │── {"method":"tools/call",            │
     │     "params":{...}} ────────────────▶│  执行工具
     │◀── {"result":{"content":[...]}} ─────│
```

### 已接入的 MCP Server

| Server | 工具数 | 条件 |
|--------|--------|------|
| **joyagent-demo**（自建） | 3 (get_weather/calculate/get_time) | 始终可用 |
| filesystem（官方） | 5+ (read_file/write_file/list_directory...) | 需要 npx + npm 包 |
| github（官方） | 20+ (search_repos/create_issue/create_pr...) | 需要 GITHUB_TOKEN |
| postgres（官方） | 5+ (query/list_tables/describe_table...) | 需要 DATABASE_URL |

### 为什么不用 mcp SDK？

面试时可以说："Python 的 `mcp` SDK 对 stdio 子进程的封装不透明——连接失败、超时、断连重试这些场景下 SDK 的内部行为难以调试。自己基于 JSON-RPC 2.0 用 `asyncio.create_subprocess_exec` 实现，只有 200 行核心代码，每一行都能在面试中讲清楚。"

---

## <a id="ch9"></a>九、Human-in-the-Loop + 工程化：生产级 Agent 还需要什么？

### Human-in-the-Loop

**三级权限，通过 ToolHook 机制挂载到工具执行链中：**

```
Agent → execute tool
  │
  ├── AUTO     → 直接执行（read_file, git_status...）
  ├── CONFIRM  → 需审批（write_file, execute_shell, git_commit...）
  └── DENY     → 直接拒绝（rm -rf, sudo, mkfs...）
```

默认自动放行模式（Docker Sandbox 已提供系统层隔离）。设置 `HITL_REQUIRE_APPROVAL=true` 可切换严格模式。

### 数据持久化

| 组件 | 技术 | 用途 |
|------|------|------|
| Session 存储 | SQLite（默认）/ PostgreSQL | 多用户会话 + Agent 状态快照 |
| TaskLog 审计 | SQLAlchemy ORM | 每条 LLM 调用/工具调用的审计日志 |
| 消息持久化 | FilePersistence / RedisPersistence | Agent 间消息跨重启恢复 |
| 任务队列 | Redis ZSET + 内置 Mock 降级 | 多用户请求排队 |

---

## <a id="ch10"></a>十、大厂 Agent 面试真题（含来源链接）

> 以下题目均来自牛客网、腾讯云开发者社区、CSDN 等平台的真实面经，已标注原文链接。

### 字节跳动

**1. Agent 工具调用死循环怎么处理？**

**来源**: [牛客网 - 字节大模型Agent算法二面-秋招面经](https://www.nowcoder.com/feed/main/detail/52021a7b98024061a3e7d83ae762465e)

**答案要点**: 迭代次数上限（本项目 `MAX_ITERATIONS=30`）、循环检测（工具调用序列去重，连续 N 次相同调用→终止）、降级策略（超过次数上限→返回已完成的部分结果，而非空响应）。

---

**2. 你的上下文压缩策略是什么？**

**来源**: [牛客网 - 字节 agent开发实习一面](https://www.nowcoder.com/feed/main/detail/075c06bf50a143a785c039f9951624f6)

**答案要点**: 四层压缩管线（参考 Claude Code 标准），"便宜的先跑，贵的后跑"：L3(tool_result_budget: 大结果 >200KB 落盘) → L1(snip_compact: 裁掉中间旧对话, 头3+尾47) → L2(micro_compact: 旧工具结果替换为占位符, 保留最近3条) → 以上三层全 0 API。还不够 → L4(递进式摘要: LLM 合并摘要, O(1) 而非 O(n) 重写)。API 报 413 → reactive_truncate 应急截断。执行顺序遵循 CC 源码 `query.ts` 的真实管线。

---

**3. Claude Code vs Codex 的差异？**

**来源**: [牛客网 - 字节 Agent开发一面](https://www.nowcoder.com/feed/main/detail/c3711862bc1f47d4914274dde70ff1de)

**答案要点**: Claude Code 的核心优势是 Anthropic 原生 API 的 `stop_reason` + `block.type` 工具调用机制，不需要 OpenAI 风格的 `tool_calls` JSON 包装。Codex 的优势是 IDE 深度集成和终端控制能力。两者的工具调用策略、上下文管理、权限模型都有差异。

---

**4. 多 Agent 编排：子 Agent 怎么分工、怎么通信、怎么终止？**

**来源**: [牛客网 - 字节大模型Agent算法二面-秋招面经](https://www.nowcoder.com/feed/main/detail/52021a7b98024061a3e7d83ae762465e)

**答案要点**: 本项目用 Mailbox 系统——每个 Agent 独立 Inbox/Outbox，MailboxManager 做消息路由。Router 发 TASK_ASSIGNMENT → Planner 拆解 → 按计划分发给 Coder/Tester/Reviewer。每条消息有 correlation_id 做线程追踪，TTL 过期自动进死信队列。终止条件：所有步骤完成 OR 总超时触发。

---

### 腾讯

**5. Agent 和传统大模型回答的本质区别是什么？**

**来源**: [牛客网 - 腾讯二面-大模型Agent面经总结](https://www.nowcoder.com/discuss/878600528970735616)

**答案要点**: 传统 LLM：一问一答，无自主行动能力。Agent：LLM + Planning + Memory + Tool Use，形成自主"感知→决策→行动→观察"闭环。核心区别在于是否存在"行动-观察"循环和自主工具调用。

---

**6. 多 Agent 怎么编排？Agent 失败/中断怎么处理？**

**来源**: [牛客网 - 腾讯 AI应用开发后端实习生面经](https://www.nowcoder.com/discuss/878600528970735616)

**答案要点**: 编排：Router 入口分析→Planner 拆解→按步骤分发。失败处理：1) 步骤级超时（每步独立 timeout）2) 异常捕获（run() 失败仍发 TASK_RESULT 含错误信息）3) 依赖检查（前置步骤失败→跳过下游）4) 死信队列（过期消息可手动 reprocess）。

---

### 淘天（淘宝天猫）

**7. Agent 应用开发中遇到的难点？**

**来源**: [牛客网 - 淘天 Agent 面经](https://www.nowcoder.com/feed/main/detail/485fbcf14893475a8dbb137064ea34f5)

**答案要点**: 1) `.pyc` 缓存导致代码不更新——最终用 `python -B` + `shutil.rmtree` 清理缓存 2) 同步 HTTP 调用阻塞事件循环——改为 `asyncio.to_thread` 3) `str.format()` 把 prompt 中的 JSON 模板当占位符——全部改为 `.replace()`。

---

**8. 工具调用的 schema 定义和失败处理？**

**来源**: [牛客网 - 淘天 Ai Agent应用开发二面](https://www.nowcoder.com/feed/main/detail/fa80eb0d4a30409b82e6cce6150920b5)

**答案要点**: Schema 用 Anthropic 原生 `input_schema` 格式，和 MCP 的 `inputSchema` 驼峰兼容。失败处理：ToolResult 含 success/error，Hook 链中 on_error 可吞异常降级。MCP 工具执行失败→MCPToolResult.error→ToolResult.error→LLM 看到错误信息决定重试或放弃。

---

### 中兴

**9. Agent 的记忆系统怎么设计的？长期记忆如何存储和检索？**

**来源**: [牛客网 - 中兴 实习 agent开发 一面凉经](https://www.nowcoder.com/feed/main/detail/81ef46253510446d93530b20fcb0815f)

**答案要点**: 三级记忆：1) Short-term：滑动窗口 + Token 管理 + 自动压缩 2) Long-term：ChromaDB 向量存储，三类 Collection（code/conversation/task），每次工具调用后异步记存 3) Reflection：错误嵌入 + 相似经验召回，Fix Loop 中用 LRU+ SHA256 缓存减少 70% 重复计算。

---

### 百度

**10. Agent Skill 开发怎么做？计划模式是什么？**

**来源**: [牛客网 - 百度 大模型生态集成实习面经](https://www.nowcoder.com/discuss/878600528970735616)

**答案要点**: Skill = 对特定任务的完整 prompt + 工具组合的封装。Plan 模式下，Agent 先输出执行计划（只读思考），用户确认后再执行。本项目的 LangGraph Planner 节点就是该模式——先拆解出 steps，用户/系统可审查，再进入 Executor 执行。

---

### 通用高频题

**11. Function Calling 的底层原理是什么？大模型为什么能"学会"调用工具？**

**来源**: [牛客网 - 大模型Agent面试全攻略](https://ac.nowcoder.com/discuss/1628704) / [腾讯云开发者社区 - 2026年5月最新AI Agent面试题汇总](https://cloud.tencent.com/developer/article/2668240)

**答案要点**: LLM 在训练时接触过带有 tool call 格式的数据，微调阶段（如 Anthropic 的 tool-use fine-tuning）教会模型何时输出 tool_use 而非 text。推理时，模型根据 system prompt + 用户输入 + 可用工具的 input_schema，决定是否需要调工具。`stop_reason == "tool_use"` 意味着模型确信它应该调工具而非直接回复。

---

**12. RAG 知识库怎么处理热更新？召回和精排怎么选型？**

**来源**: [牛客网 - 大模型Agent面试全攻略](https://ac.nowcoder.com/discuss/1628704) / [牛客网 - 大模型、Agent面经总结](https://www.nowcoder.com/discuss/878600528970735616)

**答案要点**: 热更新：哈希检测变更文档→分段重新 embedding→先删后增（delete + add 原子操作）→注意 chunk 边界偏移。召回：向量检索（embedding 语义相似度）+ BM25 关键词（互补覆盖）。精排：cross-encoder reranker 对 Top-K 重排序，因为向量相似度 ≠ 语义相关性。

---

**13. Agent 如何处理 token 超限（上下文溢出）？**

**来源**: [牛客网 - 腾讯二面](https://www.nowcoder.com/discuss/878600528970735616) / [牛客网 - 从基础到死亡追问](https://www.nowcoder.com/discuss/881823530411704320)

**答案要点**: 五层防线，参考 Claude Code 源码实现：1) TokenManager 每轮 LLM 调用前检查 token 数（`should_compress()` 用 80% 阈值）2) L3: tool_result_budget → 大结果(>200KB)落盘，上下文只留预览 3) L1: snip_compact → 裁掉中间无关对话 4) L2: micro_compact → 旧工具结果换占位符 5) L4: 递进式摘要 → LLM 合并摘要（O(1) 而非 O(n)）。还有 max_tokens 升级（4096→8192）+ reactive_truncate 应急截断。

---

**14. MCP 交互流程与通信机制？**

**来源**: [牛客网 - 字节 agent开发实习一面](https://www.nowcoder.com/feed/main/detail/075c06bf50a143a785c039f9951624f6)

**答案要点**: MCP 基于 JSON-RPC 2.0 over stdio。Client 启动 Server 子进程→发送 initialize 握手→tools/list 发现工具→缓存到 MCPTool 列表→Agent 调用时通过 MCPRegistry 路由到对应 MCPClient→MCPClient 发送 tools/call RPC→解析响应返回 ToolResult。整个链路在本项目中约 500 行纯 Python 实现，零第三方 MCP SDK 依赖。

---

### 快手

**15. MCP、Skill、Function Call 三者有什么区别？**

**来源**: [牛客网 - 快手 AI应用开发一面](https://www.nowcoder.com/discuss/882573284426932224) / [牛客网 - 28届实习拷打23个Agent问题](https://ac.nowcoder.com/discuss/1619961)

**通俗理解**: 把 Agent 比作一个厨师——**Function Call 是切菜刀**（模型决定拿什么工具干活，是"决策层"），**MCP 是菜刀和案板的接口标准**（工具怎么和模型通信，是"传输协议层"），**Skill 是一道菜的完整菜谱**（prompt + 工具 + 流程的封装，是"功能单元层"）。

| 维度 | Function Call | MCP | Skill |
|------|-------------|-----|-------|
| 定位 | 模型决策层 | 传输协议层 | 功能封装层 |
| 做什么 | 决定调哪个工具 | 工具如何通信 | 如何完成一个任务 |
| 例子 | `stop_reason=="tool_use"` → 调 `read_file` | JSON-RPC over stdio → 发现外部进程的工具 | "写单元测试" = prompt + pytest 工具 + 修复流程 |

**MCP 和 Skill 哪个上下文占用更大？** Skill 更大——因为它把整个 prompt template + 工具定义 + 流程指引全部注入 system prompt。MCP 只在 tools 列表里多几个 `name/description/input_schema` 条目。

---

**16. 上下文压缩的触发机制是什么？摘要合并还是分离？**

**来源**: [牛客网 - 快手 AI应用开发一面](https://www.nowcoder.com/discuss/882573284426932224)

**答案要点**: 触发机制：TokenManager 每轮 LLM 调用前检查 token 数，超过 Context Window 的 80% 阈值时触发压缩。策略是**递进式摘要（合并）**——不是每次从头总结全部历史（O(n)），而是把旧摘要 + 新消息一起喂给 LLM 生成新摘要（O(1)）。类比：像人类记笔记——新的一页追加到笔记本后面，而不是把整本重抄。

---

**17. Agent 沙箱怎么实现？Skill 能放在沙箱中吗？**

**来源**: [牛客网 - 快手 AI应用开发一面](https://www.nowcoder.com/discuss/882573284426932224)

**答案要点**: 沙箱 = Docker 容器六层隔离（见第五章）。Skill 本身是 prompt + 工具定义，不执行代码，不需要沙箱。但 Skill 调用的工具（如 `execute_shell`）会在沙箱中执行——工具执行层和 Skill 定义层是分离的。跨文件系统通信通过 bind mount 实现：宿主机目录同时挂载到主容器和 sandbox 容器。

---

### 蚂蚁

**18. 多 Agent 之间怎么通信？如何处理并发？**

**来源**: [牛客网 - 蚂蚁 agent开发一面](https://www.nowcoder.com/feed/main/detail/7a0ddb8e077041d4b72ba9e5290ad36a)

**答案要点**: 通信用 Mailbox 系统——不是共享 State 的 `messages.append()`，每个 Agent 有独立 Inbox/Outbox，通过 MailboxManager 做消息路由。并发处理：每个 Agent 有独立 asyncio Task + 自己的 Inbox，消息投递不阻塞其他 Agent。多 Agent 同时操作文件/DB 时，依赖操作系统的文件锁 + DB 的事务隔离。

---

**19. Agent 调工具失败了怎么处理？**

**来源**: [牛客网 - 蚂蚁 agent开发一面](https://www.nowcoder.com/feed/main/detail/7a0ddb8e077041d4b72ba9e5290ad36a)

**答案要点**: 三层处理：1) 工具层：ToolResult 含 success/error，Hook 链中 on_error 可吞异常降级 2) Agent 层：LLM 看到错误信息后自主决定重试（换参数/换工具/换策略）还是放弃 3) 兜底层：重试次数上限（3 次）→ 死信队列 → 手动 reprocess。MCP 工具失败还多一层——MCPRegistry 返回 ToolResult(success=False, error=...)，agent 看到后可以决定换一个 MCP Server 重试。

---

**20. 如何减少和规避 Agent 幻觉？**

**来源**: [牛客网 - 蚂蚁 agent开发一面](https://www.nowcoder.com/feed/main/detail/7a0ddb8e077041d4b72ba9e5290ad36a)

**通俗理解**: 幻觉就像一个人被问到不会的问题还要硬编答案。我们给它三件武器：1) **查资料**——用 RAG 检索真实知识，不靠记忆猜 2) **动手验证**——写了代码就扔 Docker 里跑测试，跑不过就改，眼见为实 3) **画红线**——在 system prompt 里明确禁止编造文件路径和接口名，允许说"我不确定，让我查一下"。

---

### 字节全栈

**21. 上下文窗口满了有哪几种压缩方式？**

**来源**: [牛客网 - 字节 agent全栈一面](https://www.nowcoder.com/feed/main/detail/b62416eaa3764a9ba46269c0058019fd)

**答案要点**: 参考 Claude Code 源码，四层："便宜的先跑，贵的后跑"——L3 budget（大结果落盘）→ L1 snip（裁旧对话）→ L2 micro（旧结果占位）→ 以上全 0 API → L4 递进式摘要（1 API）。还有 max_tokens 升级（4096→8192）+ reactive_truncate 应急。

---

**22. Agent 的 Memory 存在哪里？**

**来源**: [牛客网 - 字节 agent全栈一面](https://www.nowcoder.com/feed/main/detail/b62416eaa3764a9ba46269c0058019fd)

**答案要点**: 短期记忆（当前会话的消息和摘要）在内存中，会话结束持久化到 ChromaDB。长期记忆（嵌入向量）存在 ChromaDB（三个 Collection：code/conversation/task）。跨会话恢复：下次启动时从 ChromaDB 检索相似历史 + 从 FilePersistence 恢复未处理的 Mailbox 消息。

---

### B 站

**23. 上下文管理具体怎么做？**

**来源**: [牛客网 - B站 AI应用研发二面](https://www.nowcoder.com/feed/main/detail/a88fde0d7efc4779aa67d9841871aa90)

**答案要点**: 四条原则：1) 预防——TokenManager 每轮前检查，80% 阈值触发 2) 裁剪——四层管线（budget/snip/micro/summary）自动腾空间 3) 降级——reactive_truncate 暴截 + max_tokens 升级 4) 隔离——每个 Agent 独立 Inbox，一个 Agent 上下文炸了不影响其他。上下文持久化：FilePersistence 存未处理消息，ChromaDB 存语义记忆，SQLite 存 TaskLog 审计。

---

### 淘天（深度题）

**24. 候选工具 >100 个时怎么路由？**

**来源**: [牛客网 - 淘天 aiagent一面](https://www.nowcoder.com/feed/main/detail/a00f89eb057d4476bd67f3b24679cdb8)

**通俗理解**: 100 个工具全部塞进 system prompt 就像给厨师 100 把刀让他挑。两种解法：1) **按需暴露**——根据任务类型只给相关工具（"写代码"时给 file_write/git，"查日志"时给 search_code/execute_shell）2) **两级检索**——先用轻量语义检索定位 Top-10 候选工具，再让 LLM 从 10 个中选最合适的。本项目用方案 1，通过 AgentRole 的 `tools` 字段控制每个 Agent 能看到的工具集。

---

**25. 超长任务如何断点继续？**

**来源**: [牛客网 - 淘天 aiagent一面](https://www.nowcoder.com/feed/main/detail/a00f89eb057d4476bd67f3b24679cdb8)

**答案要点**: 三层保障：1) LangGraph Checkpoint——StateGraph 的每个节点执行完后自动保存状态到 SQLite/Postgres，中断后从最近 checkpoint 恢复 2) Mailbox 消息持久化——未处理消息存到 FilePersistence，重启后 load 回 Inbox 3) Multi-Agent 步骤追踪——correlation_id 记录每个 step 的完成情况，中断后从下一个未完成的 step 继续。

---

**26. 提示词工程 vs 上下文工程的区别？**

**来源**: [牛客网 - 快手 AI应用开发一面](https://www.nowcoder.com/discuss/882573284426932224) / [牛客网 - 28届实习23个Agent问题](https://ac.nowcoder.com/discuss/1619961)

**通俗理解**: 提示词工程是"怎么跟模型说话让它听话"（写 prompt），上下文工程是"怎么给模型安排座位让它坐得舒服"（管 context window）。前者是写一段好 prompt，后者是管一个动态变化的对话历史。面试时强调你两者都做了——Prompts 按 Agent 角色分（Planner/Coder/Tester/Reviewer 各有独立 system prompt），Context 用四层压缩管线实时管理。

---

### 通用高频题（补充）

**27. Agent 可观测性怎么做？**

**来源**: [牛客网 - 28届实习23个Agent问题](https://ac.nowcoder.com/discuss/1619961)

**答案要点**: 四层可观测：1) ToolStatsCollector——每个工具调用的次数/成功率/P50/P95/P99 耗时 2) Mailbox 审计日志——每条消息的 sender/recipient/timestamp/correlation_id 完整可追溯 3) TaskLog 数据库审计——每条 LLM 调用/工具调用/错误记录到 SQLite 4) GET /api/multi-agent/audit——按 correlation_id 查询完整执行链路。

---

**28. 多 Agent 的记忆如何隔离和共享？**

**来源**: [牛客网 - 淘天 aiagent一面](https://www.nowcoder.com/feed/main/detail/a00f89eb057d4476bd67f3b24679cdb8)

**答案要点**: 隔离——每个 Agent 有独立 Inbox（消息不串），独立 ShortTermMemory（上下文不混），独立 AgentRole（工具集隔离）。共享——MailboxManager 做消息路由（跨 Agent 通信），ChromaDB LongTermMemory 全局单例（跨 Agent 知识沉淀），correlation_id 串联同一任务的跨 Agent 消息（审计可追溯）。

---

## 附录：项目技术栈

| 层级 | 技术 |
|------|------|
| Web 框架 | FastAPI + WebSocket |
| Agent 框架 | LangGraph StateGraph |
| LLM 调用 | Anthropic Python SDK 原生（非 LangChain） |
| 工具系统 | BaseTool + ToolRegistry + Hook 中间件 |
| 安全沙箱 | Docker SDK + seccomp + resource limits |
| 向量数据库 | ChromaDB |
| 关系数据库 | SQLite / PostgreSQL + SQLAlchemy |
| MCP 协议 | 纯 Python JSON-RPC 2.0 over stdio（自实现） |
| 多 Agent 通信 | Mailbox 系统（Inbox/Outbox + MailboxManager） |

---

> 本项目完整代码：`joyagent/`，9 个 Phase 的实现参考 `dev_md/` 下的开发文档。
> 面试准备建议：第 1-5 章必讲（核心闭环），第 7-8 章是差异化亮点（Multi-Agent + MCP），
> 第 10 章真题用于模拟面试自测。
