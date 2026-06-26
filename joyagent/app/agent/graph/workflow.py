"""
Phase 3 Step 5: LangGraph StateGraph 构建与编译。

build_workflow() 构建 Plan→Execute→Reflect 三阶段工作流，
通过 Conditional Edge 实现非线性任务执行和错误恢复。

工作流拓扑：

                    ┌─────────────┐
                    │  __start__   │
                    └──────┬──────┘
                           │ (entry_point)
                    ┌──────▼──────┐
                    │   Planner    │  ← 分析任务，生成 plan[]
                    └──────┬──────┘
                           │ (add_edge)
                    ┌──────▼──────┐
                    │  Executor    │  ← 执行 plan[current_step_index]
                    └──────┬──────┘
                           │ (add_edge)
                    ┌──────▼──────┐
                    │  Reflector   │  ← 反思结果，输出 task_completed / need_replan
                    └──────┬──────┘
                           │ (add_conditional_edges)
              ┌────────────┼────────────┐
              │            │            │
         task_completed  need_replan  继续 / 错误
              │            │            │
              ▼            ▼            │
          __end__      Planner     ┌────┴────┐
                                   │         │
                              Executor   Reflector

这种设计的优势（vs Phase 1-2 的 while 循环）：
  - 显式状态流转：每一步的输入/输出都有明确的结构
  - 可暂停/恢复：LangGraph Checkpoint 支持状态快照
  - 可观测：每次 Node 执行前后的 state 都可以记录
  - 可扩展：新增 Node（如 CodeReview / Tester）只需 add_node + add_edge
"""

# ── LangGraph ──
from langgraph.graph import StateGraph, END
# StateGraph: LangGraph 核心抽象 —— 用有向图定义 Agent 工作流
# END:        LangGraph 内置终止标记 —— 路由到 END 表示工作流结束

# ── 项目内导入 ──
from app.agent.schemas.state import AgentState
# AgentState: 所有 Node 之间共享的状态 TypedDict

from app.agent.graph.nodes import planner_node, executor_node, reflector_node
# 三个 Node 函数：各自是 runtime/ 中同名函数的引用

from app.agent.graph.edges import should_continue
# Conditional Edge 回调：Reflector → 决定下一步路由


def build_workflow() -> StateGraph:
    """
    构建并编译 Plan→Execute→Reflect 工作流。

    这是 Phase 3 的核心组装函数 —— 它不包含业务逻辑，
    只负责"把三个 Node 用正确的边连接起来"。

    构建步骤：
      1. 创建 StateGraph(AgentState)      —— 声明状态类型
      2. add_node("planner", ...)         —— 注册三个 Node
      3. set_entry_point("planner")       —— 设置入口
      4. add_edge("planner", "executor")  —— 固定边（无条件）
      5. add_edge("executor", "reflector") —— 固定边
      6. add_conditional_edges(...)       —— 条件边（Reflector 后分流）

    Returns:
        CompiledStateGraph: 编译后的可执行图，支持:
          - .ainvoke(state)   → 异步执行，返回最终 state
          - .astream(state)   → 异步流式执行，yield 每个 Node 的中间 state
          - .get_state()      → 获取 Checkpoint 状态（暂停/恢复）

    使用方式：
        from app.agent.graph.workflow import agent_workflow

        initial_state = {
            "messages": [{"role": "user", "content": "..."}],
            "plan": [],
            "current_step_index": 0,
            # ... 其他 AgentState 字段
        }
        final_state = await agent_workflow.ainvoke(initial_state)
    """

    # ── 1. 创建 StateGraph ────────────────────────────────────────────
    # 传入 AgentState TypedDict 作为状态模板
    # LangGraph 会根据 TypedDict 的字段和类型自动管理状态的合并和传递
    workflow = StateGraph(AgentState)

    # ── 2. 注册 Node ──────────────────────────────────────────────────
    # 每个 Node 是一个 async 函数: (AgentState) -> dict
    # LangGraph 自动处理 async/sync 差异，无需手动包装
    workflow.add_node("planner", planner_node)
    # Planner: 接收用户消息，输出结构化步骤列表 plan[]

    workflow.add_node("executor", executor_node)
    # Executor: 取当前步骤 index，执行一个迷你 ReAct Loop

    workflow.add_node("reflector", reflector_node)
    # Reflector: 评估执行结果，输出 task_completed / need_replan

    # ── 3. 设置入口 ───────────────────────────────────────────────────
    # 工作流从 Planner 开始（先规划，再执行，最后反思）
    workflow.set_entry_point("planner")

    # ── 4. 固定边（无条件流转） ────────────────────────────────────────
    # Planner → Executor: 计划生成后自动进入执行阶段
    workflow.add_edge("planner", "executor")

    # Executor → Reflector: 每步执行完自动进入评估阶段
    workflow.add_edge("executor", "reflector")

    # ── 5. Conditional Edge（条件分流） ────────────────────────────────
    # Reflector 之后不固定走向 —— 由 should_continue(state) 根据
    # task_completed / need_replan / error_message / current_step_index
    # 动态决定下一步。
    #
    # route_map 将 should_continue() 的返回值映射到目标 Node：
    #   "planner"   → planner Node（重新规划）
    #   "executor"  → executor Node（继续执行下一步）
    #   "reflector" → reflector Node（再次评估，处理未解错误）
    #   "__end__"   → END（LangGraph 内置终止标记）
    #
    # ⚠️ route_map 中的 key 必须与 should_continue() 返回值完全匹配
    # ⚠️ value 必须是已注册的 Node 名称或 END
    workflow.add_conditional_edges(
        "reflector",                     # 条件边的出边 Node
        should_continue,                  # 路由决策函数
        {
            "planner": "planner",         # 需要 replan → 回到 Planner
            "executor": "executor",       # 继续执行 → 回到 Executor
            "reflector": "reflector",     # 错误未解 → 再次 Reflector
            "__end__": END,              # 完成/超限 → 终止工作流
        },
    )

    # ── 6. 编译返回 ───────────────────────────────────────────────────
    # compile() 做三件事：
    #   1. 校验图结构（无死循环、无孤立节点）
    #   2. 优化执行路径（合并连续的固定边）
    #   3. 返回 CompiledStateGraph（支持 ainvoke / astream / get_state）
    return workflow.compile()


# ── 全局 Agent Workflow 实例 ──────────────────────────────────────
# build_workflow() 在模块加载时执行一次，确保只编译一次
# 编译后的 graph 是线程安全的（LangGraph 内部管理 state 隔离）
agent_workflow = build_workflow()
