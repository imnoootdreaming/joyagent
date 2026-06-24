"""
Phase 3 Step 1: AgentState — LangGraph StateGraph 的核心状态容器。

AgentState 是所有 Node（Planner / Executor / Reflector）之间共享的
唯一状态对象。每个 Node 接收整个 state，返回部分更新（Partial<AgentState>），
LangGraph 自动合并到全局 state。

消息格式：纯 dict（Anthropic 原生 {"role": ..., "content": ...}），
不依赖 langchain_core.messages，保持 Phase 1-2 的简洁格式。
"""

# ── Python 标准库 ──
from typing import (
    TypedDict,        # 结构化字典类型——定义 state 字段和类型约束
    Annotated,        # 给类型附加元数据（LangGraph 用它附加 reducer 函数）
    Sequence,         # 不可变序列——消息列表的类型标识
)

# ── LangGraph ──
from langgraph.graph.message import add_messages
# add_messages 是 LangGraph 内置的消息列表 reducer。
# 作用：当 Node 返回 {"messages": [新消息]} 时，不覆盖旧消息，
# 而是自动追加到 state.messages 末尾。
# 等价于 state["messages"] = state["messages"] + 新消息


class TaskStep(TypedDict):
    """
    计划中的单个执行步骤。

    Planner Node 输出 TaskStep 列表，Executor Node 按 step_id 顺序
    逐个执行。每个步骤包括描述、推荐工具和当前状态。
    """

    step_id: int
    # 步骤序号，从 1 开始。Planner 按此字段排序，Executor 按 current_step_index 定位。

    description: str
    # 步骤的详细描述，给 LLM 看的执行指令。
    # 例如："使用 read_file 读取 pyproject.toml 获取依赖列表"

    tool_name: str | None
    # Planner 为该步骤推荐的工具名称（如 "read_file"、"execute_shell"）。
    # None 表示无需调用工具（纯 LLM 推理步骤，如"分析前面的输出"）。

    status: str
    # 步骤执行状态。允许值：
    #   "pending"      — 尚未开始
    #   "in_progress"  — 正在由 Executor 处理
    #   "completed"    — 成功完成
    #   "failed"       — 执行失败（Reflector 可能触发 replan）


class AgentState(TypedDict):
    """
    LangGraph StateGraph 的核心状态对象。

    State 在所有 Node 之间共享流转：
      Planner   → 读取 messages，写入 plan
      Executor  → 读取 plan[current_step_index]，执行工具，追加 messages
      Reflector → 读取 tool_call_history，判断 task_completed，决定路由

    字段分为四组：消息历史、任务计划、反思控制、工具记录。
    """

    # ── 消息历史 ──────────────────────────────────────
    # 使用 Annotated + add_messages reducer：
    # Node 返回 {"messages": [msg1, msg2]} 时，LangGraph 自动追加而非覆盖。
    # 消息格式为 Anthropic 原生纯 dict：
    #   {"role": "user", "content": "你好"}
    #   {"role": "assistant", "content": [{"type": "text", "text": "..."}, ...]}
    #   {"role": "user", "content": [{"type": "tool_result", ...}]}
    messages: Annotated[Sequence[dict], add_messages]

    # ── 任务计划 ──────────────────────────────────────
    plan: list[TaskStep]
    # Planner Node 输出的步骤列表，按 step_id 升序排列。
    # Executor 通过 current_step_index 索引当前要执行的步骤。

    current_step_index: int
    # 当前执行到第几步（0-based 索引）。
    # Executor 执行完一步后 +1，Reflector 判断是否需要继续。

    # ── 反思控制 ──────────────────────────────────────
    reflection_count: int
    # 已完成的反思轮次。每次 Reflector Node 执行后 +1。

    max_reflections: int
    # 允许的最大反思轮次（默认 3）。
    # 防止 Agent 陷入"反思→replan→执行→反思"的死循环。
    # 达到上限后 Conditional Edge 强制路由到 __end__。

    reflection_notes: str
    # Reflector Node 输出的分析文本，记录当前轮次的评估结论。
    # 例如："步骤 2 执行失败，原因是文件不存在。建议重新规划路径。"
    # 这些笔记会被后续 Planner Node 读取，用于 replan 决策。

    # ── 流程控制 ──────────────────────────────────────
    task_completed: bool
    # Reflector 是否判定任务已完成。
    # True → Conditional Edge 路由到 __end__（工作流终止）。

    need_replan: bool
    # Reflector 是否判定需要重新规划。
    # True → Conditional Edge 路由回 Planner（重新生成计划）。
    # TypedDict 不可声明可选键，此字段在初始 state 中设为 False，
    # 由 Reflector Node 根据执行结果设置。

    error_message: str | None
    # 最近一次错误的描述文本。
    # Planner 和 Reflector 读取此字段以理解失败原因。
    # None 表示没有错误。

    # ── 工具调用记录 ──────────────────────────────────
    tool_call_history: list[dict]
    # 所有工具调用的记录列表，供 Reflector 分析。
    # 每条记录格式：
    #   {
    #       "tool_name": str,   # 工具名称
    #       "input": dict,      # 调用参数
    #       "result": str,      # 返回结果（截断后）
    #       "success": bool,    # 是否成功
    #   }
    # Executor Node 每次工具调用后追加一条。
