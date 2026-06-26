"""
Phase 3 Step 5: Conditional Edge — Reflector 之后的路由决策。

should_continue(state) 是 LangGraph 的 Conditional Edge 函数。
每次 Reflector Node 执行完成后调用，根据 state 中的多个信号
决定工作流的下一步走向。

路由规则（按优先级排列）：
  1. task_completed == True  → "__end__"     (任务完成，终止工作流)
  2. need_replan == True     → "planner"     (当前计划有问题，重新规划)
  3. error_message 不为空      → "reflector"   (执行出错，再次评估)
  4. current_step < len(plan) → "executor"    (还有步骤未执行，继续)
  5. 上述都不满足              → "__end__"     (兜底：安全终止)

返回值是目标 Node 名称字符串，LangGraph 根据此字符串决定路由。
这些字符串必须在 workflow.py 的 add_conditional_edges(..., route_map) 中注册。
"""

# ── 项目内导入 ──
from app.agent.schemas.state import AgentState
# AgentState: LangGraph 共享状态，读取 task_completed / need_replan / plan 等字段


def should_continue(state: AgentState) -> str:
    """
    Reflector Node 之后的 Conditional Edge 路由函数。

    这是 LangGraph add_conditional_edges 的回调函数。
    每次 Reflector 完成后被调用，返回值决定下一个 Node。

    Args:
        state: 当前的完整 AgentState（LangGraph 自动传入）

    Returns:
        str: 下一个 Node 的名称，可选值：
             "__end__"   — 结束工作流（LangGraph 内置终止标记）
             "planner"   — 回到 Planner，重新生成计划
             "executor"  — 回到 Executor，继续执行下一步
             "reflector" — 回到 Reflector，再次评估（错误未解时）
    """
    # ── 优先级 1：任务完成标记 ──
    # Reflector 判断 task_completed=True → 直接终止，不再执行
    if state.get("task_completed", False):
        print("  \033[32m[route] task_completed=True → __end__\033[0m")
        return "__end__"

    # ── 优先级 2：需要重新规划 ──
    # Reflector 发现当前计划有根本性问题（如：用错工具、步骤不可能完成）
    # → 回到 Planner，将 reflection_notes 作为 context 传给 Planner
    if state.get("need_replan", False):
        print(
            f"  \033[33m[route] need_replan=True → planner "
            f"(reason: {state.get('reflection_notes', '')[:50]}...)\033[0m"
        )
        return "planner"

    # ── 优先级 3：有未处理的错误 ──
    # Executor 或 Planner 返回了 error_message 但 Reflector 尚未判定完成/重规划
    # → 再次进入 Reflector，让其基于新的错误信息重新评估
    if state.get("error_message"):
        print(
            f"  \033[33m[route] error detected → reflector "
            f"(error: {state['error_message'][:60]})\033[0m"
        )
        return "reflector"

    # ── 优先级 4：还有步骤未完成 ──
    # plan 中还有 status!="completed" 的步骤，且 current_step_index 未越界
    # → 回到 Executor，执行下一个步骤
    plan = state.get("plan", [])
    current_idx = state.get("current_step_index", 0)

    # 统计计划中的步骤状态
    pending_count = sum(
        1 for s in plan
        if s.get("status") not in ("completed", "failed")
    )
    if pending_count > 0 and current_idx < len(plan):
        print(
            f"  \033[36m[route] {pending_count} step(s) remaining → executor "
            f"(step {current_idx + 1}/{len(plan)})\033[0m"
        )
        return "executor"

    # ── 优先级 5：兜底 —— 全部步骤已执行完毕 ──
    # current_step_index >= len(plan)，所有步骤都已至少执行过一次
    # 但 task_completed != True（否则优先级 1 已拦截）
    # → 再走一次 Reflector，让 LLM 最终确认是否真的完成
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", 3)
    if reflection_count < max_reflections:
        print(
            f"  \033[36m[route] all steps executed, "
            f"re-reflecting ({reflection_count + 1}/{max_reflections}) → reflector\033[0m"
        )
        return "reflector"

    # ── 优先级 6：最终兜底 ──
    # 反思轮次也耗尽了 —— 安全终止
    print(
        f"  \033[33m[route] max reflections reached "
        f"({reflection_count}/{max_reflections}) → __end__\033[0m"
    )
    return "__end__"
