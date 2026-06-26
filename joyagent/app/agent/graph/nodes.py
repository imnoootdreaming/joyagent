"""
Phase 3 Step 5: LangGraph Node 注册入口。

将 runtime/ 中的三个 Node 实现函数导出为 LangGraph StateGraph 可用的节点函数。

运行时关系：
  graph/nodes.py （本文件）  ──import──▶  runtime/planner.py
                                          runtime/executor.py
                                          runtime/reflector.py

每个 Node 函数签名：
  async def xxx_node(state: AgentState) -> dict
    输入:  完整的 AgentState（LangGraph 自动传入当前共享状态）
    输出:  dict（部分状态更新，LangGraph 自动 merge 回全局 state）

使用方式（在 workflow.py 中）：
  workflow.add_node("planner", planner_node)
  workflow.add_node("executor", executor_node)
  workflow.add_node("reflector", reflector_node)

注意：LangGraph 支持 async Node 函数，直接传入即可，无需包装为 sync。
"""

# ── Planner Node: 任务拆解 + 生成结构化计划 ──
from app.agent.runtime.planner import planner_node
# planner_node(state): 读取 messages → LLM 分析 → 输出 plan (list[TaskStep])

# ── Executor Node: 按计划逐步执行工具调用 ──
from app.agent.runtime.executor import executor_node
# executor_node(state): 取 plan[current_step_index] → 迷你 ReAct Loop → 记录结果

# ── Reflector Node: 反思结果 + 决定下一步路由 ──
from app.agent.runtime.reflector import reflector_node
# reflector_node(state): 评估执行结果 → 输出 task_completed / need_replan

# 导出列表（供 workflow.py 使用）
__all__ = ["planner_node", "executor_node", "reflector_node"]
