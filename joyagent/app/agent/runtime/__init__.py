# app/agent/runtime/ — Phase 3: LangGraph 工作流节点实现
#
# 每个 Node 是 LangGraph StateGraph 中的一个处理单元：
#   planner   — 任务拆解 + 生成结构化计划
#   executor  — 按计划逐步执行工具调用
#   reflector — 反思执行结果 + 决定下一步路由
#
# Node 接口：async def xxx_node(state: AgentState) -> dict
#   输入：完整的 AgentState（共享状态）
#   输出：dict（部分状态更新，LangGraph 自动合并）
