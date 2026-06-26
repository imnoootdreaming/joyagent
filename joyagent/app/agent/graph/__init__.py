"""
Phase 3 Step 5: LangGraph StateGraph — Plan→Execute→Reflect 工作流。

模块结构：
  workflow.py  — build_workflow(): 构建 StateGraph + 编译为 agent_workflow
  nodes.py     — Node 函数导入（planner_node / executor_node / reflector_node）
  edges.py     — should_continue(): Conditional Edge 路由决策

使用方式：
  from app.agent.graph.workflow import agent_workflow

  final_state = await agent_workflow.ainvoke(initial_state)
  async for event in agent_workflow.astream(initial_state):
      for node_name, output in event.items():
          print(f"{node_name}: {output.keys()}")
"""

from app.agent.graph.workflow import agent_workflow, build_workflow

__all__ = ["agent_workflow", "build_workflow"]
