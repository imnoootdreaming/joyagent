"""
Phase 7 Step 3 — Planner Agent

Planner 负责：
  1. 接收 TASK_ASSIGNMENT → 拆解用户需求为可执行步骤
  2. 处理 CONFLICT_ESCALATE → 作为仲裁者解决 Agent 间冲突
  3. 通过 Outbox 发送 TASK_RESULT 回 Router
"""

from app.agent.planner.agent import PlannerAgent

__all__ = ["PlannerAgent"]
