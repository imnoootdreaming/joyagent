"""
Phase 7 Step 4 — Router Agent

Router 是所有用户请求的入口点。它分析请求复杂度，决定直接路由到
Coder/Tester/Reviewer 还是先经过 Planner 拆解。

职责：
  1. 分析用户请求 → 简单任务直接路由，复杂任务经 Planner 拆解
  2. 接收 Planner 计划 → 按依赖顺序分发到 Coder/Tester/Reviewer
  3. 收集各 Agent 的 TASK_RESULT → 聚合返回给用户
  4. 处理 ERROR_REPORT → 决定重试/重新分配/升级
"""

from app.agent.router.agent import RouterAgent

__all__ = ["RouterAgent"]
