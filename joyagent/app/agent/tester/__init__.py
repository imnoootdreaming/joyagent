"""
Phase 7 Step 3 — Tester Agent

Tester 负责：
  1. 接收 TASK_ASSIGNMENT → 执行测试验证代码
  2. 测试失败时 → 发 REVIEW_FEEDBACK 给 Coder（而非直接 TASK_RESULT）
  3. 测试通过时 → 发 TASK_RESULT 给 Router
"""

from app.agent.tester.agent import TesterAgent

__all__ = ["TesterAgent"]
