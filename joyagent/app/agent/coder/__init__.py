"""
Phase 7 Step 3 — Coder Agent

Coder 负责：
  1. 接收 TASK_ASSIGNMENT → 编写或修改代码
  2. 处理 REVIEW_FEEDBACK → 应用 Reviewer/Tester 的反馈
  3. 通过 Outbox 发送 TASK_RESULT 回 Router
"""

from app.agent.coder.agent import CoderAgent

__all__ = ["CoderAgent"]
