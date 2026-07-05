"""
Phase 7 Step 3 — ReviewerAgent

Reviewer Agent —— 代码审查 + 反馈。

职责：
  1. 接收 TASK_ASSIGNMENT → 审查代码 diff
  2. 代码有问题 → 发 REVIEW_FEEDBACK 给 Coder
  3. 代码通过   → 发 TASK_RESULT (LGTM) 给 Router

关键设计决策（与 BaseAgent 默认行为不同）：
  BaseAgent._handle_task: 完成后总是发 TASK_RESULT 回 sender
  ReviewerAgent._handle_task: 根据审查结果分别处理
    - LGTM       → TASK_RESULT 回 Router（与默认行为相同）
    - NEEDS_WORK → REVIEW_FEEDBACK 给 Coder + TASK_RESULT 给 Router
"""

from __future__ import annotations

import json

from app.agent.base import BaseAgent, extract_json, extract_text
from app.agent.mailbox import (
    MailboxManager,
    MailboxMessage,
    MessageType,
)
from app.agent.reviewer.prompts import REVIEWER_CODE_REVIEW_PROMPT
from app.agent.roles import AgentRole
from app.core.config import Config


class ReviewerAgent(BaseAgent):
    """
    Reviewer Agent —— 代码审查 + 反馈。

    相比 BaseAgent 的额外行为：
      - 覆盖 _handle_task → 发现问题时发送 REVIEW_FEEDBACK 给 Coder
      - run() 使用代码审查 prompt，返回结构化审查结果

    Example:
        reviewer = ReviewerAgent(ROLE_REVIEWER, mailbox_manager)
        reviewer.start_background()
    """

    def __init__(
        self,
        role: AgentRole,
        mailbox_manager: MailboxManager,
        agent_id: str = "",
    ):
        super().__init__(role, mailbox_manager, agent_id)

        # ── 存储当前任务上下文 ──
        self._last_task: str = ""
        self._last_correlation_id: str = ""

    # ── 核心逻辑：代码审查 ──────────────────────────────────

    async def run(self, task: str) -> dict:
        """
        审查代码变更。

        调用 LLM 使用 REVIEWER_CODE_REVIEW_PROMPT，返回结构化审查结果。

        Args:
            task: 审查任务描述（通常包含 diff 或文件路径）

        Returns:
            dict: {"verdict": "LGTM|NEEDS_WORK|REJECT", "issues": [...], ...}
        """
        focus = ""
        if isinstance(task, dict):
            focus = task.get("focus", "")
            task = task.get("task", str(task))
        prompt = REVIEWER_CODE_REVIEW_PROMPT.replace("{task}", task).replace("{focus}", focus)

        text = await self._call_llm(
            system=self.role.system_prompt,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        result = self._parse_review_result(text)
        return {
            **result,
            "agent": self.agent_id,
        }

    # ── Task handler 覆盖（核心差异化逻辑） ─────────────────

    async def _handle_task(self, msg: MailboxMessage) -> bool:
        """
        处理 TASK_ASSIGNMENT —— 审查后根据 verdict 分别处理。

        与 BaseAgent 默认行为的区别：
          - LGTM       → TASK_RESULT 给 sender（与默认相同）
          - NEEDS_WORK → REVIEW_FEEDBACK 给 coder + TASK_RESULT 给 sender
          - REJECT     → REVIEW_FEEDBACK 给 coder + TASK_RESULT 给 sender（含 reject 原因）
        """
        task = msg.body if isinstance(msg.body, str) else msg.body.get("task", "")
        subject = (
            msg.subject
            or (msg.body.get("subject", "") if isinstance(msg.body, dict) else "")
        )
        self._last_task = task if isinstance(task, str) else json.dumps(task)
        self._last_correlation_id = msg.correlation_id

        print(f"  [{self.agent_id}] TASK_ASSIGNMENT received: {subject or str(task)[:80]}")

        result = await self.run(task)
        verdict = result.get("verdict", "NEEDS_WORK")
        issues = result.get("issues", [])
        must_fix = [i for i in issues if i.get("must_fix", False)]
        non_blocking = [i for i in issues if not i.get("must_fix", False)]

        if verdict == "LGTM":
            # ── 审查通过 → TASK_RESULT 给 sender ──
            reply = self.outbox.create_message(
                recipient=msg.sender,
                msg_type=MessageType.TASK_RESULT,
                body=result,
                subject=f"Review LGTM: {subject[:60]}",
                correlation_id=msg.correlation_id,
                reply_to=msg.id,
            )
            await self.outbox.send(reply)
            print(f"  [{self.agent_id}] Review LGTM → TASK_RESULT sent to {msg.sender}")

        else:
            # ── 有问题 → REVIEW_FEEDBACK 给 Coder ──
            feedback_msg = self.outbox.create_message(
                recipient="coder",
                msg_type=MessageType.REVIEW_FEEDBACK,
                body={
                    "feedback": {
                        "verdict": verdict,
                        "must_fix": must_fix,
                        "suggestions": non_blocking,
                        "scores": result.get("scores", {}),
                        "summary": result.get("summary", ""),
                        "praise": result.get("praise", []),
                    },
                    "original_task": self._last_task,
                },
                subject=(
                    f"Review {verdict}: {len(must_fix)} must-fix, "
                    f"{len(non_blocking)} suggestions — {subject[:40]}"
                ),
                correlation_id=msg.correlation_id,
                reply_to=msg.id,
            )
            await self.outbox.send(feedback_msg)
            print(
                f"  [{self.agent_id}] Review {verdict}: "
                f"{len(must_fix)} must-fix + {len(non_blocking)} suggestions "
                f"→ REVIEW_FEEDBACK sent to coder"
            )

            # ── 同时通知 sender 审查状态 ──
            status_reply = self.outbox.create_message(
                recipient=msg.sender,
                msg_type=MessageType.TASK_RESULT,
                body={
                    **result,
                    "note": (
                        f"REVIEW_FEEDBACK sent to coder with "
                        f"{len(must_fix)} must-fix issues and "
                        f"{len(non_blocking)} suggestions"
                    ),
                },
                subject=f"Review {verdict}: {len(must_fix)}/{len(issues)} issues — {subject[:50]}",
                correlation_id=msg.correlation_id,
                reply_to=msg.id,
            )
            await self.outbox.send(status_reply)

        return True

    # ── 解析辅助 ──────────────────────────────────────────

    @staticmethod
    def _parse_review_result(text: str) -> dict:
        result = extract_json(text)
        if result.get("_parse_error"):
            return {"verdict": "NEEDS_WORK", "scores": {}, "issues": [],
                    "praise": [], "summary": text[:200]}
        return result
