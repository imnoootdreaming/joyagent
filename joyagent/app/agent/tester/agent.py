"""
Phase 7 Step 3 — TesterAgent

Tester Agent —— 测试执行 + 失败反馈。

职责：
  1. 接收 TASK_ASSIGNMENT → 执行测试验证代码
  2. 测试失败时 → 发 REVIEW_FEEDBACK 给 Coder（而非直接 TASK_RESULT）
  3. 测试通过时 → 发 TASK_RESULT 给 Router

关键设计决策（与 BaseAgent 默认行为不同）：
  BaseAgent._handle_task: 完成后总是发 TASK_RESULT 回 sender
  TesterAgent._handle_task: 根据测试结果分别处理
    - 全部通过 → TASK_RESULT 回 Router（与默认行为相同）
    - 有失败 → REVIEW_FEEDBACK 给 Coder + TASK_RESULT 给 Router（含失败摘要）
"""

from __future__ import annotations

import json

from app.agent.base import BaseAgent, extract_json, extract_text
from app.agent.mailbox import (
    MailboxManager,
    MailboxMessage,
    MessageType,
)
from app.agent.tester.prompts import TESTER_EXECUTION_PROMPT
from app.agent.roles import AgentRole
from app.core.config import Config


class TesterAgent(BaseAgent):
    """
    Tester Agent —— 测试执行 + 失败反馈。

    相比 BaseAgent 的额外行为：
      - 覆盖 _handle_task → 测试失败时发送 REVIEW_FEEDBACK 给 Coder
      - run() 使用测试执行 prompt，返回结构化测试结果

    Example:
        tester = TesterAgent(ROLE_TESTER, mailbox_manager)
        tester.start_background()
    """

    __test__ = False  # 避免 pytest 将其收集为测试类

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

    # ── 核心逻辑：测试执行 ──────────────────────────────────

    async def run(self, task: str) -> dict:
        """
        执行测试任务。

        调用 LLM 使用 TESTER_EXECUTION_PROMPT，返回结构化测试结果。

        Args:
            task: 测试任务描述

        Returns:
            dict: {"passed": bool, "failures": [...], "summary": "..."}
        """
        model_name = self.role.model or Config.DEFAULT_MODEL
        focus = ""
        if isinstance(task, dict):
            focus = task.get("focus", "")
            task = task.get("task", str(task))
        prompt = TESTER_EXECUTION_PROMPT.replace("{task}", task).replace("{focus}", focus)

        text = await self._call_llm(
            system=self.role.system_prompt,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        result = self._parse_test_result(text)
        return {
            **result,
            "agent": self.agent_id,
        }

    # ── Task handler 覆盖（核心差异化逻辑） ─────────────────

    async def _handle_task(self, msg: MailboxMessage) -> bool:
        """
        处理 TASK_ASSIGNMENT —— 测试失败时发送 REVIEW_FEEDBACK 给 Coder。

        与 BaseAgent 默认行为的区别：
          - 全部通过 → TASK_RESULT 给 sender（与默认相同）
          - 有失败   → REVIEW_FEEDBACK 给 coder + TASK_RESULT 给 sender
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

        passed = result.get("passed", False)
        failures = result.get("failures", [])

        if passed:
            # ── 全部通过 → TASK_RESULT 给 sender ──
            reply = self.outbox.create_message(
                recipient=msg.sender,
                msg_type=MessageType.TASK_RESULT,
                body=result,
                subject=f"Tests PASSED: {subject[:60]}",
                correlation_id=msg.correlation_id,
                reply_to=msg.id,
            )
            await self.outbox.send(reply)
            print(f"  [{self.agent_id}] All tests passed → TASK_RESULT sent to {msg.sender}")
        else:
            # ── 有失败 → REVIEW_FEEDBACK 给 Coder ──
            feedback = self.outbox.create_message(
                recipient="coder",
                msg_type=MessageType.REVIEW_FEEDBACK,
                body={
                    "feedback": {
                        "test_passed": False,
                        "failures": failures,
                        "summary": result.get("summary", ""),
                    },
                    "original_task": self._last_task,
                    "from_tester": True,
                },
                subject=f"Test failures: {len(failures)} failed — {subject[:50]}",
                correlation_id=msg.correlation_id,
                reply_to=msg.id,
            )
            await self.outbox.send(feedback)
            print(
                f"  [{self.agent_id}] {len(failures)} test(s) failed "
                f"→ REVIEW_FEEDBACK sent to coder"
            )

            # ── 同时通知 sender 测试状态 ──
            status_reply = self.outbox.create_message(
                recipient=msg.sender,
                msg_type=MessageType.TASK_RESULT,
                body={
                    **result,
                    "note": f"REVIEW_FEEDBACK with {len(failures)} failure(s) sent to coder",
                },
                subject=f"Tests FAILED: {len(failures)}/{result.get('total_tests', '?')} — {subject[:50]}",
                correlation_id=msg.correlation_id,
                reply_to=msg.id,
            )
            await self.outbox.send(status_reply)

        return True

    # ── 解析辅助 ──────────────────────────────────────────

    @staticmethod
    @staticmethod
    def _parse_test_result(text: str) -> dict:
        result = extract_json(text)
        if result.get("_parse_error"):
            return {"passed": False, "total_tests": 0, "passed_count": 0,
                    "failed_count": 1, "failures": [], "summary": text[:200]}
        return result
