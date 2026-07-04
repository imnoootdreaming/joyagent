"""
Phase 7 Step 3 — CoderAgent

Coder Agent —— 代码编写 + 反馈处理。

职责：
  1. 接收 TASK_ASSIGNMENT → 编写或修改代码
  2. 处理 REVIEW_FEEDBACK → 应用 Reviewer/Tester 的反馈
  3. 通过 Outbox 发送 TASK_RESULT 回 Router
"""

from __future__ import annotations

import json

from app.agent.base import BaseAgent, extract_text
from app.agent.mailbox import (
    MailboxManager,
    MailboxMessage,
    MessageType,
)
from app.agent.coder.prompts import (
    CODER_REVIEW_FEEDBACK_PROMPT,
    CODER_TASK_PROMPT,
)
from app.agent.roles import AgentRole
from app.core.config import Config


class CoderAgent(BaseAgent):
    """
    Coder Agent —— 代码编写 + 反馈处理。

    相比 BaseAgent 的额外能力：
      - 注册 REVIEW_FEEDBACK handler → 处理 Reviewer/Tester 反馈
      - run() 使用编码 prompt，返回结构化代码变更

    Example:
        coder = CoderAgent(ROLE_CODER, mailbox_manager)
        coder.start_background()
        # Coder 现在会监听 Inbox 中的 TASK_ASSIGNMENT 和 REVIEW_FEEDBACK
    """

    def __init__(
        self,
        role: AgentRole,
        mailbox_manager: MailboxManager,
        agent_id: str = "",
    ):
        super().__init__(role, mailbox_manager, agent_id)

        # ── 注册审查反馈处理器 ──
        self.inbox.register_handler(
            MessageType.REVIEW_FEEDBACK, self._handle_review_feedback
        )

        # ── 存储最近的任务上下文（供 feedback handler 使用） ──
        self._last_task: str = ""
        self._last_correlation_id: str = ""

    # ── 核心逻辑：编码 ──────────────────────────────────────

    async def run(self, task: str) -> dict:
        """
        执行编码任务。

        调用 LLM 使用 CODER_TASK_PROMPT，返回结构化代码变更。

        Args:
            task: 编码任务描述

        Returns:
            dict: {"files_created": [...], "files_modified": [...], ...}
        """
        model_name = self.role.model or Config.DEFAULT_MODEL
        prompt = CODER_TASK_PROMPT.format(task=task)

        response = self.client.messages.create(
            model=model_name,
            system=self.role.system_prompt,
            messages=[{"role": "user", "content": prompt}],
            tools=self.tools,
            max_tokens=8192,
        )

        text = extract_text(response.content)
        result = self._parse_code_result(text)
        return {
            **result,
            "agent": self.agent_id,
        }

    # ── Task handler 覆盖（保存上下文） ──────────────────────

    async def _handle_task(self, msg: MailboxMessage) -> bool:
        """
        处理 TASK_ASSIGNMENT —— 保存任务上下文后委托给父类。

        保存的任务上下文用于后续 REVIEW_FEEDBACK 处理时
        提供原始任务信息。
        """
        task = msg.body if isinstance(msg.body, str) else msg.body.get("task", "")
        self._last_task = task
        self._last_correlation_id = msg.correlation_id
        return await super()._handle_task(msg)

    # ── 反馈处理 ──────────────────────────────────────────

    async def _handle_review_feedback(self, msg: MailboxMessage) -> bool:
        """
        处理 REVIEW_FEEDBACK 消息 —— 应用审查意见。

        流程：
          1. 提取反馈内容
          2. 调用 LLM 分析反馈并生成修改
          3. 发送 TASK_RESULT 回发送方（Reviewer/Router）
        """
        body = msg.body if isinstance(msg.body, dict) else {}
        feedback = body.get("feedback", body.get("issues", str(msg.body)))
        original_task = body.get("original_task", self._last_task)

        print(f"  [{self.agent_id}] REVIEW_FEEDBACK received: {msg.subject[:80]}")

        # 调用 LLM 处理反馈
        model_name = self.role.model or Config.DEFAULT_MODEL
        prompt = CODER_REVIEW_FEEDBACK_PROMPT.format(
            original_task=original_task or "Unknown",
            feedback=json.dumps(feedback, indent=2, ensure_ascii=False)
            if isinstance(feedback, (dict, list))
            else str(feedback),
        )

        response = self.client.messages.create(
            model=model_name,
            system=self.role.system_prompt,
            messages=[{"role": "user", "content": prompt}],
            tools=self.tools,
            max_tokens=8192,
        )

        text = extract_text(response.content)
        result = self._parse_feedback_result(text)

        # 回复发送方
        reply = self.outbox.create_message(
            recipient=msg.sender,
            msg_type=MessageType.TASK_RESULT,
            body=result,
            subject=f"Feedback applied: {msg.subject[:60]}",
            correlation_id=msg.correlation_id or self._last_correlation_id,
            reply_to=msg.id,
        )
        await self.outbox.send(reply)
        print(f"  [{self.agent_id}] Feedback result sent → {msg.sender}")
        return True

    # ── 解析辅助 ──────────────────────────────────────────

    @staticmethod
    def _parse_code_result(text: str) -> dict:
        """从 LLM 响应中解析代码生成结果。"""
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end + 1]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

        return {
            "files_created": [],
            "files_modified": [],
            "summary": text[:200],
            "notes": "Parsed from unstructured response",
            "raw_output": text,
        }

    @staticmethod
    def _parse_feedback_result(text: str) -> dict:
        """从 LLM 响应中解析反馈处理结果。"""
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end + 1]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

        return {
            "changes_made": [],
            "issues_declined": [],
            "summary": text[:200],
            "raw_output": text,
        }
