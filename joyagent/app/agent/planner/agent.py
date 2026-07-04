"""
Phase 7 Step 3 — PlannerAgent

Planner Agent —— 任务拆解 + 冲突仲裁。

职责：
  1. 接收 TASK_ASSIGNMENT → 调用 LLM 拆解用户需求为可执行步骤
  2. 处理 CONFLICT_ESCALATE → 作为仲裁者解决 Agent 间冲突
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
from app.agent.planner.prompts import (
    PLANNER_CONFLICT_ARBITRATION_PROMPT,
    PLANNER_TASK_DECOMPOSE_PROMPT,
)
from app.agent.roles import AgentRole
from app.core.config import Config


class PlannerAgent(BaseAgent):
    """
    Planner Agent —— 任务拆解 + 冲突仲裁。

    相比 BaseAgent 的额外能力：
      - 注册 CONFLICT_ESCALATE handler → 作为仲裁者
      - run() 使用任务拆解 prompt，返回结构化计划

    Example:
        planner = PlannerAgent(ROLE_PLANNER, mailbox_manager)
        planner.start_background()
        # Planner 现在会监听 Inbox 中的 TASK_ASSIGNMENT 和 CONFLICT_ESCALATE
    """

    def __init__(
        self,
        role: AgentRole,
        mailbox_manager: MailboxManager,
        agent_id: str = "",
    ):
        super().__init__(role, mailbox_manager, agent_id)

        # ── 注册冲突升级处理器 ──
        self.inbox.register_handler(
            MessageType.CONFLICT_ESCALATE, self._handle_conflict_escalation
        )

    # ── 核心逻辑：任务拆解 ──────────────────────────────────

    async def run(self, task: str) -> dict:
        """
        拆解用户任务为可执行步骤。

        调用 LLM 使用 PLANNER_TASK_DECOMPOSE_PROMPT，
        返回结构化的步骤列表。

        Args:
            task: 用户需求描述文本

        Returns:
            dict: {"plan": [...], "summary": "...", "estimated_time": "..."}
        """
        model_name = self.role.model or Config.DEFAULT_MODEL
        prompt = PLANNER_TASK_DECOMPOSE_PROMPT.format(task=task)

        response = self.client.messages.create(
            model=model_name,
            system=self.role.system_prompt,
            messages=[{"role": "user", "content": prompt}],
            tools=self.tools,
            max_tokens=4096,
        )

        text = extract_text(response.content)
        plan = self._parse_plan(text)
        return {
            **plan,
            "agent": self.agent_id,
        }

    # ── 冲突仲裁 ──────────────────────────────────────────

    async def _handle_conflict_escalation(self, msg: MailboxMessage) -> bool:
        """
        处理 CONFLICT_ESCALATE 消息 —— 仲裁 Agent 间冲突。

        提取冲突详情 → 调用 LLM 仲裁 → 回复决策给发送方。

        消息被标记为 URGENT 优先级，会跳过普通消息队列优先处理。
        """
        body = msg.body if isinstance(msg.body, dict) else {}
        issue = body.get("issue", str(msg.body))
        claim_a = body.get("coder_claim", body.get("claim_a", ""))
        claim_b = body.get("reviewer_claim", body.get("claim_b", ""))
        agent_a = body.get("agent_a", msg.sender)
        agent_b = body.get("agent_b", "reviewer")

        print(f"  [{self.agent_id}] CONFLICT_ESCALATE received: {issue[:80]}")

        # 调用 LLM 进行仲裁
        model_name = self.role.model or Config.DEFAULT_MODEL
        prompt = PLANNER_CONFLICT_ARBITRATION_PROMPT.format(
            issue=issue,
            agent_a=agent_a,
            claim_a=claim_a,
            agent_b=agent_b,
            claim_b=claim_b,
        )

        response = self.client.messages.create(
            model=model_name,
            system=self.role.system_prompt,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )

        text = extract_text(response.content)
        decision = self._parse_decision(text)

        # 将仲裁结果发送回冲突发起方
        reply = self.outbox.create_message(
            recipient=msg.sender,
            msg_type=MessageType.TASK_RESULT,
            body={
                "decision": decision.get("decision", ""),
                "reasoning": decision.get("reasoning", ""),
                "action": decision.get("action", ""),
                "compromise": decision.get("compromise"),
                "arbitrated_by": self.agent_id,
            },
            subject=f"Arbitration: {issue[:60]}",
            correlation_id=msg.correlation_id,
            reply_to=msg.id,
        )
        await self.outbox.send(reply)
        print(f"  [{self.agent_id}] Arbitration decision sent → {msg.sender}")
        return True

    # ── 解析辅助 ──────────────────────────────────────────

    @staticmethod
    def _parse_plan(text: str) -> dict:
        """
        从 LLM 响应中解析结构化计划。

        尝试从响应中提取 JSON。失败时返回原始文本。
        """
        try:
            # 找到第一个 { 和最后一个 }
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end + 1]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: 返回原始文本作为非结构化计划
        return {
            "summary": "Plan (unstructured)",
            "steps": [{"step": 1, "agent": "coder", "task": text}],
            "estimated_time": "unknown",
            "raw_output": text,
        }

    @staticmethod
    def _parse_decision(text: str) -> dict:
        """
        从 LLM 响应中解析仲裁决策。

        尝试从响应中提取 JSON。失败时返回原始文本。
        """
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end + 1]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

        return {
            "decision": text.strip(),
            "reasoning": "Parsed from unstructured response",
            "action": "review decision text above",
            "compromise": None,
        }
