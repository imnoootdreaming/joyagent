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

from app.agent.base import BaseAgent, extract_json, extract_text
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
        prompt_text = PLANNER_TASK_DECOMPOSE_PROMPT.replace("{task}", task)

        try:
            text = await self._call_llm(
                system=self.role.system_prompt,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=4096,
            )
            plan = self._parse_plan(text)
        except Exception as e:
            print(f"  [{self.agent_id}] LLM call failed: {e} — "
                  f"returning fallback plan", flush=True)
            plan = {
                "summary": f"Fallback plan for: {task[:60]}",
                "steps": [{"step": 1, "agent": "coder", "task": task}],
                "estimated_time": "unknown",
            }

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
        prompt_text = (PLANNER_CONFLICT_ARBITRATION_PROMPT
                       .replace("{issue}", issue)
                       .replace("{agent_a}", agent_a)
                       .replace("{claim_a}", claim_a)
                       .replace("{agent_b}", agent_b)
                       .replace("{claim_b}", claim_b))

        text = await self._call_llm(
            system=self.role.system_prompt,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=2048,
        )

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
        result = extract_json(text)
        if result.get("_parse_error"):
            return {"summary": "Plan (unstructured)",
                    "steps": [{"step": 1, "agent": "coder", "task": text}],
                    "estimated_time": "unknown", "raw_output": text}
        return result

    @staticmethod
    def _parse_decision(text: str) -> dict:
        result = extract_json(text)
        if result.get("_parse_error"):
            return {"decision": text.strip(), "reasoning": "Unstructured",
                    "action": "review", "compromise": None}
        return result
