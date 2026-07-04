"""
Phase 7 Step 4 — RouterAgent

Router Agent —— 所有用户请求的入口点。

职责：
  1. 分析用户请求 → 简单任务直接路由，复杂任务经 Planner 拆解
  2. 接收 Planner 计划 → 按依赖顺序分发到 Coder/Tester/Reviewer
  3. 收集各 Agent 的 TASK_RESULT → 聚合返回给用户
  4. 处理 ERROR_REPORT → 决定重试/重新分配/升级

路由策略：
  - 规则路由: 关键词匹配 → 简单任务直接分发
  - LLM 路由: 复杂/模糊任务 → 发 Planner 拆解后按计划分发

与 BaseAgent 的关键区别：
  - 注册 TASK_RESULT handler → 收集 Agent 的执行结果
  - 注册 ERROR_REPORT handler → 处理 Agent 执行异常
  - handle_user_request() → 用户请求入口（同步等待结果）
  - _pending_results + _result_events → 异步结果收集机制
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from app.agent.base import BaseAgent, extract_text
from app.agent.mailbox import (
    MailboxManager,
    MailboxMessage,
    MessagePriority,
    MessageStatus,
    MessageType,
)
from app.agent.router.prompts import (
    ROUTER_AGGREGATION_PROMPT,
    ROUTER_ANALYSIS_PROMPT,
)
from app.agent.roles import AgentRole
from app.core.config import Config


class RouterAgent(BaseAgent):
    """
    Router Agent —— 所有用户请求的入口点 + 多 Agent 协调中枢。

    相比 BaseAgent 的额外能力：
      - 注册 TASK_RESULT handler → 收集 Agent 执行结果
      - 注册 ERROR_REPORT handler → 处理异常
      - handle_user_request() → 用户请求主入口
      - 规则路由 + LLM 路由双策略

    Example:
        router = RouterAgent(ROLE_ROUTER, mailbox_manager)
        result = await router.handle_user_request("Build a FastAPI user CRUD")
        print(result["summary"])
    """

    def __init__(
        self,
        role: AgentRole,
        mailbox_manager: MailboxManager,
        agent_id: str = "",
    ):
        super().__init__(role, mailbox_manager, agent_id)

        # ── 注册 Router 专属处理器 ──
        self.inbox.register_handler(MessageType.TASK_RESULT, self._handle_task_result)
        self.inbox.register_handler(MessageType.ERROR_REPORT, self._handle_error_report)

        # ── 结果收集 ──
        # _pending_results: correlation_id → list[dict]  (收集同线程的多个结果)
        # _result_events:   correlation_id → asyncio.Event (通知等待者)
        self._pending_results: dict[str, list[dict]] = {}
        self._result_events: dict[str, asyncio.Event] = {}

    # ── 用户请求入口 ─────────────────────────────────────────

    async def handle_user_request(
        self,
        user_message: str,
        timeout: float = 120.0,
    ) -> dict:
        """
        用户请求主入口 —— 分析 → 路由 → 收集 → 聚合。

        流程：
          1. 分析请求复杂度
          2. 简单 → 直接路由到 Coder/Tester/Reviewer
          3. 复杂 → 经 Planner 拆解后按计划分发
          4. 等待所有 Agent 完成
          5. 聚合结果返回

        Args:
            user_message: 用户原始请求
            timeout: 总超时秒数

        Returns:
            dict: {
                "summary": "聚合后的摘要",
                "steps": [每个步骤的结果],
                "success": True/False,
                "correlation_id": "...",
            }
        """
        correlation_id = f"user_{uuid.uuid4().hex[:8]}"
        print(f"\n{'='*60}")
        print(f"  [router] New user request: {user_message[:80]}")
        print(f"  [router] correlation_id: {correlation_id}")

        # ── Step 1: 分析请求 → 决定路由策略 ──
        analysis = await self._analyze_request(user_message)

        if analysis["complexity"] == "simple" and analysis["route"]["target"] != "planner":
            # ── 简单任务：直接路由 ──
            print(f"  [router] Simple task → routing directly to {analysis['route']['target']}")
            result = await self._route_simple(
                target=analysis["route"]["target"],
                task=analysis["route"]["task"],
                correlation_id=correlation_id,
                timeout=timeout,
            )
        else:
            # ── 复杂任务：经 Planner 拆解 ──
            print(f"  [router] Complex task → routing via Planner")
            result = await self._route_via_planner(
                user_message=user_message,
                correlation_id=correlation_id,
                timeout=timeout,
            )

        # ── Step final: 聚合 ──
        summary = await self._aggregate_results(user_message, result.get("steps", []))
        print(f"  [router] Request complete: {correlation_id}")
        print(f"{'='*60}\n")

        return {
            **summary,
            "success": result.get("success", True),
            "correlation_id": correlation_id,
        }

    # ── 请求分析 ────────────────────────────────────────────

    async def _analyze_request(self, user_message: str) -> dict:
        """
        分析用户请求 → 返回路由决策。

        策略：
          1. 先尝试规则匹配（快速路径）
          2. 规则无法匹配 → 调用 LLM 分析
        """
        # ── 规则路由：关键词匹配 ──
        rule_result = self._rule_based_route(user_message)
        if rule_result is not None:
            return rule_result

        # ── LLM 路由：调用 LLM 分析 ──
        return await self._llm_based_route(user_message)

    def _rule_based_route(self, user_message: str) -> dict | None:
        """
        规则路由 —— 基于关键词快速判断。

        匹配简单/明确的任务，直接路由到对应 Agent。
        无法匹配时返回 None，由 LLM 路由兜底。
        """
        msg_lower = user_message.lower()

        # ── 测试相关 → Tester ──
        test_keywords = ["run tests", "run test", "pytest", "unit test",
                         "integration test", "test coverage", "run the tests"]
        if any(kw in msg_lower for kw in test_keywords):
            # 纯测试任务，不涉及编码
            code_keywords = ["write", "create", "implement", "build", "generate", "fix"]
            if not any(kw in msg_lower for kw in code_keywords):
                return {
                    "complexity": "simple",
                    "reasoning": "Keyword match: test-only request → tester",
                    "route": {
                        "target": "tester",
                        "task": user_message,
                        "priority": "normal",
                    },
                }

        # ── 审查相关 → Reviewer ──
        review_keywords = ["review this", "code review", "check my code",
                           "audit", "inspect the code", "look over"]
        if any(kw in msg_lower for kw in review_keywords):
            return {
                "complexity": "simple",
                "reasoning": "Keyword match: review request → reviewer",
                "route": {
                    "target": "reviewer",
                    "task": user_message,
                    "priority": "normal",
                },
            }

        # ── 单文件、简单修改 → Coder ──
        simple_code_patterns = [
            "add a comment", "fix typo", "rename variable",
            "add docstring", "format code", "add type hint",
            "create a simple", "write a function", "add an endpoint",
            "create endpoint", "add logging",
        ]
        if any(kw in msg_lower for kw in simple_code_patterns):
            return {
                "complexity": "simple",
                "reasoning": "Keyword match: simple coding task → coder",
                "route": {
                    "target": "coder",
                    "task": user_message,
                    "priority": "normal",
                },
            }

        # ── 复杂关键词 → Planner ──
        complex_keywords = [
            "full stack", "entire system", "complete api",
            "microservice", "refactor the", "restructure",
            "authentication system", "database schema",
            "migration", "build a", "create a",
            "with tests", "and tests", "crud",
            "multi-file", "multiple files",
        ]
        if any(kw in msg_lower for kw in complex_keywords):
            return {
                "complexity": "complex",
                "reasoning": "Keyword match: multi-step/complex task → planner",
                "route": {
                    "target": "planner",
                    "task": user_message,
                    "priority": "normal",
                },
            }

        # ── 无法匹配 → LLM 兜底 ──
        return None

    async def _llm_based_route(self, user_message: str) -> dict:
        """
        LLM 路由 —— 调用 LLM 分析请求复杂度。

        规则匹配失败时的兜底策略。
        """
        model_name = self.role.model or Config.DEFAULT_MODEL
        prompt = ROUTER_ANALYSIS_PROMPT.format(user_message=user_message)

        try:
            response = self.client.messages.create(
                model=model_name,
                system=self.role.system_prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            text = extract_text(response.content)
            return self._parse_analysis(text)
        except Exception:
            # LLM 调用失败 → 默认走 Planner（安全策略）
            return {
                "complexity": "complex",
                "reasoning": "LLM analysis failed — defaulting to Planner for safety",
                "route": {
                    "target": "planner",
                    "task": user_message,
                    "priority": "normal",
                },
            }

    # ── 简单任务：直接路由 ────────────────────────────────────

    async def _route_simple(
        self,
        target: str,
        task: str,
        correlation_id: str,
        timeout: float,
    ) -> dict:
        """
        简单任务直接路由 —— 发 TASK_ASSIGNMENT 给目标 Agent，等待回复。

        Args:
            target: 目标 Agent 名称（coder/tester/reviewer）
            task: 任务描述
            correlation_id: 线程 ID
            timeout: 超时秒数

        Returns:
            dict: {"steps": [...], "success": True/False}
        """
        step_cid = f"{correlation_id}_s0"

        # 注册等待事件
        event = asyncio.Event()
        self._result_events[step_cid] = event
        self._pending_results[step_cid] = []

        # 发送任务
        msg = self.outbox.create_message(
            recipient=target,
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": task},
            subject=f"Direct task: {task[:60]}",
            correlation_id=step_cid,
        )
        await self.outbox.send(msg)

        # 等待回复
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return {
                "steps": [{"agent": target, "error": "Timeout waiting for response"}],
                "success": False,
            }

        results = self._pending_results.pop(step_cid, [])
        self._result_events.pop(step_cid, None)

        if not results:
            return {
                "steps": [{"agent": target, "error": "No result received"}],
                "success": False,
            }

        return {
            "steps": results,
            "success": all(r.get("success", True) for r in results),
        }

    # ── 复杂任务：经 Planner 拆解 ─────────────────────────────

    async def _route_via_planner(
        self,
        user_message: str,
        correlation_id: str,
        timeout: float,
    ) -> dict:
        """
        复杂任务流程：Planner 拆解 → 按步骤分发 → 收集结果。

        流程：
          1. Router → Planner: TASK_ASSIGNMENT
          2. 等待 Planner 返回计划
          3. 按依赖顺序将每个 step 分发给对应 Agent
          4. 收集所有 step 结果
          5. 返回聚合结果
        """
        # ── Phase 1: Planner 拆解 ──
        plan_cid = f"{correlation_id}_plan"
        plan_result = await self._send_and_wait_reply(
            recipient="planner",
            body={"task": user_message},
            subject=f"Plan: {user_message[:60]}",
            correlation_id=plan_cid,
            timeout=min(timeout * 0.4, 45.0),  # Planner 占 40% 超时
        )

        if plan_result is None:
            return {
                "steps": [{"agent": "planner", "error": "Planner did not respond in time"}],
                "success": False,
            }

        # ── 提取步骤 ──
        steps = plan_result.get("steps", [])
        if not steps:
            # Planner 返回空计划 → 可能是简单任务，直接发 Coder
            steps = [{"step": 1, "agent": "coder", "task": user_message}]

        print(f"  [router] Planner returned {len(steps)} step(s)")

        # ── Phase 2: 按顺序分发步骤 ──
        step_results = []
        all_success = True

        for step in steps:
            step_num = step.get("step", len(step_results) + 1)
            target = step.get("agent", "coder")
            task = step.get("task", str(step))
            depends_on = step.get("depends_on", [])

            # 检查依赖是否全部成功
            if depends_on:
                deps_failed = [
                    d for d in depends_on
                    if d <= len(step_results) and not step_results[d - 1].get("success", True)
                ]
                if deps_failed:
                    step_results.append({
                        "step": step_num,
                        "agent": target,
                        "task": task,
                        "error": f"Skipped: dependency step(s) {deps_failed} failed",
                        "success": False,
                    })
                    all_success = False
                    continue

            # 发送并等待
            step_cid = f"{correlation_id}_s{step_num}"
            result = await self._send_and_wait_reply(
                recipient=target,
                body={"task": task, "step": step_num},
                subject=f"Step {step_num}: {task[:60]}",
                correlation_id=step_cid,
                timeout=min(timeout * 0.5 / max(len(steps), 1), 60.0),
            )

            if result is None:
                step_results.append({
                    "step": step_num,
                    "agent": target,
                    "task": task,
                    "error": "Agent did not respond in time",
                    "success": False,
                })
                all_success = False
            else:
                step_results.append({
                    "step": step_num,
                    "agent": target,
                    "task": task,
                    **result,
                })
                if not result.get("success", True):
                    all_success = False

        return {
            "steps": step_results,
            "success": all_success,
            "plan": plan_result,
        }

    # ── 发送并等待回复 ───────────────────────────────────────

    async def _send_and_wait_reply(
        self,
        recipient: str,
        body: dict,
        subject: str,
        correlation_id: str,
        timeout: float = 30.0,
    ) -> dict | None:
        """
        发送 TASK_ASSIGNMENT 并等待 TASK_RESULT 回复。

        使用 _result_events 异步机制：
          1. 注册 Event + 结果槽位
          2. 发送消息
          3. 阻塞等待 Event.set() 或超时

        Args:
            recipient: 目标 Agent
            body: 消息体
            subject: 消息标题
            correlation_id: 线程 ID
            timeout: 等待超时

        Returns:
            dict | None: Agent 返回的结果 body，超时返回 None
        """
        # 注册等待机制
        event = asyncio.Event()
        self._result_events[correlation_id] = event
        self._pending_results[correlation_id] = []

        # 发送
        msg = self.outbox.create_message(
            recipient=recipient,
            msg_type=MessageType.TASK_ASSIGNMENT,
            body=body,
            subject=subject,
            correlation_id=correlation_id,
        )
        await self.outbox.send(msg)

        # 等待
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self._result_events.pop(correlation_id, None)
            self._pending_results.pop(correlation_id, None)
            return None

        # 收集结果
        results = self._pending_results.pop(correlation_id, [])
        self._result_events.pop(correlation_id, None)

        if not results:
            return None
        return results[0]  # 返回第一个（也是唯一一个）结果

    # ── 结果聚合 ────────────────────────────────────────────

    async def _aggregate_results(
        self,
        user_message: str,
        steps: list[dict],
    ) -> dict:
        """
        聚合多个 Agent 的执行结果为用户可读的摘要。

        简单情况（1-2 步）→ 直接拼接
        复杂情况（3+ 步）→ 调用 LLM 聚合
        """
        if not steps:
            return {"summary": "No steps were executed.", "details": []}

        if len(steps) <= 2:
            # 简单情况：直接拼接
            lines = []
            for s in steps:
                agent = s.get("agent", "unknown")
                if s.get("error"):
                    lines.append(f"❌ {agent}: {s['error']}")
                else:
                    summary = s.get("summary", s.get("output", "Completed"))
                    lines.append(f"✅ {agent}: {summary}")
            return {
                "summary": "\n".join(lines),
                "details": steps,
            }

        # 复杂情况：调用 LLM 聚合
        model_name = self.role.model or Config.DEFAULT_MODEL
        agent_results_str = json.dumps(steps, indent=2, ensure_ascii=False)
        prompt = ROUTER_AGGREGATION_PROMPT.format(
            user_message=user_message,
            agent_results=agent_results_str,
        )

        try:
            response = self.client.messages.create(
                model=model_name,
                system=self.role.system_prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
            text = extract_text(response.content)
            return {"summary": text, "details": steps}
        except Exception:
            # LLM 失败 → 降级为简单拼接
            lines = [f"Step {s.get('step', i+1)}: {s.get('agent', '?')} — "
                     f"{s.get('summary', s.get('error', 'unknown'))}"
                     for i, s in enumerate(steps)]
            return {"summary": "\n".join(lines), "details": steps}

    # ── TASK_RESULT 处理器 ──────────────────────────────────

    async def _handle_task_result(self, msg: MailboxMessage) -> bool:
        """
        处理 TASK_RESULT 消息 —— 收集 Agent 执行结果。

        当结果到达时：
          1. 存储到 _pending_results
          2. 设置 _result_events 中的 Event（通知等待的 handle_user_request）
        """
        body = msg.body if isinstance(msg.body, dict) else {"output": str(msg.body)}
        cid = msg.correlation_id

        print(f"  [router] TASK_RESULT received from {msg.sender} (cid={cid})")

        # 存储结果
        if cid not in self._pending_results:
            self._pending_results[cid] = []
        self._pending_results[cid].append({
            "agent": msg.sender,
            "success": body.get("success", body.get("passed", True)),
            **body,
        })

        # 通知等待者
        event = self._result_events.get(cid)
        if event is not None:
            event.set()

        return True

    # ── ERROR_REPORT 处理器 ─────────────────────────────────

    async def _handle_error_report(self, msg: MailboxMessage) -> bool:
        """
        处理 ERROR_REPORT 消息 —— Agent 执行异常。

        策略：
          1. 记录错误
          2. 如果是可重试的错误 → 重新发送任务
          3. 如果是严重错误 → 通知等待者任务失败
        """
        body = msg.body if isinstance(msg.body, dict) else {}
        error_msg = body.get("error", str(msg.body))
        cid = msg.correlation_id

        print(f"  [router] ERROR_REPORT from {msg.sender}: {error_msg[:80]}")

        # 存储错误结果
        if cid not in self._pending_results:
            self._pending_results[cid] = []
        self._pending_results[cid].append({
            "agent": msg.sender,
            "success": False,
            "error": error_msg,
            "error_details": body,
        })

        # 通知等待者（避免永久阻塞）
        event = self._result_events.get(cid)
        if event is not None:
            event.set()

        return True

    # ── 状态查询 ────────────────────────────────────────────

    @property
    def pending_request_count(self) -> int:
        """当前等待中的请求数量。"""
        return len(self._result_events)

    @property
    def stats(self) -> dict:
        """Router 状态快照（覆盖父类，增加 Router 专属指标）。"""
        base = super().stats
        base["pending_requests"] = self.pending_request_count
        base["collected_results"] = sum(len(v) for v in self._pending_results.values())
        return base

    # ── 解析辅助 ────────────────────────────────────────────

    @staticmethod
    def _parse_analysis(text: str) -> dict:
        """从 LLM 响应中解析路由分析结果。"""
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end + 1]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

        return {
            "complexity": "complex",
            "reasoning": "Could not parse LLM analysis — defaulting to Planner",
            "route": {
                "target": "planner",
                "task": text[:200],
                "priority": "normal",
            },
        }
