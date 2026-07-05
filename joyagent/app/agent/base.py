"""
Phase 7 Step 2 — BaseAgent 基类（集成 Mailbox 通信）

所有 Agent 的基类 —— 集成 Mailbox 通信。

每个 Agent 实例自动获得：
  - self.inbox:  AgentInbox（收件箱）
  - self.outbox: AgentOutbox（发件箱）
  - self._watcher: InboxWatcher（后台消息监听）

子类需要覆盖：
  - run(task) → dict    具体 Agent 的执行逻辑

子类可选覆盖：
  - _handle_task(msg)    自定义 TASK_ASSIGNMENT 处理逻辑
  - _handle_broadcast(msg)  自定义 BROADCAST 处理逻辑
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from app.agent.mailbox import (
    AgentInbox,
    AgentOutbox,
    InboxWatcher,
    MailboxManager,
    MailboxMessage,
    MessageType,
)
from app.agent.roles import AgentRole
from app.core.config import Config
from app.service.llm_service import get_or_create_client

if TYPE_CHECKING:
    pass


# ═══════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# JSON 提取工具 —— 所有 Agent 共用
# ═══════════════════════════════════════════════════════════════════════

def extract_json(text: str) -> dict:
    """
    从 LLM 响应文本中提取 JSON 对象。比简单的 find('{')/rfind('}')
    更健壮——忽略 markdown 代码块、跳过前导说明文字。

    策略：
      1. 去除 markdown ```json ... ``` 代码块标记
      2. 找到第一个 { 和最后一个 }
      3. json.loads 解析，失败返回 fallback dict
    """
    import re
    # 去掉 markdown 代码块
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = text.replace('```', '')

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end + 1]
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
    return {"_parse_error": True, "raw_text": text.strip()}


def extract_text(content: list) -> str:
    """
    从 Anthropic Messages API 响应 content 中提取纯文本。

    响应 content 是 block 列表，每个 block 有 type 字段：
      {"type": "text", "text": "LLM 的输出..."}
      {"type": "tool_use", ...}

    只提取 type=="text" 的 block。
    """
    text_parts = []
    for block in content:
        if hasattr(block, "type") and block.type == "text":
            text_parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))
    return "\n".join(text_parts).strip()


# ═══════════════════════════════════════════════════════════════════════
# BaseAgent
# ═══════════════════════════════════════════════════════════════════════

class BaseAgent:
    """
    所有 Agent 的基类 —— 集成 Mailbox 通信。

    每个 Agent 实例自动获得：
      - self.inbox:  AgentInbox（收件箱）
      - self.outbox: AgentOutbox（发件箱）
      - self._watcher: InboxWatcher（后台消息监听）

    子类需要覆盖：
      - run(task) → dict    具体 Agent 的执行逻辑

    子类可选覆盖：
      - _handle_task(msg)    自定义 TASK_ASSIGNMENT 处理逻辑
      - _handle_broadcast(msg)  自定义 BROADCAST 处理逻辑

    Example:
        class CoderAgent(BaseAgent):
            async def run(self, task: str) -> dict:
                # 编码逻辑
                return {"output": "...", "files": ["..."], "success": True}

        coder = CoderAgent(ROLE_CODER, mailbox_manager)
        coder.start_background()  # 开始监听 Inbox
    """

    def __init__(
        self,
        role: AgentRole,
        mailbox_manager: MailboxManager,
        agent_id: str = "",
    ):
        """
        Args:
            role: Agent 角色定义（AgentRole）
            mailbox_manager: 全局 MailboxManager 实例
            agent_id: Agent 唯一标识（默认使用 role.name）
        """
        self.role = role
        self.agent_id = agent_id or role.name

        # ── LLM 客户端 ──
        model_name = role.model or Config.DEFAULT_MODEL
        self.client = get_or_create_client(model_name)

        # ── 工具集 ──
        self.tools = self._resolve_tools(role.tools)

        # ── Mailbox 系统 ──
        self.inbox = AgentInbox(owner=self.agent_id)
        self.outbox = AgentOutbox(owner=self.agent_id)
        self._manager = mailbox_manager
        mailbox_manager.register_agent(self.agent_id, self.inbox, self.outbox)

        # ── 注册默认消息处理器 ──
        self.inbox.register_handler(MessageType.TASK_ASSIGNMENT, self._handle_task)
        self.inbox.register_handler(MessageType.STATUS_QUERY, self._handle_status_query)
        self.inbox.register_handler(MessageType.BROADCAST, self._handle_broadcast)

        # ── 启动 watcher ──
        self._watcher = InboxWatcher(self.inbox)

        # ── 子类可额外注册的 handler ──
        self._extra_handlers: dict[MessageType, object] = {}

    # ── 工具解析 ──────────────────────────────────────────

    def _resolve_tools(self, tool_names: list[str]) -> list[dict]:
        """
        将工具名称列表解析为 Anthropic tool schema 列表。

        如果 tool_names 为空 → 返回空列表。
        延迟导入 tool_registry（避免 docker 等可选依赖的导入错误）。
        """
        if not tool_names:
            return []
        try:
            from app.tools.registry import tool_registry
            return tool_registry.get_tool_schemas()
        except Exception:
            # 工具注册中心不可用（如 docker 未安装）→ 返回空列表
            return []

    # ── 消息处理器（默认实现，子类可覆盖） ─────────────────

    async def _handle_task(self, msg: MailboxMessage) -> bool:
        """
        处理 TASK_ASSIGNMENT 消息 —— 子类覆盖此方法实现具体逻辑。

        默认实现：调用 run() 然后回复 TASK_RESULT。
        run() 失败时仍发 TASK_RESULT（含错误信息）给 sender，避免 sender 永久等待。
        """
        task = msg.body if isinstance(msg.body, str) else msg.body.get("task", "")
        subject = (
            msg.subject
            or (msg.body.get("subject", "") if isinstance(msg.body, dict) else "")
        )
        print(f"  [{self.agent_id}] TASK_ASSIGNMENT received: {subject or task[:80]}",
              flush=True)

        try:
            result = await self.run(task)
        except Exception as e:
            print(f"  [{self.agent_id}] run() FAILED: {type(e).__name__}: {e}",
                  flush=True)
            result = {
                "error": f"{type(e).__name__}: {e}",
                "agent": self.agent_id,
                "success": False,
            }

        reply = self.outbox.create_message(
            recipient=msg.sender,
            msg_type=MessageType.TASK_RESULT,
            body=result,
            subject=f"Task result from {self.agent_id}",
            correlation_id=msg.correlation_id,
            reply_to=msg.id,
        )
        await self.outbox.send(reply)
        print(f"  [{self.agent_id}] TASK_RESULT sent → {msg.sender}", flush=True)
        return True

    async def _handle_status_query(self, msg: MailboxMessage) -> bool:
        """处理 STATUS_QUERY —— 回复当前状态。"""
        reply = self.outbox.create_message(
            recipient=msg.sender,
            msg_type=MessageType.STATUS_REPLY,
            body={
                "agent": self.agent_id,
                "status": "idle",
                "inbox_depth": self.inbox.total_count,
            },
            correlation_id=msg.correlation_id,
            reply_to=msg.id,
        )
        await self.outbox.send(reply)
        return True

    async def _handle_broadcast(self, msg: MailboxMessage) -> bool:
        """处理 BROADCAST 消息（默认：仅记录日志）。"""
        print(f"  [{self.agent_id}] RECV BROADCAST: {msg.subject}")
        return True

    # ── Agent 主逻辑（子类必须覆盖） ───────────────────────

    async def _call_llm(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        timeout: float = 30.0,
    ) -> str:
        model_name = self.role.model or Config.DEFAULT_MODEL

        def _sync_call():
            return self.client.messages.create(
                model=model_name,
                system=system,
                messages=messages,
                tools=self.tools,
                max_tokens=max_tokens,
                timeout=timeout,
            )

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(_sync_call),
                timeout=timeout + 10,
            )
            return extract_text(response.content)
        except asyncio.TimeoutError:
            print(f"  [{self.agent_id}] LLM call TIMEOUT ({timeout}s)",
                  flush=True)
            raise
        except Exception as e:
            print(f"  [{self.agent_id}] LLM call FAILED: {type(e).__name__}: {e}",
                  flush=True)
            raise

    async def run(self, task: str) -> dict:
        """
        Agent 主逻辑。子类覆盖此方法。

        子类在实现中可以通过 self.outbox 与其他 Agent 通信。

        Args:
            task: 任务描述文本（从 TASK_ASSIGNMENT 消息的 body 中提取）

        Returns:
            dict: 任务结果（作为 TASK_RESULT 的 body 发送回 Router）
        """
        text = await self._call_llm(
            system=self.role.system_prompt,
            messages=[{"role": "user", "content": f"Task: {task}"}],
        )
        return {
            "output": text,
            "agent": self.agent_id,
        }

    # ── 生命周期 ──────────────────────────────────────────

    async def start(self) -> None:
        """
        启动 Agent 的消息监听循环（阻塞当前协程）。

        调用后，Agent 持续监听 Inbox，自动处理到达的消息。
        永不主动退出，直到外部调用 stop()。
        """
        print(f"  [{self.agent_id}] Watcher started (blocking)")
        await self._watcher.run()

    def start_background(self) -> asyncio.Task:
        """
        在后台启动监听（非阻塞，用于 Agent 池同时运行）。

        Returns:
            asyncio.Task: 后台监听任务，可用于取消或等待。
        """
        task = self._watcher.start()
        print(f"  [{self.agent_id}] Watcher started (background, task={task.get_name()})")
        return task

    def stop(self) -> None:
        """停止 Agent 的消息监听。"""
        self._watcher.stop()

    # ── 诊断 ──────────────────────────────────────────────

    @property
    def is_watching(self) -> bool:
        """是否正在监听 Inbox。"""
        return self._watcher.is_running

    @property
    def inbox_depth(self) -> int:
        """当前 Inbox 中的消息数量。"""
        return self.inbox.total_count

    @property
    def stats(self) -> dict:
        """Agent 状态快照。"""
        return {
            "agent_id": self.agent_id,
            "role": self.role.name,
            "is_watching": self.is_watching,
            "inbox": self.inbox.stats,
            "outbox_sent": self.outbox.sent_count,
        }

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} id='{self.agent_id}' "
            f"role='{self.role.name}' watching={self.is_watching}>"
        )
