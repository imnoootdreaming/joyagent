"""
Phase 7 Step 5 — MultiAgentOrchestrator

多 Agent 编排器 —— 管理 Agent 池的完整生命周期 + 工作流编排。

这是整个 Phase 7 的"入口级"组件——把它挂到 FastAPI / CLI 上，
就得到了一个完整的多 Agent 协作系统。

职责：
  1. 创建 MailboxManager（消息路由中枢）
  2. 创建并注册所有 Agent 实例（Router/Planner/Coder/Tester/Reviewer）
  3. 后台启动所有 Agent 的 InboxWatcher
  4. 提供统一的 handle_user_request() 入口（委托给 Router）
  5. 提供 monitoring / shutdown / restart 等运维操作
  6. 提供死信队列检查和过期消息清理

面试要点：
  Q: "你的多 Agent 系统怎么启动和管理？"
  A: "我们有一个 MultiAgentOrchestrator，它负责 Agent 池的完整生命周期。
      它创建 MailboxManager 作为通信中枢，按 AGENT_CLASS_MAP 实例化
      每种 Agent，注册它们的 Inbox/Outbox，然后后台启动所有 Watcher。
      用户请求通过 handle_user_request() 进入，委托给 Router Agent。
      整个系统通过 asyncio 协程并发运行，Agent 之间通过 Mailbox 异步通信。"
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.agent.base import BaseAgent
from app.agent.mailbox import (
    MailboxManager,
    MailboxMessage,
    MessageType,
)
from app.agent.roles import (
    AgentRole,
    ROLE_CODER,
    ROLE_PLANNER,
    ROLE_REVIEWER,
    ROLE_ROUTER,
    ROLE_TESTER,
)


# ═══════════════════════════════════════════════════════════════════════
# Agent 工厂映射（延迟导入，避免循环依赖）
# ═══════════════════════════════════════════════════════════════════════

def _get_agent_class_map() -> dict[str, type[BaseAgent]]:
    """
    返回 role_name → Agent class 的映射（延迟导入工厂函数）。

    延迟导入的原因：
      各个 Agent 子类会 import roles.py（AgentRole/ROLE_*），
      如果在 roles.py 中定义此映射会导致循环导入。
      把它放在 orchestrator 模块中并在函数内导入，打破循环。
    """
    from app.agent.router import RouterAgent
    from app.agent.planner import PlannerAgent
    from app.agent.coder import CoderAgent
    from app.agent.tester import TesterAgent
    from app.agent.reviewer import ReviewerAgent

    return {
        "router": RouterAgent,
        "planner": PlannerAgent,
        "coder": CoderAgent,
        "tester": TesterAgent,
        "reviewer": ReviewerAgent,
    }


# ═══════════════════════════════════════════════════════════════════════
# 编排器配置
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OrchestratorConfig:
    """
    MultiAgentOrchestrator 的可调配置。

    主要是超时和并发控制参数。Agent 角色定义从 AgentRole 获取，
    不需要在这里重复。
    """
    # ── 超时 ──
    request_timeout: float = 300.0    # 单个用户请求总超时（秒）

    # ── 清理 ──
    expire_sweep_interval: float = 60.0  # 过期消息清理间隔（秒），0 = 不自动清理

    # ── 调试 ──
    verbose: bool = True              # 是否打印详细日志


# ═══════════════════════════════════════════════════════════════════════
# MultiAgentOrchestrator
# ═══════════════════════════════════════════════════════════════════════

class MultiAgentOrchestrator:
    """
    多 Agent 编排器 —— Agent 池的生命周期管理 + 工作流入口。

    典型用法::

        orchestrator = MultiAgentOrchestrator()
        orchestrator.register_all()
        await orchestrator.start_all()

        result = await orchestrator.handle_user_request("Build a REST API")
        print(result["summary"])

        await orchestrator.shutdown()

    也可以作为上下文管理器::

        async with MultiAgentOrchestrator() as orch:
            result = await orch.handle_user_request("Create a health check endpoint")
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        """
        Args:
            config: 编排器配置（可选，使用默认配置亦可）
        """
        self.config = config or OrchestratorConfig()

        # ── 核心组件 ──
        self.mailbox = MailboxManager()
        self.agents: dict[str, BaseAgent] = {}

        # ── 后台任务 ──
        self._watcher_tasks: dict[str, asyncio.Task] = {}
        self._sweeper_task: asyncio.Task | None = None

        # ── 状态 ──
        self._started = False
        self._request_count: int = 0
        self._error_count: int = 0
        self._start_time: float = 0.0

        # ── 事件回调（可选的 hooks） ──
        self._on_request_start: list[Callable] = []
        self._on_request_end: list[Callable] = []
        self._on_error: list[Callable] = []

    # ── 注册 ──────────────────────────────────────────────

    def register_all(self, agent_class_map: dict[str, type[BaseAgent]] | None = None) -> None:
        """
        创建并注册所有 Agent 到 MailboxManager。

        按 AGENT_ROLES 中的定义，为每种角色创建一个 Agent 实例。
        每个 Agent 自动获得独立的 Inbox/Outbox 并注册到 MailboxManager。

        Args:
            agent_class_map: role_name → Agent class 映射。
                            默认使用内置的 _get_agent_class_map()。
        """
        class_map = agent_class_map or _get_agent_class_map()
        roles = {
            "router": ROLE_ROUTER,
            "planner": ROLE_PLANNER,
            "coder": ROLE_CODER,
            "tester": ROLE_TESTER,
            "reviewer": ROLE_REVIEWER,
        }

        for role_name, role in roles.items():
            agent_cls = class_map.get(role_name)
            if agent_cls is None:
                if self.config.verbose:
                    print(f"  [orchestrator] ⚠ No class for '{role_name}' — skipping")
                continue

            agent = agent_cls(role, self.mailbox, agent_id=role_name)
            self.agents[role_name] = agent

        if self.config.verbose:
            registered = self.mailbox.registered_agents
            print(f"  [orchestrator] {len(self.agents)} agents registered: "
                  f"{', '.join(registered)}")

    def register_agent(
        self,
        name: str,
        agent: BaseAgent,
    ) -> None:
        """
        手动注册一个 Agent（用于自定义 Agent 或测试注入）。

        Agent 必须已经关联到同一个 MailboxManager（即构造时传入了
        self.mailbox）。

        Args:
            name: Agent 名称
            agent: Agent 实例
        """
        self.agents[name] = agent
        if self.config.verbose:
            print(f"  [orchestrator] Agent '{name}' registered manually")

    # ── 启动 / 停止 ───────────────────────────────────────

    async def start_all(self) -> None:
        """
        后台启动所有 Agent 的 InboxWatcher。

        启动后，每个 Agent 开始监听自己的 Inbox，自动处理到达的消息。
        所有 Agent 并发运行（通过 asyncio 协程），互不阻塞。

        这是幂等的——多次调用不会重复启动已运行的 Agent。
        """
        tasks = {}
        for name, agent in self.agents.items():
            if name in self._watcher_tasks and not self._watcher_tasks[name].done():
                continue
            task = agent.start_background()
            tasks[name] = task

        self._watcher_tasks.update(tasks)
        self._started = True
        self._start_time = time.time()

        # 启动过期消息清理器
        if self.config.expire_sweep_interval > 0:
            self._sweeper_task = asyncio.create_task(
                self._sweep_loop(self.config.expire_sweep_interval)
            )

        if self.config.verbose:
            print(f"  [orchestrator] All {len(tasks)} agents started (background)")

    async def stop_all(self) -> None:
        """
        停止所有 Agent 的 InboxWatcher。

        停止后 Agent 不再处理新消息，但已注册的 Inbox/Outbox 仍保留
        （可以再次 start_all 恢复）。
        """
        for name, agent in self.agents.items():
            agent.stop()

        # 取消 sweeper
        if self._sweeper_task and not self._sweeper_task.done():
            self._sweeper_task.cancel()
            try:
                await self._sweeper_task
            except asyncio.CancelledError:
                pass
        self._sweeper_task = None

        # 等待所有 watcher 任务完成
        for name, task in self._watcher_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._watcher_tasks.clear()
        self._started = False

        if self.config.verbose:
            print(f"  [orchestrator] All agents stopped")

    # ── 用户请求入口 ──────────────────────────────────────

    async def handle_user_request(self, user_message: str) -> dict:
        """
        用户请求入口 —— 委托给 Router Agent 处理。

        这是 MultiAgentOrchestrator 对外的核心 API。
        上层（FastAPI / CLI）只需调用这一个方法即可。

        流程（由 RouterAgent 内部完成）：
          1. 分析请求复杂度（规则匹配 + LLM 兜底）
          2. 简单任务 → 直接路由到 Coder/Tester/Reviewer
          3. 复杂任务 → Planner 拆解 → 按计划分发给各 Agent
          4. 收集所有结果 → 聚合返回

        Args:
            user_message: 用户原始请求文本

        Returns:
            dict: {
                "summary": "聚合后的摘要",
                "details": [每个步骤的详情],
                "success": True/False,
                "correlation_id": "...",
            }
        """
        router = self.agents.get("router")
        if router is None:
            return {
                "summary": "Error: Router agent not registered.",
                "success": False,
                "error": "no_router",
            }

        if not self.is_running:
            # 自动启动（如果尚未启动）
            await self.start_all()

        # ── 触发 on_request_start 回调 ──
        for cb in self._on_request_start:
            try:
                cb(user_message)
            except Exception:
                pass

        self._request_count += 1

        try:
            result = await router.handle_user_request(
                user_message,
                timeout=self.config.request_timeout,
            )
        except Exception as e:
            self._error_count += 1
            for cb in self._on_error:
                try:
                    cb(user_message, e)
                except Exception:
                    pass
            return {
                "summary": f"Request failed: {type(e).__name__}: {e}",
                "success": False,
                "error": str(e),
            }

        # ── 触发 on_request_end 回调 ──
        for cb in self._on_request_end:
            try:
                cb(user_message, result)
            except Exception:
                pass

        return result

    # ── 消息等待辅助（用于外部注入的步骤间等待） ────────────

    async def wait_for_reply(
        self,
        agent_name: str,
        correlation_id: str,
        timeout: float = 30.0,
    ) -> MailboxMessage | None:
        """
        等待指定 Agent 收到特定 correlation_id 的回复。

        用于需要精确控制步骤间等待的场景（如测试或自定义编排逻辑）。

        Args:
            agent_name: 等待消息的 Agent 名称
            correlation_id: 要等待的 correlation_id
            timeout: 超时秒数

        Returns:
            MailboxMessage | None: 匹配的消息，超时返回 None
        """
        agent = self.agents.get(agent_name)
        if agent is None:
            return None

        deadline = time.time() + timeout
        while time.time() < deadline:
            # 搜索 inbox 中匹配的消息
            candidates = agent.inbox.fetch_by_correlation(correlation_id)
            for msg in candidates:
                if msg.status.value in ("processed", "failed"):
                    return msg
            await asyncio.sleep(0.3)

        return None

    # ── 监控与运维 ────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """所有 Agent 是否正在运行。"""
        return self._started

    @property
    def uptime_seconds(self) -> float:
        """从 start_all() 到现在的秒数。"""
        if not self._started or self._start_time == 0:
            return 0.0
        return time.time() - self._start_time

    def get_stats(self) -> dict:
        """
        获取完整的系统状态快照。

        Returns:
            dict: 包含 global stats、per-agent stats、审计日志大小等。
        """
        mailbox_stats = self.mailbox.get_all_stats()
        agent_stats = {}
        for name, agent in self.agents.items():
            agent_stats[name] = agent.stats

        return {
            "system": {
                "is_running": self.is_running,
                "uptime_seconds": self.uptime_seconds,
                "request_count": self._request_count,
                "error_count": self._error_count,
            },
            "mailbox": mailbox_stats,
            "agents": agent_stats,
            "audit_log_size": self.mailbox.audit_log_size,
        }

    def get_audit_trail(
        self,
        correlation_id: str = "",
        agent_name: str = "",
    ) -> list[dict]:
        """
        查询消息审计日志。

        Args:
            correlation_id: 按线程 ID 过滤（查看一个完整任务的链路）
            agent_name: 按 Agent 名称过滤（查看某个 Agent 的收发记录）

        Returns:
            list[dict]: 审计日志条目
        """
        return self.mailbox.get_audit_trail(
            correlation_id=correlation_id,
            agent_name=agent_name,
        )

    def sweep_expired(self) -> int:
        """手动触发全局过期消息清理。返回清理总数。"""
        return self.mailbox.sweep_expired()

    def get_dead_letter_count(self) -> int:
        """获取全局死信队列中的消息数量。"""
        return len(self.mailbox._global_dead_letter)

    # ── 事件回调 ──────────────────────────────────────────

    def on_request_start(self, callback: Callable) -> None:
        """注册请求开始回调。"""
        self._on_request_start.append(callback)

    def on_request_end(self, callback: Callable) -> None:
        """注册请求结束回调。"""
        self._on_request_end.append(callback)

    def on_error(self, callback: Callable) -> None:
        """注册错误回调。"""
        self._on_error.append(callback)

    # ── 生命周期 ──────────────────────────────────────────

    async def shutdown(self) -> None:
        """
        优雅关闭整个系统。

        1. 停止所有 Agent 的 InboxWatcher
        2. 清理过期消息
        3. 注销所有 Agent
        4. 解除所有 Outbox 绑定
        """
        if self.config.verbose:
            print(f"  [orchestrator] Shutting down...")

        await self.stop_all()

        # 最后一次过期清理
        self.mailbox.sweep_expired()

        # 注销所有 Agent
        for name in list(self.agents.keys()):
            self.mailbox.unregister_agent(name)

        self.agents.clear()
        self._started = False

        if self.config.verbose:
            print(f"  [orchestrator] Shutdown complete")

    async def __aenter__(self):
        """Async context manager entry —— 自动注册 + 启动。"""
        self.register_all()
        await self.start_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit —— 自动关闭。"""
        await self.shutdown()
        return False  # 不吞异常

    # ── 内部 ──────────────────────────────────────────────

    async def _sweep_loop(self, interval: float) -> None:
        """后台过期消息清理循环。"""
        while True:
            await asyncio.sleep(interval)
            swept = self.mailbox.sweep_expired()
            if swept > 0 and self.config.verbose:
                print(f"  [orchestrator] Swept {swept} expired messages")
