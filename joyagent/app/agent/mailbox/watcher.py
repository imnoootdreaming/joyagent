"""
Phase 7 Step 1 — InboxWatcher 异步消息监听器

异步消息监听器 —— Agent 不需要手动轮询。

每个 Agent 启动一个 watcher 协程，在后台持续监听 Inbox。
新消息到达 → 自动 dispatch 到对应 handler。

这是 Agent 主循环的核心：
  BaseAgent.run() 内部启动 watcher
  → watcher 持续 fetch_next() + process()
  → Agent 收到 TASK_ASSIGNMENT 时自动执行任务
  → 执行完毕后通过 Outbox 发送 TASK_RESULT
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable, Awaitable

if TYPE_CHECKING:
    from app.agent.mailbox.inbox import AgentInbox

# ── Idle 回调类型 ──
IdleCallback = Callable[[], Awaitable[None]]


class InboxWatcher:
    """
    异步消息监听器 —— Agent 不需要手动轮询。

    每个 Agent 启动一个 watcher 协程，在后台持续监听 Inbox。
    新消息到达 → 自动 dispatch 到对应 handler。

    这是 Agent 主循环的核心：
      BaseAgent.run() 内部启动 watcher
      → watcher 持续 fetch_next() + process()
      → Agent 收到 TASK_ASSIGNMENT 时自动执行任务
      → 执行完毕后通过 Outbox 发送 TASK_RESULT
    """

    def __init__(
        self,
        inbox: "AgentInbox",
        idle_callback: IdleCallback | None = None,
    ):
        """
        Args:
            inbox: 要监听的 AgentInbox 实例
            idle_callback: 空闲时回调（如心跳上报、状态同步）
        """
        self.inbox = inbox
        self.idle_callback = idle_callback
        self._running = False
        self._task: asyncio.Task | None = None

    async def run(self, poll_interval: float = 0.5) -> None:
        """
        启动监听循环（永不退出，直到外部调用 stop()）。

        流程：
          1. 阻塞等待下一条消息（无消息时由 asyncio.Event 挂起）
          2. 消息到达 → process(msg)
          3. 处理完毕 → 继续等待下一条
          4. 过期消息自动清理

        Args:
            poll_interval: 保留参数，当前版本使用 Event 驱动（非轮询）
        """
        self._running = True
        while self._running:
            msg = await self.inbox.fetch_next(timeout=5.0)

            if msg is None:
                # 超时——无新消息
                self.inbox.clean_expired()
                if self.idle_callback:
                    await self.idle_callback()
                continue

            await self.inbox.process(msg)

    def start(self) -> asyncio.Task:
        """
        启动 watcher（非阻塞）。

        Returns:
            创建的 asyncio.Task，可用于取消或等待。
        """
        self._task = asyncio.create_task(self.run())
        return self._task

    def stop(self) -> None:
        """停止 watcher。"""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    @property
    def is_running(self) -> bool:
        """watcher 是否正在运行。"""
        return self._running

    @property
    def task(self) -> asyncio.Task | None:
        """返回内部的 asyncio.Task（用于调试）。"""
        return self._task
