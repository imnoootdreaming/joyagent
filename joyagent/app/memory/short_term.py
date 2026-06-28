from __future__ import annotations
"""
Phase 6 Step 3b: Short-term Memory — 滑动窗口 + 递进式摘要压缩。

ShortTermMemory 管理单个 Agent 会话的工作记忆。它实现两层策略：
  1. 滑动窗口 — 只保留最近 N 条消息（默认 50），防止对话历史无限膨胀
  2. 递进式摘要 — 当 Token 数超过阈值时，将旧消息压缩为 LLM 生成的摘要

与 Long-term Memory 的区别：
  Short-term Memory 是"当前会话的工作台"——Agent 用它来跟踪本轮对话的上下文。
  会话结束后数据不持久化。Long-term Memory 是"跨会话的知识库"——持久化到
  ChromaDB，下次启动时可检索。

设计原则：
  ┌───────────────────────────────────────────────────────┐
  │  ShortTermMemory 内部状态                              │
  │                                                       │
  │  messages: [msg50, msg51, ..., msg99]  ← 滑动窗口      │
  │  summary:  "Step 1-3 完成了用户管理模块..."  ← 递进摘要 │
  │                                                       │
  │  get_context() 返回:                                   │
  │    [summary_msg, msg80, msg81, ..., msg99]             │
  │    ↑ 摘要注入为 user 消息      ↑ 最近 50 条原始消息      │
  └───────────────────────────────────────────────────────┘

与 LangGraph 的兼容性：
  get_context() 返回纯 dict 列表，不包含 system prompt。
  摘要作为 user 消息注入（{"role": "user", "content": "[Summary]: ..."}），
  而非放在 system 参数中。这样 LangGraph 的 add_messages reducer 能正确
  处理它（system prompt 由 Agent 层单独传给 client.messages.create(system=...))。

使用方法：
  from app.memory.short_term import ShortTermMemory
  from app.memory.token_manager import get_token_manager

  stm = ShortTermMemory(max_messages=50)
  tm = get_token_manager()

  # 在 ReAct Loop 中
  stm.add_message({"role": "user", "content": "创建一个文件"})
  # LLM 响应后...
  stm.add_message({"role": "assistant", "content": [TextBlock(...)]})

  # 需要压缩时
  if tm.should_compress(stm.messages, model="claude-sonnet-4-6"):
      await stm.compress()

  # 构建 LLM 上下文
  context = stm.get_context()
  response = client.messages.create(
      model=MODEL,
      system=SYSTEM_PROMPT,     # system 是独立参数
      messages=context,          # context = [summary_msg] + recent_msgs
  )
"""

# ── Python 标准库 ──
import asyncio
import time
from typing import Optional

# ── 项目内导入 ──
from app.memory.token_manager import get_token_manager, TokenManager
from app.memory.summary import generate_summary


# ═══════════════════════════════════════════════════════════════════════════════
# ShortTermMemory — 会话工作记忆
# ═══════════════════════════════════════════════════════════════════════════════

class ShortTermMemory:
    """
    当前会话的工作记忆 —— 滑动窗口 + 递进式摘要压缩。

    职责：
      1. 存储当前会话的消息历史（Anthropic 原生 dict 格式）
      2. 当 Token 数超过阈值时，将前一半消息压缩为递进式摘要
      3. 提供 get_context() 构建 LLM 上下文（摘要 + 最近消息）
      4. 追踪压缩统计（次数、耗时、保存的 Token 数）

    压缩触发策略：
      - 自动检测（try_auto_compress）：Token 数 > summary_trigger_tokens → 压缩
      - 手动触发（compress）：调用方自己判断时机（推荐，与 TokenManager 配合）

    为什么用递进式摘要而非每次从头总结？
      - 成本：递进式 = O(1) LLM 调用，每次只处理增量；从头总结 = O(n)
      - 质量：递进式摘要累积上下文，不会"忘记"早期决策
      - 面试表达："progressive summarization — 类似人类记笔记，逐页追加而非重写全书"
    """

    def __init__(
        self,
        max_messages: int = 50,
        summary_trigger_tokens: int = 6000,
        token_manager: Optional[TokenManager] = None,
        model: str = None,
    ):
        """
        初始化 Short-term Memory。

        Args:
            max_messages:           滑动窗口最多保留的消息数（默认 50）
            summary_trigger_tokens: Token 数达到此阈值时触发压缩（默认 6000）
            token_manager:          TokenManager 实例。为 None 时自动获取全局单例。
            model:                  关联的 LLM 模型（用于摘要生成）

        参数调优建议：
          - 简短对话（10 轮以内）：max_messages=30, trigger=3000
          - 中等对话（20-50 轮）：max_messages=50, trigger=6000（默认）
          - 长对话（50+ 轮）：max_messages=100, trigger=12000
            注意：trigger 值需 < 模型 Context Window 的 80%
        """
        self.max_messages = max_messages
        self.summary_trigger_tokens = summary_trigger_tokens
        self.model = model

        # ── 核心状态 ──
        self.messages: list[dict] = []
        self.summary: str = ""

        # ── Token 管理器 ──
        self._token_manager = token_manager or get_token_manager()

        # ── 统计信息 ──
        self._stats = {
            "total_messages_added": 0,
            "total_compressions": 0,
            "total_compression_time_ms": 0.0,
            "total_tokens_saved": 0,           # 通过压缩节省的 Token 估算值
            "last_compression_at": None,       # 上次压缩时间（epoch）
            "last_compression_msg_count": 0,   # 上次压缩时的消息数
        }

    # ── 消息管理 ──────────────────────────────────────────────────

    def add_message(self, message: dict) -> bool:
        """
        向工作记忆追加一条消息。

        这是快速路径（同步）——只做追加和基础记录，不触发 LLM 调用。
        压缩决策交给 try_auto_compress() 或外部 TokenManager.should_compress()。

        Args:
            message: Anthropic 原生 dict 格式的消息：
                     {"role": "user", "content": "..."}
                     {"role": "assistant", "content": [...]}

        Returns:
            True 如果消息数超过 max_messages（调用方可能需要压缩），
            False 表示仍在窗口内。

        Example:
            stm.add_message({"role": "user", "content": "创建 main.py"})
            stm.add_message({"role": "assistant", "content": [{"type": "text", "text": "..."}]})
        """
        self.messages.append(message)
        self._stats["total_messages_added"] += 1

        # 返回是否超过窗口（供调用方决策，不自动压缩）
        return len(self.messages) > self.max_messages

    def add_messages_batch(self, messages: list[dict]) -> bool:
        """
        批量追加消息（比逐条 add_message 少做 len() 检查，适合初始化恢复场景）。

        Args:
            messages: 消息列表

        Returns:
            True 如果总消息数超过 max_messages
        """
        self.messages.extend(messages)
        self._stats["total_messages_added"] += len(messages)
        return len(self.messages) > self.max_messages

    # ── Token 计数 ───────────────────────────────────────────────

    def count_tokens(self) -> int:
        """
        计算当前消息列表的总 Token 数。

        使用简化的快速计数（count_messages_simple），因为压缩决策
        不需要逐 block 精确计数——±15% 误差对阈值判断来说足够了。
        """
        return self._token_manager.count_messages_simple(self.messages)

    def needs_compression(self) -> bool:
        """
        快速判断是否需要压缩（同步，不调 LLM）。

        判断条件：
          1. 消息数超过 max_messages，或
          2. Token 数超过 summary_trigger_tokens

        调用方可以在此方法返回 True 后，选择合适的时机调用 compress()。
        """
        if len(self.messages) > self.max_messages:
            return True
        return self.count_tokens() > self.summary_trigger_tokens

    # ── 上下文构建 ───────────────────────────────────────────────

    def get_context(self) -> list[dict]:
        """
        构建 LLM 上下文：摘要消息 + 最近消息。

        摘要不放在 system 参数中，而是作为 user 消息注入 ——
        保持 messages 列表的自包含性，便于 LangGraph 的 add_messages reducer。

        Returns:
            Anthropic 原生 dict 格式的消息列表，可直接传给
            client.messages.create(messages=...)

        Example:
            stm = ShortTermMemory()
            # ... 对话进行中 ...
            messages = stm.get_context()
            # messages = [
            #     {"role": "user", "content": "[Previous context summary]: ..."},
            #     {"role": "user", "content": "最后一条用户消息"},
            #     {"role": "assistant", "content": [...]},
            # ]
            response = client.messages.create(
                model=MODEL,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        """
        result = []

        # ── 1. 注入递进式摘要（如果有） ──
        if self.summary:
            result.append({
                "role": "user",
                "content": f"[Previous context summary]: {self.summary}",
            })

        # ── 2. 追加最近的消息（滑动窗口） ──
        result.extend(self.messages[-self.max_messages:])

        return result

    def get_context_with_system(
        self,
        system_prompt: str,
    ) -> tuple[str, list[dict]]:
        """
        返回分离的 (system_prompt, messages) —— 符合 Anthropic API 调用模式。

        与 get_context() 的区别：
          get_context() 把摘要注入到 messages 中。
          get_context_with_system() 返回分离的两部分，摘要仍在 messages 中，
          system_prompt 由调用方单独提供。

        Args:
            system_prompt: Agent 的 System Prompt 文本

        Returns:
            (system_prompt, messages) 元组
        """
        return system_prompt, self.get_context()

    # ── 压缩逻辑 ─────────────────────────────────────────────────

    async def compress(self, model: str = None) -> str:
        """
        执行递进式摘要压缩（异步，调用 LLM）。

        压缩策略：
          1. 将消息列表分为两半
          2. 前半部分用 LLM 生成递进式摘要
          3. 丢弃前半部分，保留后半部分
          4. 更新 self.summary

        如果当前消息数 ≤ 10，跳过压缩（不值得为少量消息调 LLM）。

        Args:
            model: 用于摘要的 LLM 模型（默认使用 self.model 或 Config.DEFAULT_MODEL）

        Returns:
            新生成的摘要文本

        Raises:
            RuntimeError: LLM 调用失败时抛出（调用方应处理降级）

        Example:
            try:
                new_summary = await stm.compress()
            except RuntimeError:
                # 降级：丢弃最旧的消息，保留摘要
                stm.messages = stm.messages[-stm.max_messages:]
        """
        if len(self.messages) <= 10:
            # 消息太少，不值得压缩
            return self.summary

        start_time = time.time()

        # ── 1. 二分消息 ──
        half = len(self.messages) // 2
        old_messages = self.messages[:half]
        kept_messages = self.messages[half:]

        pre_token_count = self.count_tokens()

        # ── 2. LLM 生成递进式摘要 ──
        new_summary = await generate_summary(
            old_messages=old_messages,
            existing_summary=self.summary,
            model=model or self.model,
        )

        # ── 3. 更新状态 ──
        self.messages = kept_messages
        self.summary = new_summary

        # ── 4. 更新统计 ──
        post_token_count = self.count_tokens()
        tokens_saved = max(0, pre_token_count - post_token_count)
        elapsed_ms = (time.time() - start_time) * 1000

        self._stats["total_compressions"] += 1
        self._stats["total_compression_time_ms"] += elapsed_ms
        self._stats["total_tokens_saved"] += tokens_saved
        self._stats["last_compression_at"] = time.time()
        self._stats["last_compression_msg_count"] = len(old_messages)

        return new_summary

    async def try_auto_compress(self, model: str = None) -> bool:
        """
        自动检测并触发压缩（如果条件满足）。

        这是"懒人 API"——调用方不需要自己判断 needs_compression()，
        直接调用此方法即可。适合在每轮 ReAct Loop 结束后使用。

        Args:
            model: 摘要 LLM 模型

        Returns:
            True 如果执行了压缩，False 如果不需要压缩

        Example:
            # 每轮循环后
            stm.add_message(assistant_msg)
            stm.add_message(tool_result_msg)
            await stm.try_auto_compress()  # 自动判断，需要就压
        """
        if not self.needs_compression():
            return False

        try:
            await self.compress(model)
            return True
        except RuntimeError:
            # 压缩失败 → 降级：保留最新消息
            if len(self.messages) > self.max_messages:
                self.messages = self.messages[-self.max_messages:]
            return False

    # ── 统计与诊断 ───────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """获取压缩统计信息。"""
        s = dict(self._stats)
        s["current_message_count"] = len(self.messages)
        s["current_token_estimate"] = self.count_tokens()
        s["summary_length"] = len(self.summary)
        s["has_summary"] = bool(self.summary)
        s["needs_compression"] = self.needs_compression()
        if self._stats["total_compressions"] > 0:
            s["avg_compression_time_ms"] = round(
                self._stats["total_compression_time_ms"]
                / self._stats["total_compressions"],
                1,
            )
            s["avg_tokens_saved"] = round(
                self._stats["total_tokens_saved"]
                / self._stats["total_compressions"],
                0,
            )
        else:
            s["avg_compression_time_ms"] = 0.0
            s["avg_tokens_saved"] = 0.0
        return s

    def get_diagnostic_snapshot(self) -> str:
        """
        生成诊断快照——用于调试和日志。

        Returns:
            多行诊断文本，显示当前消息数、Token 估算、摘要大小等。
        """
        tokens = self.count_tokens()
        lines = [
            "── ShortTermMemory Snapshot ──",
            f"  Messages: {len(self.messages)} / {self.max_messages} (max)",
            f"  Token estimate: {tokens:,} / {self.summary_trigger_tokens:,} (trigger)",
            f"  Summary: {len(self.summary)} chars / {self._token_manager.count_tokens(self.summary)} tokens",
            f"  Compressions: {self._stats['total_compressions']} total, "
            f"{self._stats['total_tokens_saved']:,} tokens saved",
            f"  Needs compression: {self.needs_compression()}",
        ]
        return "\n".join(lines)

    # ── 重置 ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        重置 Short-term Memory 到初始状态。

        清空所有消息和摘要，保留配置参数和统计信息。
        用于新会话开始或手动清空上下文。
        """
        self.messages.clear()
        self.summary = ""
        # 不清空 stats——保留历史统计用于诊断

    def full_reset(self) -> None:
        """
        完全重置——包括统计信息。

        用于测试环境或需要完全干净的起点。
        """
        self.messages.clear()
        self.summary = ""
        self._stats = {
            "total_messages_added": 0,
            "total_compressions": 0,
            "total_compression_time_ms": 0.0,
            "total_tokens_saved": 0,
            "last_compression_at": None,
            "last_compression_msg_count": 0,
        }
