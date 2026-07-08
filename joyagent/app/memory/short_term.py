from __future__ import annotations
"""
Phase 6 Step 3b: Short-term Memory — 四层上下文压缩管线。

ShortTermMemory 管理单个 Agent 会话的工作记忆，实现四层压缩策略：

  L3: tool_result_budget        — 大结果(>200KB)落盘，上下文只留预览
  L1: snip_compact              — 保留头 3 + 尾 47，裁掉中间无关对话
  L2: micro_compact             — 旧工具结果替换为占位符，只保留最近 3 条
  L4: 递进式摘要（compress）     — LLM 生成递进式摘要（O(1) 而非 O(n)）
  应急: reactive_truncate        — API 报 prompt_too_long 时暴力截断

设计理念（Claude Code 标准）：
  "便宜的先跑，贵的后跑" — L3/L1/L2 都是 0 API 调用的纯文本操作，
  全部跑完后 token 仍超阈值才触发 L4（1 次 LLM 调用）。
  执行顺序: L3 → L1 → L2 → (still over?) → L4
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

    # ── 四层压缩管线 ──────────────────────────────────────────
    # 设计理念: "便宜的先跑，贵的后跑"
    # L1/L2/L3 都是 0 API 调用的纯文本操作，L4 才调 LLM
    # 执行顺序: L3 → L1 → L2 → (still over?) → L4

    def tool_result_budget(self, max_bytes: int = 200_000) -> int:
        """
        L3: 大工具结果落盘——最贵的文本操作最先跑（保护后续裁剪）。

        统计最后一条 user 消息里所有 tool_result 的总大小。
        超过 max_bytes → 按大小排序，从最大的开始落盘到
        data/tool_outputs/，上下文里只留 <persisted-output> 标记
        + 前 2000 字符预览。

        必须最先跑：因为 L2（micro_compact）会把旧的大 tool_result
        替换成一行占位符，budget 需要在被替换前把完整内容落盘。

        Args:
            max_bytes: 单条 user 消息的 tool_result 总大小上限（字符数）

        Returns:
            落盘的文件数（0 = 不需要）
        """
        import os
        import hashlib

        if not self.messages:
            return 0

        persisted = 0
        output_dir = os.path.join("data", "tool_outputs")
        os.makedirs(output_dir, exist_ok=True)

        for msg in self.messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue

            # 收集所有 tool_result blocks
            blocks = [(i, b) for i, b in enumerate(content)
                      if isinstance(b, dict) and b.get("type") == "tool_result"]
            if not blocks:
                continue

            total = sum(len(str(b[1].get("content", ""))) for b in blocks)
            if total <= max_bytes:
                continue

            # 按大小降序——最大的先落盘
            ranked = sorted(blocks, key=lambda p: len(str(p[1].get("content", ""))),
                            reverse=True)
            for idx, block in ranked:
                if total <= max_bytes:
                    break
                raw = str(block.get("content", ""))
                if len(raw) < 500:
                    continue  # 太小不值得落盘

                # 落盘
                tid = block.get("tool_use_id", f"unknown_{idx}")
                file_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
                disk_path = os.path.join(output_dir,
                                         f"tool_{tid}_{file_hash}.txt")
                with open(disk_path, "w", encoding="utf-8") as f:
                    f.write(raw)

                # 上下文里只留标记 + 预览
                preview = raw[:2000]
                block["content"] = (
                    f"<persisted-output file='{disk_path}'>\n"
                    f"{preview}\n"
                    f"... ({len(raw) - len(preview)} more chars)"
                    f"</persisted-output>"
                )
                total = sum(len(str(b[1].get("content", ""))) for b in blocks)
                persisted += 1

        if persisted:
            self._stats.setdefault("tool_outputs_persisted", 0)
            self._stats["tool_outputs_persisted"] += persisted
        return persisted

    def micro_compact(self, keep_recent: int = 3) -> int:
        """
        L2: 旧工具结果占位——只保留最近 keep_recent 条完整内容。

        Agent 连续读了 10 个文件。第 1-7 次的完整内容还躺在上下文里，
        早就不需要了，但占着大量空间。将更旧的 tool_result 替换为
        "[Earlier tool result compacted. Re-run if needed.]"

        必须在 L3 之后跑：L3 已经把大结果落盘了，L2 把旧结果
        替换为占位符释放上下文。不在白名单中的工具（如 read_file
        的 FILE_UNCHANGED_STUB）不压缩——可以用 metadata 跳过。

        Args:
            keep_recent: 保留最近几条完整 tool_result

        Returns:
            替换的块数
        """
        if not self.messages:
            return 0

        replaced = 0
        tool_result_blocks: list[tuple[int, int, dict]] = []  # (msg_i, block_i, block)
        for mi, msg in enumerate(self.messages):
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue
            for bi, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_result_blocks.append((mi, bi, block))

        if len(tool_result_blocks) <= keep_recent:
            return 0

        to_replace = tool_result_blocks[:-keep_recent]
        for _mi, _bi, block in to_replace:
            raw = str(block.get("content", ""))
            if len(raw) > 120:
                block["content"] = "[Earlier tool result compacted. Re-run if needed.]"
                replaced += 1

        if replaced:
            self._stats.setdefault("micro_compactions", 0)
            self._stats["micro_compactions"] += replaced
        return replaced

    def snip_compact(self, max_messages: int = 50) -> int:
        """
        L1: 裁掉无关的旧对话——保留头部 3 条 + 尾部 (max-3) 条。

        消息超过 max_messages → 保留头 3（初始上下文: system/user/task）
        和尾 (max-3)（当前工作），中间裁掉。切口保护：不会把
        assistant(tool_use) 和紧接的 user(tool_result) 拆开。

        Args:
            max_messages: 触发裁剪的阈值

        Returns:
            裁掉的消息数（0 = 不需要）
        """
        if len(self.messages) <= max_messages:
            return 0

        head_keep = 3
        tail_keep = max_messages - head_keep
        head_end = head_keep
        tail_start = len(self.messages) - tail_keep

        # 切口保护: head 末端不能是拆开的 tool_use → tool_result
        head_msg = self.messages[head_end - 1]
        if _msg_has_tool_use(head_msg):
            while head_end < len(self.messages) and _is_tool_result_message(self.messages[head_end]):
                head_end += 1

        # tail 开头不能是孤立的 tool_result
        if _is_tool_result_message(self.messages[tail_start]) and \
           _msg_has_tool_use(self.messages[tail_start - 1]):
            tail_start -= 1

        snipped = tail_start - head_end
        if snipped <= 0:
            return 0

        placeholder = {
            "role": "user",
            "content": f"[snipped {snipped} messages from conversation middle]",
        }
        self.messages = (
            self.messages[:head_end]
            + [placeholder]
            + self.messages[tail_start:]
        )
        self._stats.setdefault("snip_compactions", 0)
        self._stats["snip_compactions"] += snipped
        return snipped

    def reactive_truncate(self) -> int:
        """
        应急: API 返回 prompt_too_long (413) 时暴力截断。

        compact_history 可能还来不及跑——上下文增长速度快于压缩触发速度。
        此时从尾部保留最后 10 条消息，丢弃前面所有，
        但仍要避免留下孤立的 tool_result 无对应的 tool_use。

        Returns:
            丢弃的消息数
        """
        if len(self.messages) <= 10:
            return 0

        tail_start = max(0, len(self.messages) - 10)
        if _is_tool_result_message(self.messages[tail_start]) and \
           _msg_has_tool_use(self.messages[tail_start - 1]):
            tail_start -= 1

        discarded = len(self.messages) - (len(self.messages) - tail_start)
        self.messages = self.messages[tail_start:]
        self._stats.setdefault("reactive_truncations", 0)
        self._stats["reactive_truncations"] += discarded
        return discarded

    def run_cheap_compaction(self) -> dict:
        """
        运行三层 0-API 压缩管线: L3 → L1 → L2。

        返回每个操作的计数，用于日志。
        """
        result = {}
        result["budget_files"] = self.tool_result_budget()
        result["snipped"] = self.snip_compact(self.max_messages)
        result["micro_replaced"] = self.micro_compact(keep_recent=3)
        return result


# ═══════════════════════════════════════════════════════════════════════
# 辅助: 消息类型判断
# ═══════════════════════════════════════════════════════════════════════

def _msg_has_tool_use(msg: dict) -> bool:
    """消息的 content 列表中是否包含 tool_use block。"""
    content = msg.get("content", "")
    if not isinstance(content, list):
        return False
    return any(
        (isinstance(b, dict) and b.get("type") == "tool_use")
        for b in content
    )


def _is_tool_result_message(msg: dict) -> bool:
    """消息是否为 user 角色且包含 tool_result block。"""
    if msg.get("role") != "user":
        return False
    content = msg.get("content", "")
    if not isinstance(content, list):
        return False
    return any(
        (isinstance(b, dict) and b.get("type") == "tool_result")
        for b in content
    )
