from __future__ import annotations
"""
Phase 6 Step 6: Memory Manager — 三级记忆系统的统一协调整合层。

MemoryManager 是 Memory System 与 Agent Runtime 之间的桥梁。它将 Short-term、
Long-term、Reflection 三个子系统封装为统一的 API，提供会话生命周期管理、
LLM 调用前/后 Hook、工具调用后自动记存、Fix Loop 错误经验记录等一站式接口。

Step 6 的四个集成目标：
  1. 每次对话开始时 → 从 Long-term Memory 检索相关历史上下文
  2. 每次工具调用后   → 判断是否值得存入 Long-term Memory
  3. Fix Loop 每次修复 → 记录错误经验到 Reflection Memory
  4. STM 触发压缩时   → 将摘要存入 Long-term Memory（避免摘要丢失）

架构位置：
  ┌──────────────────────────────────────────────────────────┐
  │                      Agent Runtime                       │
  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
  │  │ Simple  │  │ LangGraph│  │ Fix Loop │  │   API    │ │
  │  │ Agent   │  │ Executor │  │ (Phase5) │  │  Layer   │ │
  │  └────┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
  │       │            │             │              │       │
  │       └────────────┴──────┬──────┴──────────────┘       │
  │                           │                              │
  │                    ┌──────▼──────┐                       │
  │                    │ MemoryManager│  ← Step 6 新模块     │
  │                    └──────┬──────┘                       │
  │           ┌───────────────┼───────────────┐              │
  │    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐      │
  │    │ Short-term  │ │ Long-term   │ │ Reflection  │      │
  │    │ Memory      │ │ Memory      │ │ Memory      │      │
  │    └─────────────┘ └─────────────┘ └─────────────┘      │
  └──────────────────────────────────────────────────────────┘

使用方式：
  from app.memory.manager import MemoryManager

  # 创建会话级 Manager（每次新对话一个实例）
  mm = MemoryManager(session_id="sess_abc123")

  # 1. 会话开始 — 检索历史上下文
  context = await mm.begin_session("创建一个 User 模型")

  # 2. 每次 LLM 调用前 — 检查是否需要压缩
  await mm.before_llm_call()

  # 3. LLM 调用后 — 追加消息
  mm.after_llm_response(assistant_msg)

  # 4. 工具调用后 — 判断是否值得记存
  await mm.after_tool_call(tool_name, tool_input, tool_result)

  # 5. Fix Loop 中 — 记录错误经验
  await mm.on_fix_attempt(error_type, error_msg, file_path, context_snippet, fix_desc, success)

  # 6. 会话结束 — 持久化摘要
  await mm.end_session()
"""

# ── Python 标准库 ──
import time
import threading
from typing import Optional

# ── 项目内导入 ──
from app.memory.short_term import ShortTermMemory
from app.memory.long_term import (
    LongTermMemory, MemoryEntry,
    get_long_term_memory, retrieve_relevant_context,
)
from app.memory.reflection import (
    ReflectionMemory, ErrorExperience,
    get_reflection_memory,
)
from app.memory.token_manager import get_token_manager, TokenManager
from app.memory.embeddings import get_embedding_service
from app.core.config import Config


# ═══════════════════════════════════════════════════════════════════════════════
# MemoryManager — 三级记忆系统的会话级协调器
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryManager:
    """
    会话级的 Memory System 统一入口。

    Lifecycle:
      begin_session()          → 检索历史上下文，初始化 STM
      before_llm_call()        → 检查/触发压缩
      after_llm_response(msg)  → 追加 assistant 消息到 STM
      after_tool_call(...)     → 追加 tool_result 到 STM + 判断是否存入 LTM
      on_fix_attempt(...)      → 记录错误经验到 Reflection Memory
      end_session()            → 持久化 STM 摘要到 LTM

    设计原则：
      - ShortTermMemory 为会话级实例（每会话独立）
      - LongTermMemory / ReflectionMemory 为全局单例（跨会话共享）
      - 所有异步方法都是 non-blocking——失败时降级不中断主流程
    """

    def __init__(
        self,
        session_id: str = "",
        model: str = None,
        max_messages: int = 50,
        summary_trigger_tokens: int = 6000,
        auto_store_tool_results: bool = True,
    ):
        """
        初始化 MemoryManager。

        Args:
            session_id:              会话 ID（用于标注 LTM 记忆来源）
            model:                   使用的 LLM 模型
            max_messages:            STM 滑动窗口大小
            summary_trigger_tokens:  STM 压缩触发阈值
            auto_store_tool_results: 是否自动将文件写入等关键工具结果存入 LTM
        """
        self.session_id = session_id or f"sess_{int(time.time())}"
        self.model = model or Config.DEFAULT_MODEL
        self.auto_store_tool_results = auto_store_tool_results

        # ── 获取共享单例（Long-term + Reflection） ──
        self.ltm = get_long_term_memory()
        self.rm = get_reflection_memory()
        self.tm = get_token_manager()
        self.emb = get_embedding_service()

        # ── 创建会话级实例（Short-term — 每会话独立） ──
        self.stm = ShortTermMemory(
            max_messages=max_messages,
            summary_trigger_tokens=summary_trigger_tokens,
            token_manager=self.tm,
            model=self.model,
        )

        # ── 会话元数据 ──
        self._context_injected = False       # 是否已注入历史上下文
        self._started_at = time.time()
        self._tool_call_count = 0
        self._llm_call_count = 0
        self._stored_to_ltm_count = 0

        print(
            f"  [memory_manager] session={self.session_id} initialized, "
            f"stm_max_msgs={max_messages}, auto_store={auto_store_tool_results}"
        )

    # ── 会话生命周期 ───────────────────────────────────────────────

    async def begin_session(self, user_message: str = "") -> list[dict]:
        """
        会话开始：检索历史上下文，返回应注入到对话开头的消息列表。

        做的事情：
          1. 从 Long-term Memory 检索与 user_message 相关的历史
          2. 构建上下文注入消息（代码、对话、错误经验）
          3. 返回 context messages 列表，调用方追加到对话开头

        Args:
            user_message: 用户的新请求文本（用于语义检索）

        Returns:
            上下文消息列表（可直接 extend 到 messages 开头）

        Example:
            mm = MemoryManager(session_id="sess_123")
            context = await mm.begin_session("add email field to User")
            messages = []
            messages.extend(context)           # 历史上下文在最前
            messages.append({"role": "user", "content": "add email field to User"})
        """
        if self._context_injected:
            return []

        self._context_injected = True
        context_msgs: list[dict] = []

        # ── 1. 从 Long-term Memory 检索历史 ──
        try:
            context_text = await retrieve_relevant_context(
                query=user_message or "general coding task",
                ltm=self.ltm,
                top_k_code=3,
                top_k_conv=2,
                top_k_task=3,
            )

            if context_text:
                # 作为 user 消息注入（保持自包含性）
                context_msgs.append({
                    "role": "user",
                    "content": context_text,
                })
                print(
                    f"  [memory_manager] retrieved historical context "
                    f"({len(context_text)} chars)"
                )
        except Exception as e:
            print(f"  [memory_manager] context retrieval failed (non-fatal): {e}")

        return context_msgs

    async def end_session(self) -> None:
        """
        会话结束：持久化 STM 摘要到 Long-term Memory。

        将当前 STM 的摘要（如果有）存入 LTM 的 "conversation" 集合，
        供未来会话检索。即使 STM 没有摘要也会存储最近对话的简要记录。
        """
        try:
            # ── 1. 如果 STM 有递进式摘要 → 存入 LTM ──
            if self.stm.summary:
                entry = MemoryEntry(
                    id=f"conv_{self.session_id}",
                    content=(
                        f"Session: {self.session_id}\n"
                        f"Summary: {self.stm.summary}"
                    ),
                    embedding=self.emb.embed(self.stm.summary),
                    memory_type="conversation",
                    metadata={
                        "session_id": self.session_id,
                        "message_count": str(len(self.stm.messages)),
                        "tool_calls": str(self._tool_call_count),
                        "llm_calls": str(self._llm_call_count),
                        "stored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                )
                await self.ltm.store(entry)
                self._stored_to_ltm_count += 1
                print(
                    f"  [memory_manager] session summary stored to LTM "
                    f"({len(self.stm.summary)} chars)"
                )

            # ── 2. 如果没有摘要但消息较多 → 生成简要记录 ──
            elif len(self.stm.messages) > 10:
                # 取最近几条用户/assistant 消息作为简要记录
                recent = self.stm.messages[-10:]
                content_parts = []
                for msg in recent[-5:]:
                    role = msg.get("role", "?")
                    c = msg.get("content", "")
                    if isinstance(c, str):
                        content_parts.append(f"[{role}]: {c[:150]}")
                    elif isinstance(c, list):
                        for block in c[-2:]:
                            if isinstance(block, dict) and block.get("type") == "text":
                                content_parts.append(f"[{role}]: {block.get('text', '')[:150]}")
                            elif hasattr(block, 'type') and getattr(block, 'type') == 'text':
                                content_parts.append(f"[{role}]: {getattr(block, 'text', '')[:150]}")

                brief = f"Session {self.session_id} summary:\n" + "\n".join(content_parts[:5])
                entry = MemoryEntry(
                    id=f"conv_{self.session_id}_brief",
                    content=brief,
                    embedding=self.emb.embed(brief),
                    memory_type="conversation",
                    metadata={
                        "session_id": self.session_id,
                        "message_count": str(len(self.stm.messages)),
                        "summary_type": "auto_brief",
                        "stored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                )
                await self.ltm.store(entry)
                self._stored_to_ltm_count += 1
                print(f"  [memory_manager] session brief stored to LTM")

        except Exception as e:
            print(f"  [memory_manager] end_session failed (non-fatal): {e}")

    # ── LLM 调用 Hook ─────────────────────────────────────────────

    async def before_llm_call(self) -> bool:
        """
        每次 LLM 调用前调用 —— 检查是否需要压缩上下文。

        Returns:
            True 如果执行了压缩（调用方可能需要刷新 messages 引用）

        Example:
            await mm.before_llm_call()
            response = client.messages.create(
                messages=mm.get_context(),  # 使用压缩后的上下文
                ...
            )
        """
        self._llm_call_count += 1

        # ── 检查 Token 是否接近限制 ──
        if self.tm.should_compress(self.stm.messages, self.model):
            try:
                await self.stm.compress(self.model)
                print(
                    f"  [memory_manager] auto-compressed STM "
                    f"(msgs={len(self.stm.messages)}, "
                    f"summary={len(self.stm.summary)} chars)"
                )
                return True
            except Exception as e:
                print(f"  [memory_manager] compress failed (non-fatal): {e}")
                # 降级：紧急截断
                if len(self.stm.messages) > self.stm.max_messages:
                    self.stm.messages = self.stm.messages[-self.stm.max_messages:]

        return False

    def after_llm_response(self, message: dict) -> None:
        """
        LLM 响应后：将 assistant 消息追加到 STM。

        Args:
            message: Anthropic 格式的 assistant 消息
        """
        self.stm.add_message(message)

    # ── 工具调用 Hook ─────────────────────────────────────────────

    async def after_tool_call(
        self,
        tool_name: str,
        tool_input: dict,
        tool_result: str,
        success: bool,
    ) -> None:
        """
        工具调用后：追加 tool_result 到 STM + 判断是否存入 LTM。

        LTM 存储策略（自动判断）：
          - 文件写入操作（file_write/write_file/apply_patch）→ 存入 "code"
          - 代码生成相关工具                   → 存入 "code"
          - 文件读取（大文件、有价值内容）     → 存入 "code"
          - Shell 执行（含测试结果）           → 存入 "task"
          - 其他（默认）                       → 不自动存入

        面试要点：
          "不是所有工具调用都值得存入 LTM —— 我们根据工具类型和结果大小
           做自动筛选，避免噪音记忆污染向量空间。"

        Args:
            tool_name:   工具名称
            tool_input:  工具参数
            tool_result: 工具执行结果文本
            success:     是否成功
        """
        self._tool_call_count += 1

        # ── 1. 追加到 STM ──
        tool_result_msg = {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": f"{tool_name}_{self._tool_call_count}",
                "content": tool_result,
            }],
        }
        self.stm.add_message(tool_result_msg)

        # ── 2. 判断是否存入 LTM ──
        if not self.auto_store_tool_results:
            return

        store_type = _should_store_in_ltm(tool_name, tool_result, success)
        if not store_type:
            return

        try:
            # 构建存储内容：工具名 + 关键输入 + 关键输出
            input_summary = _format_tool_input_summary(tool_input)
            result_summary = tool_result[:500]
            if len(tool_result) > 500:
                result_summary += "\n... (truncated)"

            content = (
                f"Tool: {tool_name}({input_summary})\n"
                f"Success: {success}\n"
                f"Result: {result_summary}"
            )

            entry = MemoryEntry(
                id=f"tool_{self.session_id}_{self._tool_call_count}",
                content=content,
                embedding=self.emb.embed(content),
                memory_type=store_type,
                metadata={
                    "tool_name": tool_name,
                    "success": str(success),
                    "session_id": self.session_id,
                    "call_index": str(self._tool_call_count),
                    "stored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )

            # 提取文件路径（如果有）
            if isinstance(tool_input, dict):
                for key in ("file_path", "path", "target_file"):
                    if key in tool_input:
                        entry.metadata[key] = str(tool_input[key])
                        break

            await self.ltm.store(entry)
            self._stored_to_ltm_count += 1

        except Exception as e:
            print(f"  [memory_manager] LTM store failed (non-fatal): {e}")

    # ── Fix Loop Hook ─────────────────────────────────────────────

    async def on_fix_attempt(
        self,
        error_type: str,
        error_message: str,
        file_path: str,
        context_snippet: str,
        fix_description: str,
        fix_success: bool,
    ) -> str:
        """
        Fix Loop 每次修复后调用 —— 记录错误经验到 Reflection Memory。

        这是 §4.3 和 Step 5 的核心集成点：每次 Fix Loop 尝试修复后，
        无论成功还是失败，都将经验记录到 Reflection Memory，
        让 Agent 在未来遇到类似错误时能够从历史中学习。

        Args:
            error_type:      错误类型（"ImportError" / "SyntaxError" / ...）
            error_message:   错误消息
            file_path:       出错文件路径
            context_snippet: 出错代码片段
            fix_description: 修复描述
            fix_success:     是否修复成功

        Returns:
            存储的记忆 ID

        Example:
            # 在 FixLoop._fix_one_round() 中
            await mm.on_fix_attempt(
                error_type="ImportError",
                error_message="No module named 'numpy'",
                file_path="calc.py",
                context_snippet="import numpy as np",
                fix_description="pip install numpy",
                fix_success=True,
            )
        """
        try:
            eid = await self.rm.record_error(ErrorExperience(
                error_type=error_type,
                error_message=error_message,
                file_path=file_path,
                context_snippet=context_snippet or "",
                fix_description=fix_description or "",
                fix_success=fix_success,
            ))
            print(
                f"  [memory_manager] error experience recorded: "
                f"{error_type} in {file_path} -> "
                f"{'[OK]' if fix_success else '[FAILED]'} ({eid})"
            )
            return eid
        except Exception as e:
            print(f"  [memory_manager] error recording failed (non-fatal): {e}")
            return ""

    async def get_fix_suggestion(
        self,
        error_text: str,
        file_path: str = "",
    ) -> str:
        """
        Fix Loop 修复前调用 —— 从历史中查找类似错误的修复建议。

        这是 §4.3 的另一半集成点：在 LLM 生成修复代码之前，
        先检索历史上类似错误的修复方案，注入到 Fixer Prompt 中。

        Args:
            error_text: 当前错误的描述
            file_path:  出错文件

        Returns:
            修复建议文本（可直接追加到 Fixer Prompt）
        """
        try:
            result = await self.rm.find_similar_errors(error_text, file_path, top_k=5)
            if result.has_relevant_history:
                print(
                    f"  [memory_manager] found {len(result.similar_errors)} "
                    f"similar past errors for fix suggestion"
                )
                return result.suggestion
            return ""
        except Exception as e:
            print(f"  [memory_manager] fix suggestion failed (non-fatal): {e}")
            return ""

    # ── 上下文构建 ───────────────────────────────────────────────

    def get_context(self) -> list[dict]:
        """
        构建 LLM 上下文（摘要 + 最近消息）。

        代理到 ShortTermMemory.get_context()，返回自包含的消息列表。
        直接传给 client.messages.create(messages=...)。

        Returns:
            Anthropic dict 格式的消息列表
        """
        return self.stm.get_context()

    def get_context_with_system(self, system_prompt: str) -> tuple[str, list[dict]]:
        """
        返回分离的 (system_prompt, messages) 元组。

        符合 Anthropic API 调用模式：
          response = client.messages.create(
              model=MODEL,
              system=system_prompt,
              messages=messages,
          )
        """
        return self.stm.get_context_with_system(system_prompt)

    # ── 诊断 ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """获取会话的 Memory System 统计信息。"""
        stm_stats = self.stm.stats
        return {
            "session_id": self.session_id,
            "model": self.model,
            "llm_calls": self._llm_call_count,
            "tool_calls": self._tool_call_count,
            "stored_to_ltm": self._stored_to_ltm_count,
            "stm_messages": len(self.stm.messages),
            "stm_tokens": self.stm.count_tokens(),
            "stm_summary_chars": len(self.stm.summary),
            "stm_compressions": stm_stats.get("total_compressions", 0),
            "ltm_total_entries": self.ltm.count(),
            "rm_errors_recorded": self.rm.stats.get("total_errors_recorded", 0),
            "embedding_cache_hit_rate": round(self.emb.cache_hit_rate, 3),
            "session_duration_s": round(time.time() - self._started_at, 1),
        }

    def diagnostic_snapshot(self) -> str:
        """生成诊断快照文本。"""
        s = self.stats
        lines = [
            "══════ MemoryManager Snapshot ══════",
            f"  Session:       {s['session_id']}",
            f"  Model:         {s['model']}",
            f"  Duration:      {s['session_duration_s']}s",
            f"  LLM calls:     {s['llm_calls']}",
            f"  Tool calls:    {s['tool_calls']}",
            f"  Stored to LTM: {s['stored_to_ltm']}",
            "  ── Short-term Memory ──",
            f"    Messages:     {s['stm_messages']}",
            f"    Tokens:       {s['stm_tokens']:,}",
            f"    Summary:      {s['stm_summary_chars']} chars",
            f"    Compressions: {s['stm_compressions']}",
            "  ── Long-term Memory ──",
            f"    Total entries:{s['ltm_total_entries']}",
            "  ── Reflection Memory ──",
            f"    Errors:       {s['rm_errors_recorded']}",
            f"  ── Embedding Cache ──",
            f"    Hit rate:     {s['embedding_cache_hit_rate']:.1%}",
            "══════════════════════════════════════",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

# 自动存入 LTM 的工具白名单
_LTM_STORE_WHITELIST: dict[str, str] = {
    # 文件操作 → "code"
    "write_file": "code",
    "file_write": "code",
    "apply_patch": "code",
    "generate_diff": "code",
    # Shell 执行 → "task"
    "execute_shell": "task",
    "shell_execute": "task",
    # 代码分析 → "code"
    "analyze_code": "code",
    "search_code": "code",
    "read_file": "code",
    "file_read": "code",
    # Git → "task"
    "git_commit": "task",
    "git_branch": "task",
}


def _should_store_in_ltm(
    tool_name: str,
    tool_result: str,
    success: bool,
) -> str | None:
    """
    判断工具调用结果是否值得存入 Long-term Memory。

    规则：
      1. 工具名在白名单中 → 按映射存入对应集合
      2. 不在白名单中 → 不自动存储（避免噪音）
      3. 结果太短（< 20 字符）→ 不存储（无价值）
    """
    # 结果太短，不值得存储
    if len(tool_result) < 20:
        return None

    # 白名单匹配
    store_type = _LTM_STORE_WHITELIST.get(tool_name)
    if store_type:
        return store_type

    # 前缀匹配（"file_read_..." 这类变体）
    for prefix, stype in _LTM_STORE_WHITELIST.items():
        if tool_name.startswith(prefix):
            return stype

    return None


def _format_tool_input_summary(tool_input: dict) -> str:
    """
    格式化工具参数为简短摘要（用于 LTM 存储）。

    Args:
        tool_input: 工具参数字典

    Returns:
        简短的参数字符串，如 "file=main.py, content_len=1234"
    """
    if not isinstance(tool_input, dict):
        return str(tool_input)[:100]

    parts = []
    for k, v in tool_input.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)[:200]
