from __future__ import annotations
"""
Phase 6 Step 3a: Context Compression — LLM 驱动的对话摘要与上下文压缩。

Summary 模块实现了 §6.1 的上下文压缩策略——让 Agent 在长对话中保持"记忆连续"。
当消息 Token 数接近 Context Window 限制时，不直接丢弃旧消息（那会导致 Agent
"失忆"），而是用 LLM 将旧消息压缩为递进式摘要，保留关键决策、错误和修复信息。

核心策略（三层压缩）：
  1. System Prompt 不动         — 不压缩（它是 Agent 的行为准则）
  2. 中间层 → LLM 生成渐进式摘要 — 旧消息被压缩为"叙事"而非数据
  3. 最新层 → 保留原始消息       — 保证最近上下文的完整性和工具调用精度

递进式摘要（Progressive Summary）：
  不是每次从头总结所有历史消息（那样 O(n²)），而是：
    new_summary = summarize(existing_summary + new_old_messages)
  这样每次压缩只消耗 O(1) 的 LLM 调用成本，摘要随对话推进逐步累积。

工具调用结果截断：
  tool_result 可能包含大段文件内容或命令输出（数千 token），压缩时只保留：
    - 工具名称
    - 成功/失败状态
    - 前 200 字符的输出
  丢弃冗长输出，但保留"哪些工具被调用、结果如何"的语义信息。

使用方式：
  from app.memory.summary import generate_summary, compress_context

  new_summary = await generate_summary(old_messages, existing_summary="")
  compressed, new_summary = await compress_context(messages, existing_summary, keep_recent=20)
"""

# ── Python 标准库 ──
from typing import Optional

# ── 项目内导入 ──
from app.service.llm_service import get_or_create_client
from app.core.config import Config


# ═══════════════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════════════

# 摘要 LLM 的 system prompt —— 专注、简洁
SUMMARY_SYSTEM_PROMPT = (
    "You are a conversation summarizer for an autonomous coding agent. "
    "Your job is to compress old conversation history into a concise, "
    "information-dense summary that preserves all critical context.\n\n"
    "Rules:\n"
    "1. Keep it concise — 300 words max\n"
    "2. Include: key decisions, file changes, errors & fixes, current task state\n"
    "3. Write in the language of the original conversation\n"
    "4. Do NOT narrate what you're doing — just output the summary\n"
    "5. Use bullet points for clarity"
)

# 摘要生成最大 token 数（摘要本身不应太长）
SUMMARY_MAX_TOKENS = 2048

# 工具结果截断长度（保留开头 + 结尾关键信息）
TOOL_RESULT_TRUNCATE_LENGTH = 200


# ═══════════════════════════════════════════════════════════════════════════════
# 消息格式化工具
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_text_from_content(content) -> str:
    """
    从 Anthropic 消息的 content 字段安全提取纯文本。

    content 可能是：
      - str: "hello"
      - list[dict]: [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]
      - list[SDKBlock]: TextBlock / ToolUseBlock / ThinkingBlock 对象

    对于 tool_use 和 tool_result，生成简短的描述文本（不包含完整输入/输出）。
    """
    # ── 纯字符串 ──
    if isinstance(content, str):
        return content

    # ── 列表 ──
    if isinstance(content, list):
        parts = []
        for block in content:
            # ── Anthropic SDK Block 对象 ──
            if hasattr(block, 'type'):
                block_type = block.type
                if block_type == "text":
                    parts.append(getattr(block, 'text', ''))
                elif block_type == "tool_use":
                    name = getattr(block, 'name', 'unknown')
                    # 截断 input
                    import json
                    try:
                        inp = json.dumps(getattr(block, 'input', {}))
                        if len(inp) > TOOL_RESULT_TRUNCATE_LENGTH:
                            inp = inp[:TOOL_RESULT_TRUNCATE_LENGTH] + "..."
                    except Exception:
                        inp = "..."
                    parts.append(f"[tool_use: {name}({inp})]")
                elif block_type == "tool_result":
                    tc = str(getattr(block, 'content', ''))
                    if len(tc) > TOOL_RESULT_TRUNCATE_LENGTH:
                        tc = tc[:TOOL_RESULT_TRUNCATE_LENGTH] + "..."
                    parts.append(f"[tool_result: {tc}]")
                elif block_type == "thinking":
                    parts.append("[thinking omitted]")
                else:
                    parts.append(str(block)[:TOOL_RESULT_TRUNCATE_LENGTH])

            # ── Dict 格式 ──
            elif isinstance(block, dict):
                bt = block.get("type", "")
                if bt == "text":
                    parts.append(block.get("text", ""))
                elif bt == "tool_use":
                    name = block.get("name", "unknown")
                    import json
                    try:
                        inp = json.dumps(block.get("input", {}))
                        if len(inp) > TOOL_RESULT_TRUNCATE_LENGTH:
                            inp = inp[:TOOL_RESULT_TRUNCATE_LENGTH] + "..."
                    except Exception:
                        inp = "..."
                    parts.append(f"[tool_use: {name}({inp})]")
                elif bt == "tool_result":
                    tc = str(block.get("content", ""))
                    if len(tc) > TOOL_RESULT_TRUNCATE_LENGTH:
                        tc = tc[:TOOL_RESULT_TRUNCATE_LENGTH] + "..."
                    parts.append(f"[tool_result: {tc}]")
                elif bt == "thinking":
                    parts.append("[thinking omitted]")
                else:
                    parts.append(str(block)[:TOOL_RESULT_TRUNCATE_LENGTH])
            else:
                parts.append(str(block)[:TOOL_RESULT_TRUNCATE_LENGTH])

        return " ".join(parts)

    return str(content)


def _format_message_for_summary(msg: dict | object) -> str:
    """
    将单条消息格式化为摘要 LLM 可读的一行文本。

    Args:
        msg: Anthropic dict 消息 或 LangChain SDK 消息对象

    Returns:
        "role: content_summary" 格式的字符串
    """
    # ── Anthropic 原生 dict ──
    if isinstance(msg, dict):
        role = msg.get("role", "unknown")
        content = _extract_text_from_content(msg.get("content", ""))
        return f"[{role}]: {content}"

    # ── SDK / LangChain 消息对象 ──
    if hasattr(msg, 'content'):
        role = getattr(msg, 'role', getattr(msg, 'type', 'unknown'))
        content = _extract_text_from_content(msg.content)
        return f"[{role}]: {content}"

    return str(msg)[:500]


def format_messages_for_summary(messages: list) -> str:
    """
    将消息列表格式化为摘要 Prompt 中的消息历史文本。

    工具调用结果会被截断（保留前 200 字符），避免摘要 Prompt 过长。

    Args:
        messages: 待压缩的消息列表

    Returns:
        格式化的多行文本，可直接嵌入摘要 Prompt
    """
    lines = []
    for msg in messages:
        line = _format_message_for_summary(msg)
        if len(line) > 500:
            line = line[:500] + "..."
        lines.append(line)

    # 对过长的消息列表做二次截断 — 控制摘要 Prompt 总长度
    if len(lines) > 100:
        lines = lines[:100]
        lines.append("... (additional messages omitted for brevity)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 核心：LLM 摘要生成
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_summary(
    old_messages: list,
    existing_summary: str = "",
    model: str = None,
) -> str:
    """
    调用 LLM 生成递进式对话摘要。

    递进式摘要策略：
      不是从头总结全部历史，而是基于 existing_summary（上次压缩的产物）
      增量追加 new messages 的信息。这样每次调用成本恒定，不会随对话增长。

    Args:
        old_messages:  需要压缩的旧消息列表（超过滑动窗口的部分）
        existing_summary: 已有的摘要文本（空字符串表示首次压缩）
        model:         用于摘要生成的模型（默认 Config.DEFAULT_MODEL）

    Returns:
        新的递进式摘要文本

    Raises:
        RuntimeError: LLM 调用失败时抛出（调用方应捕获并处理降级）

    Example:
        stm = ShortTermMemory()
        # 当消息过多时...
        new_summary = await generate_summary(
            old_messages=stm.messages[:50],
            existing_summary=stm.summary,
        )
        stm.summary = new_summary
        stm.messages = stm.messages[50:]
    """
    model = model or Config.DEFAULT_MODEL

    # ── 格式化消息 ──
    formatted = format_messages_for_summary(old_messages)
    if not formatted.strip():
        return existing_summary  # 空消息 → 无需更新摘要

    # ── 构建摘要 Prompt ──
    if existing_summary:
        summary_prompt = (
            f"## Existing Summary (from earlier compression)\n\n"
            f"{existing_summary}\n\n"
            f"## New Messages to Summarize\n\n"
            f"{formatted}\n\n"
            f"Update the existing summary by merging in the key information "
            f"from the new messages above. Keep it under 300 words. "
            f"Output ONLY the updated summary — no preamble, no meta-commentary."
        )
    else:
        summary_prompt = (
            f"## Messages to Summarize\n\n"
            f"{formatted}\n\n"
            f"Write a concise summary (under 300 words) that captures:\n"
            f"1. Key decisions made\n"
            f"2. Important file changes (created/modified/deleted)\n"
            f"3. Errors encountered and how they were fixed\n"
            f"4. Current task state and pending work\n\n"
            f"Output ONLY the summary — no preamble, no meta-commentary."
        )

    # ── 调用 LLM ──
    try:
        client = get_or_create_client(model)
        response = client.messages.create(
            model=model,
            system=SUMMARY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=SUMMARY_MAX_TOKENS,
            # 注意：不传 tools——摘要不触发工具调用
        )
    except Exception as e:
        raise RuntimeError(
            f"Summary generation failed: {e}\n"
            f"Model: {model}, messages count: {len(old_messages)}"
        ) from e

    # ── 提取摘要文本 ──
    text_parts = []
    for block in response.content:
        if hasattr(block, 'type'):
            if block.type == "text":
                text_parts.append(getattr(block, 'text', ''))
        elif isinstance(block, dict):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        else:
            text_parts.append(str(block))

    return "".join(text_parts).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 上下文压缩（完整策略）
# ═══════════════════════════════════════════════════════════════════════════════

async def compress_context(
    messages: list,
    existing_summary: str = "",
    keep_recent: int = 20,
    model: str = None,
) -> tuple[list, str]:
    """
    压缩对话上下文 —— 实现 §6.1 的完整压缩策略。

    策略：
      1. 保留最近 keep_recent 条消息（保证最近上下文完整）
      2. 前面的消息用 LLM 生成递进式摘要
      3. 工具调用结果截断（保留关键输出，丢弃冗长内容）
      4. 返回压缩后的消息列表 + 新摘要

    与 ShortTermMemory._compress() 的关系：
      ShortTermMemory._compress() 按"前 50% / 后 50%"分割，
      compress_context() 按"最近 N 条"分割。
      前者适合自动触发（不需要知道消息总数），
      后者适合手动调用（更精确的控制）。

    Args:
        messages:         完整消息列表
        existing_summary: 已有的递进式摘要
        keep_recent:      保留最近多少条消息（默认 20）
        model:            摘要用的 LLM 模型

    Returns:
        (compressed_messages, new_summary)
          - compressed_messages: 压缩后的消息列表（摘要消息 + 最近消息）
          - new_summary:         更新后的递进式摘要

    Example:
        # 在每次 LLM 调用前检查是否需要压缩
        if tm.should_compress(messages, model):
            messages, summary = await compress_context(messages, summary)
    """
    if len(messages) <= keep_recent:
        # 消息还不够多 → 不需要压缩
        result = list(messages)
        if existing_summary:
            result.insert(0, {"role": "user", "content": f"[Previous context summary]: {existing_summary}"})
        return result, existing_summary

    # ── 分层 ──
    recent_msgs = messages[-keep_recent:]     # 保留原样
    old_msgs = messages[:-keep_recent]         # 用于生成摘要

    if not old_msgs:
        return recent_msgs, existing_summary

    # ── 生成递进式摘要 ──
    try:
        new_summary = await generate_summary(old_msgs, existing_summary, model)
    except RuntimeError:
        # 降级：LLM 摘要失败 → 保留现有摘要 + 粗暴截断旧消息
        new_summary = existing_summary
        if not new_summary and old_msgs:
            # 连现有摘要都没有 → 用最后几条 old 消息的文本作为降级摘要
            fallback_parts = []
            for msg in old_msgs[-3:]:
                fallback_parts.append(_format_message_for_summary(msg)[:150])
            new_summary = " | ".join(fallback_parts)
            if len(new_summary) > 500:
                new_summary = new_summary[:500] + "..."

    # ── 构建压缩后的消息列表（Anthropic dict 格式） ──
    compressed = []
    if new_summary:
        compressed.append({
            "role": "user",
            "content": f"[Previous context summary]: {new_summary}",
        })
    compressed.extend(recent_msgs)

    return compressed, new_summary


# ═══════════════════════════════════════════════════════════════════════════════
# 紧急压缩（token budget 严重不足时的降级策略）
# ═══════════════════════════════════════════════════════════════════════════════

def emergency_truncate(
    messages: list,
    keep_recent: int = 10,
    existing_summary: str = "",
) -> list:
    """
    紧急截断（无需 LLM 调用的降级策略）。

    当 token budget 严重不足、来不及调 LLM 生成摘要时使用。
    直接丢弃旧消息，只保留最近 N 条 + 现有摘要（如果有）。

    注意：这会丢失旧消息中的细节信息，仅作为最后手段。

    Args:
        messages:         完整消息列表
        keep_recent:      保留最近 N 条（默认 10）
        existing_summary: 已有摘要（不会被丢弃）

    Returns:
        截断后的消息列表
    """
    if len(messages) <= keep_recent:
        result = list(messages)
        if existing_summary:
            result.insert(0, {"role": "user", "content": f"[Context]: {existing_summary}"})
        return result

    recent_msgs = messages[-keep_recent:]

    result = []
    if existing_summary:
        result.append({
            "role": "user",
            "content": f"[Context (emergency truncated)]: {existing_summary}",
        })
    result.extend(recent_msgs)

    return result
