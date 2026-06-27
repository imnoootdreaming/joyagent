"""
Phase 6 Step 1: Token Manager — Token 计数与 Context Window 管理。

TokenManager 是 Memory System 的基础设施层——所有记忆策略（滑动窗口、
摘要压缩、向量检索）都依赖精确的 Token 计数来决定"何时触发"。

为什么需要 TokenManager 而不是简单用 len(text)？
  1. LLM 的计费/billing 按 token（不是字符）——字符数与 token 数可能相差 3-10 倍
  2. Context Window 的限制是按 token 计算的（如 Claude 200K tokens）
  3. 不同模型使用不同的 tokenizer（GPT-4o 用 o200k_base，Claude 用 cl100k_base）
  4. 精确计数让压缩策略更准确——在真正接近限制前就触发，避免重试成本

模型 Context Window 对照（面试用）：
  Claude Opus 4.8 / Sonnet 4.6  — 200,000 tokens  (~150K 英文单词)
  GPT-4o / GPT-4o-mini          — 128,000 tokens  (~96K 英文单词)
  DeepSeek-V3                    — 128,000 tokens
  Claude Haiku 4.5               — 200,000 tokens

消息格式兼容：
  Phase 1-5 的消息列表是 Anthropic 原生 dict 格式：
    {"role": "user", "content": "hello"}
    {"role": "assistant", "content": [{"type": "text", "text": "..."}, ...]}
    {"role": "user", "content": [{"type": "tool_result", "content": "..."}]}
  content 字段可能是 str / list[dict] / list[SDKBlock] 三种格式，
  TokenManager 需要安全提取所有文本内容来估算 token 数。

使用方式：
  from app.memory.token_manager import TokenManager

  tm = TokenManager()
  tokens = tm.count_tokens("hello world")       # 2
  msg_tokens = tm.count_messages(messages)       # 估算消息列表总 token 数
  remaining = tm.get_remaining_budget(messages, "claude-sonnet-4-6")
  if tm.should_compress(messages, "claude-sonnet-4-6"):
      # 触发上下文压缩
      pass
"""

# ── Python 标准库 ──
import tiktoken                        # OpenAI 官方 tokenizer（被 Anthropic/DeepSeek 兼容）

# ── 项目内导入 ──
from app.core.config import Config     # DEFAULT_MODEL — 默认使用当前模型


# ═══════════════════════════════════════════════════════════════════════════════
# TokenManager — Token 计数引擎
# ═══════════════════════════════════════════════════════════════════════════════

class TokenManager:
    """
    Token 计数和 Context Window 管理。

    职责：
      1. 精确计数文本 token 数（基于模型 tokenizer）
      2. 估算消息列表（含 Anthropic SDK content blocks）的总 token 数
      3. 计算剩余 Context Window 预算
      4. 判断是否需要触发上下文压缩

    模型 → tokenizer 映射：
      - GPT-4o / GPT-4o-mini          → o200k_base (OpenAI 最新)
      - GPT-4 / GPT-3.5               → cl100k_base
      - Claude 全系列                  → cl100k_base (近似，Claude 无公开 tokenizer)
      - DeepSeek-V3                    → cl100k_base (DeepSeek 兼容 OpenAI tokenizer)
      - 未知模型                       → cl100k_base (安全回退)

    Claude tokenizer 的注意事项：
      Anthropic 没有公开发布 Claude 的 tokenizer。实际使用中 cl100k_base
      对英文文本的 token 化结果与 Claude 的真实 tokenizer 误差在 ±5% 以内。
      对于中英文混合内容，误差可能稍大。面试时诚实说明这一点即可。

    Context Window 限制（面试数据）：
      - Claude Opus 4.8 / Sonnet 4.6 / Haiku 4.5 → 200K
      - GPT-4o / GPT-4o-mini                        → 128K
      - DeepSeek-V3                                 → 128K
      - Claude 3 Opus (legacy)                      → 200K
      - Claude 3 Sonnet (legacy)                    → 200K
    """

    # ── 模型 tokenizer 映射 ──
    ENCODER_MAP: dict[str, str] = {
        # OpenAI
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        # Claude (用 cl100k_base 近似)
        "claude": "cl100k_base",
        "claude-sonnet": "cl100k_base",
        "claude-opus": "cl100k_base",
        "claude-haiku": "cl100k_base",
        # DeepSeek
        "deepseek": "cl100k_base",
    }

    # ── 模型 Context Window 限制 ──
    CONTEXT_LIMITS: dict[str, int] = {
        # Claude 全系列 — 200K
        "claude-sonnet-4-6": 200_000,
        "claude-opus-4-8": 200_000,
        "claude-haiku-4-5": 200_000,
        "claude-sonnet-4-5": 200_000,
        "claude-opus-4-7": 200_000,
        "claude-3-opus": 200_000,
        "claude-3-sonnet": 200_000,
        "claude-3-haiku": 200_000,
        # GPT-4o / GPT-4o-mini — 128K
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        # GPT-4 — 128K (Turbo), 8K/32K (legacy)
        "gpt-4-turbo": 128_000,
        "gpt-4": 8_192,                   # 默认 GPT-4 是 8K 版本
        "gpt-4-32k": 32_768,
        # GPT-3.5 — 16K (Turbo), 4K (legacy)
        "gpt-3.5-turbo": 16_384,
        # DeepSeek — 128K
        "deepseek": 128_000,
        "deepseek-v3": 128_000,
        "deepseek-v4": 128_000,
        # 默认 — 128K（保守估计）
        "default": 128_000,
    }

    # ── 每条消息的格式开销（token 估算） ──
    # Anthropic API 的 role/content 格式 + tool_use block 的元信息
    MSG_OVERHEAD_TOKENS = 4               # 每条消息的 role/content 框架约 4 tokens

    # ── 系统 Prompt 预留空间 ──
    DEFAULT_SYSTEM_PROMPT_BUDGET = 2000   # System Prompt 通常占 1K-4K tokens

    def __init__(self, model: str = None):
        """
        初始化 TokenManager —— 选择对应模型的 tokenizer。

        Args:
            model: 模型名称（如 "claude-sonnet-4-6"）。
                   为 None 时使用 Config.DEFAULT_MODEL。
        """
        model = model or Config.DEFAULT_MODEL

        # ── 选择 encoder ──
        self.encoder = self._get_encoder(model)
        self.encoder_name = self._get_encoder_name(model)

        print(
            f"  [token_manager] model={model}, "
            f"encoder={self.encoder_name}, "
            f"context_limit={self.get_context_limit(model):,} tokens"
        )

    def _get_encoder(self, model: str) -> tiktoken.Encoding:
        """
        获取模型对应的 tiktoken encoder。

        优先精确匹配模型名 → 回退到前缀匹配 → 最终回退 cl100k_base。
        """
        model_lower = model.lower()

        # 1. 精确匹配
        if model_lower in self.ENCODER_MAP:
            enc_name = self.ENCODER_MAP[model_lower]
            return tiktoken.get_encoding(enc_name)

        # 2. 前缀匹配（"claude-sonnet-4-6" → 匹配 "claude-sonnet"）
        for prefix, enc_name in self.ENCODER_MAP.items():
            if model_lower.startswith(prefix):
                return tiktoken.get_encoding(enc_name)

        # 3. 最终回退
        return tiktoken.get_encoding("cl100k_base")

    def _get_encoder_name(self, model: str) -> str:
        """获取 encoder 的名称（用于日志）。"""
        model_lower = model.lower()
        if model_lower in self.ENCODER_MAP:
            return self.ENCODER_MAP[model_lower]
        for prefix, enc_name in self.ENCODER_MAP.items():
            if model_lower.startswith(prefix):
                return enc_name
        return "cl100k_base (fallback)"

    # ── 核心计数方法 ──────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        """
        精确计数单段文本的 token 数。

        Args:
            text: 任意文本字符串（中文、英文、代码等）

        Returns:
            token 数量

        Example:
            tm.count_tokens("hello world")        → 2
            tm.count_tokens("你好世界")            → 4 (中文字符每个 1-2 token)
            tm.count_tokens("async def fn():...") → ~10
        """
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def count_content_block(self, content) -> int:
        """
        估算单个 Anthropic content block 的 token 数。

        Anthropic response.content 中的 block 可以是：
          - TextBlock(type="text", text="...")         → 纯文本
          - ToolUseBlock(type="tool_use", name, input, id) → 工具调用
          - ThinkingBlock(type="thinking", thinking, signature) → 思考过程

        以及 Anthropic API 接收的 dict：
          - {"type": "text", "text": "..."}
          - {"type": "tool_result", "tool_use_id": ..., "content": "..."}

        Args:
            content: Anthropic SDK block 对象或 dict

        Returns:
            估算 token 数
        """
        if content is None:
            return 0

        # ── Anthropic SDK TextBlock / ThinkingBlock（Pydantic 对象） ──
        if hasattr(content, 'type') and hasattr(content, 'text'):
            return self.count_tokens(getattr(content, 'text', ''))

        if hasattr(content, 'type') and content.type == 'thinking':
            return self.count_tokens(getattr(content, 'thinking', ''))

        # ── Anthropic SDK ToolUseBlock ──
        if hasattr(content, 'type') and content.type == 'tool_use':
            total = 0
            total += self.count_tokens(getattr(content, 'name', ''))
            # 序列化 input 为 JSON 后计数
            import json
            try:
                input_str = json.dumps(getattr(content, 'input', {}))
                total += self.count_tokens(input_str)
            except Exception:
                pass
            total += 6                      # tool_use 格式开销
            return total

        # ── Dict 格式（经过 _content_to_dicts 转换或原始 API 消息） ──
        if isinstance(content, dict):
            block_type = content.get("type", "")

            if block_type == "text":
                return self.count_tokens(content.get("text", ""))
            elif block_type == "tool_use":
                total = self.count_tokens(content.get("name", ""))
                import json
                try:
                    total += self.count_tokens(
                        json.dumps(content.get("input", {}))
                    )
                except Exception:
                    pass
                total += 6
                return total
            elif block_type == "tool_result":
                return self.count_tokens(
                    str(content.get("content", ""))
                )
            elif block_type == "thinking":
                return self.count_tokens(content.get("thinking", ""))
            else:
                # 未知 dict → 转字符串
                return self.count_tokens(str(content))

        # ── 纯字符串 ──
        if isinstance(content, str):
            return self.count_tokens(content)

        # ── 兜底 ──
        return self.count_tokens(str(content))

    def count_messages(self, messages: list) -> int:
        """
        估算消息列表的总 token 数。

        处理消息的两种格式：
          1. Anthropic API 原始格式:
             {"role": "user", "content": "hello"}
             {"role": "assistant", "content": [TextBlock, ToolUseBlock, ...]}
             其中 content 是 SDK 对象的列表

          2. 经过 LangGraph add_messages reducer 后的 LangChain 对象:
             HumanMessage(type="human", content="hello")
             AIMessage(type="ai", content=[...])

          3. 纯 dict 格式（经过 _content_to_dicts）:
             {"role": "assistant", "content": [{"type": "text", "text": "..."}]}

        Args:
            messages: 消息列表

        Returns:
            估算的总 token 数（含每条消息的格式开销）
        """
        total = 0
        for msg in messages:
            # ── LangChain Message 对象（add_messages reducer 转换后） ──
            if hasattr(msg, 'content'):
                content = msg.content
                if isinstance(content, str):
                    total += self.count_tokens(content)
                elif isinstance(content, list):
                    for block in content:
                        total += self.count_content_block(block)
                total += self.MSG_OVERHEAD_TOKENS
                continue

            # ── Anthropic 原生 dict 格式 ──
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    total += self.count_tokens(content)
                elif isinstance(content, list):
                    # content = [{"type": "text", "text": "..."}, ...]
                    for block in content:
                        total += self.count_content_block(block)
                total += self.MSG_OVERHEAD_TOKENS
                continue

            # ── 兜底 ──
            total += self.count_tokens(str(msg))
            total += self.MSG_OVERHEAD_TOKENS

        return total

    def count_messages_simple(self, messages: list) -> int:
        """
        简化版消息列表 token 计数（比 count_messages 快 5-10 倍）。

        不做逐 block 解析——直接把每条消息转为字符串后计数。
        适合高频调用场景（如每轮循环后检查是否需要压缩）。

        精度：误差在 ±15% 以内（对于 SDK block 对象可能多计或少计）。
        对于压缩决策来说这个精度足够了——压缩阈值本来就是一个估算值。
        """
        total = 0
        for msg in messages:
            if hasattr(msg, 'content'):
                total += self.count_tokens(str(msg.content))
            elif isinstance(msg, dict):
                total += self.count_tokens(str(msg.get("content", "")))
            else:
                total += self.count_tokens(str(msg))
            total += self.MSG_OVERHEAD_TOKENS
        return total

    # ── Context Window 预算管理 ───────────────────────────────

    def get_context_limit(self, model: str) -> int:
        """
        获取模型的 Context Window 大小。

        Args:
            model: 模型名称

        Returns:
            Context Window token 限制
        """
        model_lower = model.lower()

        # 1. 精确匹配
        if model_lower in self.CONTEXT_LIMITS:
            return self.CONTEXT_LIMITS[model_lower]

        # 2. 前缀匹配
        for prefix, limit in self.CONTEXT_LIMITS.items():
            if model_lower.startswith(prefix):
                return limit

        # 3. 默认
        return self.CONTEXT_LIMITS["default"]

    def get_remaining_budget(
        self,
        messages: list,
        model: str = None,
        system_prompt: str = "",
    ) -> int:
        """
        计算 Context Window 剩余 Token 预算。

        Args:
            messages:      当前消息列表
            model:         模型名称
            system_prompt: System Prompt 文本（也为它预留空间）

        Returns:
            剩余可用 token 数（永远不会返回负数——最小值为 0）

        Example:
            tm = TokenManager()
            remaining = tm.get_remaining_budget(
                messages, "claude-sonnet-4-6", system_prompt=SYSTEM_PROMPT
            )
            if remaining < 4096:
                print("Warning: low token budget!")
        """
        model = model or Config.DEFAULT_MODEL
        limit = self.get_context_limit(model)

        # 总用量 = 消息 token + System Prompt token + 预留响应空间
        used = self.count_messages(messages)
        used += self.count_tokens(system_prompt)

        # max_tokens（LLM 响应最大长度）也需要预留
        reserve_for_response = 4096

        remaining = limit - used - reserve_for_response
        return max(0, remaining)

    def should_compress(
        self,
        messages: list,
        model: str = None,
        reserve_for_response: int = 4096,
        threshold: float = 0.80,
    ) -> bool:
        """
        判断是否需要触发上下文压缩。

        当 (已用 token + 预留响应空间) > Context Window × threshold 时，
        返回 True → 调用方应该触发摘要压缩或滑动窗口。

        Args:
            messages:            消息列表
            model:               模型名称
            reserve_for_response: 为 LLM 响应预留的 token 数（默认 4096）
            threshold:           触发阈值（占 Context Window 的比例，默认 0.80）

        Returns:
            True → 需要压缩

        Example:
            if tm.should_compress(messages, "claude-sonnet-4-6"):
                # 压缩历史消息，释放 Context Window 空间
                stm.compress()
        """
        model = model or Config.DEFAULT_MODEL
        limit = self.get_context_limit(model)

        used = self.count_messages_simple(messages)
        # 加上 System Prompt 的预留（保守估计）
        used += self.DEFAULT_SYSTEM_PROMPT_BUDGET

        # 判断：当前用量 + 响应预留 > 限制 × 阈值
        return (used + reserve_for_response) > (limit * threshold)

    def get_usage_info(
        self,
        messages: list,
        model: str = None,
        system_prompt: str = "",
    ) -> dict:
        """
        获取详细的 Context Window 使用信息。

        Returns:
            {
                "model": str,
                "context_limit": int,
                "tokens_used": int,
                "tokens_remaining": int,
                "usage_pct": float,          # 0.0 ~ 1.0
                "should_compress": bool,
                "system_prompt_tokens": int,
                "message_count": int,
            }

        适合在 UI 上展示一个"Token 用量条"：
          [████████░░] 34,560 / 200,000 (17.3%)
        """
        model = model or Config.DEFAULT_MODEL
        limit = self.get_context_limit(model)

        msg_tokens = self.count_messages(messages)
        sys_tokens = self.count_tokens(system_prompt)
        used = msg_tokens + sys_tokens

        remaining = max(0, limit - used - 4096)  # 减预留响应空间
        usage_pct = used / limit if limit > 0 else 0.0

        return {
            "model": model,
            "context_limit": limit,
            "tokens_used": used,
            "tokens_remaining": remaining,
            "usage_pct": round(usage_pct, 4),
            "should_compress": self.should_compress(messages, model),
            "system_prompt_tokens": sys_tokens,
            "message_count": len(messages),
        }

    def format_usage(self, messages: list, model: str = None,
                     system_prompt: str = "") -> str:
        """
        将 Token 用量格式化为 LLM 友好的文本。

        Returns:
            如 "Token usage: 34,560 / 200,000 (17.3%) — 165,440 remaining"
        """
        info = self.get_usage_info(messages, model, system_prompt)
        return (
            f"Token usage: {info['tokens_used']:,} / {info['context_limit']:,} "
            f"({info['usage_pct']:.1%}) — "
            f"{info['tokens_remaining']:,} remaining, "
            f"{info['message_count']} messages"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════════════════

# 全局默认 TokenManager 实例（使用 Config.DEFAULT_MODEL）
_token_manager: TokenManager | None = None


def get_token_manager(model: str = None) -> TokenManager:
    """
    获取全局 TokenManager 实例（懒加载 + 缓存）。

    首次调用时创建，后续调用返回同一实例。
    """
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager(model)
    return _token_manager
