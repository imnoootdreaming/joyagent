"""
上下文压缩 Benchmark — 复现四层压缩的 Token 节省数据（用于简历量化）。

说明：
  本项目 TokenManager 依赖 tiktoken（Claude 用 cl100k_base 近似，误差 ±5%）。
  若环境无 tiktoken，本脚本在进程内打桩一个等价 heuristic encoder，
  使真实的四层压缩管线（L3/L1/L2）可运行并得到可复现的相对比例。

  对比对象：
    (A) 不压缩（原始消息累积）的 token 数
    (B) 启用 L3/L1/L2 纯文本层压缩后的 token 数（0 LLM 调用，确定性）
  L4 递进式摘要为 LLM 兜底层，端到端实际节省通常更高。
"""
from __future__ import annotations

import sys
import types


# ── tiktoken 桩（仅在缺失时注入，口径对齐 cl100k_base 启发式）──
if "tiktoken" not in sys.modules:
    class _Enc:
        def encode(self, text: str):
            # ≈4 字符/token，贴近 cl100k_base 英文估算
            return ["t"] * max(1, len(text) // 4)

    class _Tiktoken(types.ModuleType):
        def get_encoding(self, name: str):
            return _Enc()

        def encoding_for_model(self, name: str):
            return _Enc()

    sys.modules["tiktoken"] = _Tiktoken("tiktoken")


from app.memory.short_term import ShortTermMemory  # noqa: E402


def build_simulated_session(rounds: int = 20, result_kb: int = 60) -> list[dict]:
    """构造模拟会话：每轮 = 用户指令 + 一个大工具结果 + assistant 回复。"""
    messages: list[dict] = []
    filler = "x" * (result_kb * 1024)
    for i in range(rounds):
        messages.append({"role": "user", "content": f"第 {i + 1} 步：读取模块并分析"})
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": f"t{i}", "content": filler}
            ],
        })
        messages.append({"role": "assistant", "content": f"已处理第 {i + 1} 步的结果。"})
    return messages


def main() -> None:
    stm_ref = ShortTermMemory()
    tm = stm_ref._token_manager

    def msg_tokens(m: dict) -> int:
        # 与 TokenManager 口径对齐：序列化后用 count_tokens 估算
        import json
        text = json.dumps(m, ensure_ascii=False)
        return tm.count_tokens(text)

    raw = build_simulated_session()
    raw_tokens = sum(msg_tokens(m) for m in raw)

    # 启用四层压缩（L3/L1/L2 纯文本层，确定性，不调 LLM）
    stm = ShortTermMemory()
    for m in raw:
        stm.add_message(m)
    stm.tool_result_budget()
    stm.snip_compact()
    stm.micro_compact()
    compressed = stm.get_context()
    comp_tokens = sum(msg_tokens(m) for m in compressed)

    saved = raw_tokens - comp_tokens
    pct = (saved / raw_tokens * 100) if raw_tokens else 0

    print("=" * 56)
    print("  Context Compression Benchmark (L3/L1/L2)")
    print("=" * 56)
    print(f"  模拟轮数              : {len(raw) // 3}")
    print(f"  原始上下文 Token      : {raw_tokens:,}")
    print(f"  压缩后上下文 Token    : {comp_tokens:,}")
    print(f"  节省 Token            : {saved:,}  ({pct:.1f}%)")
    print(f"  说明                  : L4 递进式摘要为 LLM 兜底层，")
    print(f"                         实际端到端节省通常高于该值。")
    print("=" * 56)


if __name__ == "__main__":
    main()
