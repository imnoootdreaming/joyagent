from app.agent.prompt import SYSTEM_PROMPT
from app.service.llm_service import get_or_create_client
from app.tools.registry import tool_registry
from app.core.config import Config
from app.tools.base import ToolResult

# ── Agent 核心循环常量 ──
DEFAULT_MAX_TOKENS = 4096
ESCALATED_MAX_TOKENS = DEFAULT_MAX_TOKENS * 2    # max_tokens 升级上限
MAX_RECOVERY_RETRIES = 3        # 续写最多尝试次数
CONTINUATION_PROMPT = (
    "Output token limit hit. Resume directly — "
    "no apology, no recap. Pick up mid-thought."
)


class RecoveryState:
    """
    Trace for max_tokens recovery attempts, recovery times and compact attempts
    within a single agent_loop execution.
    """

    def __init__(self):
        self.has_escalated = False           # 是否已升级过 max_tokens
        self.recovery_count = 0              # 续写次数
        self.has_attempted_compact = False   # 是否已尝试紧急压缩


class Agent:
    """
    Autonomous coding agent — Anthropic style ReAct Loop.

    Phase 2: 使用 ToolRegistry 统一管理工具（替代 Phase 1 硬编码 TOOLS/TOOL_HANDLERS）。
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.DEFAULT_MODEL
        self.client = get_or_create_client(self.model_name)
        self.max_iterations = Config.MAX_ITERATIONS

    async def agent_loop(self, user_message: str, context: list = None) -> dict:
        """
        Core ReAct Loop（Anthropic native style）：

            while stop_reason == "tool_use":
                response = client.messages.create(messages, tools)
                tool_results = execute_via_registry(response.content)
                messages.append({"role": "user", "content": tool_results})

        Phase 2 变化：
          - tools:        tool_registry.get_tool_schemas() 替代硬编码 TOOLS
          - 工具执行:      tool_registry.execute(name, **input) 替代 TOOL_HANDLERS
          - is_dangerous:  危险工具执行时打印黄色警告日志
        """
        # 构建消息历史（纯 dict，非 LangChain 对象）
        messages = []
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": user_message})

        iterations = 0
        tool_calls_made = []
        state = RecoveryState()
        max_tokens = DEFAULT_MAX_TOKENS

        while iterations < self.max_iterations:
            iterations += 1

            # ── 1. 调用 LLM ──
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    system=SYSTEM_PROMPT,       # system 是独立参数
                    messages=messages,           # 纯 dict 列表
                    tools=tool_registry.get_tool_schemas(),  # ⬅ Phase 2: Registry
                    max_tokens=max_tokens,
                )
            except Exception as e:
                return {
                    "response": f"[Error] {type(e).__name__}: {e}",
                    "tool_calls": tool_calls_made,
                    "iterations": iterations,
                }

            # ── 2. 追加 assistant 回复到消息历史 ──
            messages.append({"role": "assistant", "content": response.content})

            # ── 3. max_tokens 恢复 ──
            if response.stop_reason == "max_tokens":
                if not state.has_escalated:
                    # 第一次：升级 max_tokens 后重试同一请求
                    max_tokens = ESCALATED_MAX_TOKENS
                    state.has_escalated = True
                    messages.pop()  # 移除截断的 assistant 消息
                    print(f"  \033[33m[max_tokens] escalating to {max_tokens}\033[0m")
                    continue
                # 已升级仍截断：追加续写提示
                if state.recovery_count < MAX_RECOVERY_RETRIES:
                    messages.append({
                        "role": "user",
                        "content": CONTINUATION_PROMPT,
                    })
                    state.recovery_count += 1
                    print(
                        f"  \033[33m[max_tokens] continuation "
                        f"{state.recovery_count}/{MAX_RECOVERY_RETRIES}\033[0m"
                    )
                    continue
                return {
                    "response": "Task output exceeded max token limits.",
                    "tool_calls": tool_calls_made,
                    "iterations": iterations,
                }

            # ── 4. 检查 stop_reason：没有 tool_use → 任务完成 ──
            if response.stop_reason != "tool_use":
                text_output = ""
                for block in response.content:
                    if block.type == "text":
                        text_output += block.text
                return {
                    "response": text_output or "Task completed.",
                    "stop_reason": response.stop_reason,
                    "tool_calls": tool_calls_made,
                    "iterations": iterations,
                }

            # ── 5. 执行工具调用（Phase 2: 统一通过 ToolRegistry） ──
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input  # Anthropic 用 .input，不是 ["args"]

                # ── 5a. is_dangerous 检查（Phase 2: 打黄色警告日志） ──
                tool = tool_registry.get_tool(tool_name)
                if tool and tool.is_dangerous:
                    print(
                        f"  \033[33m[!] dangerous {tool_name}("
                        f"{_format_args(tool_input)}) — requires confirmation\033[0m"
                    )

                # ── 5b. 通过 Registry 执行工具 ──
                result = await tool_registry.execute(tool_name, **tool_input)
                if isinstance(result, ToolResult):
                    # 成功/失败都从 ToolResult 提取 LLM 可读文本
                    content_for_llm = (
                        result.message if result.success
                        else f"Error: {result.error or result.message}"
                    )
                else:
                    # 未知工具降级（ToolRegistry 返回带 error 的 ToolResult，一般不会到这）
                    content_for_llm = str(result)

                # ── 5c. 记录工具调用（供 API 返回和日志审计） ──
                tool_calls_made.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "result": content_for_llm[:500],
                    "execution_mode": result.metadata.get("execution_mode", "unknown")
                    if isinstance(result, ToolResult) and result.metadata
                    else "unknown",
                })

                # ── 5d. 构造 Anthropic 格式的 tool_result ──
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content_for_llm,
                })

            # ── 6. 工具结果作为 user 消息追加，循环继续 ──
            messages.append({"role": "user", "content": tool_results})

        # 超出最大迭代次数
        return {
            "response": "Task exceeded maximum iterations.",
            "tool_calls": tool_calls_made,
            "iterations": iterations,
        }


def _format_args(kwargs: dict, max_len: int = 80) -> str:
    """格式化工具参数为简短字符串（用于日志输出）。"""
    parts = []
    for k, v in kwargs.items():
        s = str(v)
        if len(s) > 40:
            s = s[:37] + "..."
        parts.append(f"{k}={s}")
    joined = ", ".join(parts)
    if len(joined) > max_len:
        joined = joined[:max_len - 3] + "..."
    return joined
