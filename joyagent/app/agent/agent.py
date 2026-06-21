from app.agent.prompt import SYSTEM_PROMPT
from app.service.llm_service import get_or_create_client
from app.tools.schemas import TOOLS, TOOL_HANDLERS
from app.core.config import Config

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
    Trace for max_tokens recovery attempts, recovery times and compact attempts within a single agent_loop execution.
    """

    def __init__(self):
        self.has_escalated = False           # 是否已升级过 max_tokens
        self.recovery_count = 0              # 续写次数
        self.has_attempted_compact = False   # 是否已尝试紧急压缩


class Agent:
    """
    Autonomous coding agent — Anthropic style ReAct Loop
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.DEFAULT_MODEL
        self.client = get_or_create_client(self.model_name)
        self.max_iterations = Config.MAX_ITERATIONS

    async def agent_loop(self, user_message: str, history: list = None) -> dict:
        """ 
        Core（Anthropic style）：
            while stop_reason == "tool_use":
                response = client.messages.create(messages, tools)
                tool_results = execute_tools(response.content)
                messages.append({"role": "user", "content": tool_results})
        """
        # 构建消息历史（纯 dict，非 LangChain 对象）
        messages = []
        if history:
            messages.extend(history)
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
                    tools=TOOLS,                 # Anthropic 原生工具格式
                    max_tokens=max_tokens,
                )
            except Exception as e:
                # 简单错误处理（后续 Phase 补充完整 error recovery）
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

            # ── 4. 检查该轮是否有工具调用（没有 tool_use 代表 LLM 看过了一切内容） ──
            if response.stop_reason != "tool_use":
                # 没有工具调用 → 模型认为任务完成
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

            # ── 5. 执行工具调用 ──
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input  # Anthropic 用 .input，不是 .args

                # 查找并执行 handler
                handler = TOOL_HANDLERS.get(tool_name)
                if handler:
                    result = handler(**tool_input)
                else:
                    result = f"Error: Unknown tool '{tool_name}'"

                tool_calls_made.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "result": str(result)[:500],
                })

                # 构造 Anthropic 格式的 tool_result
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })

            # ── 6. 工具结果作为 user 消息追加，循环继续 ──
            messages.append({"role": "user", "content": tool_results})

        # 超出最大迭代次数
        return {
            "response": "Task exceeded maximum iterations.",
            "tool_calls": tool_calls_made,
            "iterations": iterations,
        }
