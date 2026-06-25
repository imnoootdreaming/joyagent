"""
Phase 3 Step 3: Executor Node — 按计划逐步执行工具调用。

Executor 是 LangGraph Plan→Execute→Reflect 工作流的第二个 Node。
它从 state["plan"] 中取出当前步骤（由 current_step_index 索引），
运行一个迷你 ReAct Loop（LLM + Tool Use）执行该步骤，
将执行结果记录到 tool_call_history，然后推进 current_step_index。

与 Phase 1-2 Agent 的区别：
  - Phase 1-2 Agent 是无结构的"大 while 循环"，LLM 自己决定何时停止
  - Executor 只执行一步，然后交还控制权给 LangGraph 工作流
  - 是否继续 / replan / 结束 由后续的 Reflector Node 决定
"""

# ── Python 标准库 ──
from pathlib import Path             # 读取 Executor 专用 Prompt 模板文件

# ── 项目内导入 ──
from app.agent.schemas.state import AgentState, TaskStep
# AgentState: LangGraph 共享状态（输入整个 state，返回部分更新）
# TaskStep:   当前要执行步骤的类型

from app.service.llm_service import get_or_create_client
# 获取 Anthropic 客户端（按模型名分流 Claude / DeepSeek）

from app.core.config import Config
# 全局配置：DEFAULT_MODEL、MAX_ITERATIONS 等

from app.tools.registry import tool_registry
# 全局工具注册中心：get_tool_schemas() 取 Schema、execute() 执行工具调用

from app.tools.base import ToolResult
# 统一工具返回格式：success + message + error + metadata


# ── 常量 ─────────────────────────────────────────────────────────────────

# Executor 专用 Prompt 模板路径（与 planner.txt 同目录）
EXECUTOR_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "executor.txt"

# Executor 每步内 LLM 调用默认 max_tokens
EXECUTOR_DEFAULT_MAX_TOKENS = 4096

# max_tokens 升级上限（同 Phase 1-2 的 ESCALATED_MAX_TOKENS）
EXECUTOR_ESCALATED_MAX_TOKENS = EXECUTOR_DEFAULT_MAX_TOKENS * 2

# 单步内最多工具调用轮次（防止单步消耗过多迭代资源）
MAX_TOOL_ROUNDS_PER_STEP = 5

# 单步内 max_tokens 续写最多尝试次数
MAX_CONTINUATION_RETRIES = 2

# max_tokens 续写提示（与 Phase 1-2 保持一致）
CONTINUATION_PROMPT = (
    "Output token limit hit. Resume directly — "
    "no apology, no recap. Pick up mid-thought."
)


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _load_executor_prompt() -> str:
    """
    加载 Executor 专用的 System Prompt 模板。

    为什么从文件加载而非硬编码？
      - Prompt 会随调试迭代频繁修改，文件修改不触及代码逻辑
      - 模板可独立做版本管理（git diff 只看 prompt 变化）
      - 与 planner.txt / reflector.txt 统一管理
    """
    if EXECUTOR_PROMPT_PATH.exists():
        return EXECUTOR_PROMPT_PATH.read_text(encoding="utf-8")
    # 兜底：文件缺失时使用内置简版 Prompt
    return (
        "You are a step executor. Execute the given step using the available tools. "
        "Report what you did and whether it succeeded. Be concise."
    )


def _extract_text_from_response(response) -> str:
    """
    从 Anthropic Messages API 响应中提取纯文本。

    响应 response.content 是 block 列表，每个 block 有 type 字段：
      {"type": "text", "text": "LLM 输出..."}
      {"type": "tool_use", ...}

    只拼接 type=="text" 的 block，忽略 tool_use。
    """
    text_parts = []
    for block in response.content:
        if hasattr(block, 'type') and block.type == "text":
            # Anthropic SDK 返回对象，用属性访问
            text_parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            # 备选：纯 HTTP 响应返回 dict
            text_parts.append(block.get("text", ""))
    return "\n".join(text_parts).strip()


def _format_step_prompt(step: TaskStep) -> str:
    """
    构建发送给 LLM 的单步执行指令。

    将 TaskStep 的字段组装为清晰的文本指令，
    告诉 LLM 当前要做什么、推荐用什么工具。
    """
    lines = [
        f"## Current Step ({step['step_id']})",
        f"**Description:** {step['description']}",
    ]
    # 如果 Planner 推荐了工具，显式提示
    if step.get("tool_name"):
        lines.append(f"**Recommended Tool:** {step['tool_name']}")
    lines.append("")
    lines.append("Execute this step now. Use tools as needed, then report the result.")
    return "\n".join(lines)


def _format_args(kwargs: dict, max_len: int = 80) -> str:
    """
    格式化工具参数为简短字符串（用于危险工具警告日志）。

    超过 max_len 字符的参数值会被截断并附加 "..."。
    """
    parts = []
    for k, v in kwargs.items():
        s = str(v)
        if len(s) > 40:                      # 单个值超过 40 字符 → 截断
            s = s[:37] + "..."
        parts.append(f"{k}={s}")
    joined = ", ".join(parts)
    if len(joined) > max_len:                # 整行超过 max_len → 截断
        joined = joined[:max_len - 3] + "..."
    return joined


# ═══════════════════════════════════════════════════════════════════════════════
# Executor Node 主函数（LangGraph Node 接口）
# ═══════════════════════════════════════════════════════════════════════════════

async def executor_node(state: AgentState) -> dict:
    """
    LangGraph Executor Node —— 执行计划中的当前步骤。

    这是 Plan→Execute→Reflect 工作流的第二个 Node，由 Planner 无条件进入，
    或由 Reflector 的 Conditional Edge 路由回来继续执行下一步。

    工作流程：
      1. 从 state["plan"] 取 current_step_index 指向的步骤
      2. 运行迷你 ReAct Loop（LLM + Tool Use）执行该步骤
      3. 将工具调用记录追加到 tool_call_history
      4. 返回 state 更新（messages、current_step_index++、tool_call_history）

    输入：完整的 AgentState
    输出：dict（部分状态更新，LangGraph 自动合并到全局 state）

    输出字段：
      - messages:           追加的 user/assistant/tool_result 消息
      - current_step_index: 执行指针 +1（指向下一步）
      - tool_call_history:  追加的工具调用记录
      - error_message:      如果执行失败则设置错误文本
    """

    # ── 0. 参数校验 ──────────────────────────────────────────────────────
    plan = state.get("plan", [])
    if not plan:
        # 空计划 → 无法执行，直接返回错误
        return {
            "error_message": "Executor: plan is empty, nothing to execute.",
            "task_completed": True,              # 标记完成以避免空循环
        }

    step_idx = state.get("current_step_index", 0)
    if step_idx >= len(plan):
        # 索引越界 → 所有步骤已执行完毕（正常情况）
        return {
            "error_message": None,               # 不是错误，只是没有更多步骤
            "task_completed": True,              # 标记完成
        }

    # 取出当前步骤
    current_step = plan[step_idx]

    # ── 1. 准备 LLM 调用资源 ────────────────────────────────────────────
    client = get_or_create_client(Config.DEFAULT_MODEL)
    system_prompt = _load_executor_prompt()
    step_prompt = _format_step_prompt(current_step)

    # 构建消息历史：
    #   messages[0] = user: 单步执行指令
    #   messages[1..n] = (ReAct 循环追加): assistant + tool_result
    messages = [{"role": "user", "content": step_prompt}]

    # 本步骤内的工具调用记录（最后统一追加到 state.tool_call_history）
    step_tool_log: list[dict] = []

    # max_tokens 恢复状态（与 Phase 1-2 RecoveryState 逻辑一致）
    max_tokens = EXECUTOR_DEFAULT_MAX_TOKENS
    has_escalated = False                # 是否已升级过 max_tokens
    continuation_count = 0               # 续写次数计数

    # ── 2. 迷你 ReAct Loop（单步内最多 MAX_TOOL_ROUNDS_PER_STEP 轮） ───
    for _ in range(MAX_TOOL_ROUNDS_PER_STEP):

        # ── 2a. 调用 LLM ──────────────────────────────────────────────
        try:
            response = client.messages.create(
                model=Config.DEFAULT_MODEL,
                system=system_prompt,        # Executor 的 System Prompt
                messages=messages,           # 当前步骤的消息上下文
                tools=tool_registry.get_tool_schemas(),  # 全部可用工具
                max_tokens=max_tokens,
            )
        except Exception as e:
            # LLM 调用失败 → 记录错误并终止本步
            return {
                "error_message": f"Executor LLM call failed at step {step_idx}: {e}",
                "tool_call_history": step_tool_log,  # 保留已记录的部分
            }

        # ── 2b. 追加 assistant 回复到消息历史 ─────────────────────────
        messages.append({
            "role": "assistant",
            "content": response.content,     # Anthropic 格式 content 列表
        })

        # ── 2c. max_tokens 恢复（与 Phase 1-2 完全一致） ─────────────
        if response.stop_reason == "max_tokens":
            if not has_escalated:
                # 第一次截断：升级 max_tokens 后重试同一请求
                max_tokens = EXECUTOR_ESCALATED_MAX_TOKENS
                has_escalated = True
                messages.pop()               # 移除截断的 assistant 消息
                print(
                    f"  \033[33m[executor] max_tokens escalated to "
                    f"{max_tokens}\033[0m"
                )
                continue                     # 不消耗 round_num

            # 已升级仍截断：追加续写提示后继续
            if continuation_count < MAX_CONTINUATION_RETRIES:
                messages.append({
                    "role": "user",
                    "content": CONTINUATION_PROMPT,
                })
                continuation_count += 1
                print(
                    f"  \033[33m[executor] continuation "
                    f"{continuation_count}/{MAX_CONTINUATION_RETRIES}\033[0m"
                )
                continue
            # 续写次数耗尽 → 记录截断错误并退出本步
            step_tool_log.append({
                "tool_name": "_max_tokens_exhausted",
                "input": {},
                "result": "Output exceeded max token limits after retries.",
                "success": False,
            })
            break                          # 退出 ReAct 循环

        # ── 2d. 无工具调用 → LLM 认为本步骤完成 ─────────────────────
        if response.stop_reason != "tool_use":
            # 提取文本输出并记录到 tool_call_history
            text_output = _extract_text_from_response(response)
            step_tool_log.append({
                "tool_name": "_llm_response",
                "input": {"step_id": current_step["step_id"]},
                "result": text_output[:500],    # 截断避免过长的 LLM 输出
                "success": True,
            })
            break                          # 退出 ReAct 循环

        # ── 2e. 有工具调用 → 逐一执行 ───────────────────────────────
        tool_results = []
        for block in response.content:
            # 跳过非 tool_use 类型的 block（如 text）
            if hasattr(block, 'type') and block.type != "tool_use":
                continue
            if isinstance(block, dict) and block.get("type") != "tool_use":
                continue

            # Anthropic SDK: block.name / block.input / block.id
            tool_name = block.name if hasattr(block, 'name') else block.get("name")
            tool_input = block.input if hasattr(block, 'input') else block.get("input", {})
            tool_id = block.id if hasattr(block, 'id') else block.get("id")

            # ── 危险工具警告日志（Phase 2 规范） ──
            tool = tool_registry.get_tool(tool_name)
            if tool and tool.is_dangerous:
                print(
                    f"  \033[33m[!] dangerous {tool_name}("
                    f"{_format_args(tool_input)})"
                    f" — requires confirmation\033[0m"
                )

            # ── 通过 ToolRegistry 执行工具 ──
            result = await tool_registry.execute(tool_name, **tool_input)

            # 提取 LLM 可读的文本内容
            if isinstance(result, ToolResult):
                content_for_llm = (
                    result.message if result.success
                    else f"Error: {result.error or result.message}"
                )
            else:
                content_for_llm = str(result)

            # ── 构造 Anthropic 格式的 tool_result ──
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content_for_llm,
            })

            # ── 记录到本步骤的 tool_call_history ──
            step_tool_log.append({
                "tool_name": tool_name,          # 工具名称
                "input": tool_input,             # LLM 传入的参数
                "result": content_for_llm[:500], # 结果（截断 500 字符）
                "success": (
                    result.success if isinstance(result, ToolResult)
                    else True                    # 非 ToolResult → 假定成功
                ),
            })

        # ── 2f. 工具结果作为 user 消息追加，循环继续 ─────────────────
        messages.append({"role": "user", "content": tool_results})

    # ── 3. 构建返回的 state 更新 ─────────────────────────────────────────
    return {
        # 将本步产生的所有消息追加到全局 messages
        "messages": messages,
        # 执行指针 +1 → 下一步
        "current_step_index": step_idx + 1,
        # 追加本步的工具调用记录
        "tool_call_history": state.get("tool_call_history", []) + step_tool_log,
        # 清除之前的错误（本步正常完成）
        "error_message": None,
    }
