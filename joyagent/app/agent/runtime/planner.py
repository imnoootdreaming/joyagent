"""
Phase 3 Step 2: Planner Node — 任务拆解 + 生成结构化计划。

Planner 是 LangGraph Plan→Execute→Reflect 工作流的第一个 Node。
它的职责是接收用户的自然语言请求，调用 LLM 将其拆解为有序的、
可执行的步骤列表（list[TaskStep]），为后续 Executor Node 提供执行脚本。

Planner 自身不调用工具——它只做"规划"，不做"执行"。
"""

# ── Python 标准库 ──
import json                          # 解析 LLM 输出的 JSON 计划
from pathlib import Path             # 读取 Planner 专用 Prompt 模板文件

# json_repair 是可选的第三方库——能修复 LLM 输出的畸形 JSON。
# 未安装时自动降级，跳过修复步骤，不影响核心功能。
try:
    import json_repair               # 修复畸形 JSON（缺逗号、多引号等）
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False           # 降级：跳过修复，依赖其他解析策略

# ── 项目内导入 ──
from app.agent.schemas.state import AgentState, TaskStep
# AgentState: LangGraph 共享状态（输入整个 state，返回部分更新）
# TaskStep:   计划中单个步骤的类型

from app.service.llm_service import get_or_create_client
# 获取 Anthropic 客户端（按模型名分流 Claude / DeepSeek）

from app.core.config import Config
# 全局配置：DEFAULT_MODEL、MAX_ITERATIONS 等

from app.tools.registry import tool_registry
# 全局工具注册中心——获取可用工具列表，供 LLM 规划时参考


# ── 常量 ──
# Planner 专用 Prompt 模板路径
PLANNER_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "planner.txt"

# Planner 调用 LLM 的 max_tokens 上限
# 计划通常是简短的 JSON 数组，4096 足以覆盖 20+ 步骤
PLANNER_MAX_TOKENS = 4096


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _load_planner_prompt() -> str:
    """
    加载 Planner 专用的 System Prompt 模板。

    为什么从文件加载而非硬编码？
      - Prompt 会随调试迭代频繁修改，文件修改不触及代码逻辑
      - 模板可独立做版本管理（git diff 只看 prompt 变化）
      - 后续可扩展为多语言模板（planner_cn.txt / planner_en.txt）
    """
    if PLANNER_PROMPT_PATH.exists():
        return PLANNER_PROMPT_PATH.read_text(encoding="utf-8")
    # 兜底：文件缺失时使用内置简版 Prompt
    return (
        "You are a task planner. Break down user requests into ordered steps. "
        "Output only valid JSON with a 'plan' array."
    )


def _extract_user_request(messages: list[dict]) -> str:
    """
    从 AgentState.messages 中提取用户原始请求文本。

    messages 为 Anthropic 原生格式的 dict 列表：
      {"role": "user", "content": "你好"}
      {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
      {"role": "user", "content": [{"type": "tool_result", ...}]}

    策略：取第一条 user 消息的文本（通常就是用户输入）。
    后续可扩展为拼接所有 user 消息以支持多轮对话。
    """
    for msg in messages:
        if msg.get("role") != "user":
            continue                      # 只看 user 消息

        content = msg.get("content", "")
        # Anthropic 的 content 可能是纯字符串，也可能是 block 列表
        if isinstance(content, str):
            return content.strip()        # 简单情况：content 就是一个 str

        # 复杂情况：content 是 [{"type": "text", "text": "..."}, ...]
        # 只提取 type=="text" 的 block
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return " ".join(text_parts).strip()

    return ""                            # 没有 user 消息（异常情况）


def _build_plan_prompt(state: AgentState) -> str:
    """
    构建发送给 LLM 的完整 plan 请求文本。

    包含三部分信息：
      1. 可用工具列表 —— 让 LLM 知道能调什么
      2. 用户原始请求 —— 要拆解的目标
      3. 反思笔记（如有）—— 告诉 LLM 上次为什么失败，避免重蹈覆辙
    """
    # 1. 收集可用工具名称
    tool_names = tool_registry.list_tools()
    tools_info = "\n".join(f"- {name}" for name in tool_names)

    # 2. 提取用户请求
    user_request = _extract_user_request(state.get("messages", []))

    # 3. 检查是否需要 replan
    replan_context = ""
    if state.get("need_replan") and state.get("reflection_notes"):
        # Reflection 阶段发现计划有问题 → 告诉 Planner 上次出了什么错
        replan_context = (
            f"\n## ⚠️ REPLAN MODE\n"
            f"The previous plan failed. Reasons:\n"
            f"{state['reflection_notes']}\n\n"
            f"Please create a DIFFERENT approach — do NOT repeat the same steps.\n"
        )

    return (
        f"## Available Tools\n{tools_info}\n\n"
        f"## User Request\n{user_request}\n"
        f"{replan_context}"
    )


def _extract_text_from_response(response) -> str:
    """
    从 Anthropic Messages API 响应中提取纯文本。

    响应格式：response.content 是 block 列表，每个 block 有 type 字段：
      {"type": "text", "text": "这是 LLM 的输出..."}
      {"type": "tool_use", ...}

    Planner 不带工具调用，所以只关心 type=="text" 的 block。
    """
    text_parts = []
    for block in response.content:
        if hasattr(block, 'type') and block.type == "text":
            # Anthropic SDK 返回的 block 是对象，用属性访问
            text_parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            # 备选：如果 block 是 dict（纯 HTTP 响应）
            text_parts.append(block.get("text", ""))
    return "\n".join(text_parts).strip()


def _parse_plan_json(raw_text: str) -> list[TaskStep]:
    """
    将 LLM 输出的 JSON 字符串解析为 TaskStep 列表。

    解析策略（由宽松到严格）：
      1. 直接 json.loads —— LLM 输出正确 JSON
      2. json_repair.repair_json —— 修复畸形 JSON（缺逗号、多引号）
      3. 从代码块中提取 —— LLM 输出 ```json ... ```
      4. 兜底：构造一个"分析任务"步骤（优雅降级，不崩工作流）

    json_repair 是 LLM 工程中的常用库，能修复：
      - 末尾多余逗号：{"a": 1,}
      - 缺少闭合括号：{"a": [1,2
      - 单引号替代双引号：{'a': 'b'}
    """
    text = raw_text.strip()

    # ── 尝试 1：直接解析 ──
    try:
        data = json.loads(text)
        return _validate_and_convert(data)
    except json.JSONDecodeError:
        pass

    # ── 尝试 2：json_repair 修复后解析（仅库已安装时） ──
    if HAS_JSON_REPAIR:
        try:
            repaired = json_repair.repair_json(text)
            data = json.loads(repaired)
            return _validate_and_convert(data)
        except Exception:
            pass

    # ── 尝试 3：从 ```json ... ``` 代码块中提取 ──
    if "```json" in text:
        # 找到 ```json 和对应的 ```
        start = text.find("```json") + len("```json")
        end = text.find("```", start)
        if end > start:
            inner = text[start:end].strip()
            try:
                data = json.loads(inner)
                return _validate_and_convert(data)
            except json.JSONDecodeError:
                if HAS_JSON_REPAIR:
                    try:
                        repaired = json_repair.repair_json(inner)
                        data = json.loads(repaired)
                        return _validate_and_convert(data)
                    except Exception:
                        pass

    # ── 尝试 4：如果 LLM 输出了 ``` 但没有 json 标记 ──
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            inner = text[start:end].strip()
            try:
                data = json.loads(inner)
                return _validate_and_convert(data)
            except json.JSONDecodeError:
                pass

    # ── 兜底：无法解析 → 返回一条"分析任务"的默认步骤 ──
    return [
        TaskStep(
            step_id=1,
            description=(
                "The planner failed to generate a structured plan. "
                "Analyze the user request and proceed step by step."
            ),
            tool_name=None,               # 不给工具建议，让 Executor 自行判断
            status="pending",
        )
    ]


def _validate_and_convert(data: dict) -> list[TaskStep]:
    """
    校验解析后的 JSON 结构，转换为 TaskStep 列表。

    处理两种情况：
      1. {"plan": [{...}, {...}]} → 取 plan 键
      2. [{...}, {...}]            → 直接作为步骤列表
    """
    # LLM 可能输出 {"plan": [...]} 或直接输出 [...]
    if isinstance(data, dict) and "plan" in data:
        raw_steps = data["plan"]
    elif isinstance(data, list):
        raw_steps = data
    else:
        # 无法识别格式 → 兜底
        return [
            TaskStep(
                step_id=1,
                description="Unexpected planner output format. Proceed with manual analysis.",
                tool_name=None,
                status="pending",
            )
        ]

    steps: list[TaskStep] = []
    for i, item in enumerate(raw_steps):
        if not isinstance(item, dict):
            continue                       # 跳过非 dict 条目

        steps.append(TaskStep(
            step_id=item.get("step_id", i + 1),
            # LLM 可能用了不同字段名：description / desc / task
            description=item.get("description")
                     or item.get("desc")
                     or item.get("task")
                     or f"Step {i + 1}",
            # tool_name: LLM 可能输出 null / None / "" / 或者不包含此键
            tool_name=item.get("tool_name") or None,
            status="pending",              # 全部初始化为 pending
        ))

    return steps


# ═══════════════════════════════════════════════════════════════════════════════
# Planner Node 主函数（LangGraph Node 接口）
# ═══════════════════════════════════════════════════════════════════════════════

async def planner_node(state: AgentState) -> dict:
    """
    LangGraph Planner Node —— 将用户请求拆解为有序执行步骤。

    这是 Plan→Execute→Reflect 工作流的第一个 Node，由 __start__ 无条件进入。

    输入：完整的 AgentState（读取 messages、reflection_notes 等字段）
    输出：dict（部分状态更新，LangGraph 自动合并到全局 state）

    输出字段：
      - plan: list[TaskStep]  — 新生成的步骤列表
      - current_step_index: 0 — 重置执行指针到第一步
      - need_replan: False    — 清除 replan 标记
      - messages: [...]       — 追加 Planner 的分析消息（可选）

    调用链路：
      HTTP POST /api/chat
        → agent_workflow.ainvoke(initial_state)
          → planner_node(state)
          → executor_node(state)
          → reflector_node(state)
          → should_continue() → planner / executor / __end__
    """

    # ── 0. 参数检查 ──
    if not state.get("messages"):
        # 空消息 → 无法规划，返回错误标记
        return {
            "plan": [],
            "current_step_index": 0,
            "error_message": "No user messages found in state.",
            "task_completed": True,          # 标记完成（避免空循环）
        }

    # ── 1. 加载 System Prompt ──
    system_prompt = _load_planner_prompt()

    # ── 2. 构建用户消息（含工具列表 + 用户请求 + replan 上下文） ──
    user_prompt = _build_plan_prompt(state)

    # ── 3. 调用 LLM 生成计划（不带 tools——Planner 只规划不执行） ──
    try:
        client = get_or_create_client(Config.DEFAULT_MODEL)
        response = client.messages.create(
            model=Config.DEFAULT_MODEL,
            system=system_prompt,            # Planner 的 System Prompt（角色定义 + 输出格式）
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=PLANNER_MAX_TOKENS,
            # 注意：这里不传 tools= 参数，因为 Planner 不调用工具
            # Planner 只输出文本（JSON 计划），由后续 Executor 调用工具
        )
    except Exception as e:
        # LLM 调用失败 → 返回错误，让 Reflector 决定下一步
        return {
            "plan": [],
            "current_step_index": 0,
            "error_message": f"Planner LLM call failed: {e}",
        }

    # ── 4. 提取 LLM 输出的纯文本 ──
    plan_text = _extract_text_from_response(response)

    # ── 5. 解析 JSON → list[TaskStep] ──
    plan = _parse_plan_json(plan_text)

    # ── 6. 构建返回的 state 更新 ──
    return {
        "plan": plan,                        # 新生成的步骤列表
        "current_step_index": 0,             # 重置执行指针到第一步
        "need_replan": False,                # 清除 replan 标记
        "error_message": None,               # 清除之前的错误（新计划开始了）
        # 追加 Planner 的输出到消息历史（供 Executor / Reflector 参考）
        "messages": [
            {
                "role": "assistant",
                "content": (
                    f"[Planner] Task decomposed into {len(plan)} step(s):\n" +
                    "\n".join(
                        f"  {s['step_id']}. [{s.get('tool_name') or 'no tool'}] "
                        f"{s['description']}"
                        for s in plan
                    )
                ),
            }
        ],
    }
