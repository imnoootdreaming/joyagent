"""
Phase 3 Step 4: Reflector Node — 反思执行结果 + 决定下一步路由。

Reflector 是 LangGraph Plan→Execute→Reflect 工作流的第三个 Node。
在 Executor 完成一个或多个步骤后，Reflector 检查执行结果，
由 LLM 评估：任务是否完成？是否需要重新规划？还是继续执行？

Reflector 的产出是 StateGraph 的路由决策依据 ——
它的 task_completed / need_replan 字段被 should_continue() 条件边读取，
决定工作流的下一步走向：
  - task_completed == True  → __end__
  - need_replan == True     → Planner（重新生成计划）
  - 否则                     → Executor（继续执行下一步）
"""

# ── Python 标准库 ──
import json                          # 解析 LLM 输出的 JSON 评估结果
from pathlib import Path             # 读取 Reflector 专用 Prompt 模板文件

# json_repair 是可选的第三方库 —— 能修复 LLM 输出的畸形 JSON
try:
    import json_repair               # 修复畸形 JSON（缺逗号、多余引号等）
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False           # 降级：跳过修复，依赖其他解析策略

# ── 项目内导入 ──
from app.agent.schemas.state import AgentState
# AgentState: LangGraph 共享状态（读取 plan、tool_call_history、error_message）

from app.service.llm_service import get_or_create_client
# 获取 Anthropic 客户端（按模型名分流 Claude / DeepSeek）

from app.core.config import Config
# 全局配置：DEFAULT_MODEL


# ── 常量 ─────────────────────────────────────────────────────────────────

# Reflector 专用 Prompt 模板路径
REFLECTOR_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "reflector.txt"

# Reflector 调用 LLM 的 max_tokens（评估结果通常是简短的 JSON）
REFLECTOR_MAX_TOKENS = 4096


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _load_reflector_prompt() -> str:
    """
    加载 Reflector 专用的 System Prompt 模板。

    为什么从文件加载而非硬编码？
      - 评估维度可能会随迭代增加（如：代码质量、安全性等）
      - 模板独立做版本管理，与 planner.txt / executor.txt 统一
      - 易于做 A/B 测试（切换不同 Prompt 版本对比 Reflector 准确度）
    """
    if REFLECTOR_PROMPT_PATH.exists():
        return REFLECTOR_PROMPT_PATH.read_text(encoding="utf-8")
    # 兜底：文件缺失时使用内置简版 Prompt
    return (
        "You are a quality inspector. Evaluate execution results and output "
        "only valid JSON with keys: task_completed, analysis, need_replan, suggestion."
    )


def _extract_text_from_response(response) -> str:
    """
    从 Anthropic Messages API 响应中提取纯文本。

    Reflector 不带工具调用，所以只关心 type=="text" 的 block。
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


def _build_reflection_prompt(state: AgentState) -> str:
    """
    构建发送给 LLM 的完整评估上下文。

    包含四部分信息：
      1. 原始计划及每步状态 —— 让 LLM 知道"目标是什么、完成了多少"
      2. 工具调用历史 —— 让 LLM 知道"具体发生了什么"
      3. 反思轮次 —— 提醒 LLM 如果接近上限应趋向终止
      4. 错误信息（如有）—— 帮助 LLM 定位根因
    """
    plan = state.get("plan", [])
    tool_history = state.get("tool_call_history", [])
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", 3)
    error_msg = state.get("error_message")

    # ── 1. 格式化计划状态 ──
    plan_lines = ["## Original Plan"]
    if not plan:
        plan_lines.append("  (No plan was generated)")
    else:
        for step in plan:
            # 步骤状态标记：✓ 完成 / → 待执行 / ✗ 失败
            status_marker = {
                "completed":   "[DONE]",
                "pending":     "[TODO]",
                "in_progress": "[NOW]",
                "failed":      "[FAIL]",
            }.get(step.get("status", ""), step.get("status", "?"))
            plan_lines.append(
                f"  {status_marker} Step {step.get('step_id', '?')}: "
                f"{step.get('description', 'no description')}"
            )
    plan_text = "\n".join(plan_lines)

    # ── 2. 格式化工具调用历史 ──
    history_lines = ["## Tool Call History"]
    if not tool_history:
        history_lines.append("  (No tools were called)")
    else:
        # 统计成功/失败比例，帮助 LLM 快速获取整体印象
        success_count = sum(1 for t in tool_history if t.get("success"))
        fail_count = len(tool_history) - success_count
        history_lines.append(
            f"  Summary: {len(tool_history)} calls — "
            f"{success_count} succeeded, {fail_count} failed"
        )
        history_lines.append("")
        for i, tc in enumerate(tool_history, 1):
            # 每行展示：序号. 工具名 (成功/失败): 结果摘要
            status = "OK" if tc.get("success") else "FAIL"
            result_preview = str(tc.get("result", ""))[:120]  # 截断过长结果
            history_lines.append(
                f"  {i}. [{status}] {tc.get('tool_name', '?')}"
            )
            if result_preview:
                history_lines.append(f"     → {result_preview}")

    history_text = "\n".join(history_lines)

    # ── 3. 反思轮次信息 ──
    progress_info = (
        f"## Progress\n"
        f"  Reflection round: {reflection_count + 1} / {max_reflections}\n"
    )
    if reflection_count >= max_reflections - 1:
        # 最后一轮反思 → 提示 LLM 应趋向终止，防止无限循环
        progress_info += (
            f"  ⚠ This is the LAST reflection round. "
            f"If the task is not clearly incomplete, set task_completed=true.\n"
        )

    # ── 4. 错误信息（如有） ──
    error_section = ""
    if error_msg:
        error_section = (
            f"## Recent Error\n"
            f"  {error_msg}\n"
        )

    return f"{plan_text}\n\n{history_text}\n\n{progress_info}\n{error_section}"


def _parse_reflection_json(raw_text: str) -> dict:
    """
    将 LLM 输出的 JSON 字符串解析为评估结果 dict。

    解析策略（由宽松到严格）：
      1. 直接 json.loads —— LLM 输出正确 JSON
      2. json_repair.repair_json —— 修复畸形 JSON
      3. 从 ```json ... ``` 代码块中提取
      4. 兜底：返回安全的默认值（任务完成，避免无限循环）

    预期输出格式：
      {
        "task_completed": true/false,
        "analysis": "...",
        "need_replan": true/false,
        "suggestion": "..."
      }
    """
    text = raw_text.strip()

    # ── 尝试 1：直接解析 ──
    try:
        data = json.loads(text)
        return _normalize_reflection(data)
    except json.JSONDecodeError:
        pass

    # ── 尝试 2：json_repair 修复后解析 ──
    if HAS_JSON_REPAIR:
        try:
            repaired = json_repair.repair_json(text)
            data = json.loads(repaired)
            return _normalize_reflection(data)
        except Exception:
            pass

    # ── 尝试 3：从 ```json ... ``` 代码块中提取 ──
    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.find("```", start)
        if end > start:
            inner = text[start:end].strip()
            try:
                data = json.loads(inner)
                return _normalize_reflection(data)
            except json.JSONDecodeError:
                if HAS_JSON_REPAIR:
                    try:
                        repaired = json_repair.repair_json(inner)
                        data = json.loads(repaired)
                        return _normalize_reflection(data)
                    except Exception:
                        pass

    # ── 尝试 4：从 ``` ... ``` 中提取（无 json 标记） ──
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            inner = text[start:end].strip()
            try:
                data = json.loads(inner)
                return _normalize_reflection(data)
            except json.JSONDecodeError:
                pass

    # ── 尝试 5：从文本中匹配布尔关键词（LLM 输出纯文本而非 JSON） ──
    text_lower = text.lower()
    if "task_completed" in text_lower:
        # LLM 可能输出了类似 "task_completed: true" 的 YAML 风格
        # 尝试提取布尔值
        task_completed = (
            "true" in text_lower.split("task_completed")[-1][:30]
        )
        need_replan = (
            "true" in text_lower.split("need_replan")[-1][:30]
            if "need_replan" in text_lower else False
        )
        return {
            "task_completed": task_completed,
            "analysis": "Extracted from non-JSON output.",
            "need_replan": need_replan,
            "suggestion": "",
        }

    # ── 兜底：无法解析 → 安全默认值（保守判断"未完成"由后续逻辑处理） ──
    return {
        "task_completed": False,           # 无法判断 → 假定未完成
        "analysis": "Failed to parse reflection output. Assuming not completed.",
        "need_replan": False,              # 不轻易触发 replan
        "suggestion": "",
    }


def _normalize_reflection(data: dict) -> dict:
    """
    规范化 LLM 输出的评估 JSON。

    处理常见的 LLM 输出异常：
      - task_completed 可能是字符串 "true"/"yes"/"done"
      - need_replan 可能是字符串
      - 缺少某些键则填充默认值
    """
    # ── 规范化 task_completed ──
    tc = data.get("task_completed", False)
    if isinstance(tc, str):
        # LLM 可能输出 "true" / "yes" / "done" / "completed"
        tc = tc.lower().strip() in ("true", "yes", "done", "completed", "1")
    data["task_completed"] = bool(tc)

    # ── 规范化 need_replan ──
    nr = data.get("need_replan", False)
    if isinstance(nr, str):
        nr = nr.lower().strip() in ("true", "yes", "1")
    data["need_replan"] = bool(nr)

    # ── 确保所有键存在 ──
    data.setdefault("analysis", "")        # 分析文本（缺失 → 空字符串）
    data.setdefault("suggestion", "")      # 改进建议（缺失 → 空字符串）

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Reflector Node 主函数（LangGraph Node 接口）
# ═══════════════════════════════════════════════════════════════════════════════

async def reflector_node(state: AgentState) -> dict:
    """
    LangGraph Reflector Node —— 评估执行结果并决定下一步。

    这是 Plan→Execute→Reflect 工作流的第三个 Node，由 Executor 无条件进入。
    其返回值直接影响 should_continue() 条件边的路由决策。

    工作流程：
      1. 检查是否已达到最大反思轮次（硬上限保护）
      2. 构建评估 Prompt（计划状态 + 工具历史 + 错误信息）
      3. 调用 LLM 进行评估（不带工具 —— Reflector 不做执行）
      4. 解析 JSON 评估结果
      5. 返回 state 更新（task_completed / need_replan / reflection_notes）

    输入：完整的 AgentState（读取 plan、tool_call_history、error_message）
    输出：dict（部分状态更新，LangGraph 自动合并到全局 state）

    输出字段：
      - task_completed:    bool  — True → __end__（工作流终止）
      - need_replan:       bool  — True → Planner（重新生成计划）
      - reflection_notes:  str   — LLM 的分析文本
      - reflection_count:  int   — +1（递增反思轮次）
      - suggestion:        str   — 供 Planner replan 使用的改进建议

    路由关系（在 graph/edges.py 的 should_continue() 中定义）：
      task_completed == True  → "__end__"
      need_replan == True     → "planner"
      否则                     → "executor"（继续执行下一步）
    """

    # ── 0. 反思轮次硬上限检查 ──────────────────────────────────────────
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", 3)

    if reflection_count >= max_reflections:
        # 达到最大反思轮次 → 强制终止，避免无限循环
        print(
            f"  \033[33m[reflector] max reflections reached "
            f"({reflection_count}/{max_reflections}) — forcing end\033[0m"
        )
        return {
            "task_completed": True,              # 强制结束
            "reflection_notes": (
                f"Reached maximum reflection rounds ({max_reflections}). "
                f"Forcing termination."
            ),
            "reflection_count": reflection_count,  # 不再递增
            "need_replan": False,                # 不再 replan
        }

    # ── 1. 加载 System Prompt ───────────────────────────────────────────
    system_prompt = _load_reflector_prompt()

    # ── 2. 构建用户消息（计划状态 + 工具历史 + 反思轮次 + 错误） ───────
    user_prompt = _build_reflection_prompt(state)

    # ── 3. 调用 LLM 进行评估（不带 tools —— Reflector 不做执行） ────────
    try:
        client = get_or_create_client(Config.DEFAULT_MODEL)
        response = client.messages.create(
            model=Config.DEFAULT_MODEL,
            system=system_prompt,            # Reflector 的 System Prompt
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=REFLECTOR_MAX_TOKENS,
            # 注意：不传 tools= 参数，Reflector 只输出评估 JSON
        )
    except Exception as e:
        # LLM 调用失败 → 保守处理：不标记完成、不 replan，继续尝试执行
        return {
            "task_completed": False,
            "reflection_notes": f"Reflector LLM call failed: {e}",
            "reflection_count": reflection_count + 1,
            "need_replan": False,            # 不触发 replan（可能是临时网络问题）
        }

    # ── 4. 提取 LLM 输出的纯文本 ───────────────────────────────────────
    reflection_text = _extract_text_from_response(response)

    # ── 5. 解析 JSON → 结构化评估结果 ────────────────────────────────────
    result = _parse_reflection_json(reflection_text)

    # ── 6. 如果当前计划所有步骤都是 completed 但 LLM 说未完成 ─────────
    # 这种情况通常是 LLM 误判或者是最后一步执行结果不够理想
    # 我们需要检查所有步骤状态来辅助判断
    plan = state.get("plan", [])
    all_steps_done = (
        len(plan) > 0 and
        all(s.get("status") in ("completed", None) for s in plan)
    )

    # 强制一致性：如果所有步骤都标记完成且没有出错，则 task_completed=True
    if all_steps_done and not state.get("error_message"):
        # 检查是否有工具调用失败
        tool_history = state.get("tool_call_history", [])
        all_tools_ok = all(tc.get("success") for tc in tool_history)
        if all_tools_ok:
            result["task_completed"] = True
            if not result["analysis"]:
                result["analysis"] = "All steps executed successfully with no errors."

    # ── 7. 构建返回的 state 更新 ────────────────────────────────────────
    # 保存 suggestion 到 reflection_notes（供 Planner replan 时使用）
    suggestion = result.get("suggestion", "")
    analysis = result.get("analysis", "")
    reflection_notes = analysis
    if suggestion:
        reflection_notes += f"\n[Suggestion for replan]: {suggestion}"

    return {
        "task_completed": result["task_completed"],
        "need_replan": result["need_replan"],
        "reflection_notes": reflection_notes,
        "reflection_count": reflection_count + 1,     # 递增反思轮次
        "error_message": None,                         # 清除错误（已评估）
    }
