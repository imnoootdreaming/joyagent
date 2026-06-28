"""
Phase 5 Step 5.5: Bad Case Analyzer — 失败案例分析 + Prompt 优化闭环。

BadCaseAnalyzer 是 Agent 从"能跑"到"跑得好"的关键基础设施。
它系统性地收集 Fix Loop 耗尽所有尝试仍失败的案例，
按多维度聚合分析，定位高频失败模式，输出可操作的改进建议。

分析流水线（4 步）：
  1. 收集 — Fix Loop 失败 → 自动创建 BadCase 记录
  2. 聚合 — 按错误类型 + 任务类别 + 交叉维度聚合
  3. 诊断 — 检测高频失败模式（FailurePattern），定位可疑的 Prompt 段
  4. 输出 — 生成分析报告 + 改进建议 + Prompt 回归提醒

在整体 Agent 优化循环中的位置：
  Fix Loop 失败
    → BadCaseAnalyzer.collect()    ← 收集病例
    → detect_patterns()            ← 诊断高频失败模式
    → generate_report()            ← 输出改进建议
    → 修改 Prompt / Skill
    → Benchmark 回放（Phase 9B）   ← 验证改进效果
    → 循环（直到完成率达到目标）

面试要点：
  面试官问"你怎么优化 Agent 的表现"——
  这不是玄学调 prompt，而是数据驱动的系统性方法：
  收集失败案例 → 聚合分析 → 定位根因 → 针对性改进 → 回归验证。

使用方式：
  from app.sandbox.bad_case_analyzer import bad_case_analyzer

  # Fix Loop 失败后自动收集
  bad_case_analyzer.collect(task, category, fix_state, code, stderr)

  # 定期查看分析报告
  print(bad_case_analyzer.generate_report())

  # 查询统计数据（通过 /api/tools/stats 暴露给前端）
  stats = bad_case_analyzer.get_summary()
"""
from __future__ import annotations

# ── Python 标准库 ──
import datetime                       # 时间戳
from collections import defaultdict   # 自动创建不存在的 key
from dataclasses import dataclass, field  # 数据类

# ── 项目内导入 ──
from app.sandbox.error_parser import TestFailure, ErrorParser  # 错误解析
from app.sandbox.security import FixLoopState  # 修复循环状态


# ═══════════════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BadCase:
    """
    单个 Bad Case 记录 —— 一次 Fix Loop 失败的完整快照。

    包含：原始任务是什么、什么类别的任务、犯了哪些错、
    尝试了多少次仍然失败、最后一次生成的代码是什么。

    这些数据是后续聚合分析和模式检测的原材料。
    """
    task_description: str              # 用户原始任务描述（如 "创建 calculator.py"）
    task_category: str                 # 任务分类: crud | algorithm | file_op | api_dev | other
    failures: list                     # 解析后的测试失败列表 (list[TestFailure])
    error_categories: list[str]        # 每个失败对应的分类 (syntax/import/assertion/...)
    fix_attempts: int                  # Fix Loop 实际尝试次数
    final_state: str                   # 终止原因: exhausted | stalled | gave_up | timeout
    source_code: str                   # 最后一次修复后的代码（最长存 5000 字符）
    timestamp: str = field(            # 收集时间（ISO 8601）
        default_factory=lambda: datetime.datetime.now().isoformat()
    )

    def summary(self) -> str:
        """单行摘要。"""
        cats = ", ".join(sorted(set(self.error_categories)))
        return (
            f"[{self.task_category}] {self.task_description[:50]}... "
            f"— {self.fix_attempts} attempts, "
            f"errors: {cats}, "
            f"state: {self.final_state}"
        )


@dataclass
class FailurePattern:
    """
    高频失败模式 —— 同一个 (任务类别 × 错误类型) 组合多次出现。

    detect_patterns() 的输出单位。frequency 越高 → 越值得关注。
    ratio = frequency / 该类别总 case 数 → 反映影响面。
    """
    pattern_name: str                  # 可读名称，如 "CRUD 任务中的 ImportError"
    task_category: str                 # 任务类别
    error_type: str                    # 错误类型 (syntax/import/assertion/runtime/timeout)
    frequency: int                     # 发生次数
    ratio: float                       # 占同类任务的比例 (0.0 ~ 1.0)
    example_cases: list                # 前 3 个样例（BadCase 列表）


@dataclass
class PromptImprovement:
    """
    一个可操作的 Prompt 改进建议。

    _generate_suggestions() 对每个高频失败模式生成一条建议。
    """
    suggestion: str                    # 改进建议文本
    action: str                        # 建议操作: add_prompt_rule | restrict_tools | add_syntax_check | manual_review
    priority: str                      # 优先级: high | medium | low
    affected_pattern: str              # 关联的失败模式名称


# ═══════════════════════════════════════════════════════════════════════════════
# BadCaseAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

class BadCaseAnalyzer:
    """
    Bad Case 分析引擎 — 数据驱动的 Agent 优化工具。

    四个核心职责：
      1. 收集 (collect)           → Fix Loop 失败 → 自动记录
      2. 聚合 (aggregate_* )      → 按错误类型/任务类别/交叉维度分组
      3. 诊断 (detect_patterns)   → 检测高频失败模式
      4. 输出 (generate_report)   → 分析报告 + 改进建议
      5. 反馈 (to_prompt_context) → 以 Few-shot 负例形式注入后续任务的 Prompt

    Prompt 改进→回归工作流：
      收集 ≥ 20 个 case → detect_patterns → generate_report
      → 读出 report + suggestions → 人/LLM 修改 Prompt
      → 重新跑 Benchmark + Bad Case 回放（Phase 9B）
      → 完成率提升 → 记录新 Prompt 版本
    """

    # ── 任务类别关键词（用于自动分类） ──────────────────────

    CATEGORY_KEYWORDS: dict[str, list[str]] = {
        "crud": ["crud", "api", "endpoint", "create", "read", "update", "delete",
                 "user", "model", "database", "postgres", "sql"],
        "algorithm": ["algorithm", "sort", "search", "fibonacci", "recursion",
                      "binary", "tree", "graph", "calculate", "compute"],
        "file_op": ["file", "read", "write", "csv", "json", "yaml", "parse",
                    "process", "transform", "convert"],
        "api_dev": ["fastapi", "flask", "django", "router", "middleware",
                    "endpoint", "websocket", "deploy", "server"],
        "testing": ["test", "pytest", "unittest", "mock", "fixture", "coverage"],
        "refactor": ["refactor", "rename", "extract", "move", "simplify", "clean"],
    }

    def __init__(self):
        # 全量 case 列表（按收集顺序）
        self.cases: list[BadCase] = []
        # ErrorParser 实例（用于 _determine_final_state 等）
        self._error_parser = ErrorParser()
        # Prompt 改进建议历史
        self.improvements: list[PromptImprovement] = []

    # ═══════════════════════════════════════════════════════════════
    # 1. 收集
    # ═══════════════════════════════════════════════════════════════

    def collect(
        self,
        task: str,
        category: str,
        fix_state: FixLoopState,
        source_code: str,
        stderr: str,
    ) -> BadCase:
        """
        从 Fix Loop 失败结果中收集一个 Bad Case。

        调用时机：FixLoopState.should_escalate() == True 时自动调用
        （即修复循环耗尽所有尝试后仍未通过）。

        Args:
            task:        用户原始任务描述
            category:    任务类别。空字符串 → 自动检测
            fix_state:   FixLoop 返回的最终状态
            source_code: 最后一次修复后的代码
            stderr:      pytest/测试的完整错误输出

        Returns:
            BadCase — 已追加到 self.cases 中

        Example:
            state = await fix_loop.fix_and_retest(...)
            if state.should_escalate():
                bad_case_analyzer.collect(
                    task="Create a calculator",
                    category="algorithm",
                    fix_state=state,
                    source_code=broken_code,
                    stderr=last_test_stderr,
                )
        """
        # 解析 stderr → 结构化 TestFailure 列表
        failures = self._error_parser.parse(stderr)

        # 每个 failure 归类
        error_cats = [
            self._error_parser.categorize_error(f) for f in failures
        ]

        # 自动检测任务类别（如果未指定）
        if not category:
            category = self._detect_category(task)

        # 截断过长的源码（5KB 足够分析）
        truncated_code = (source_code or "")[:5000]

        case = BadCase(
            task_description=task,
            task_category=category,
            failures=failures,
            error_categories=error_cats,
            fix_attempts=fix_state.current_attempt,
            final_state=self._determine_final_state(fix_state),
            source_code=truncated_code,
        )

        self.cases.append(case)
        return case

    def collect_from_fix_loop(
        self,
        fix_state: FixLoopState,
        task: str,
        category: str = "",
        source_files: dict[str, str] = None,
    ) -> BadCase | None:
        """
        从完整的 FixLoopState 中自动提取 Bad Case。

        比 collect() 更方便——只需传入 state 和 task，
        自动从 state.fix_history 中提取最后失败的 stderr 和代码。

        Args:
            fix_state:    FixLoop 返回的最终状态
            task:         任务描述
            category:     任务类别（空 → 自动检测）
            source_files: 源代码文件映射（用于提取最后一次的代码）

        Returns:
            BadCase 或 None（如果 fix_state 表示成功则返回 None）
        """
        if not fix_state.should_escalate():
            return None                  # 成功了，不收集

        # 从 fix_history 中提取最后一次失败的 stderr 和代码
        history = fix_state.fix_history or []
        last_failure = None
        for entry in reversed(history):
            if entry.get("result") not in ("", "all_passed"):
                last_failure = entry
                break

        # 提取错误文本（从 failures 字段重建 stderr）
        failures_list = last_failure.get("failures", []) if last_failure else []
        stderr_text = "\n".join(
            f"{f.get('error_type', '?')}: {f.get('error_message', '')}"
            for f in failures_list
        ) if failures_list else "No parseable errors."

        # 提取最后一次的源代码
        source_code = ""
        if source_files:
            parts = []
            for fp, code in source_files.items():
                parts.append(f"# {fp}\n{code}")
            source_code = "\n\n".join(parts)

        return self.collect(
            task=task,
            category=category,
            fix_state=fix_state,
            source_code=source_code,
            stderr=stderr_text,
        )

    # ═══════════════════════════════════════════════════════════════
    # 2. 聚合分析
    # ═══════════════════════════════════════════════════════════════

    def aggregate_by_error_type(self) -> dict[str, list[BadCase]]:
        """
        按错误类型聚合所有 Bad Case。

        Returns:
            {"syntax": [case1, ...], "import": [case3, ...], ...}
        """
        by_type: dict[str, list[BadCase]] = defaultdict(list)
        for case in self.cases:
            for cat in case.error_categories:
                by_type[cat].append(case)
        return dict(by_type)

    def aggregate_by_task_category(self) -> dict[str, list[BadCase]]:
        """
        按任务类别聚合。

        Returns:
            {"crud": [case1, ...], "algorithm": [case5, ...], ...}
        """
        by_cat: dict[str, list[BadCase]] = defaultdict(list)
        for case in self.cases:
            by_cat[case.task_category].append(case)
        return dict(by_cat)

    def cross_aggregate(self) -> dict[str, dict[str, int]]:
        """
        交叉聚合：任务类别 × 错误类型 = 频次矩阵。

        这是最有价值的聚合维度——它告诉你"哪种任务最容易出哪种错误"。

        Returns:
            {
              "crud":      {"import": 5, "assertion": 3},
              "algorithm": {"syntax": 2, "runtime": 7},
              ...
            }
        """
        matrix: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for case in self.cases:
            for cat in case.error_categories:
                matrix[case.task_category][cat] += 1
        return {k: dict(v) for k, v in matrix.items()}

    # ═══════════════════════════════════════════════════════════════
    # 3. 诊断
    # ═══════════════════════════════════════════════════════════════

    def detect_patterns(self, min_frequency: int = 2) -> list[FailurePattern]:
        """
        检测高频失败模式。

        在 (任务类别 × 错误类型) 的交叉矩阵中，找出出现 ≥ min_frequency 次
        的组合。每个组合代表一个"失败模式"。

        Args:
            min_frequency: 最少出现次数才被列为"模式"（默认 2）

        Returns:
            list[FailurePattern] — 按 frequency 降序排列
            空列表表示数据不足，尚无明确模式
        """
        cross = self.cross_aggregate()
        # 每个类别的总 case 数（用于计算 ratio）
        total_by_cat = {
            cat: len(cases)
            for cat, cases in self.aggregate_by_task_category().items()
        }

        patterns: list[FailurePattern] = []
        for task_cat, error_counts in cross.items():
            total = total_by_cat.get(task_cat, 1)  # 避免除零
            for error_type, count in error_counts.items():
                if count >= min_frequency:
                    # 找最多 3 个样例
                    examples = [
                        c for c in self.cases
                        if c.task_category == task_cat
                        and error_type in c.error_categories
                    ][:3]

                    # 中文友好的模式名称
                    cn_task = self._category_name(task_cat)
                    cn_error = self._error_name(error_type)
                    name = f"{cn_task}任务中的{cn_error}"

                    patterns.append(FailurePattern(
                        pattern_name=name,
                        task_category=task_cat,
                        error_type=error_type,
                        frequency=count,
                        ratio=count / total if total > 0 else 1.0,
                        example_cases=examples,
                    ))

        # 按频率降序排列
        return sorted(patterns, key=lambda p: p.frequency, reverse=True)

    # ═══════════════════════════════════════════════════════════════
    # 4. 报告 + 建议
    # ═══════════════════════════════════════════════════════════════

    def generate_report(self) -> str:
        """
        生成完整分析报告。

        报告包含四部分：
          1. 概览 — 总 case 数、错误类型分布
          2. Top 失败模式 — 高频组合
          3. Prompt 改进建议 — 基于模式的可操作建议
          4. 下一步行动 — 具体的优化步骤

        Returns:
            Markdown 格式的完整报告文本
        """
        total = len(self.cases)
        if total == 0:
            return "## Bad Case Analysis\n\n*No bad cases collected yet.*"

        patterns = self.detect_patterns()

        # ── 标题 ──
        report = f"""## Bad Case Analysis Report

**Total cases:** {total}
**Collection period:** {self._get_time_range()}

---

### 1. Error Distribution by Type

| Error Type  | Count | Ratio |
|-------------|-------|-------|
"""
        for err_type, cases in sorted(
            self.aggregate_by_error_type().items(),
            key=lambda kv: -len(kv[1]),
        ):
            pct = len(cases) / total * 100
            report += f"| {err_type:12s} | {len(cases):>5d} | {pct:5.1f}% |\n"

        # ── 任务类别分布 ──
        report += "\n### 2. Task Category Distribution\n\n"
        report += "| Category   | Count | Ratio |\n"
        report += "|------------|-------|-------|\n"
        for cat, cases in sorted(
            self.aggregate_by_task_category().items(),
            key=lambda kv: -len(kv[1]),
        ):
            pct = len(cases) / total * 100
            cn = self._category_name(cat)
            report += f"| {cn:11s} | {len(cases):>5d} | {pct:5.1f}% |\n"

        # ── Top 失败模式 ──
        report += "\n### 3. Top Failure Patterns\n\n"
        if patterns:
            report += "| # | Pattern | Freq | Ratio |\n"
            report += "|---|---------|------|-------|\n"
            for i, p in enumerate(patterns[:10], 1):
                report += (
                    f"| {i} | {p.pattern_name[:45]:45s} "
                    f"| {p.frequency:>4d} "
                    f"| {p.ratio*100:>4.0f}% |\n"
                )
        else:
            report += (
                "*Not enough data to detect patterns yet.* "
                f"(need at least {total + 1} cases with the same error type)\n"
            )

        # ── 改进建议 ──
        suggestions = self._generate_suggestions(patterns)
        report += "\n### 4. Improvement Suggestions\n\n"
        if suggestions:
            for i, s in enumerate(suggestions, 1):
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "⚪"}
                emoji = priority_emoji.get(s.priority, "⚪")
                report += (
                    f"{i}. {emoji} **[{s.priority.upper()}]** "
                    f"{s.suggestion}\n"
                    f"   - Action: `{s.action}`\n"
                    f"   - Related: *{s.affected_pattern}*\n\n"
                )
        else:
            report += "*No specific suggestions yet.*\n"

        # ── 下一步行动 ──
        report += self._next_steps(total, len(patterns))

        return report

    def get_summary(self) -> dict:
        """
        返回 JSON 格式的统计摘要（供 API 暴露）。

        Returns:
            dict with keys: total_cases, top_errors, top_categories, patterns
        """
        total = len(self.cases)
        patterns = self.detect_patterns()
        return {
            "total_cases": total,
            "top_errors": {
                et: len(cases)
                for et, cases in sorted(
                    self.aggregate_by_error_type().items(),
                    key=lambda kv: -len(kv[1]),
                )[:5]
            },
            "top_categories": {
                cat: len(cases)
                for cat, cases in sorted(
                    self.aggregate_by_task_category().items(),
                    key=lambda kv: -len(kv[1]),
                )[:5]
            },
            "patterns": [
                {
                    "name": p.pattern_name,
                    "frequency": p.frequency,
                    "ratio": round(p.ratio, 2),
                }
                for p in patterns[:5]
            ],
            "suggestions": [
                {
                    "suggestion": s.suggestion,
                    "priority": s.priority,
                    "action": s.action,
                }
                for s in self.improvements[-5:]
            ],
        }

    def to_prompt_context(self, max_examples: int = 3) -> str:
        """
        将 Bad Case 分析结果转换为 Prompt 上下文（Few-shot 负例）。

        在 Agent 执行类似任务前，将相关的失败案例作为"请避免以下错误"
        的 Few-shot 示例注入 System Prompt 或用户消息。

        Args:
            max_examples: 最多输出几个案例

        Returns:
            可拼接到 Prompt 中的文本
        """
        if not self.cases:
            return ""

        patterns = self.detect_patterns(min_frequency=1)
        if not patterns:
            return ""

        parts = [
            "## Previous Failures to Avoid\n",
            "The following patterns have caused failures in similar tasks. "
            "Please avoid these mistakes:\n",
        ]

        for p in patterns[:max_examples]:
            parts.append(f"### {p.pattern_name}")
            parts.append(f"- Occurred {p.frequency} time(s) "
                         f"({p.ratio*100:.0f}% of {p.task_category} tasks)")
            if p.example_cases:
                example = p.example_cases[0]
                parts.append(f"- Example task: *{example.task_description[:100]}*")
                parts.append(f"- Final state: {example.final_state} "
                             f"after {example.fix_attempts} attempts")
                if example.failures:
                    for f in example.failures[:2]:
                        desc = self._error_parser.get_error_description(f)
                        parts.append(f"  - {f.error_type}: {f.error_message[:80]}")
            parts.append("")

        return "\n".join(parts)

    # ═══════════════════════════════════════════════════════════════
    # 管理方法
    # ═══════════════════════════════════════════════════════════════

    def clear(self):
        """清空所有已收集的 Bad Case（重置分析）。"""
        self.cases.clear()
        self.improvements.clear()

    def export_cases(self) -> list[dict]:
        """导出所有 case 为 JSON-serializable 列表。"""
        return [
            {
                "task_description": c.task_description,
                "task_category": c.task_category,
                "error_categories": c.error_categories,
                "fix_attempts": c.fix_attempts,
                "final_state": c.final_state,
                "timestamp": c.timestamp,
                "failure_count": len(c.failures),
            }
            for c in self.cases
        ]

    def count_by_category(self) -> dict[str, int]:
        """每个类别的 case 数量。"""
        counts: dict[str, int] = defaultdict(int)
        for case in self.cases:
            counts[case.task_category] += 1
        return dict(counts)

    # ═══════════════════════════════════════════════════════════════
    # 私有辅助
    # ═══════════════════════════════════════════════════════════════

    def _detect_category(self, task: str) -> str:
        """
        从任务描述中自动检测任务类别。

        基于关键词匹配（CATEGORY_KEYWORDS）。
        所有类别都不匹配 → 返回 "other"。
        """
        task_lower = task.lower()
        scores: dict[str, int] = {}
        for cat, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in task_lower)
            if score > 0:
                scores[cat] = score
        if scores:
            return max(scores, key=scores.get)  # 最高分的类别
        return "other"

    def _determine_final_state(self, state: FixLoopState) -> str:
        """
        确定 Fix Loop 的终止原因。

        优先级：
          1. failed + errors > 0 → "gave_up"（有失败但无法修复）
          2. current_attempt >= max_attempts → "exhausted"（资源耗尽）
          3. fix_history 最后一条为 stalled → "stalled"
          4. 兜底 → "unknown"
        """
        # 检查 test_results 中最后一次的结果
        if state.test_results:
            last = state.test_results[-1]
            if last.failed > 0 or last.errors > 0:
                return "gave_up"         # 有测试失败，但修复循环退出了

        # 检查 fix_history 中最后一次
        if state.fix_history:
            last_entry = state.fix_history[-1]
            if last_entry.get("result") == "stalled":
                return "stalled"

        if state.current_attempt >= state.max_attempts:
            return "exhausted"

        return "unknown"

    def _generate_suggestions(
        self,
        patterns: list[FailurePattern],
    ) -> list[PromptImprovement]:
        """
        根据检测到的失败模式生成可操作的 Prompt 改进建议。

        每条建议包含：
          - suggestion: 具体改进内容
          - action:     建议操作类型（方便程序化执行）
          - priority:   优先级（high/medium/low）
          - affected_pattern: 关联的失败模式名称

        优先级判断：
          - frequency >= 5 或 ratio >= 50% → high
          - frequency >= 3 或 ratio >= 30% → medium
          - 其他 → low
        """
        suggestions: list[PromptImprovement] = []

        if not patterns:
            return suggestions

        # 按模式生成建议
        for p in patterns:
            # 判定优先级
            if p.frequency >= 5 or p.ratio >= 0.5:
                priority = "high"
            elif p.frequency >= 3 or p.ratio >= 0.3:
                priority = "medium"
            else:
                priority = "low"

            if p.error_type == "import":
                suggestions.append(PromptImprovement(
                    suggestion="在 Prompt 中增加明确指示：生成代码前先检查并列出所有 "
                               "需要的 import 语句，确保依赖已在 requirements.txt 中声明。",
                    action="add_prompt_rule",
                    priority=priority,
                    affected_pattern=p.pattern_name,
                ))

            elif p.error_type == "assertion":
                suggestions.append(PromptImprovement(
                    suggestion="在 Prompt 中增加输出格式约束：明确返回值类型、"
                               "边界条件（如空列表、负数、None）、精度要求（浮点数）。",
                    action="add_prompt_rule",
                    priority=priority,
                    affected_pattern=p.pattern_name,
                ))

            elif p.error_type == "syntax":
                suggestions.append(PromptImprovement(
                    suggestion="考虑在代码生成后增加 compile() 语法检查步骤，"
                               "或在 Executor 生成代码后立即做一次快速语法验证。",
                    action="add_syntax_check",
                    priority=priority,
                    affected_pattern=p.pattern_name,
                ))

            elif p.error_type == "runtime":
                suggestions.append(PromptImprovement(
                    suggestion="在 Prompt 中增加防御性编程要求：生成的代码必须"
                               "包含必要的 try/except、None 检查、空列表保护。",
                    action="add_prompt_rule",
                    priority=priority,
                    affected_pattern=p.pattern_name,
                ))

            elif p.error_type == "timeout":
                suggestions.append(PromptImprovement(
                    suggestion="考虑对任务增加复杂度评估：如果预期代码含大量循环，"
                               "提示 Agent 使用优化的算法或拆分任务。",
                    action="add_prompt_rule",
                    priority=priority,
                    affected_pattern=p.pattern_name,
                ))

            elif p.error_type == "unknown":
                suggestions.append(PromptImprovement(
                    suggestion="出现多次无法分类的错误——建议人工 review 这些 case "
                               "的原始 stderr，确定是否需要新增错误分类规则。",
                    action="manual_review",
                    priority=priority,
                    affected_pattern=p.pattern_name,
                ))

        # 全局建议（与具体模式无关）
        if len(patterns) >= 3:
            suggestions.append(PromptImprovement(
                suggestion="存在 ≥ 3 种不同的失败模式，建议做一次系统性 Prompt Review，"
                           "对比成功和失败 case 的差异，迭代一版新 Prompt。",
                action="prompt_review",
                priority="high",
                affected_pattern="(multiple patterns)",
            ))

        if len(self.cases) >= 20 and any(
            p.ratio >= 0.5 for p in patterns
        ):
            suggestions.append(PromptImprovement(
                suggestion="某个失败模式占比超过 50%——这可能是某个 Skill/Prompt 的"
                           "系统性缺陷，建议回溯相关的 System Prompt 段落并修正。",
                action="prompt_review",
                priority="high",
                affected_pattern="(dominant pattern)",
            ))

        # 存储建议
        self.improvements = suggestions
        return suggestions

    def _next_steps(self, total_cases: int, pattern_count: int) -> str:
        """生成"下一步行动"建议。"""
        steps = ["### 5. Next Steps\n"]

        if total_cases < 5:
            steps.append(
                f"📊 *Only {total_cases} cases collected — continue collecting "
                f"until you have at least 10-20 cases for meaningful analysis.*\n"
            )
        elif pattern_count == 0:
            steps.append(
                "🔍 *No clear patterns detected — failures appear random. "
                "Consider: (1) checking if the test suite itself is flaky, "
                "(2) increasing LLM temperature to 0 for more deterministic output.*\n"
            )
        else:
            steps.append(
                "🔧 **Recommended actions:**\n"
                "1. Review the top failure patterns above\n"
                "2. Select 1-2 high-priority suggestions to implement\n"
                "3. Modify the relevant System Prompt or Skill configuration\n"
                "4. Re-run the Benchmark suite to measure the impact\n"
                "5. Compare pass rates: before vs after the Prompt change\n"
                "6. If pass rate improves, commit the new Prompt version\n"
            )

        if total_cases >= 10:
            steps.append(
                "💡 *With 10+ cases, you can start using Few-shot prompting: "
                "inject the most relevant Bad Cases as negative examples "
                "into the Planner/Executor System Prompt via `to_prompt_context()`.*\n"
            )

        return "\n".join(steps) + "\n"

    def _get_time_range(self) -> str:
        """获取 case 收集的时间范围。"""
        if not self.cases:
            return "N/A"
        first = self.cases[0].timestamp[:10]
        last = self.cases[-1].timestamp[:10]
        if first == last:
            return first
        return f"{first} ~ {last}"

    @staticmethod
    def _category_name(cat: str) -> str:
        """任务类别英文 → 中文可读名称。"""
        names = {
            "crud": "CRUD/建模",
            "algorithm": "算法实现",
            "file_op": "文件操作",
            "api_dev": "API开发",
            "testing": "测试编写",
            "refactor": "代码重构",
            "other": "其他",
        }
        return names.get(cat, cat)

    @staticmethod
    def _error_name(err: str) -> str:
        """错误类型英文 → 中文可读名称。"""
        names = {
            "syntax": "语法错误",
            "import": "导入/依赖错误",
            "assertion": "断言失败",
            "runtime": "运行时错误",
            "timeout": "超时",
            "unknown": "未分类错误",
        }
        return names.get(err, err)


# ── 全局单例 ──
# Agent 每次 Fix Loop 失败后自动向此实例收集 Bad Case
# 通过 app/api 暴露 GET /api/bad-cases/stats 供前端查询
bad_case_analyzer = BadCaseAnalyzer()
