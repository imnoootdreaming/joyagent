"""
Phase 5 Step 5: Fix Loop — 自动修复循环（"编程能力闭环"的核心）。

FixLoop 将 Phase 5 的所有组件串联为一个完整的"测试→失败→修复→重测"循环：
  PytestRunner (Step 3)  →  运行测试  →  TestResult + TestFailure 列表
  ErrorParser  (Step 4)  →  解析错误  →  分类 + 修复 Prompt
  LLM (Phase 1/2)       →  生成修复  →  修复后的代码
  DiffGenerator (P4)    →  生成 Diff  →  精准差异
  PatchApplier  (P4)    →  应用 Patch →  增量修改文件
  循环直到: 所有测试通过 OR 达到最大尝试次数 OR 修复停滞

修复停滞检测（stall detection）：
  如果连续两轮产生完全相同的错误 → 说明 LLM 一直在重复无效修复
  → 提前终止循环，避免浪费 Token。

与 Phase 1-2 Simple Agent 的区别：
  - Simple Agent: 单轮"读→写"，没有验证循环
  - Fix Loop:    多轮"测→修→测"，有明确的退出条件和停滞检测

使用方式:
  from app.sandbox.fix_loop import FixLoop

  loop = FixLoop(docker_runner, max_attempts=3)
  final_state = await loop.fix_and_retest(
      test_path="tests/",
      source_files={"tests/test_calc.py": original_test_code},
      task_context="User wants a calculator with add/sub/mul/div",
  )
  if final_state.all_passed:
      print("All tests pass!")
  else:
      print(f"Failed after {final_state.current_attempt} attempts.")
"""
from __future__ import annotations

# ── Python 标准库 ──
import hashlib                        # 错误指纹计算（用于停滞检测）
from pathlib import Path              # 路径操作

# ── 项目内导入 ──
from app.sandbox.security import (
    TestResult,                        # 测试结果聚合
    TestFailure,                       # 单个测试失败
    FixLoopState,                      # 修复循环状态机
)
from app.sandbox.docker_runner import DockerRunner  # Docker 执行器
from app.sandbox.pytest_runner import PytestRunner  # pytest 测试执行器
from app.sandbox.error_parser import ErrorParser     # 错误解析 + 分类 + Prompt 生成

from app.coding.diff_generator import DiffGenerator  # unified diff 生成
from app.coding.patch_apply import PatchApplier      # diff 应用

from app.service.llm_service import get_or_create_client  # Anthropic 客户端
from app.core.config import Config                  # DEFAULT_MODEL 等

from app.tools.base import ToolResult               # 工具返回格式（兼容判断）

from typing import Optional
from app.memory.reflection import get_reflection_memory  # Phase 6: Reflection Memory


# ── 常量 ──

FIXER_MAX_TOKENS = 8192              # Fixer LLM 的 max_tokens
FIXER_SYSTEM_PROMPT = (
    "You are an expert code fixer. Your job is to fix test failures. "
    "You receive: the test failure details, the source code, and a fix strategy. "
    "Output ONLY the corrected Python code for each file. "
    "Do NOT output explanations—only code. "
    "Wrap each file in ```python path/to/file.py ... ``` blocks."
)


# ═══════════════════════════════════════════════════════════════════════════════
# _extract_text — 从 Anthropic SDK 响应中提取纯文本
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_text(response) -> str:
    """从 Anthropic Messages API 响应中提取纯文本（只取 type=='text' 的 block）。"""
    parts = []
    for block in response.content:
        if hasattr(block, 'type') and block.type == "text":
            parts.append(block.text)
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# _parse_fixed_code — 从 LLM 输出中提取修复后的代码
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_fixed_code(response_text: str) -> dict[str, str]:
    """
    从 LLM 响应中提取修复后的代码（按文件分）。

    LLM 输出格式：
      ```python app/main.py
      from fastapi import FastAPI
      ...
      ```

      ```python tests/test_calc.py
      from calculator import add
      ...
      ```

    或者纯代码（无 markdown 包裹）→ 作为单个文件的修复结果。

    Returns:
        {file_path: code_string} — 按文件路径索引的修复后代码
    """
    import re

    files: dict[str, str] = {}

    # ── 模式 A: ```python path/to/file.py ... ``` ──
    pattern = re.compile(
        r'```(?:python\s+)?(\S+\.py)\s*\n(.*?)```',
        re.DOTALL,
    )
    matches = pattern.findall(response_text)
    if matches:
        for file_path, code in matches:
            files[file_path.strip()] = code.strip()
        return files

    # ── 模式 B: 纯代码（无 markdown） ──
    # 清理可能的 markdown 残留
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    # 无法确定具体文件 → 用 "_fixed.py" 作为默认 key
    files["_fixed.py"] = cleaned
    return files


# ═══════════════════════════════════════════════════════════════════════════════
# _compute_error_fingerprint — 错误指纹（用于停滞检测）
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_error_fingerprint(failures: list[TestFailure]) -> str:
    """
    计算当前失败集合的"指纹"。

    指纹 = SHA256(test_name + error_type + error_message) 的前 16 位。

    如果两轮的指纹相同 → 说明 LLM 的修复完全没有改变错误集合
    → 可能陷入了重复无效修复 → 应提前终止循环。
    """
    if not failures:
        return ""
    # 按测试名排序（消除顺序差异）
    sorted_failures = sorted(failures, key=lambda f: f.test_name)
    content = "|".join(
        f"{f.test_name}:{f.error_type}:{f.error_message[:80]}"
        for f in sorted_failures
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════════
# FixLoop — 自动修复循环
# ═══════════════════════════════════════════════════════════════════════════════

class FixLoop:
    """
    自动修复循环 —— Agent 编程能力闭环的核心组件。

    组件集成图：
      ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
      │ ① 运行   │───▶│ ② 解析   │───▶│ ③ LLM    │───▶│ ④ 生成   │
      │   测试    │    │   错误    │    │   修复    │    │   Diff   │
      └──────────┘    └──────────┘    └──────────┘    └────┬─────┘
           ▲                                               │
           │          ┌──────────┐                         │
           └──────────│ ⑤ 应用   │◀────────────────────────┘
                      │   Patch  │
                      └──────────┘
      循环直到: all_passed=True OR max_attempts 耗尽 OR 修复停滞

    退出条件（优先级从高到低）：
      1. 所有测试通过             → SUCCESS — 任务完成
      2. LLM 连续无效修复（停滞）  → STALLED — 避免浪费 Token
      3. 达到最大尝试次数          → EXHAUSTED — 升级到人工/BadCaseAnalyzer

    面试要点：
      Fix Loop 不是简单的 "while not pass: fix()"——
      它有停滞检测（error fingerprinting）、策略分类（13 种错误类型）、
      增量修复（Diff→Patch 而非重写文件）三方面的设计深度。
    """

    def __init__(
        self,
        docker_runner: DockerRunner,
        max_attempts: int = 3,
        max_stall_retries: int = 2,
        bad_case_analyzer=None,         # Optional[BadCaseAnalyzer] — Phase 5 Step 5.5
        auto_collect_bad_cases: bool = True,
        memory_manager: Optional[object] = None,  # Phase 6: MemoryManager
    ):
        """
        Args:
            docker_runner:    已配置好的 DockerRunner 实例
            max_attempts:     最多修复尝试次数（默认 3）
            max_stall_retries:停滞检测阈值——连续 N 次相同错误指纹后终止
            bad_case_analyzer: BadCaseAnalyzer 实例（Phase 5.5）。
                              传入 None → 不收集 Bad Case。
            auto_collect_bad_cases: 是否在 Fix Loop 失败后自动收集 Bad Case。
                                    True（默认）→ 自动收集并打印报告摘要。
            memory_manager:   MemoryManager 实例（Phase 6）。
                             传入 None → 不集成记忆系统（向后兼容）。
        """
        self.test_runner = PytestRunner(docker_runner)
        self.error_parser = ErrorParser()
        self.diff_gen = DiffGenerator()
        self.patch_applier = PatchApplier(backup=True)

        # LLM 客户端（按配置模型分流 Claude/DeepSeek）
        self.client = get_or_create_client(Config.DEFAULT_MODEL)

        self.max_attempts = max_attempts
        self.max_stall_retries = max_stall_retries

        # Bad Case 收集器（Phase 5 Step 5.5）
        self.bad_case_analyzer = bad_case_analyzer
        self.auto_collect = auto_collect_bad_cases

        # Phase 6: Memory Manager（可选集成）
        self.mm = memory_manager
        # 即使不传 MemoryManager，也初始化 ReflectionMemory 单例
        self.rm = get_reflection_memory()

    # ── 主循环 ──────────────────────────────────────────────

    async def fix_and_retest(
        self,
        test_path: str,
        source_files: dict[str, str],
        task_context: str = "",
    ) -> FixLoopState:
        """
        自动修复主循环：测试 → 修复 → 重测 → ... → 终止。

        Args:
            test_path:    测试文件路径（如 "tests/" 或 "tests/test_calc.py"）
            source_files: 要修复的源文件内容 {file_path: content}
                          （通常是 Agent 生成的代码）
            task_context: 任务原始描述（帮助 LLM 理解修复目标）

        Returns:
            FixLoopState — 包含：
              - current_attempt: 实际尝试次数
              - test_results:    每次尝试的测试结果
              - fix_history:     每次尝试的修复详情
              - should_escalate(): 是否需要人工介入

        Example:
            loop = FixLoop(runner)
            state = await loop.fix_and_retest(
                test_path="tests/",
                source_files={
                    "tests/test_calc.py": test_code,
                    "calculator.py": src_code,
                },
                task_context="Create a calculator with add/sub/mul/div",
            )
            if state.all_passed:
                print("✅ All tests pass!")
            elif state.should_escalate():
                print("⚠ Need human review")
        """
        state = FixLoopState(max_attempts=self.max_attempts)

        # 跟踪错误指纹（用于停滞检测）
        prev_fingerprint: str | None = None
        stall_count = 0

        while state.should_continue():
            state.current_attempt += 1
            attempt = state.current_attempt
            print(f"\n  {'─' * 40}")
            print(f"  Fix Loop attempt {attempt}/{self.max_attempts}")

            # ── 1. 运行测试 ──────────────────────────────
            test_result, failures = await self.test_runner.run_tests(test_path)
            state.test_results.append(test_result)

            print(f"  Test result: {test_result.to_summary()}")

            # ── 2. 检查是否全部通过 ──────────────────────
            if test_result.all_passed:
                print(f"  ✅ All tests passed on attempt {attempt}!")
                state.fix_history.append({
                    "attempt": attempt,
                    "result": "all_passed",
                })
                break

            # ── 3. 如果没有 failure 详情，尝试从 stderr 解析 ──
            # （PytestRunner 返回的 failures 可能为空但 test_result 显示有失败）
            if not failures and test_result.failed + test_result.errors > 0:
                # 运行一次测试收集 stderr 用于 ErrorParser
                _, failures = await self._retry_parse_errors(test_path)

            if not failures:
                # 测试有失败但无法提取详情 → 记录但继续
                print(f"  ⚠ No parseable failures, but {test_result.failed + test_result.errors} tests failed")
                state.fix_history.append({
                    "attempt": attempt,
                    "result": "unparseable_errors",
                    "test_result": test_result,
                })
                continue

            # ── 4. 停滞检测 ─────────────────────────────
            fingerprint = _compute_error_fingerprint(failures)
            if fingerprint == prev_fingerprint:
                stall_count += 1
                print(f"  ⚠ Same error pattern ({stall_count}/{self.max_stall_retries})")
                if stall_count >= self.max_stall_retries:
                    print(f"  ❌ Repair stalled — same errors after {stall_count} attempts")
                    state.fix_history.append({
                        "attempt": attempt,
                        "result": "stalled",
                        "fingerprint": fingerprint,
                        "failures": failures,
                    })
                    break
            else:
                stall_count = 0               # 新模式 → 重置停滞计数
            prev_fingerprint = fingerprint

            # ── 5. 分类错误 ──────────────────────────────
            categories = [self.error_parser.categorize_error(f) for f in failures]
            error_cats = sorted(set(categories))
            print(f"  Error categories: {error_cats}")

            # ── 6. LLM 生成修复 ──────────────────────────
            # 构建修复 Prompt（使用 ErrorParser 的分类模板）
            source_code = self._format_source_files(source_files)
            fix_prompt = self.error_parser.generate_batch_fix_prompt(
                failures, source_code
            )

            # 添加上下文信息
            full_prompt = fix_prompt
            if task_context:
                full_prompt = (
                    f"## Original Task\n{task_context}\n\n" + fix_prompt
                )

            # ── Phase 6: 检索历史类似错误的修复建议 ──
            if self.mm is not None:
                # 构建错误描述（取第一个failure的完整信息）
                first_f = failures[0]
                error_desc = (
                    f"{first_f.error_type}: {first_f.error_message}\n"
                    f"File: {first_f.test_name}\n"
                    f"{first_f.traceback[:500]}"
                )
                suggestion = await self.mm.get_fix_suggestion(
                    error_text=error_desc,
                    file_path=first_f.test_name,
                )
                if suggestion:
                    full_prompt = (
                        f"## Historical Fix Suggestions\n{suggestion}\n\n"
                        + full_prompt
                    )
                    print(f"  \033[36m[memory] injected historical fix suggestion\033[0m")

            try:
                response = self.client.messages.create(
                    model=Config.DEFAULT_MODEL,
                    system=FIXER_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=FIXER_MAX_TOKENS,
                )
            except Exception as e:
                print(f"  ❌ LLM call failed: {e}")
                state.fix_history.append({
                    "attempt": attempt,
                    "result": "llm_error",
                    "error": str(e),
                })
                continue

            fixed_text = _extract_text(response)

            # ── 7. 解析修复后的代码 ──────────────────────
            fixed_files = _parse_fixed_code(fixed_text)
            if not fixed_files:
                print(f"  ❌ Could not parse fixed code from LLM response")
                state.fix_history.append({
                    "attempt": attempt,
                    "result": "parse_error",
                    "llm_response_preview": fixed_text[:300],
                })
                continue

            # ── 8. 生成 Diff + 应用 Patch ────────────────
            patch_results = []
            all_patches_ok = True

            for file_path, new_content in fixed_files.items():
                if file_path not in source_files:
                    # LLM 修复了未提供的文件 → 跳过（避免误写）
                    print(f"  ⚠ LLM fixed unknown file '{file_path}' — skipping")
                    continue

                original = source_files[file_path]

                # 8a. 生成 diff
                diff_result = self.diff_gen.generate(file_path, original, new_content)
                if diff_result.is_empty:
                    print(f"  ⚠ No changes for {file_path}")
                    continue

                # 8b. 应用 patch
                patch_result = self.patch_applier.apply_manual(
                    diff_result.diff_text, "."
                )

                patch_results.append({
                    "file": file_path,
                    "hunks": len(diff_result.hunks),
                    "applied": patch_result.success,
                    "added": diff_result.lines_added,
                    "removed": diff_result.lines_removed,
                })

                if patch_result.success:
                    # 更新 source_files 缓存（下次 LLM 看到的是修复后的代码）
                    source_files[file_path] = new_content
                    print(
                        f"  ✅ Patch applied: {file_path} "
                        f"(+{diff_result.lines_added} -{diff_result.lines_removed})"
                    )
                else:
                    all_patches_ok = False
                    print(
                        f"  ❌ Patch failed for {file_path}: "
                        f"{patch_result.error_message[:100]}"
                    )

            # ── 9. 记录本轮修复历史 ──────────────────────
            state.fix_history.append({
                "attempt": attempt,
                "result": "patches_applied" if all_patches_ok else "partial_patch_failure",
                "fingerprint": fingerprint,
                "failures": [{
                    "test_name": f.test_name,
                    "error_type": f.error_type,
                    "error_message": f.error_message[:120],
                } for f in failures],
                "categories": error_cats,
                "patches": patch_results,
                "llm_response_preview": fixed_text[:300],
            })

            # ── Phase 6: 记录错误经验到 Reflection Memory ──
            # 每个失败的测试都记录为其独立的 ErrorExperience
            if self.mm is not None:
                for f in failures:
                    try:
                        await self.mm.on_fix_attempt(
                            error_type=f.error_type,
                            error_message=f.error_message[:200],
                            file_path=f.test_name,
                            context_snippet=f.traceback[:500] if f.traceback else "",
                            fix_description=fixed_text[:300],
                            fix_success=all_patches_ok,
                        )
                    except Exception:
                        pass  # 静默降级

            if not all_patches_ok:
                # 部分 patch 失败 → 可能文件状态不一致 → 继续重试
                print(f"  ⚠ Some patches failed — will retry")
                continue

        # ── 循环结束 ──────────────────────────────────────
        final_passed = (
            state.test_results[-1].all_passed
            if state.test_results else False
        )
        if final_passed:
            print(f"\n  ✅ Fix Loop SUCCESS — all tests pass!")
        elif state.should_escalate():
            print(f"\n  ⚠ Fix Loop EXHAUSTED — {state.current_attempt}/{self.max_attempts} attempts used")
        else:
            print(f"\n  ❌ Fix Loop terminated — see fix_history for details")

        # ── Phase 5.5: Auto-collect Bad Case ─────────────────
        if (
            self.auto_collect
            and self.bad_case_analyzer
            and state.should_escalate()
        ):
            # Build stderr from last failure
            last_stderr = ""
            if state.fix_history:
                last_entry = state.fix_history[-1]
                failures_list = last_entry.get("failures", [])
                last_stderr = "\n".join(
                    f"{f.get('error_type', '?')}: {f.get('error_message', '')}"
                    for f in failures_list
                )

            source_code = self._format_source_files(source_files)

            self.bad_case_analyzer.collect_from_fix_loop(
                fix_state=state,
                task=task_context,
                category="",
                source_files=source_files,
            )

            total = len(self.bad_case_analyzer.cases)
            print(f"  \033[35m📊 Bad Case #{total} collected — "
                  f"call bad_case_analyzer.generate_report() "
                  f"for analysis\033[0m")

            # Auto-print patterns if enough data
            if total >= 3:
                patterns = self.bad_case_analyzer.detect_patterns()
                if patterns:
                    print(f"  \033[35m   Top pattern: {patterns[0].pattern_name} "
                          f"({patterns[0].frequency}x)\033[0m")

        return state

    # ── 辅助方法 ──────────────────────────────────────────

    async def _retry_parse_errors(
        self, test_path: str
    ) -> tuple[TestResult, list[TestFailure]]:
        """
        PytestRunner 的 run_tests 可能返回空 failures 列表但 test_result
        显示有失败（如 JSON 报告解析失败时）。此时重新运行并收集原始输出
        传给 ErrorParser 解析。
        """
        # 直接用 DockerRunner 执行并收集完整 stderr
        command = (
            f"python3 -m pytest {test_path} -v --tb=long 2>&1"
        )
        exec_result = await self.test_runner.runner.run_command(command)

        # ErrorParser 解析完整输出
        failures = self.error_parser.parse(exec_result.combined_output)

        # 重建 TestResult
        if failures:
            total = len(failures)
            errors = sum(1 for f in failures if f.error_type not in ("AssertionError",))
            failed = total - errors
            return TestResult(
                total=total, passed=0, failed=failed, errors=errors,
            ), failures

        return TestResult(), []

    @staticmethod
    def _format_source_files(source_files: dict[str, str]) -> str:
        """
        将多文件源码格式化为 LLM 消费的文本块。

        Returns:
            带文件标记的源码文本
        """
        parts = []
        for file_path, content in source_files.items():
            parts.append(f"### {file_path}\n```python\n{content}\n```\n")
        return "\n".join(parts)

    # ── 便捷方法 ──────────────────────────────────────────

    async def fix_single_file(
        self,
        test_path: str,
        file_path: str,
        code: str,
        task_context: str = "",
    ) -> FixLoopState:
        """
        单文件修复便捷方法。

        Args:
            test_path:  测试文件路径
            file_path:  要修复的源文件路径
            code:       源代码内容
            task_context: 任务原始描述

        Returns:
            FixLoopState
        """
        return await self.fix_and_retest(
            test_path=test_path,
            source_files={file_path: code},
            task_context=task_context,
        )

    async def fix_with_generated_test(
        self,
        source_file: str,
        source_code: str,
        test_file: str,
        test_code: str,
        task_context: str = "",
    ) -> FixLoopState:
        """
        Agent 同时生成了代码和测试——对两者一起修复。

        适用场景：Agent 生成了 calculator.py 和 test_calculator.py，
        测试失败可能是代码问题或测试问题。

        Args:
            source_file: 源代码文件路径
            source_code: 源代码内容
            test_file:   测试文件路径
            test_code:   测试代码内容
            task_context: 任务描述

        Returns:
            FixLoopState
        """
        return await self.fix_and_retest(
            test_path=test_file,
            source_files={
                source_file: source_code,
                test_file: test_code,
            },
            task_context=task_context,
        )
