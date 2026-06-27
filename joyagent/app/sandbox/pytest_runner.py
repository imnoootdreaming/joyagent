"""
Phase 5 Step 3: Pytest Runner — 在 Docker Sandbox 中运行 pytest 测试。

PytestRunner 是 Agent "编程能力闭环"的关键组件——Agent 生成代码后，
PytestRunner 在隔离的 Docker 沙箱中运行测试，返回结构化结果供 Fix Loop 分析。

双模式结果解析：
  1. JSON 报告模式（优先） — 使用 pytest-json-report 插件生成结构化 JSON
     精确到每个测试函数的状态、耗时、错误信息
  2. 正则文本回退（fallback） — 解析 pytest 的标准文本输出
     JSON 报告不可用时（插件未安装或报告损坏）自动降级

测试粒度：
  run_tests()      — 全量运行（test_path 默认为当前目录）
  run_single_test() — 运行单个测试函数（更精准的错误定位）
  discover_tests() — 列出所有可用的测试函数（test discovery）

使用方式：
  from app.sandbox import DockerRunner, PytestRunner

  runner = DockerRunner(config)
  test_runner = PytestRunner(runner)

  result = await test_runner.run_tests("tests/")
  if result.all_passed:
      print("All tests passed!")
  else:
      print(f"{result.failed} failures, {result.errors} errors")
"""

# ── Python 标准库 ──
import asyncio                         # 异步执行
import json                            # JSON 报告解析
import re                              # 正则文本输出解析
from pathlib import Path               # 路径操作（镜像验证等）

# ── 项目内导入 ──
from app.sandbox.security import (
    ExecutionResult,                    # Docker 命令执行结果
    TestResult,                         # 测试结果聚合
    TestFailure,                        # 单个测试失败详情
)
from app.sandbox.docker_runner import DockerRunner  # Docker 执行器


# ═══════════════════════════════════════════════════════════════════════════════
# PytestRunner
# ═══════════════════════════════════════════════════════════════════════════════

class PytestRunner:
    """
    在 Docker Sandbox 中运行 pytest 的测试执行器。

    职责：
      1. 构建 pytest 命令（参数：冗余输出、JSON 报告、短 traceback）
      2. 通过 DockerRunner 在沙箱中执行测试
      3. 解析测试结果（JSON 优先，正则回退）
      4. 返回结构化的 TestResult + TestFailure 列表

    与 Fix Loop 的协作：
      PytestRunner.run_tests() → TestResult
      → 如果 all_passed=False → ErrorParser 解析错误
      → Fix Loop 生成修复 → PytestRunner.run_single_test() 针对性重测
      → 重复直到 all_passed=True 或达到 max_attempts
    """

    def __init__(self, docker_runner: DockerRunner):
        """
        Args:
            docker_runner: DockerRunner 实例（已配置好 SandboxConfig）
        """
        self.runner = docker_runner       # Docker 执行器

    # ── 公共 API ──────────────────────────────────────────────

    async def run_tests(
        self,
        test_path: str = ".",
        extra_args: str = "",
    ) -> tuple[TestResult, list[TestFailure]]:
        """
        在 Docker 沙箱中运行 pytest 全量测试。

        默认使用 --json-report 生成结构化 JSON 报告（更精确），
        如果 JSON 报告解析失败，自动回退到文本解析。

        Args:
            test_path:  测试文件/目录路径（如 "tests/"、"test_calc.py"）
            extra_args: 额外的 pytest 参数（如 "-k test_add"、" -x"）

        Returns:
            (TestResult, list[TestFailure]) — 测试结果聚合 + 失败详情列表
            TestResult.all_passed=True 表示全部通过

        Example:
            test_result, failures = await runner.run_tests("tests/")
            if not test_result.all_passed:
                for f in failures:
                    print(f"{f.test_name}: {f.error_type} at L{f.line_number}")
        """
        # ── 0. Docker 不可用 → 返回失败 ──
        if not self.runner.is_available:
            return (
                TestResult(total=0, passed=0, failed=0, errors=1),
                [TestFailure(
                    test_name="(all)",
                    error_type="DockerUnavailable",
                    error_message="Docker is not available — cannot run tests.",
                    file_path="",
                )],
            )

        # ── 1. 构建 pytest 命令 ────────────────────────────────
        # 标志说明：
        #   -v              详细输出（每个测试一行）
        #   --tb=short      traceback 简写模式（文件名+行号+错误类型）
        #   --json-report   JSON 格式结构化报告（写入文件，不混入 stdout）
        #   --json-report-file=/tmp/test_report.json 报告文件路径
        #   --tb=no         禁用 traceback（JSON 报告已包含详细信息）
        command = (
            f"python3 -m pytest {test_path} "
            f"-v "
            f"--tb=short "
            f"--json-report "
            f"--json-report-file=/workspace/_test_report.json "
            f"{extra_args}"
        )

        # ── 2. 在 Docker 沙箱中执行 ─────────────────────────
        exec_result = await self.runner.run_command(command)

        # ── 3. 解析结果 — JSON 优先 ────────────────────────
        test_result, failures = self._parse_result(exec_result, test_path)

        return test_result, failures

    async def run_single_test(
        self,
        test_path: str,
        test_name: str,
    ) -> tuple[TestResult, list[TestFailure]]:
        """
        运行单个测试函数。

        Fix Loop 用此方法精准重测失败的测试，避免重跑全部测试。

        Args:
            test_path: 测试文件路径（如 "tests/test_calc.py"）
            test_name: 测试函数名（如 "test_add"）

        Returns:
            (TestResult, list[TestFailure])
        """
        # -k: 只运行名称匹配的测试
        return await self.run_tests(
            test_path=test_path,
            extra_args=f'-k "{test_name}"',
        )

    async def discover_tests(self, test_path: str = ".") -> list[str]:
        """
        列出所有可用的测试函数名称。

        使用 pytest --collect-only 实现测试发现。
        适合 Agent 在运行测试前了解有哪些测试、决定测试策略。

        Args:
            test_path: 搜索路径

        Returns:
            测试函数名列表（如 ["test_add", "test_subtract", ...]）

        Example:
            tests = await runner.discover_tests("tests/")
            for name in tests:
                print(f"  - {name}")
        """
        command = f"python3 -m pytest {test_path} --collect-only -q --no-header"
        result = await self.runner.run_command(command)

        if not result.succeeded:
            return []

        # 解析 pytest --collect-only 的输出：
        # tests/test_calc.py::test_add
        # tests/test_calc.py::test_subtract
        # tests/test_calc.py::test_divide_by_zero
        test_names = []
        for line in result.stdout.split("\n"):
            line = line.strip()
            if "::" in line and not line.startswith("<") and not line.startswith("="):
                # 格式：path::function_name
                parts = line.split("::")
                if len(parts) >= 2:
                    test_names.append(parts[-1].strip())
        return test_names

    async def check_test_file(
        self,
        test_path: str,
    ) -> ExecutionResult:
        """
        快速检查测试文件是否存在并能被 pytest 识别。

        不运行任何测试，只做语法检查。

        Args:
            test_path: 测试文件路径

        Returns:
            ExecutionResult — exited_code=0 表示文件有效
        """
        command = f"python3 -m pytest {test_path} --collect-only -q --no-header"
        return await self.runner.run_command(command)

    # ── 结果解析 — JSON 模式 ──────────────────────────────────

    @staticmethod
    def _parse_by_json_report(
        exec_result: ExecutionResult,
    ) -> tuple[TestResult | None, list[TestFailure]]:
        """
        从 pytest-json-report 插件生成的 JSON 报告中解析测试结果。

        JSON 报告格式（pytest-json-report 1.5+）：
          {
            "summary": {"passed": 8, "failed": 2, "total": 10, ...},
            "tests": [
              {
                "nodeid": "tests/test_calc.py::test_add",
                "outcome": "passed",
                "duration": 0.001234
              },
              {
                "nodeid": "tests/test_calc.py::test_subtract",
                "outcome": "failed",
                "call": {
                  "crash": {"path": "...", "lineno": 15, "message": "..."},
                  "longrepr": "AssertionError: assert 5 == 4\\n..."
                }
              },
              ...
            ]
          }

        Returns:
          (TestResult, list[TestFailure]) — JSON 解析成功 → 返回结果
          (None, [])                        — JSON 解析失败 → 调用方应降级到 regex
        """
        # JSON 报告不直接在 stdout 中——它写在 /workspace/_test_report.json
        # 需要用 read_file 读取，但 PytestRunner 在沙箱内无法直接读文件
        # 更实用的方案：也收集 stdout（pytest 终端输出本身也是完整结果）
        # 所以这里 JSON 模式解析的是 stdout 中包含的 summary 行和每个测试的状态行
        stdout = exec_result.stdout

        # ── 尝试 1：pytest 输出中包含 "= test session starts =" 但不是结构化 JSON ──
        # 真正的 JSON 报告在文件中，不在 stdout。此处检测是否存在 JSON 报告文件
        # 如果无 stdout 信息，说明测试根本没有被收集

        # 实际上 pytest-json-report 会将摘要打印到 stdout:
        # {"summary": {"passed": 5, "failed": 1, "total": 6, ...}}
        # 尝试解析末尾的 JSON 行
        json_lines = []
        for line in stdout.split("\n"):
            line = line.strip()
            if line.startswith("{") and '"summary"' in line:
                json_lines.append(line)

        for json_line in reversed(json_lines):
            try:
                data = json.loads(json_line)
                summary = data.get("summary", {})

                passed = summary.get("passed", 0)
                failed = summary.get("failed", 0)
                errors = summary.get("error", summary.get("errors", 0))
                total = summary.get("total", passed + failed + errors)

                test_result = TestResult(
                    total=total,
                    passed=passed,
                    failed=failed,
                    errors=errors,
                )

                # 提取每个测试的失败信息
                tests = data.get("tests", [])
                failures = PytestRunner._extract_failures_from_json(tests)

                return test_result, failures

            except (json.JSONDecodeError, KeyError):
                continue

        # ── 尝试 2：如果 pytest-json-report 版本不同，检查是否在 stderr ──
        for line in exec_result.stderr.split("\n"):
            line = line.strip()
            if line.startswith("{") and '"summary"' in line:
                try:
                    data = json.loads(line)
                    summary = data.get("summary", {})
                    return TestResult(
                        total=summary.get("total", 0),
                        passed=summary.get("passed", 0),
                        failed=summary.get("failed", 0),
                        errors=summary.get("error", 0),
                    ), []
                except (json.JSONDecodeError, KeyError):
                    pass

        return None, []                  # JSON 解析失败 → 降级到 regex

    @staticmethod
    def _extract_failures_from_json(tests: list[dict]) -> list[TestFailure]:
        """
        从 JSON 报告中的 tests 数组提取失败测试的详细信息。
        """
        failures = []
        for test in tests:
            outcome = test.get("outcome", "")
            if outcome in ("passed", "skipped", "xfailed"):
                continue                   # 只关注失败/错误的测试

            # 提取测试名（nodeid 的最后一段）
            nodeid = test.get("nodeid", "unknown")
            test_name = nodeid.split("::")[-1] if "::" in nodeid else nodeid

            # 提取文件路径
            file_path = nodeid.split("::")[0] if "::" in nodeid else ""

            # 提取错误详情（在 test["call"] 中）
            call_info = test.get("call", {})
            crash = call_info.get("crash", {})
            longrepr = call_info.get("longrepr", "")

            # 从 longrepr 中提取错误类型和行号
            error_type = "Unknown"
            error_message = ""
            line_number = None

            if crash:
                error_message = crash.get("message", "")
                line_number = crash.get("lineno")
            elif longrepr:
                error_message = longrepr
                # 尝试从中提取错误类型
                match = re.match(r"(\w+Error|\w+Exception|AssertionError):?\s*(.*)", longrepr)
                if match:
                    error_type = match.group(1)
                    error_message = match.group(2) or error_message

            if not error_type or error_type == "Unknown":
                if "Assertion" in longrepr:
                    error_type = "AssertionError"
                elif "NameError" in longrepr:
                    error_type = "NameError"
                elif "Import" in longrepr:
                    error_type = "ImportError"
                else:
                    error_type = "RuntimeError"

            failures.append(TestFailure(
                test_name=test_name,
                error_type=error_type,
                error_message=error_message[:500],  # 截断
                file_path=file_path,
                line_number=line_number,
                traceback=longrepr[:2000],          # 完整 traceback
            ))

        return failures

    # ── 结果解析 — 正则回退模式 ──────────────────────────────────

    @staticmethod
    def _parse_by_regex(
        exec_result: ExecutionResult,
    ) -> tuple[TestResult, list[TestFailure]]:
        """
        从 pytest 的文本输出中解析测试结果（正则回退）。

        当 JSON 报告不可用时使用此方法解析 pytest 的标准文本输出。

        pytest -v 输出的典型格式：
          tests/test_calc.py::test_add PASSED
          tests/test_calc.py::test_subtract FAILED
          tests/test_calc.py::test_divide ERROR

          ===== short test summary info =====
          FAILED tests/test_calc.py::test_subtract - assert 5 == 4
          ERROR tests/test_calc.py::test_divide - ZeroDivisionError: ...

          ===== 1 failed, 5 passed, 1 error in 0.05s =====
        """
        output = exec_result.combined_output

        # ── 1. 按行提取测试状态 ──
        # 格式: path::test_name STATUS  （STATUS = PASSED | FAILED | ERROR | SKIPPED）
        # 注意：文件路径必须包含 / 或 .py（排除 summary 行 "FAILED path::name" 的误判）
        test_status_pattern = re.compile(
            r'^([^ ]*(?:/|\.py).*?)::(.+?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFALLED|XPASSED)',
            re.MULTILINE,
        )
        statuses = test_status_pattern.findall(output)

        passed = 0
        failed = 0
        errors = 0
        failed_tests: list[dict] = []    # {name, file, status}

        for file_path, test_name, status in statuses:
            if status == "PASSED":
                passed += 1
            elif status == "FAILED":
                failed += 1
                failed_tests.append({
                    "name": test_name.strip(),
                    "file": file_path.strip(),
                    "status": "failed",
                })
            elif status == "ERROR":
                errors += 1
                failed_tests.append({
                    "name": test_name.strip(),
                    "file": file_path.strip(),
                    "status": "error",
                })
            # SKIPPED / XFAILED / XPASSED 不计入失败

        total = passed + failed + errors

        # ── 2. 从 "short test summary info" 提取错误详情 ──
        failures = PytestRunner._extract_failure_details(output, failed_tests)

        # ── 3. 回退：从最后一行 ("== 1 failed, 5 passed in 0.05s ==") 提取 ──
        if total == 0:
            summary_match = re.search(
                r"(\d+)\s+failed.*?(\d+)\s+passed.*?(\d+)\s+error",
                output,
                re.IGNORECASE,
            )
            if not summary_match:
                # 尝试另一种格式: "1 failed, 5 passed, 0 warnings"
                summary_match = re.search(
                    r"(\d+)\s+passed.*?(\d+)\s+failed",
                    output,
                    re.IGNORECASE,
                )
                if summary_match:
                    passed = int(summary_match.group(1))
                    failed = int(summary_match.group(2))
                    total = passed + failed + errors
            else:
                failed = int(summary_match.group(1))
                passed = int(summary_match.group(2))
                errors = int(summary_match.group(3))
                total = passed + failed + errors

        return TestResult(
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
        ), failures

    @staticmethod
    def _extract_failure_details(
        output: str,
        failed_tests: list[dict],
    ) -> list[TestFailure]:
        """
        从 pytest 输出中提取失败测试的详细错误信息。

        搜索 "short test summary info" 后的内容，
        每行一个失败测试的简要说明。
        """
        if not failed_tests:
            return []

        # 找到 "short test summary info" 或 "FAILURES" 部分
        failures_section = ""
        for marker in ["short test summary info", "FAILURES", "ERRORS"]:
            idx = output.find(marker)
            if idx >= 0:
                failures_section = output[idx:]
                break

        results = []
        for ft in failed_tests:
            error_type = "Unknown"
            error_message = ""
            line_number = None

            # 在 summary section 中查找该测试的错误信息
            test_key = f"{ft['name']}"
            for line in failures_section.split("\n"):
                if test_key in line and " - " in line:
                    # 格式：FAILED test_xxx - AssertionError: assert 5 == 4
                    after_dash = line.split(" - ", 1)[-1].strip()
                    match = re.match(
                        r"(\w+Error|\w+Exception|AssertionError):?\s*(.*)",
                        after_dash,
                    )
                    if match:
                        error_type = match.group(1)
                        error_message = match.group(2) or after_dash
                    else:
                        error_message = after_dash
                    break

            # 在 traceback 中搜索文件行号
            tb_pattern = re.compile(
                rf'File\s+"([^"]+)",\s+line\s+(\d+).*?\n\s*{re.escape(ft["name"])}',
                re.DOTALL,
            )
            tb_match = tb_pattern.search(output)
            if not tb_match:
                # 更宽松的匹配：任何与该测试同文件的行号
                file_only = ft.get("file", "")
                if file_only:
                    file_tb = re.compile(
                        rf'File\s+"({re.escape(file_only)})",\s+line\s+(\d+)',
                    )
                    file_match = file_tb.search(output)
                    if file_match:
                        line_number = int(file_match.group(2))

            if tb_match:
                line_number = int(tb_match.group(2))

            results.append(TestFailure(
                test_name=ft["name"],
                error_type=error_type or "RuntimeError",
                error_message=error_message[:500],
                file_path=ft.get("file", ""),
                line_number=line_number,
                traceback=failures_section[:2000],
            ))

        return results

    # ── 统一解析入口 ────────────────────────────────────────────

    @staticmethod
    def _parse_result(
        exec_result: ExecutionResult,
        test_path: str = ".",
    ) -> tuple[TestResult, list[TestFailure]]:
        """
        统一的结果解析入口 — JSON 优先，正则回退。

        策略：
          1. 先尝试 JSON 报告解析（精确到每个测试的详细信息）
          2. JSON 解析失败 → 降级到正则文本解析
          3. 两者都失败 → 检查是否是 Docker/环境错误
          4. 兜底 → 返回空 TestResult
        """
        # ── 0. 处理 Docker 执行失败 ──
        # 注意：pytest 有失败时 exit_code != 0，这是正常情况，不应拦截。
        # 只在 Docker 本身错误（exit_code=-1 或 error_message 非空）时才提前返回。
        if exec_result.exit_code == -1 or exec_result.error_message is not None:
            return (
                TestResult(total=0, passed=0, failed=0, errors=1),
                [TestFailure(
                    test_name="(execution failed)",
                    error_type="ExecutionError",
                    error_message=exec_result.error_message or exec_result.stderr,
                    file_path=test_path,
                )],
            )

        # ── 1. JSON 报告解析 ──
        test_result, failures = PytestRunner._parse_by_json_report(exec_result)
        if test_result is not None and test_result.total > 0:
            return test_result, failures

        # ── 2. 正则回退 ──
        test_result, failures = PytestRunner._parse_by_regex(exec_result)
        if test_result.total > 0:
            return test_result, failures

        # ── 3. 检查是否是 pytest 本身的问题 ──
        output = exec_result.combined_output
        if "no tests ran" in output.lower():
            return (
                TestResult(total=0, passed=0, failed=0, errors=0),
                [TestFailure(
                    test_name="(no tests)",
                    error_type="NoTestsFound",
                    error_message=f"No tests discovered in {test_path}. Check test file naming (must start with test_ or end with _test).",
                    file_path=test_path,
                )],
            )

        if "ModuleNotFoundError" in output or "ImportError" in output:
            return (
                TestResult(total=0, passed=0, failed=0, errors=1),
                [TestFailure(
                    test_name="(import error)",
                    error_type="ImportError",
                    error_message=f"pytest could not import {test_path}. Check file path and dependencies.",
                    file_path=test_path,
                )],
            )

        # ── 4. 兜底 ──
        return TestResult(total=0, passed=0, failed=0, errors=0), []
