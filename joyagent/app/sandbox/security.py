"""
Phase 5 Step 1-2: Sandbox 安全配置 + 执行/测试结果数据模型。

本文件定义了 Phase 5 所有核心数据模型：
  SandboxConfig   — Docker 沙箱六层安全配置（面试重点）
  ExecutionResult — 容器命令执行结果
  TestResult      — pytest 测试结果聚合
  TestFailure     — 单个测试失败的详细信息
  FixLoopState    — 自动修复循环的状态机

面试要点（SandboxConfig）：
  面试官必问："Agent 生成代码中可能包含 rm -rf /、curl evil.com、
  死循环等危险操作，你的系统如何防护？"

  标准答案结构（六层防御）：
    1. 容器隔离       — 宿主机文件系统与容器隔离，删除操作只影响容器
    2. 资源限制       — CPU/内存上限防止 DoS，禁止 swap 防止绕过内存限制
    3. 网络隔离       — network_mode=none，防止下载恶意脚本、数据泄露
    4. 文件系统       — 根文件只读，只挂载必要的工作目录
    5. 权限限制       — no_new_privileges + cap_drop=ALL 禁止提权
    6. 超时控制       — 每次执行 60s 超时 + 容器用完即销毁
    7. Human-in-the-Loop — 危险命令需要用户审批（Phase 9）
"""

# ── Python 标准库 ──
import os                              # 文件路径检查和 seccomp profile 读取
from dataclasses import dataclass, field  # 数据类装饰器
from pathlib import Path               # 路径对象（seccomp profile 路径解析）


# ═══════════════════════════════════════════════════════════════════════════════
# SandboxConfig — 沙箱安全配置（面试重点）
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SandboxConfig:
    """
    Docker 沙箱的完整安全配置。

    每个字段对应一层安全防御。设计原则：
      - 每个限制都有明确的"防什么"（见字段注释中的 ⚠ 标注）
      - 所有限制可独立调节（宽松 → 测试/调试；严格 → 生产）
      - 默认值采用严格模式（生产环境安全第一）

    安全层次（由外到内）：
      Level 1 — 容器隔离（Docker 进程/Mount 隔离）
      Level 2 — 资源限制（CPU 限额 + 内存上限 + 禁止 swap）
      Level 3 — 网络隔离（network_mode=none）
      Level 4 — 文件系统（只读根 + 最小权限挂载）
      Level 5 — 权限限制（no_new_privileges + cap_drop=ALL）
      Level 6 — 超时控制（执行超时 + 容器自动销毁）

    Example:
      # 安全模式（默认，生产环境）
      config = SandboxConfig()

      # 宽松模式（本地调试）
      config = SandboxConfig(
          network_disabled=False,
          read_only_root=False,
          timeout_seconds=300,
      )
    """

    # ── 镜像 ───────────────────────────────────────────────────────
    image: str = "joyagent-sandbox:latest"
    # Docker 镜像标签。此镜像由 sandbox_config/Dockerfile 构建。
    # 包含非 root 用户 sandbox + 预装的 pytest。

    # ── Level 2: 资源限制 ─────────────────────────────────────────
    cpu_limit: float = 1.0
    # ⚠ 防什么：Agent 生成死循环代码占用 100% CPU → 宿主机其他进程受影响
    # 值含义：Docker 用 CPU shares 机制，1.0 = 1 个 CPU 核心
    # 0.5 = 半核，2.0 = 双核

    memory_limit: str = "512m"
    # ⚠ 防什么：Agent 生成内存泄漏代码分配大量内存 → 宿主机 OOM
    # 格式：数字+单位 (m=MB, g=GB)

    memory_swap: str = "0m"
    # ⚠ 防什么：通过 swap 绕过 memory_limit（容器用磁盘模拟内存）
    # 0m = 禁止使用 swap。设为与 memory_limit 相同 = 允许 swap。

    # ── Level 6: 时间限制 ─────────────────────────────────────────
    timeout_seconds: int = 60
    # ⚠ 防什么：Agent 生成死循环 → 容器一直运行消耗资源
    # 执行超过此时间后 Docker 强制终止容器（SIGKILL）

    # ── Level 4: 文件系统 ─────────────────────────────────────────
    read_only_root: bool = True
    # ⚠ 防什么：Agent 生成代码删除/修改系统文件（如 /etc/passwd）
    # True → 根文件系统只读（/bin /usr /etc 等不可修改）
    # 工作目录 /workspace 通过 bind mount 独立挂载为可读写

    working_dir: str = "/workspace"
    # 容器的默认工作目录。宿主机的项目代码通过 bind mount 挂载到此路径。

    mount_path: str | None = None
    # 宿主机上要挂载到工作目录的路径。None 表示不挂载（容器使用空目录）。
    # Phase 5 中通常设为仓库根目录的绝对路径。

    # ── Level 3: 网络隔离 ─────────────────────────────────────────
    network_disabled: bool = True
    # ⚠ 防什么：Agent 生成代码下载恶意脚本、泄露数据到外部服务器
    # True → 容器内无网络接口（ping/curl/wget 全部失败）

    network_mode: str = "none"
    # Docker 的网络模式：none = 无网络，bridge = Docker 默认网桥

    # ── Level 5: 权限限制 ─────────────────────────────────────────
    no_new_privileges: bool = True
    # ⚠ 防什么：容器内进程通过 setuid/setgid 二进制文件提权
    # True → 即使执行了 chmod +s 的文件，内核也拒绝提升权限

    drop_capabilities: list[str] = field(default_factory=list)
    # ⚠ 防什么：容器内进程利用 Linux capabilities 执行特权操作
    # 默认在 __post_init__ 中设为 ["ALL"] — 丢弃所有 capabilities
    # 包括：CAP_SYS_ADMIN（挂载）、CAP_NET_RAW（原始套接字）、
    #       CAP_SYS_PTRACE（调试其他进程）等

    # ── 可选：seccomp profile ─────────────────────────────────────
    seccomp_profile: str | None = None
    # seccomp 系统调用过滤策略的 JSON 文件路径
    # None → Docker 默认 seccomp profile（禁止 ~40 个危险系统调用）
    # sandbox_config/seccomp_profile.json → 更严格的策略
    # ⚠ 面试加分项：能够讨论 seccomp 如何限制系统调用级别攻击面

    def __post_init__(self):
        """
        初始化后处理：设置默认值。

        drop_capabilities 默认设为 ["ALL"] — 丢弃所有 Linux capabilities。
        注意：此默认值在 __init__ 参数中不能用 field(default=["ALL"])
        因为 Python 的 dataclass 不允许 mutable default，所以在这里设置。
        """
        if not self.drop_capabilities:
            self.drop_capabilities = ["ALL"]

    def to_docker_params(self) -> dict:
        """
        将 SandboxConfig 转换为 Docker SDK 的容器参数字典。

        此方法桥接了 SandboxConfig 数据模型和 docker-py 的 API，
        避免在 DockerRunner 中散落配置转换逻辑。

        Returns:
            dict: 可以直接解包传给 client.containers.create(**result)
                  的容器参数字典。
        """
        params: dict = {}

        # ── 资源限制 ──
        # Docker 的 CPU 限制基于 CFS scheduler：cpu_quota / cpu_period = CPU 核心数
        params["cpu_period"] = 100000         # CFS 周期（100ms = 100000μs）
        params["cpu_quota"] = int(self.cpu_limit * 100000)  # 1.0 核 = 100000μs
        params["mem_limit"] = self.memory_limit     # 内存硬上限
        params["memswap_limit"] = self.memory_swap   # Swap 限制

        # ── 安全配置 ──
        params["read_only"] = self.read_only_root    # 根文件只读
        params["network_mode"] = self.network_mode    # 网络隔离
        params["no_new_privileges"] = self.no_new_privileges  # 禁止提权

        # Docker SDK 的 cap_drop 参数
        if self.drop_capabilities:
            params["cap_drop"] = self.drop_capabilities

        # ── 容器生命周期 ──
        params["auto_remove"] = True                 # 容器退出后自动删除
        params["detach"] = True                      # 后台运行（需要 wait 等待结果）

        return params

    def to_security_summary(self) -> str:
        """
        生成安全配置的文本摘要（用于注入 LLM 上下文或日志）。

        让 Agent 了解当前沙箱的限制，避免生成超出限制范围的代码。
        """
        lines = [
            "## Sandbox Security Configuration",
            f"  CPU Limit:           {self.cpu_limit} core(s)",
            f"  Memory Limit:        {self.memory_limit}",
            f"  Swap:                {'Disabled' if self.memory_swap == '0m' else self.memory_swap}",
            f"  Network:             {'Disabled' if self.network_disabled else 'Enabled'}",
            f"  Root FS:             {'Read-only' if self.read_only_root else 'Writable'}",
            f"  Timeout:             {self.timeout_seconds}s",
            f"  Privilege Escalation: {'Blocked' if self.no_new_privileges else 'Allowed'}",
            f"  Capabilities:        {'All dropped' if 'ALL' in self.drop_capabilities else ', '.join(self.drop_capabilities)}",
            f"  Working Dir:         {self.working_dir}",
            "",
            "## Implications for Code Execution",
            "- Network calls (HTTP, sockets) WILL FAIL (network disabled).",
            "- Cannot modify system files outside /workspace (read-only root).",
            "- Cannot install system packages (no root, read-only filesystem).",
            "- pip install --user packages IS allowed (user packages go to home dir).",
            "- Maximum execution time per command: 60 seconds.",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# ExecutionResult — 容器命令执行结果
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionResult:
    """
    Docker 容器命令执行的结果。

    每次 DockerRunner.run_command() 返回一个 ExecutionResult，
    包含命令的退出码、标准输出、标准错误、耗时和超时标记。

    字段：
      exit_code       — 命令退出码（0=成功，-1=Docker 异常）
      stdout          — 标准输出文本
      stderr          — 标准错误文本
      elapsed_seconds — 执行耗时（秒）
      timed_out       — 是否因超时被杀
      error_message   — Docker SDK 异常的错误描述
    """
    exit_code: int                       # 退出码：0=成功，非0=失败，-1=Docker异常
    stdout: str                          # 标准输出（UTF-8 decode，不可解码字节替换）
    stderr: str                          # 标准错误输出
    elapsed_seconds: float               # 使用 time.time() 计时的实际耗时
    timed_out: bool = False              # Docker wait() timeout → 容器被强制 kill
    error_message: str | None = None     # Docker SDK 抛出的异常描述

    @property
    def succeeded(self) -> bool:
        """命令是否成功执行（exit_code == 0 且未超时）。"""
        return self.exit_code == 0 and not self.timed_out

    @property
    def combined_output(self) -> str:
        """合并 stdout 和 stderr（用于日志/LLM 上下文）。"""
        parts = [self.stdout]
        if self.stderr:
            parts.append(f"\n[STDERR]\n{self.stderr}")
        return "".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# TestResult — pytest 测试结果聚合
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    """
    pytest 测试运行的结果摘要。

    PytestRunner.run_tests() 返回此类型。包含测试总数、通过数、
    失败数和错误数的计数，以及全部通过的便捷判断。

    Example:
      TestResult(total=10, passed=8, failed=1, errors=1)
      → all_passed = False
    """
    total: int = 0                       # 测试总数（passed + failed + errors）
    passed: int = 0                      # 通过的测试数
    failed: int = 0                      # 断言失败的测试数（AssertionError）
    errors: int = 0                      # 运行时错误的测试数（ImportError, NameError等）

    @property
    def all_passed(self) -> bool:
        """所有测试是否全部通过。面试时用此属性判断 Fix Loop 是否终止。"""
        return self.failed == 0 and self.errors == 0

    @property
    def pass_rate(self) -> float:
        """通过率（0.0 ~ 1.0）。total=0 时返回 0.0。"""
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    def to_summary(self) -> str:
        """生成单行测试摘要文本。"""
        if self.all_passed:
            return f"✅ All {self.total} test(s) passed."
        parts = [f"❌ {self.passed}/{self.total} passed"]
        if self.failed:
            parts.append(f"{self.failed} failed")
        if self.errors:
            parts.append(f"{self.errors} errors")
        return ", ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# TestFailure — 单个测试失败详情
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestFailure:
    """
    单个测试失败的详细结构化信息。

    ErrorParser.parse(stderr) 从 pytest 的 traceback 中提取这些信息。
    每条 TestFailure 对应一个独立的测试失败。

    为什么需要结构化？
      - 文本 traceback 有几百行，LLM 直接看难以定位根因
      - 结构化后可以按错误类型分类，不同类有不同的修复策略
      - error_type 决定 Fix Loop 的修复方向（见 ErrorParser.categorize_error）

    Example:
      TestFailure(
          test_name="test_add",
          error_type="AssertionError",
          error_message="assert 4 == 5",
          file_path="test_calc.py",
          line_number=15,
          traceback="...full traceback text...",
      )
    """
    test_name: str                       # 失败的测试函数名（如 "test_add"）
    error_type: str                      # 异常类型（"AssertionError" | "NameError" | "ImportError" | ...）
    error_message: str                   # 异常消息（如 "assert 4 == 5"）
    file_path: str                       # 错误所在文件路径
    line_number: int | None = None       # 错误所在行号（部分错误如 ImportError 可能无行号）
    traceback: str = ""                  # 完整 traceback 文本（供 LLM 分析细节）


# ═══════════════════════════════════════════════════════════════════════════════
# FixLoopState — 自动修复循环状态
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FixLoopState:
    """
    自动修复循环（Fix Loop）的状态机。

    每次测试失败 → Agent 修复 → 重新测试的循环由 FixLoopState 跟踪。
    它决定：是否继续尝试、是否升级到人工介入、总共尝试了多少次。

    状态流转（面试时画这个图）：
      START → [Run Tests] → All Pass? → YES → SUCCESS
                  │                         │
                  └── NO ──→ [LLM Fix] ──→ [Apply Fix] ──→ [Run Tests]
                                                              │
                                                  MAX reached? ──→ ESCALATE

    Example:
      state = FixLoopState(max_attempts=3)
      while state.should_continue():
          test_result = run_tests()
          if test_result.all_passed:
              break
          fix_and_apply()
          state.current_attempt += 1
      if state.should_escalate():
          notify_user()
    """
    max_attempts: int = 3                # 最多尝试次数（默认 3）
    current_attempt: int = 0             # 当前尝试次数（0-based）
    test_results: list[TestResult] = field(default_factory=list)
    # 每轮测试的结果列表（按尝试顺序，索引 0 = 第一次测试的结果）

    fix_history: list[dict] = field(default_factory=list)
    # 每轮修复的详细记录：
    #   [{"attempt": 1, "diff": "...", "test_result": TestResult, "errors": [...]}]

    def should_continue(self) -> bool:
        """
        是否应该继续尝试修复。

        使用条件：
          current_attempt < max_attempts → 还可以尝试
          current_attempt >= max_attempts → 停止循环

        这是 while 循环的主条件：
          while state.should_continue():
              ...
        """
        return self.current_attempt < self.max_attempts

    def should_escalate(self) -> bool:
        """
        是否应该升级到人工介入。

        连续尝试 max_attempts 次仍未通过 → 说明问题可能不是简单的 bug 修复：
          - 设计层面有根本问题
          - LLM 一直生成同一错误修复
          - 任务需求本身不合理

        此时应由 Human-in-the-Loop（Phase 9）或 BadCaseAnalyzer（Step 5.5）接管。
        """
        return self.current_attempt >= self.max_attempts
