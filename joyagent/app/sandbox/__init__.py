"""
Phase 5: Docker Sandbox + Auto Testing — 安全代码执行与自动化测试闭环。

模块结构：
  security.py      — Step 1+2: SandboxConfig 六层安全配置 + 执行结果/测试结果数据模型
  docker_runner.py — Step 2:   DockerRunner 容器管理（创建/运行/销毁）
  pytest_runner.py — Step 3:   PytestRunner 在沙箱中运行 pytest
  error_parser.py  — Step 4:   ErrorParser 结构化解析 Python traceback
  fix_loop.py      — Step 5:   FixLoop 自动修复循环
  bad_case_analyzer.py — 5.5:  Bad Case 收集与分析

安全模型（六层防御，由外到内）：
  Level 1: 容器隔离 — 进程/文件系统/Mount namespace 隔离
  Level 2: 资源限制 — CPU 限额 + 内存上限 + 禁止 swap
  Level 3: 网络隔离 — network_mode=none（容器内无网络接口）
  Level 4: 文件系统 — 根文件只读 + 只挂载必要的工作目录
  Level 5: 权限限制 — no_new_privileges + 丢弃所有 Linux capabilities
  Level 6: 超时控制 — 每次执行强制超时 + 容器自动销毁
"""

from app.sandbox.security import (
    SandboxConfig,
    ExecutionResult,
    TestResult,
    TestFailure,
    FixLoopState,
)

from app.sandbox.docker_runner import (
    DockerRunner,
    create_default_runner,
)

from app.sandbox.pytest_runner import PytestRunner

__all__ = [
    "SandboxConfig",
    "ExecutionResult",
    "TestResult",
    "TestFailure",
    "FixLoopState",
    "DockerRunner",
    "create_default_runner",
    "PytestRunner",
]
