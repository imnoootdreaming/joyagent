"""
Phase 5 Step 2: Docker Runner — Docker 沙箱安全执行器（⭐ 核心）。

DockerRunner 是 Phase 5 的核心组件，负责在 Docker 容器中安全地执行
Agent 生成的代码。每个命令在独立容器中运行，执行完立即销毁，确保：
  1. 进程隔离 — 恶意代码无法影响宿主机
  2. 资源限制 — CPU/内存有上限，防止 DoS
  3. 网络隔离 — 容器内无网络，防止数据泄露
  4. 文件隔离 — 只挂载必要的工作目录
  5. 无状态 — 每次执行使用全新容器（无历史污染）

安全设计原则（面试必问）：
  - 每次执行 = 新容器创建 → 执行 → 销毁（不留痕迹）
  - try/finally 确保容器一定被清理（即使异常也销毁）
  - 超时控制：Docker wait(timeout=N) + 过期 SIGKILL
  - Docker 不可用时优雅降级（返回错误而非崩溃）

面试要点：
  面试官："如果 Agent 执行 'rm -rf /'，你的系统怎么防护？"
  答案：Docker 隔离 — 操作在容器内，容器根文件是只读的 (read_only=True)，
  rm 无法删除宿主机文件。即使删了容器内的文件，容器也不共享宿主机文件系统。
  唯一可读写的是通过 bind mount 挂载的 /workspace 目录。
"""

# ── Python 标准库 ──
import asyncio                         # 异步运行 + 线程池执行器
import os                              # 路径操作和文件存在检查
import time                            # 执行耗时计算
from typing import Optional            # 可选类型标注

# ── 第三方库 ──
import docker                          # Docker SDK for Python (docker-py)
from docker import errors as docker_errors  # Docker 专用异常类型
from docker.types import Mount         # 文件挂载配置（bind mount / tmpfs）

# ── 项目内导入 ──
from app.sandbox.security import SandboxConfig, ExecutionResult
# SandboxConfig:   六层安全防御配置（Step 1）
# ExecutionResult: 容器命令执行结果数据模型


# ═══════════════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════════════

# 命令输出最大长度（字符）—— 防止大文件/无限输出撑爆 LLM 上下文
MAX_OUTPUT_CHARS = 10_000

# 容器日志读取超时（秒）
LOG_READ_TIMEOUT = 5

# Docker 健康检查超时（秒）
DOCKER_PING_TIMEOUT = 3


# ═══════════════════════════════════════════════════════════════════════════════
# DockerRunner
# ═══════════════════════════════════════════════════════════════════════════════

class DockerRunner:
    """
    Docker 沙箱安全执行器（Phase 5 核心）。

    职责：
      1. 为每次命令执行创建独立的 Docker 容器
      2. 应用 SandboxConfig 的全部六层安全限制
      3. 挂载必要的文件目录（最小权限原则）
      4. 等待命令执行完成（带超时控制）
      5. 收集 stdout/stderr 并返回结构化结果
      6. 确保容器被销毁（try/finally + force remove）

    线程模型：
      Docker SDK (docker-py) 是同步的，但 Agent 是异步的 (asyncio)。
      通过 asyncio.to_thread() 将同步 Docker 操作放入线程池执行，
      避免阻塞事件循环。

    使用方式：
      config = SandboxConfig(
          mount_path="/path/to/repo",
          timeout_seconds=60,
      )
      runner = DockerRunner(config)
      result = await runner.run_command("python -m pytest test_calc.py -v")
      if result.succeeded:
          print(result.stdout)
    """

    def __init__(self, config: SandboxConfig = None):
        """
        初始化 Docker Runner。

        Args:
            config: SandboxConfig 安全配置。为 None 时使用默认安全配置。
        """
        self.config = config or SandboxConfig()

        # ── 1. 尝试连接 Docker daemon ──
        self._client: docker.DockerClient | None = None
        self._docker_available: bool = False
        self._docker_error: str = ""       # 连接失败的原因

        try:
            # docker.from_env() 从环境变量 / docker config 自动获取连接参数
            self._client = docker.from_env(timeout=DOCKER_PING_TIMEOUT)
            # 快速 ping 验证 daemon 是否可达
            self._client.ping()
            self._docker_available = True
        except docker_errors.DockerException as e:
            self._docker_error = f"Docker unavailable: {e}"
            print(f"  \033[33m[!] {self._docker_error}\033[0m")
            print(f"  \033[33m[!] DockerRunner will return error results. "
                  f"Install Docker to enable sandbox execution.\033[0m")
        except Exception as e:
            self._docker_error = f"Docker unavailable: {e}"
            print(f"  \033[33m[!] {self._docker_error}\033[0m")

    # ── 公共 API ──────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Docker 是否可用（daemon 可达且客户端可用）。"""
        return self._docker_available and self._client is not None

    async def run_command(
        self,
        command: str,
        working_dir: str | None = None,
    ) -> ExecutionResult:
        """
        在 Docker 沙箱中安全执行一个命令。

        完整执行流程：
          1. Docker 可用性检查
          2. 构建容器安全参数（六层防御）
          3. 配置文件挂载（仅工作目录）
          4. 创建容器 → 启动 → 等待 → 收集日志 → 销毁
          5. 返回结构化 ExecutionResult

        每次调用创建全新容器，执行完立即销毁——无状态、无污染。

        Args:
            command:     要执行的 shell 命令（如 "python test_calc.py"）
            working_dir: 容器内的工作目录（默认 /workspace）

        Returns:
            ExecutionResult — 包含 exit_code, stdout, stderr, elapsed, timed_out
            即使 Docker 不可用也返回 ExecutionResult（含错误信息）

        Example:
            runner = DockerRunner(config)
            result = await runner.run_command("python -m pytest tests/ -v")
            print(f"Exit: {result.exit_code}, Time: {result.elapsed_seconds:.1f}s")
        """
        start_time = time.time()           # 计时起点
        wd = working_dir or self.config.working_dir  # 默认 /workspace

        # ── 0. Docker 不可用 → 直接返回错误 ──
        if not self.is_available:
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=self._docker_error,
                elapsed_seconds=0,
                error_message=self._docker_error,
            )

        # ── 1. 构建容器参数 ──────────────────────────────────
        container_kwargs = self._build_container_kwargs(command, wd)

        container = None                   # 容器引用（用于 finally 清理）

        try:
            # ── 2. 创建容器（在线程池中执行同步 Docker API） ──
            container = await asyncio.to_thread(
                self._client.containers.create,
                **container_kwargs,
            )

            # ── 3. 启动容器 ──
            await asyncio.to_thread(container.start)

            # ── 4. 等待执行完成（带超时） ──
            try:
                exit_result = await asyncio.to_thread(
                    container.wait,
                    timeout=self.config.timeout_seconds,
                )
                exit_code = exit_result.get("StatusCode", -1)
                timed_out = False
            except asyncio.TimeoutError:
                # Docker wait timeout → 容器被 asyncio 终止
                # 注意: 这里不是 Docker 的超时，是 asyncio.wait_for 的超时
                timed_out = True
                exit_code = -1

            except docker_errors.APIError as e:
                # Docker API 层面的错误
                return ExecutionResult(
                    exit_code=-1,
                    stdout="",
                    stderr=str(e),
                    elapsed_seconds=time.time() - start_time,
                    timed_out=False,
                    error_message=str(e),
                )

            # ── 5. 收集日志 ──
            try:
                stdout_raw = await asyncio.to_thread(
                    container.logs, stdout=True, stderr=False
                )
                stderr_raw = await asyncio.to_thread(
                    container.logs, stdout=False, stderr=True
                )
                stdout = self._decode_logs(stdout_raw)[:MAX_OUTPUT_CHARS]
                stderr = self._decode_logs(stderr_raw)[:MAX_OUTPUT_CHARS]
            except Exception:
                # 日志收集失败（容器已被销毁或其他异常）
                stdout = ""
                stderr = "Failed to collect container logs."

            # ── 6. 如果超时 → 附加说明 ──
            if timed_out:
                timeout_msg = (
                    f"Execution timed out after "
                    f"{self.config.timeout_seconds}s."
                )
                if stderr:
                    stderr = timeout_msg + "\n" + stderr
                else:
                    stderr = timeout_msg

            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                elapsed_seconds=time.time() - start_time,
                timed_out=timed_out,
            )

        except docker_errors.ImageNotFound as e:
            # 镜像不存在 → 友好的错误提示（告知用户如何构建）
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=(
                    f"Docker image '{self.config.image}' not found. "
                    f"Build it with: docker build -t {self.config.image} "
                    f"-f sandbox_config/Dockerfile ."
                ),
                elapsed_seconds=time.time() - start_time,
                error_message=str(e),
            )

        except docker_errors.APIError as e:
            # Docker daemon 返回的错误（如资源不足、权限不足）
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                elapsed_seconds=time.time() - start_time,
                error_message=str(e),
            )

        except Exception as e:
            # 未预期的异常（兜底）
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                elapsed_seconds=time.time() - start_time,
                timed_out="timeout" in str(e).lower(),
                error_message=str(e),
            )

        finally:
            # ── 7. 清理容器（无论如何都执行） ─────────────────
            await self._cleanup_container(container)

    async def run_python(
        self,
        code: str,
        working_dir: str | None = None,
    ) -> ExecutionResult:
        """
        在 Docker 沙箱中执行一段 Python 代码。

        将代码写入临时文件 → 执行 python <file> → 删除临时文件。
        适合 Agent 生成单文件脚本后快速测试。

        Args:
            code:        Python 源代码字符串
            working_dir: 容器内工作目录

        Returns:
            ExecutionResult

        Example:
            code = '''
        import json
        data = {"name": "test", "value": 42}
        print(json.dumps(data))
        '''
            result = await runner.run_python(code)
        """
        # 将代码写入工作目录的临时文件
        tmp_filename = "_joyagent_tmp_exec.py"
        wd = working_dir or self.config.working_dir
        code_path = f"{wd}/{tmp_filename}"

        # 先写入代码文件（使用 echo 写入简单代码，或用 base64 编码复杂代码）
        import base64
        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")

        # 在容器中执行：解码 → 写入文件 → 运行 → 清理
        command = (
            f"python3 -c \"import base64; "
            f"open('{tmp_filename}', 'w').write("
            f"base64.b64decode('{encoded}').decode())\" "
            f"&& python3 {tmp_filename} "
            f"&& rm {tmp_filename}"
        )

        return await self.run_command(command, working_dir=wd)

    async def ensure_image(self) -> bool:
        """
        确保沙箱镜像存在。

        如果镜像存在于本地 Docker 仓库，返回 True。
        如果不存在，打印构建命令提示并返回 False。
        （不会自动执行 docker build——构建应由用户手动完成。）

        Returns:
            True → 镜像可用
        """
        if not self.is_available:
            return False

        try:
            await asyncio.to_thread(
                self._client.images.get,
                self.config.image,
            )
            return True                    # 镜像存在
        except docker_errors.ImageNotFound:
            print(
                f"  \033[33m[!] Image '{self.config.image}' not found.\033[0m\n"
                f"  \033[33m    Build it: docker build -t "
                f"{self.config.image} "
                f"-f sandbox_config/Dockerfile .\033[0m"
            )
            return False
        except Exception:
            return False

    # ── 私有辅助方法 ──────────────────────────────────────────

    def _build_container_kwargs(
        self,
        command: str,
        working_dir: str,
    ) -> dict:
        """
        构建传给 client.containers.create(**kwargs) 的参数字典。

        整合三个来源的配置：
          1. SandboxConfig.to_docker_params() — 安全 + 资源限制
          2. 执行特定参数 — image / command / working_dir
          3. 文件挂载 — 仅挂载工作目录（最小权限）

        Args:
            command:     要执行的命令
            working_dir: 容器内工作目录
        Returns:
            完整的容器创建参数字典
        """
        # ── 获取 SandboxConfig 的安全参数 ──
        kwargs = self.config.to_docker_params()

        # ── 覆盖/补充执行特定参数 ──
        kwargs["image"] = self.config.image      # 使用的镜像名
        kwargs["command"] = f"/bin/bash -c '{self._escape_command(command)}'"
        kwargs["working_dir"] = working_dir       # 容器内工作目录

        # user 设为 sandbox 用户（非 root）
        kwargs["user"] = "sandbox"

        # ── 文件挂载（仅挂载必要的工作目录） ──
        if self.config.mount_path:
            kwargs["mounts"] = [
                Mount(
                    target=working_dir,          # 容器内路径
                    source=self.config.mount_path,  # 宿主机路径
                    type="bind",                 # bind mount（直接映射）
                    read_only=False,             # 工作目录可写
                )
            ]

        return kwargs

    @staticmethod
    def _escape_command(command: str) -> str:
        """
        转义 shell 命令中的特殊字符。

        命令作为 /bin/bash -c '...' 的参数传入容器。
        command 中的单引号会导致 shell 提前结束引号。

        策略：将 ' 替换为 '\''（结束引号 → 转义单引号 → 重启引号）

        Example:
          "echo 'hello'" → "echo '\\''hello'\\''" (但保持可读)
          实际上是: echo '\''hello'\''  → shell 解析为 echo 'hello'
        """
        return command.replace("'", "'\\''")

    @staticmethod
    def _decode_logs(raw: bytes) -> str:
        """Docker 日志字节流 → UTF-8 字符串。不可解码字节用 U+FFFD 替换。"""
        return raw.decode("utf-8", errors="replace")

    async def _cleanup_container(self, container) -> None:
        """
        清理 Docker 容器（强制移除）。

        此方法在 run_command 的 finally 块中调用，
        无论执行成功/失败/超时，都确保容器被移除。

        Args:
            container: Docker 容器对象（可为 None）
        """
        if container is None:
            return

        try:
            # 检查容器是否还存在（可能已被 auto_remove 清理）
            await asyncio.to_thread(
                container.remove, force=True    # force=True: 即使运行中也强制移除
            )
        except docker_errors.NotFound:
            pass                               # 容器已被清理（auto_remove 生效）
        except docker_errors.APIError:
            # 容器可能已经退出/被删，尝试再次 force remove
            try:
                await asyncio.to_thread(container.remove, force=True)
            except Exception:
                pass
        except Exception:
            # 兜底：即使清理失败也不抛异常（不干扰主流程）
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# 便捷工厂函数
# ═══════════════════════════════════════════════════════════════════════════════

def create_default_runner(mount_path: str | None = None) -> DockerRunner:
    """
    创建使用默认安全配置的 DockerRunner。

    适合快速集成——一行代码获得安全的代码执行环境。

    Args:
        mount_path: 要挂载到容器 /workspace 的宿主机路径。
                    None → 当前工作目录。
    Returns:
        配置好的 DockerRunner 实例

    Example:
        runner = create_default_runner(mount_path="/path/to/repo")
        result = await runner.run_command("pytest tests/ -v")
    """
    config = SandboxConfig(
        mount_path=mount_path or os.getcwd(),
    )
    return DockerRunner(config)
