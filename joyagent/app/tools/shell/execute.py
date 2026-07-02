# ── 标准库导入 ──
import asyncio                     # 异步子进程管理（create_subprocess_shell）
import subprocess                  # 子进程常量（PIPE）
import os                          # 路径检查和 working_dir 验证

# ── 项目内导入 ──
from app.tools.base import BaseTool, ToolResult  # 工具基类和统一返回格式

# ── Phase 5: Docker Sandbox 集成 ──
from app.sandbox.docker_runner import DockerRunner  # Docker 沙箱安全执行器
from app.sandbox.security import SandboxConfig       # 六层安全防御配置


# ── 模块级缓存的 DockerRunner 实例 ──
# 只创建一次，后续调用复用同一个客户端连接
_runner_cache: dict = {}                             # {"key": DockerRunner}


def _get_sandbox_runner(working_dir: str = None) -> DockerRunner | None:
    """
    获取可用的 Docker Sandbox 执行器。

    如果 Docker 可用 → 返回配置好的 DockerRunner。
    如果 Docker 不可用 → 返回 None，调用方降级到宿主机 subprocess。

    缓存策略：
      同一 mount_path 的 runner 只创建一次，避免反复 docker.from_env()。

    mount_path 解析：
      - Docker 容器内运行时，容器内路径（如 /app）≠ 宿主机路径。
      - 通过环境变量 SANDBOX_HOST_MOUNT_PATH 指定宿主机上的真实项目路径。
      - 未设置时，默认使用当前工作目录（适用于直接在宿主机运行）。
    """
    wd = working_dir or os.getcwd()

    # Docker 容器内运行时，宿主机路径通过环境变量指定
    host_mount = os.environ.get("SANDBOX_HOST_MOUNT_PATH", "")
    if host_mount:
        mount_path = host_mount
    else:
        mount_path = os.path.abspath(wd)

    cache_key = mount_path

    if cache_key not in _runner_cache:
        config = SandboxConfig(mount_path=mount_path)
        runner = DockerRunner(config)
        _runner_cache[cache_key] = runner

    runner = _runner_cache[cache_key]
    if runner.is_available:
        return runner
    return None


class ShellExecuteTool(BaseTool):
    """
    Phase 2 → Phase 5: Shell 命令执行工具。

    执行策略（自动选择）：
      Phase 5 模式（Docker 可用）→ DockerRunner 沙箱隔离执行
        ✅ 六层安全防御（容器隔离/资源限制/网络隔离/只读文件/权限限制/超时控制）
        ✅ 每次执行创建新容器 → 执行 → 销毁（无状态、无污染）
        ✅ 非 root 用户执行（user="sandbox"）

      Phase 2 降级模式（Docker 不可用）→ asyncio.create_subprocess_shell
        ⚠️ 无安全隔离——仅在开发阶段使用
        ⚠️ 命令直接在宿主机执行

    为什么不在 Agent 层新增独立工具，而是修改现有工具？
      - Sandbox 是基础设施，不是可选功能（类比：安全带默认扣上）
      - Agent 不应该关心“是用 Docker 还是 subprocess”——它只管执行命令
      - 唯一命令执行通道 = 无法绕过沙箱 = 安全设计更可靠
      - Claude Code 也是所有 Bash 走同一个工具入口，沙箱是透明实现细节
    """

    # ─── 工具标识 ───
    name = "execute_shell"                         # LLM 调用的工具名

    description = (
        "Execute a shell command and return its stdout and stderr output. "
        "Use this to list files, check Python versions, install packages, "
        "run scripts, or any other command-line operation."
    )

    # ─── JSON Schema 定义 ───
    @property
    def input_schema(self) -> dict:
        """
        输入参数：
          - command:    要执行的 shell 命令（必填）
          - working_dir: 命令执行的工作目录（可选，默认当前目录）
        """
        return {
            "type": "object",                      # JSON Schema 根类型
            "properties": {
                "command": {
                    "type": "string",              # 命令字符串，如 "ls -la *.py"
                    "description": "The shell command to execute. Can include pipes and redirections.",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command. Defaults to current directory.",
                },
            },
            "required": ["command"],               # 只有 command 是必填的
        }

    # ─── 安全标记 ───
    @property
    def is_dangerous(self) -> bool:
        """
        Shell 命令是最高危操作——可以删除文件、安装恶意软件、泄露数据。
        Phase 2: 只打黄色警告日志。
        Phase 5: Docker 沙箱提供六层隔离防护（容器/资源/网络/文件/权限/超时）。
        Phase 9: 接入用户确认流程，每次执行前弹出预览。
        """
        return True

    # ─── 核心执行逻辑 ───
    async def execute(self, command: str, working_dir: str = None,
                      **kwargs) -> ToolResult:
        """
        执行 shell 命令（自动选择 Docker Sandbox 或宿主机 subprocess）。

        Args:
            command:     要执行的 shell 命令（支持管道和重定向）
            working_dir: 命令执行的工作目录（可选，默认当前目录）

        Returns:
            ToolResult — success=True 表示 exit_code==0 且未超时
        """
        # ── 0. 校验 working_dir ──
        if working_dir and not os.path.isdir(working_dir):
            return ToolResult(
                success=False,
                message=f"Error: Working directory '{working_dir}' does not exist.",
                error=f"DirectoryNotFound: {working_dir}",
            )

        # ── 1. 获取沙箱执行器 ──
        sandbox_runner = _get_sandbox_runner(working_dir)

        if sandbox_runner is not None:
            # ── Phase 5: Docker Sandbox 执行 ──────────────
            return await self._execute_in_sandbox(
                command, working_dir, sandbox_runner
            )
        else:
            # ── Phase 2 降级: 宿主机 subprocess ──────────
            return await self._execute_on_host(command, working_dir)

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Docker Sandbox 路径
    # ═══════════════════════════════════════════════════════════

    async def _execute_in_sandbox(
        self,
        command: str,
        working_dir: str | None,
        runner: DockerRunner,
    ) -> ToolResult:
        """
        在 Docker 沙箱中执行命令。

        容器内执行环境：
          - 用户: sandbox（非 root）
          - 工作目录: /workspace（宿主机项目目录 bind mount）
          - 网络: none（无外部连接）
          - 根文件系统: 只读（/workspace 除外）
          - CPU: 1 核, 内存: 512M
          - 超时: 60 秒
        """
        # 映射 working_dir: 宿主机路径 → 容器内 /workspace
        exec_result = await runner.run_command(
            command,
            working_dir="/workspace",              # 容器内工作目录固定为 /workspace
        )

        # Docker 连接失败或镜像不存在 → 降级到宿主机执行
        if exec_result.exit_code == -1 and exec_result.error_message:
            err = exec_result.error_message

            # ── Image not found → 打印构建提示后降级 ──
            if "ImageNotFound" in err or "not found" in err.lower():
                print(
                    f"  \033[33m[!] Sandbox image not found. "
                    f"Falling back to host execution.\033[0m\n"
                    f"  \033[33m    Build it: docker build -t "
                    f"joyagent-sandbox:latest "
                    f"-f sandbox_config/Dockerfile .\033[0m"
                )
                return await self._execute_on_host(command, working_dir)

            # ── Docker daemon 不可用 → 降级 ──
            if "Docker unavailable" in err:
                print(
                    f"  \033[33m[!] Docker unavailable — "
                    f"falling back to host execution.\033[0m"
                )
                return await self._execute_on_host(command, working_dir)

            # ── 其他 Docker 错误 → 返回错误 ──
            return ToolResult(
                success=False,
                message=f"Error: [Sandbox] {err[:300]}",
                error=f"DockerError: {err}",
            )

        # ── 成功：将 ExecutionResult 转为 ToolResult ──
        output = exec_result.combined_output

        was_truncated = len(output) > 5000
        truncated_output = output[:5000]

        return ToolResult(
            success=exec_result.succeeded,          # exit_code==0 且未超时
            message=truncated_output,
            metadata={
                "command": command,
                "working_dir": working_dir,
                "exit_code": exec_result.exit_code,
                "truncated": was_truncated,
                "original_length": len(output),
                "execution_mode": "docker_sandbox",  # ⬅ 标记执行模式
                "container_elapsed_s": round(exec_result.elapsed_seconds, 3),
                "timed_out": exec_result.timed_out,
            },
        )

    # ═══════════════════════════════════════════════════════════
    # Phase 2 降级: 宿主机 subprocess 路径
    # ═══════════════════════════════════════════════════════════

    async def _execute_on_host(
        self,
        command: str,
        working_dir: str | None,
    ) -> ToolResult:
        """
        Phase 2 降级模式：直接在宿主机执行命令。

        使用 asyncio.create_subprocess_shell 实现异步非阻塞执行，
        带 30 秒超时保护和输出截断。

        ⚠️ 安全性警告：
          此模式无任何隔离保护——Agent 生成的代码直接在宿主机运行。
          仅在以下情况使用：
            1. Docker 未安装/不可用（开发阶段）
            2. 沙箱镜像未构建
            3. DockerRunner 主动降级（如 ImageNotFound）
        """
        try:
            # 1. 启动异步子进程
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir,
            )

            # 2. 等待子进程结束并读取输出（带 30 秒超时）
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    success=False,
                    message=f"Error: Command timed out after 30 seconds: '{command}'",
                    error=f"TimeoutError: {command}",
                )

            # 3. 解码输出
            output = stdout_bytes.decode("utf-8", errors="replace")
            if stderr_bytes:
                stderr_text = stderr_bytes.decode("utf-8", errors="replace")
                output += "\n[STDERR]\n" + stderr_text

            # 4. 截断过长输出
            was_truncated = len(output) > 5000
            truncated_output = output[:5000]

            return ToolResult(
                success=process.returncode == 0,
                message=truncated_output,
                metadata={
                    "command": command,
                    "working_dir": working_dir,
                    "exit_code": process.returncode,
                    "truncated": was_truncated,
                    "original_length": len(output),
                    "execution_mode": "host_subprocess",  # ⬅ 标记降级模式
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error: Failed to execute '{command}': {e}",
                error=str(e),
            )
