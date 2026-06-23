# ── 标准库导入 ──
import asyncio                     # 异步子进程管理（create_subprocess_shell）
import subprocess                  # 子进程常量（PIPE）
import os                          # 路径检查和 working_dir 验证

# ── 项目内导入 ──
from app.tools.base import BaseTool, ToolResult  # 工具基类和统一返回格式


class ShellExecuteTool(BaseTool):
    """
    Phase 2: Shell 命令执行工具。
    使用 asyncio.create_subprocess_shell 实现异步非阻塞执行。

    ⚠️ 安全设计债务（Phase 5 偿还）：
      - 当前无命令白名单/黑名单
      - 无资源限制（CPU/内存/磁盘）
      - 无网络访问控制
      - 无超时强制（Phase 5 加 Docker Sandbox + 超时 kill）
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
        Phase 9: 接入用户确认流程，每次执行前弹出预览。
        """
        return True

    # ─── 核心执行逻辑 ───
    async def execute(self, command: str, working_dir: str = None, **kwargs) -> ToolResult:
        """
        异步执行 shell 命令，捕获 stdout 和 stderr。

        参数映射机制：
          ToolRegistry.execute(name="execute_shell", command="ls", working_dir="/tmp")
          → self.execute(command="ls", working_dir="/tmp")

        为什么用 asyncio.create_subprocess_shell 而不是 subprocess.run？
          - 异步非阻塞：Agent 可以在等待命令完成时处理其他任务
          - 同时捕获 stdout + stderr：PIPE 模式两个流独立
          - 可扩展：Phase 5 可以通过 .kill() 实现超时强制终止
        """
        # 1. 校验 working_dir（如果指定了）
        if working_dir and not os.path.isdir(working_dir):
            return ToolResult(
                success=False,
                message=f"Error: Working directory '{working_dir}' does not exist.",
                error=f"DirectoryNotFound: {working_dir}",
            )

        try:
            # 2. 启动异步子进程
            process = await asyncio.create_subprocess_shell(
                command,                             # 要执行的命令字符串（支持管道和重定向）
                stdout=subprocess.PIPE,              # 捕获标准输出到管道
                stderr=subprocess.PIPE,              # 捕获标准错误到管道
                cwd=working_dir,                     # 指定工作目录（None 表示使用当前目录）
            )

            # 3. 等待子进程结束并读取输出（带 30 秒超时）
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),           # communicate() 返回 (stdout, stderr) 字节元组
                    timeout=30.0,                    # 30 秒后抛出 TimeoutError
                )
            except asyncio.TimeoutError:
                # 超时 → 杀死子进程，返回错误
                process.kill()                       # 发送 SIGKILL（Windows 上是 TerminateProcess）
                await process.wait()                 # 等待进程彻底结束（回收僵尸进程）
                return ToolResult(
                    success=False,
                    message=f"Error: Command timed out after 30 seconds: '{command}'",
                    error=f"TimeoutError: {command}",
                )

            # 4. 解码输出（用 errors="replace" 避免非 UTF-8 字节导致崩溃）
            output = stdout_bytes.decode("utf-8", errors="replace")
            # errors="replace": 遇到无法解码的字节用 U+FFFD 替换，而不是抛 UnicodeDecodeError

            # 5. 如果 stderr 有内容，附加到输出末尾
            if stderr_bytes:
                stderr_text = stderr_bytes.decode("utf-8", errors="replace")
                output += "\n[STDERR]\n" + stderr_text  # 标记区分 stdout 和 stderr

            # 6. 截断过长输出——防止撑爆 LLM 上下文窗口
            was_truncated = len(output) > 5000
            truncated_output = output[:5000]          # 最多返回 5000 字符

            # 7. 构建返回结果
            return ToolResult(
                success=process.returncode == 0,      # returncode=0 表示命令成功退出
                message=truncated_output,             # 给 LLM 看的输出文本
                metadata={
                    "command": command,               #   - 原始命令
                    "working_dir": working_dir,       #   - 工作目录
                    "exit_code": process.returncode,  #   - 进程退出码（0=成功，非0=失败）
                    "truncated": was_truncated,       #   - 输出是否被截断
                    "original_length": len(output),   #   - 原始输出长度
                },
            )

        except Exception as e:
            # 兜底：子进程启动失败（如命令语法错误）或其他未预期异常
            return ToolResult(
                success=False,
                message=f"Error: Failed to execute '{command}': {e}",
                error=str(e),
            )
