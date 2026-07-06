"""
Phase 9A-4 — Human-in-the-Loop 权限系统

通过 Phase 2 的 ToolHook 机制，在工具执行前拦截危险操作，
实现 AUTO / CONFIRM / DENY 三级权限控制。

设计原理（面试必问）：
  "Agent 安全 = 三层防线：Docker Sandbox（系统层隔离）
   + SafetyCheckHook（关键词拦截）+ PermissionManager（流程层审批）"

权限级别：
  AUTO     → 直接执行（read_file, git_status 等只读操作）
  CONFIRM  → 需要用户确认（write_file, execute_shell 等危险操作）
  DENY     → 直接拒绝（rm -rf, sudo, curl 等明确禁止的模式）

使用方式：
  1. 注册为 ToolHook（和 SafetyCheckHook 共存）：
     tool_registry.register_hook(PermissionManager())

  2. 默认行为：无外部审批者时，CONFIRM 自动拒绝，
     有审批者时，请求审批并等待结果。

  3. 设置审批回调（生产环境接 WebSocket）：
     pm = PermissionManager()
     pm.set_approver(my_ws_approver)

与 Phase 2 SafetyCheckHook 的分工：
  SafetyCheckHook → 关键词黑名单（rm -rf → 直接拒绝）
  PermissionManager → 按工具类型分级（write_file → 需确认）

面试要点：
  Q: "你的 Agent 怎么防止执行危险操作？"
  A: "三层防线：1) Docker Sandbox 系统隔离 2) SafetyCheckHook 关键词拦截
       3) PermissionManager 流程审批。工具按风险分为 AUTO/CONFIRM/DENY 三级，
       CONFRIMM 级操作必须经过用户审批才能执行。"
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

# ═══════════════════════════════════════════════════════════════════════
# 延迟导入 ToolHook —— 避免 app.tools.__init__ 的 docker/chromadb
# 链式导入。PermissionManager 不继承 ToolHook，而是 duck typing
# 实现 on_pre_execute / on_post_execute / on_error 三个方法。
# 在 register_all_tools() 中才通过 ToolHook.register() 绑定。
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# 权限级别
# ═══════════════════════════════════════════════════════════════════════

class PermissionLevel(Enum):
    """三级权限。"""
    AUTO = "auto"          # 自动放行，不询问用户
    CONFIRM = "confirm"    # 需要用户确认
    DENY = "deny"          # 直接拒绝


# ═══════════════════════════════════════════════════════════════════════
# 权限规则
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PermissionRule:
    """
    单个工具的权限规则。

    condition 是可选的 Python 表达式字符串，用于细粒度控制。
    例如：write_file 默认需要确认，但写入 .env 文件直接拒绝。
    """
    tool_name: str
    level: PermissionLevel
    reason: str = ""
    condition: str | None = None  # 如 "path.endswith('.env')"


# ═══════════════════════════════════════════════════════════════════════
# 审批请求/响应
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ApprovalRequest:
    """发给用户的审批请求。"""
    request_id: str
    tool_name: str
    tool_args: dict
    risk_level: str           # "low" | "medium" | "high"
    reason: str
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        """人类可读的审批摘要。"""
        args_str = ", ".join(f"{k}={str(v)[:40]}" for k, v in self.tool_args.items())
        return (
            f"[{self.risk_level.upper()} RISK] {self.tool_name}"
            f"{'(' + args_str + ')' if args_str else ''}"
            f"\n  Reason: {self.reason}"
        )


@dataclass
class ApprovalResponse:
    """用户的审批结果。"""
    request_id: str
    approved: bool
    comment: str = ""


# ── 审批回调类型 ──
Approver = Callable[[ApprovalRequest], Awaitable[ApprovalResponse]]


# ═══════════════════════════════════════════════════════════════════════
# 默认权限规则表
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_RULES: list[PermissionRule] = [
    # ── 只读操作：自动放行 ──
    PermissionRule("read_file",   PermissionLevel.AUTO, "只读操作"),
    PermissionRule("search_code", PermissionLevel.AUTO, "只读操作"),
    PermissionRule("load_repo",   PermissionLevel.AUTO, "只读操作"),
    PermissionRule("analyze_code",PermissionLevel.AUTO, "只读操作"),
    PermissionRule("git_status",  PermissionLevel.AUTO, "只读操作"),
    PermissionRule("git_diff",    PermissionLevel.AUTO, "只读操作"),
    PermissionRule("git_log",     PermissionLevel.AUTO, "只读操作"),
    PermissionRule("remember",    PermissionLevel.AUTO, "记忆操作"),

    # ── 写操作：需要确认 ──
    PermissionRule("write_file",    PermissionLevel.CONFIRM, "写入文件可能修改项目代码"),
    PermissionRule("execute_shell", PermissionLevel.CONFIRM, "Shell 命令执行"),
    PermissionRule("apply_patch",   PermissionLevel.CONFIRM, "应用补丁可能修改多个文件"),
    PermissionRule("git_commit",    PermissionLevel.CONFIRM, "Git 提交是永久操作"),
    PermissionRule("git_branch",    PermissionLevel.CONFIRM, "Git 分支操作"),

    # ── MCP 工具（外部进程）：需要确认 ──
    PermissionRule("filesystem__write_file",       PermissionLevel.CONFIRM,
                   "MCP 文件写入"),
    PermissionRule("filesystem__create_directory", PermissionLevel.CONFIRM,
                   "MCP 目录创建"),
    PermissionRule("filesystem__move_file",        PermissionLevel.CONFIRM,
                   "MCP 文件移动"),

    # ── 高风险 MCP 操作（默认拒绝，需手动加入白名单） ──
    PermissionRule("github__create_issue",     PermissionLevel.CONFIRM,
                   "GitHub Issue 创建"),
    PermissionRule("github__create_pull_request", PermissionLevel.CONFIRM,
                   "GitHub PR 创建"),

    # ── 明确拒绝的操作模式 ──
    PermissionRule("execute_shell", PermissionLevel.DENY,
                   "命令包含危险操作", condition="rm -rf"),
    PermissionRule("execute_shell", PermissionLevel.DENY,
                   "命令包含危险操作", condition="sudo"),
    PermissionRule("execute_shell", PermissionLevel.DENY,
                   "命令包含危险操作", condition="mkfs"),
    PermissionRule("execute_shell", PermissionLevel.DENY,
                   "命令包含危险操作", condition="dd if="),
    PermissionRule("execute_shell", PermissionLevel.DENY,
                   "命令包含危险操作", condition="> /dev/sda"),
]


# ═══════════════════════════════════════════════════════════════════════
# PermissionManager（作为 ToolHook）
# ═══════════════════════════════════════════════════════════════════════

class PermissionManager:
    """
    Human-in-the-Loop 权限管理器 —— duck typing 实现 ToolHook 接口。

    核心逻辑（面试流程图）：
      Agent 要执行工具
          │
          ▼
      check(tool_name, args)
          │
          ├── AUTO ────▶ 直接执行
          │
          ├── CONFIRM ──▶ 有外部审批者？──▶ 请求审批 ──▶ approve/deny
          │               │
          │               └── 无外部审批者 → 返回自动拒绝
          │
          └── DENY ────▶ 阻止执行，返回 ToolResult(error=...)

    使用方式：:

        pm = PermissionManager()
        tool_registry.register_hook(pm)

        # 可选：注入外部审批者（如 WebSocket 推送 + 用户交互）
        pm.set_approver(my_approver)

        # 查询审批历史
        for req in pm.approval_history:
            print(req.summary())
    """

    def __init__(
        self,
        rules: list[PermissionRule] | None = None,
        auto_approve_in_test: bool = False,
    ):
        """
        Args:
            rules: 权限规则表，默认 DEFAULT_RULES
            auto_approve_in_test: True = 测试环境自动批准 CONFIRM
        """
        self._rules: dict[str, list[PermissionRule]] = {}
        for r in (rules or DEFAULT_RULES):
            self._rules.setdefault(r.tool_name, []).append(r)

        self._auto_approve = auto_approve_in_test
        self._approver: Approver | None = None

        # ── 统计 ──
        self.auto_count: int = 0
        self.confirm_approved: int = 0
        self.confirm_denied: int = 0
        self.denied_count: int = 0

        # ── 审批历史 ──
        self.approval_history: list[tuple[ApprovalRequest, ApprovalResponse]] = []

    # ── 审批者配置 ─────────────────────────────────────────

    def set_approver(self, approver: Approver) -> None:
        """
        注入外部审批者。设置后 CONFIRM 级操作会请求审批。

        不设置时：CONFIRM 会因无审批者而自动拒绝。

        Approver 签名：async def my_approver(req: ApprovalRequest) -> ApprovalResponse
        """
        self._approver = approver

    # ── 权限检查 ──────────────────────────────────────────

    def check(self, tool_name: str, tool_args: dict) -> tuple[PermissionLevel, str]:
        """
        检查一个工具调用是否需要审批。

        按规则表匹配（先精确匹配 → 再条件匹配）。
        无规则匹配的工具默认 CONFIRM（安全优先）。

        Returns:
            (PermissionLevel, reason)
        """
        rules = self._rules.get(tool_name, [])
        if not rules:
            return PermissionLevel.CONFIRM, f"Unknown tool '{tool_name}' — requires confirmation"

        # 先检查有条件的规则（如 execute_shell + "rm -rf" → DENY）
        for rule in rules:
            if rule.condition and self._evaluate_condition(rule.condition, tool_args):
                return rule.level, rule.reason

        # 再检查无条件规则
        for rule in rules:
            if rule.condition is None:
                return rule.level, rule.reason

        # 兜底
        return PermissionLevel.CONFIRM, f"Tool '{tool_name}' requires confirmation"

    # ── ToolHook 实现 ──────────────────────────────────────

    async def on_pre_execute(self, tool_name: str, kwargs: dict) -> dict | None:
        """
        Hook：工具执行前检查权限。

        Returns:
            None → 继续执行
            dict → 阻止执行，dict 转为 ToolResult(**dict) 返回
        """
        level, reason = self.check(tool_name, kwargs)

        if level == PermissionLevel.AUTO:
            self.auto_count += 1
            return None  # 放行

        if level == PermissionLevel.DENY:
            self.denied_count += 1
            print(f"  [permission] DENIED: {tool_name} — {reason}", flush=True)
            return {
                "success": False,
                "message": "",
                "error": f"[HITL DENIED] {tool_name}: {reason}",
            }

        # CONFIRM
        if self._auto_approve:
            self.confirm_approved += 1
            return None  # 放行

        if self._approver is None:
            # 无审批者 → 自动拒绝（安全优先）
            self.confirm_denied += 1
            print(f"  [permission] CONFIRM DENIED (no approver): {tool_name} — {reason}",
                  flush=True)
            return {
                "success": False,
                "message": "",
                "error": f"[HITL CONFIRM REQUIRED] {tool_name}: {reason}. "
                         f"No approver configured — execution blocked.",
            }

        # 有审批者 → 请求审批
        req = ApprovalRequest(
            request_id=uuid.uuid4().hex[:12],
            tool_name=tool_name,
            tool_args=kwargs,
            risk_level=self._assess_risk(tool_name, kwargs),
            reason=reason,
        )

        try:
            # 支持同步和异步审批者
            result = self._approver(req)
            if asyncio.iscoroutine(result):
                response = await asyncio.wait_for(result, timeout=60.0)
            else:
                response = result  # 同步回调直接取返回值
        except asyncio.TimeoutError:
            response = ApprovalResponse(
                request_id=req.request_id,
                approved=False,
                comment="Approval timeout (60s)",
            )

        self.approval_history.append((req, response))

        if response.approved:
            self.confirm_approved += 1
            print(f"  [permission] APPROVED: {tool_name}", flush=True)
            return None  # 放行
        else:
            self.confirm_denied += 1
            print(f"  [permission] DENIED by user: {tool_name} — "
                  f"{response.comment or 'no comment'}", flush=True)
            return {
                "success": False,
                "message": "",
                "error": f"[HITL DENIED] {tool_name} was denied by user"
                         f"{': ' + response.comment if response.comment else ''}",
            }

    # ── 统计 ──────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "auto_passed": self.auto_count,
            "confirm_approved": self.confirm_approved,
            "confirm_denied": self.confirm_denied,
            "denied": self.denied_count,
            "total_checked": (self.auto_count + self.confirm_approved
                              + self.confirm_denied + self.denied_count),
            "approver_configured": self._approver is not None,
            "auto_approve_mode": self._auto_approve,
        }

    # ═══════════════════════════════════════════════════════════
    # 内部
    # ═══════════════════════════════════════════════════════════

    def _evaluate_condition(self, condition: str, args: dict) -> bool:
        """评估规则条件表达式（安全子集）。"""
        cmd = str(args.get("command", "")).lower()
        path = str(args.get("path", "")).lower()
        file_path = str(args.get("file_path", "")).lower()

        if condition == "rm -rf":
            return "rm -rf" in cmd or "rm  -rf" in cmd
        if condition == "sudo":
            return "sudo" in cmd
        if condition == "mkfs":
            return "mkfs" in cmd
        if condition == "dd if=":
            return "dd if=" in cmd or "dd  if=" in cmd
        if condition == "> /dev/sda":
            return "> /dev/sda" in cmd or ">/dev/sda" in cmd

        # 通用后缀匹配
        if condition.startswith("path.endswith"):
            suffix = condition.split("'")[1] if "'" in condition else ""
            return path.endswith(suffix) or file_path.endswith(suffix)

        return False

    @staticmethod
    def _assess_risk(tool_name: str, args: dict) -> str:
        """评估操作风险等级。"""
        cmd = str(args.get("command", "")).lower()
        if any(kw in cmd for kw in ("rm ", "delete", "drop", "truncate")):
            return "high"
        if any(kw in cmd for kw in ("pip install", "npm install", "apt-get")):
            return "high"
        if "curl" in cmd or "wget" in cmd:
            return "medium"
        return "low"
