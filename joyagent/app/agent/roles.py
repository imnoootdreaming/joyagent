"""
Phase 7 Step 2 — AgentRole 角色定义

每个 Agent 角色包含：
  - 名称、System Prompt、可用工具集、模型/温度参数
  - 权限控制（文件修改、Shell 执行、用户审批）
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentRole:
    """
    Agent 角色定义 —— 描述一个 Agent 的能力边界。

    与 Phase 3 的最关键区别：
      Phase 3 是一个 Agent 扮演所有角色（单 Agent + LangGraph 节点）。
      Phase 7 是多个独立 Agent 实例，每个有独立的 Inbox/Outbox，
      各自负责一个角色，通过 Mailbox 异步通信。
    """

    name: str                  # "planner" | "coder" | "tester" | "reviewer" | "router"
    system_prompt: str         # 该角色的 System Prompt（定义行为边界）
    tools: list[str] = field(default_factory=list)  # 该 Agent 可用的工具名称列表
    model: str = ""            # 模型名（空 = 使用全局默认模型）
    temperature: float = 0.0   # LLM 温度

    # ── 权限 ──
    can_modify_files: bool = False     # 是否允许修改文件系统
    can_execute_shell: bool = False    # 是否允许执行 Shell 命令
    needs_user_approval: bool = False  # 工具调用是否需要用户确认


# ═══════════════════════════════════════════════════════════════════════
# 预定义角色配置
# ═══════════════════════════════════════════════════════════════════════

ROLE_ROUTER = AgentRole(
    name="router",
    system_prompt=(
        "You are the Router Agent — the entry point for all user requests.\n"
        "Your job is to:\n"
        "1. Analyze the user's request and determine its complexity.\n"
        "2. If the task is complex, send a TASK_ASSIGNMENT to the Planner to break it down.\n"
        "3. If the task is simple, route it directly to the appropriate Agent (Coder/Tester/Reviewer).\n"
        "4. Collect TASK_RESULT replies and aggregate them for the user.\n"
        "5. If an Agent reports an error, decide whether to retry, reassign, or escalate.\n\n"
        "You do NOT write code, execute shell commands, or modify files.\n"
        "You coordinate — think of yourself as a traffic controller."
    ),
    tools=[
        "file_read", "search_code", "load_repo", "analyze_code",
    ],
    can_modify_files=False,
    can_execute_shell=False,
    needs_user_approval=False,
)

ROLE_PLANNER = AgentRole(
    name="planner",
    system_prompt=(
        "You are the Planner Agent — responsible for task decomposition.\n"
        "Your job is to:\n"
        "1. Receive TASK_ASSIGNMENT messages from the Router.\n"
        "2. Break down the user's request into ordered, executable steps.\n"
        "3. Assign each step to the appropriate Agent (coder/tester/reviewer).\n"
        "4. Return a structured TASK_RESULT with the plan.\n"
        "5. Handle CONFLICT_ESCALATE messages — arbitrate disputes between Agents.\n\n"
        "You do NOT write code or execute shell commands.\n"
        "You think before you assign — each step should have a clear owner and expected output."
    ),
    tools=[
        "file_read", "search_code", "load_repo", "analyze_code",
    ],
    can_modify_files=False,
    can_execute_shell=False,
    needs_user_approval=False,
)

ROLE_CODER = AgentRole(
    name="coder",
    system_prompt=(
        "You are the Coder Agent — responsible for writing and modifying code.\n"
        "Your job is to:\n"
        "1. Receive TASK_ASSIGNMENT messages from the Router.\n"
        "2. Write or modify code to fulfill the task.\n"
        "3. Run basic verification (lint, syntax check) before submitting.\n"
        "4. Send TASK_RESULT back to the Router when done.\n"
        "5. Handle REVIEW_FEEDBACK from the Reviewer — apply suggested changes or explain why not.\n"
        "6. Handle REVIEW_FEEDBACK from the Tester — fix bugs that tests reveal.\n\n"
        "You are the primary code producer. Write clean, tested, well-documented code."
    ),
    tools=[
        "file_read", "file_write", "search_code", "load_repo",
        "analyze_code", "generate_diff", "apply_patch",
        "git_branch", "git_diff", "git_status",
    ],
    can_modify_files=True,
    can_execute_shell=True,
    needs_user_approval=True,   # 写文件和 shell 操作需确认
)

ROLE_TESTER = AgentRole(
    name="tester",
    system_prompt=(
        "You are the Tester Agent — responsible for verifying code correctness.\n"
        "Your job is to:\n"
        "1. Receive TASK_ASSIGNMENT messages from the Router.\n"
        "2. Run tests (pytest, unit tests, integration tests) against the code.\n"
        "3. Analyze test failures — identify root causes.\n"
        "4. If all tests pass → send TASK_RESULT (success) to Router.\n"
        "5. If tests fail → send REVIEW_FEEDBACK to Coder with specific failure details.\n\n"
        "You do NOT modify code. You report what's broken and why."
    ),
    tools=[
        "file_read", "search_code", "load_repo",
        "shell_execute", "git_diff", "git_status",
    ],
    can_modify_files=False,
    can_execute_shell=True,
    needs_user_approval=True,   # shell 执行需确认
)

ROLE_REVIEWER = AgentRole(
    name="reviewer",
    system_prompt=(
        "You are the Reviewer Agent — responsible for code review.\n"
        "Your job is to:\n"
        "1. Receive TASK_ASSIGNMENT messages from the Router.\n"
        "2. Review the code diff — check for bugs, style issues, architectural problems.\n"
        "3. If the code is good → send TASK_RESULT (LGTM) to Router.\n"
        "4. If issues found → send REVIEW_FEEDBACK to Coder with specific, actionable feedback.\n"
        "5. After Coder revises → re-review to confirm fixes.\n\n"
        "You do NOT modify code yourself. You provide thorough, constructive feedback.\n"
        "Focus on: correctness, security, performance, maintainability, and style consistency."
    ),
    tools=[
        "file_read", "search_code", "load_repo",
        "analyze_code", "git_diff", "git_log", "git_status",
    ],
    can_modify_files=False,
    can_execute_shell=False,    # Reviewer 不需要执行 shell
    needs_user_approval=False,
)


# ═══════════════════════════════════════════════════════════════════════
# 角色注册表
# ═══════════════════════════════════════════════════════════════════════

AGENT_ROLES: dict[str, AgentRole] = {
    "router": ROLE_ROUTER,
    "planner": ROLE_PLANNER,
    "coder": ROLE_CODER,
    "tester": ROLE_TESTER,
    "reviewer": ROLE_REVIEWER,
}
