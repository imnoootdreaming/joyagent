# app/agent — Multi-Agent + Mailbox 通信系统
#
# Phase 1-2: Simple Agent (agent.py)
# Phase 3:   LangGraph Workflow (graph/, runtime/, schemas/)
# Phase 7:   Multi-Agent + Mailbox (base.py, roles.py, mailbox/)
# Phase 7 Step 3: Specialized Agents (planner/, coder/, tester/, reviewer/)

from app.agent.base import BaseAgent, extract_text
from app.agent.roles import AgentRole, AGENT_ROLES, ROLE_ROUTER, ROLE_PLANNER, ROLE_CODER, ROLE_TESTER, ROLE_REVIEWER

# ── Step 3: Specialized Agents ──
from app.agent.planner import PlannerAgent
from app.agent.coder import CoderAgent
from app.agent.tester import TesterAgent
from app.agent.reviewer import ReviewerAgent

__all__ = [
    # Base
    "BaseAgent",
    "extract_text",
    # Roles
    "AgentRole",
    "AGENT_ROLES",
    "ROLE_ROUTER",
    "ROLE_PLANNER",
    "ROLE_CODER",
    "ROLE_TESTER",
    "ROLE_REVIEWER",
    # Step 3: Specialized Agents
    "PlannerAgent",
    "CoderAgent",
    "TesterAgent",
    "ReviewerAgent",
]
