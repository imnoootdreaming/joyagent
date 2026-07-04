# app/agent — Multi-Agent + Mailbox 通信系统
#
# Phase 1-2: Simple Agent (agent.py)
# Phase 3:   LangGraph Workflow (graph/, runtime/, schemas/)
# Phase 7:   Multi-Agent + Mailbox (base.py, roles.py, mailbox/)

from app.agent.base import BaseAgent, extract_text
from app.agent.roles import AgentRole, AGENT_ROLES, ROLE_ROUTER, ROLE_PLANNER, ROLE_CODER, ROLE_TESTER, ROLE_REVIEWER

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
]
