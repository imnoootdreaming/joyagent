"""
Phase 9A-4 — Human-in-the-Loop 权限系统测试

全部纯 Python + 内存 mock，零外部依赖。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from app.core.permissions import (
    ApprovalRequest,
    ApprovalResponse,
    PermissionLevel,
    PermissionManager,
    PermissionRule,
)


# ═══════════════════════════════════════════════════════════════════════
# 内联 BaseTool / ToolResult / ToolRegistry（避免 app.tools 链式导入）
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class _Result:
    success: bool
    message: str = ""
    error: str | None = None

class _Tool:
    def __init__(self, name): self._n = name
    @property
    def name(self): return self._n
    async def execute(self, **kw): return _Result(True, "ok")

class _Registry:
    def __init__(self):
        self._tools = {}
        self._hooks = []
    def add(self, tool):
        self._tools[tool.name] = tool
    def hook(self, h):
        self._hooks.append(h)
    async def run(self, name, **kw):
        t = self._tools.get(name)
        if not t: return _Result(False, error=f"Unknown: {name}")
        for h in self._hooks:
            if hasattr(h, "on_pre_execute"):
                ov = await h.on_pre_execute(name, kw)
                if ov is not None: return _Result(**ov)
        return await t.execute(**kw)


# ═══════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEnums:
    def test_levels(self):
        assert PermissionLevel.AUTO.value == "auto"
        assert PermissionLevel.CONFIRM.value == "confirm"
        assert PermissionLevel.DENY.value == "deny"

class TestRule:
    def test_create(self):
        r = PermissionRule("w", PermissionLevel.CONFIRM, "reason")
        assert r.level == PermissionLevel.CONFIRM

class TestModels:
    def test_req(self):
        r = ApprovalRequest("id", "w", {"p": "x"}, "medium", "reason")
        assert "x" in r.summary()
    def test_resp(self):
        r = ApprovalResponse("id", True, "ok")
        assert r.approved

class TestCheck:
    @pytest.fixture
    def pm(self): return PermissionManager()
    def test_auto(self, pm): assert pm.check("read_file", {})[0] == PermissionLevel.AUTO
    def test_confirm(self, pm): assert pm.check("write_file", {})[0] == PermissionLevel.CONFIRM
    def test_deny_rm(self, pm): assert pm.check("execute_shell", {"command": "rm -rf /"})[0] == PermissionLevel.DENY
    def test_deny_sudo(self, pm): assert pm.check("execute_shell", {"command": "sudo rm"})[0] == PermissionLevel.DENY
    def test_shell_ok(self, pm): assert pm.check("execute_shell", {"command": "pytest"})[0] == PermissionLevel.CONFIRM
    def test_unknown(self, pm): assert pm.check("xyz", {})[0] == PermissionLevel.CONFIRM
    def test_custom(self):
        pm = PermissionManager(rules=[PermissionRule("t", PermissionLevel.AUTO)])
        assert pm.check("t", {})[0] == PermissionLevel.AUTO

class TestHook:
    @pytest.fixture
    def pm(self): return PermissionManager()

    @pytest.mark.asyncio
    async def test_auto_ok(self, pm): assert await pm.on_pre_execute("read_file", {}) is None

    @pytest.mark.asyncio
    async def test_deny(self, pm):
        r = await pm.on_pre_execute("execute_shell", {"command": "rm -rf /"})
        assert r and "DENIED" in r["error"]

    @pytest.mark.asyncio
    async def test_confirm_no_approver(self, pm):
        r = await pm.on_pre_execute("write_file", {})
        assert r and "CONFIRM REQUIRED" in r["error"]

    @pytest.mark.asyncio
    async def test_confirm_approved(self):
        pm = PermissionManager()
        pm.set_approver(lambda req: ApprovalResponse(req.request_id, True))
        assert await pm.on_pre_execute("write_file", {}) is None

    @pytest.mark.asyncio
    async def test_confirm_denied(self):
        pm = PermissionManager()
        pm.set_approver(lambda req: ApprovalResponse(req.request_id, False, "no"))
        r = await pm.on_pre_execute("write_file", {})
        assert r and "DENIED" in r["error"] and "no" in r["error"]

    @pytest.mark.asyncio
    async def test_auto_approve(self):
        pm = PermissionManager(auto_approve_in_test=True)
        assert await pm.on_pre_execute("write_file", {}) is None

    @pytest.mark.asyncio
    async def test_stats(self):
        pm = PermissionManager()
        await pm.on_pre_execute("read_file", {})
        await pm.on_pre_execute("write_file", {})
        await pm.on_pre_execute("execute_shell", {"command": "rm -rf /"})
        s = pm.stats
        assert s["auto_passed"] == 1 and s["confirm_denied"] == 1 and s["denied"] == 1

    @pytest.mark.asyncio
    async def test_history(self):
        pm = PermissionManager()
        pm.set_approver(lambda req: ApprovalResponse(req.request_id, True, "ok"))
        await pm.on_pre_execute("write_file", {"path": "a.py"})
        assert len(pm.approval_history) == 1

    @pytest.mark.asyncio
    async def test_timeout(self):
        pm = PermissionManager()
        async def _slow(req):
            await asyncio.sleep(999)
            return ApprovalResponse(req.request_id, True)
        pm.set_approver(_slow)
        r = await pm.on_pre_execute("write_file", {})
        assert r and "timeout" in r["error"].lower()

    @pytest.mark.asyncio
    async def test_args_seen(self):
        pm = PermissionManager()
        s = []
        pm.set_approver(lambda req: (s.append(req.tool_args), ApprovalResponse(req.request_id, True))[1])
        await pm.on_pre_execute("write_file", {"path": "x"})
        assert s[0] == {"path": "x"}

    @pytest.mark.asyncio
    async def test_registry_auto(self):
        tr = _Registry()
        tr.hook(PermissionManager())
        tr.add(_Tool("read_file"))
        assert (await tr.run("read_file")).success

    @pytest.mark.asyncio
    async def test_registry_confirm_blocked(self):
        tr = _Registry()
        tr.hook(PermissionManager())
        tr.add(_Tool("write_file"))
        r = await tr.run("write_file")
        assert not r.success and "CONFIRM REQUIRED" in r.error

    @pytest.mark.asyncio
    async def test_registry_approved(self):
        tr = _Registry()
        pm = PermissionManager()
        pm.set_approver(lambda req: ApprovalResponse(req.request_id, True))
        tr.hook(pm)
        tr.add(_Tool("write_file"))
        assert (await tr.run("write_file")).success
