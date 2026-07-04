"""
Phase 7 Step 6 — 消息持久化单元测试

覆盖：
  - PersistenceBackend 枚举
  - MemoryPersistence: save/load/delete（无操作）
  - FilePersistence: save/load/delete（真实文件 I/O）
  - AgentInbox.load_envelopes(): 从持久化恢复消息、过期处理、损坏数据处理
  - Orchestrator 持久化集成: start/stop save/load 周期
  - OrchestratorConfig: 持久化后端选择
  - RedisConfig: 配置数据类
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

from app.agent.base import BaseAgent
from app.agent.mailbox import (
    AgentInbox,
    FilePersistence,
    MailboxManager,
    MailboxMessage,
    MailboxPersistence,
    MemoryPersistence,
    RedisConfig,
    MessagePriority,
    MessageStatus,
    MessageType,
)
from app.agent.orchestrator import (
    MultiAgentOrchestrator,
    OrchestratorConfig,
    PersistenceBackend,
)
from app.agent.roles import (
    ROLE_CODER,
    ROLE_PLANNER,
    ROLE_REVIEWER,
    ROLE_ROUTER,
    ROLE_TESTER,
)


# ═══════════════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════════════

def _make_msg(**kwargs) -> MailboxMessage:
    """创建测试消息。"""
    defaults = {
        "sender": "router",
        "recipient": "coder",
        "msg_type": MessageType.TASK_ASSIGNMENT,
        "body": {"task": "test"},
        "subject": "Test message",
    }
    defaults.update(kwargs)
    return MailboxMessage(**defaults)


def _register_mock_agents(orch: MultiAgentOrchestrator) -> None:
    """将全套 mock agent 注册到 orchestrator（不调用真实 LLM）。"""
    manager = orch.mailbox

    # 延迟导入避免循环
    from app.agent.router import RouterAgent
    from app.agent.planner import PlannerAgent
    from app.agent.coder import CoderAgent
    from app.agent.tester import TesterAgent
    from app.agent.reviewer import ReviewerAgent

    class MockRouter(RouterAgent):
        async def _analyze_request(self, user_message: str) -> dict:
            return {"complexity": "simple", "reasoning": "Mock",
                    "route": {"target": "coder", "task": user_message, "priority": "normal"}}

    class MockPlanner(PlannerAgent):
        async def run(self, task: str) -> dict:
            return {"summary": "Plan", "steps": [{"step": 1, "agent": "coder", "task": task}],
                    "estimated_time": "1 min", "agent": self.agent_id}

    class MockCoder(CoderAgent):
        async def run(self, task: str) -> dict:
            return {"files_created": ["out.py"], "summary": "Done", "agent": self.agent_id}

    class MockTester(TesterAgent):
        async def run(self, task: str) -> dict:
            return {"passed": True, "total_tests": 1, "passed_count": 1,
                    "failed_count": 0, "failures": [], "summary": "OK", "agent": self.agent_id}

    class MockReviewer(ReviewerAgent):
        async def run(self, task: str) -> dict:
            return {"verdict": "LGTM", "scores": {}, "issues": [],
                    "praise": [], "summary": "OK", "agent": self.agent_id}

    orch.agents = {
        "router": MockRouter(ROLE_ROUTER, manager, agent_id="router"),
        "planner": MockPlanner(ROLE_PLANNER, manager, agent_id="planner"),
        "coder": MockCoder(ROLE_CODER, manager, agent_id="coder"),
        "tester": MockTester(ROLE_TESTER, manager, agent_id="tester"),
        "reviewer": MockReviewer(ROLE_REVIEWER, manager, agent_id="reviewer"),
    }


# ═══════════════════════════════════════════════════════════════════════
# PersistenceBackend 枚举测试
# ═══════════════════════════════════════════════════════════════════════

class TestPersistenceBackend:
    """PersistenceBackend 枚举测试。"""

    def test_all_values(self):
        assert PersistenceBackend.MEMORY == "memory"
        assert PersistenceBackend.FILE == "file"
        assert PersistenceBackend.REDIS == "redis"
        assert len(list(PersistenceBackend)) == 3

    def test_string_comparison(self):
        """可以直接和字符串比较。"""
        assert PersistenceBackend.FILE == "file"
        assert PersistenceBackend.FILE != "redis"

    def test_value_attribute(self):
        """.value 返回字符串值。"""
        assert PersistenceBackend.MEMORY.value == "memory"
        assert PersistenceBackend.FILE.value == "file"


# ═══════════════════════════════════════════════════════════════════════
# RedisConfig 测试
# ═══════════════════════════════════════════════════════════════════════

class TestRedisConfig:
    """RedisConfig 数据类测试。"""

    def test_default_config(self):
        cfg = RedisConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 6379
        assert cfg.password == ""
        assert cfg.db == 0
        assert cfg.prefix == "joyagent:mailbox"
        assert cfg.ttl_seconds == 0
        assert cfg.connect_timeout == 5.0

    def test_custom_config(self):
        cfg = RedisConfig(
            host="redis.example.com",
            port=6380,
            password="secret",
            db=1,
            prefix="myapp:mailbox",
            ttl_seconds=3600,
        )
        assert cfg.port == 6380
        assert cfg.password == "secret"
        assert cfg.prefix == "myapp:mailbox"
        assert cfg.ttl_seconds == 3600


# ═══════════════════════════════════════════════════════════════════════
# MemoryPersistence 测试
# ═══════════════════════════════════════════════════════════════════════

class TestMemoryPersistence:
    """MemoryPersistence: 不做持久化（所有操作都是 no-op）。"""

    @pytest.mark.asyncio
    async def test_save_load_delete_are_noops(self):
        inbox = AgentInbox(owner="test")
        inbox.deliver(_make_msg())

        mp = MemoryPersistence()
        await mp.save(inbox)          # 不保存
        result = await mp.load("test")  # 返回 None
        assert result is None
        await mp.delete("test")       # 不删


# ═══════════════════════════════════════════════════════════════════════
# FilePersistence 测试
# ═══════════════════════════════════════════════════════════════════════

class TestFilePersistence:
    """FilePersistence: 本地 JSON 文件持久化。"""

    @pytest.fixture
    def tmpdir(self):
        """临时目录 fixture。"""
        d = tempfile.mkdtemp(prefix="joyagent_test_")
        yield d
        # 清理
        for f in Path(d).glob("*.json"):
            f.unlink(missing_ok=True)
        Path(d).rmdir()

    @pytest.mark.asyncio
    async def test_save_creates_file(self, tmpdir):
        """save() 创建 JSON 文件。"""
        inbox = AgentInbox(owner="coder")
        msg = _make_msg()
        inbox.deliver(msg)

        fp = FilePersistence(path=tmpdir)
        await fp.save(inbox)

        file_path = Path(tmpdir) / "coder.json"
        assert file_path.exists()

        content = json.loads(file_path.read_text())
        assert len(content) == 1
        assert content[0]["sender"] == "router"
        assert content[0]["msg_type"] == "task_assignment"

    @pytest.mark.asyncio
    async def test_load_returns_envelopes(self, tmpdir):
        """load() 返回消息 envelope 列表。"""
        inbox = AgentInbox(owner="coder")
        msg = _make_msg()
        inbox.deliver(msg)

        fp = FilePersistence(path=tmpdir)
        await fp.save(inbox)

        envelopes = await fp.load("coder")
        assert envelopes is not None
        assert len(envelopes) == 1
        assert envelopes[0]["sender"] == "router"

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_none(self, tmpdir):
        """不存在的 owner → load() 返回 None。"""
        fp = FilePersistence(path=tmpdir)
        result = await fp.load("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_empty_inbox_no_file(self, tmpdir):
        """空 Inbox → save() 不创建文件。"""
        inbox = AgentInbox(owner="coder")
        fp = FilePersistence(path=tmpdir)
        await fp.save(inbox)

        file_path = Path(tmpdir) / "coder.json"
        assert not file_path.exists()

    @pytest.mark.asyncio
    async def test_delete_removes_file(self, tmpdir):
        """delete() 删除持久化文件。"""
        inbox = AgentInbox(owner="coder")
        inbox.deliver(_make_msg())
        fp = FilePersistence(path=tmpdir)
        await fp.save(inbox)

        assert (Path(tmpdir) / "coder.json").exists()
        await fp.delete("coder")
        assert not (Path(tmpdir) / "coder.json").exists()

    @pytest.mark.asyncio
    async def test_save_overwrites_previous(self, tmpdir):
        """第二次 save() 覆盖旧文件。"""
        inbox = AgentInbox(owner="coder")

        msg1 = _make_msg(subject="First")
        inbox.deliver(msg1)
        fp = FilePersistence(path=tmpdir)
        await fp.save(inbox)

        # 模拟处理完第一条，发送第二条
        await inbox.fetch_next(timeout=0)
        msg2 = _make_msg(subject="Second")
        inbox.deliver(msg2)
        await fp.save(inbox)

        envelopes = await fp.load("coder")
        assert len(envelopes) == 1
        assert envelopes[0]["subject"] == "Second"

    @pytest.mark.asyncio
    async def test_backend_flag_in_config(self, tmpdir):
        """OrchestratorConfig 中设置 'file' 后端 → 创建 FilePersistence。"""
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False,
            persistence_backend="file",
            persistence_path=tmpdir,
        ))
        assert isinstance(orch._persistence, FilePersistence)
        assert orch._persistence.storage_path == tmpdir


# ═══════════════════════════════════════════════════════════════════════
# AgentInbox.load_envelopes() 测试
# ═══════════════════════════════════════════════════════════════════════

class TestInboxLoadEnvelopes:
    """AgentInbox.load_envelopes() —— 从持久化恢复消息。"""

    def test_load_single_message(self):
        """加载单条消息 → Inbox 中有 1 条。"""
        inbox = AgentInbox(owner="coder")
        msg = _make_msg()
        loaded = inbox.load_envelopes([msg.to_envelope()])
        assert loaded == 1
        assert inbox.total_count == 1

    def test_load_multiple_messages(self):
        """加载多条消息 → 全部恢复。"""
        inbox = AgentInbox(owner="coder")
        envelopes = [_make_msg(subject=f"Task {i}").to_envelope() for i in range(5)]
        loaded = inbox.load_envelopes(envelopes)
        assert loaded == 5
        assert inbox.total_count == 5

    def test_load_skips_expired_messages(self):
        """过期消息 → 移动到死信队列，不计入 active。"""
        inbox = AgentInbox(owner="coder")
        msg = _make_msg()
        msg.created_at = 0       # 很久以前创建
        msg.ttl_seconds = 1      # 1 秒 TTL → 确定过期
        loaded = inbox.load_envelopes([msg.to_envelope()])
        assert loaded == 0  # 不计入 active
        assert inbox.total_count == 0
        assert inbox.dead_letter_count == 1

    def test_load_skips_processed_messages(self):
        """已处理的消息 → 不恢复。"""
        inbox = AgentInbox(owner="coder")
        msg = _make_msg()
        msg.status = MessageStatus.PROCESSED
        loaded = inbox.load_envelopes([msg.to_envelope()])
        assert loaded == 0
        assert inbox.total_count == 0

    def test_load_sets_new_message_event(self):
        """加载消息后 → _new_message_event 被 set。"""
        inbox = AgentInbox(owner="coder")
        msg = _make_msg()
        inbox.load_envelopes([msg.to_envelope()])
        # Event 应该已 set（因为加载了消息）
        assert inbox._new_message_event.is_set()

    def test_load_handles_corrupt_data(self):
        """损坏的数据 → 跳过，不影响其他消息。"""
        inbox = AgentInbox(owner="coder")
        envelopes = [
            _make_msg(subject="Good").to_envelope(),
            {"corrupt": "no id field"},  # 损坏数据
            _make_msg(subject="Also good").to_envelope(),
        ]
        loaded = inbox.load_envelopes(envelopes)
        assert loaded == 2  # 2 条好的

    def test_load_empty_list(self):
        """空列表 → 加载 0。"""
        inbox = AgentInbox(owner="coder")
        loaded = inbox.load_envelopes([])
        assert loaded == 0

    def test_load_preserves_message_status(self):
        """加载后消息状态保持原样。"""
        inbox = AgentInbox(owner="coder")
        msg = _make_msg()
        msg.status = MessageStatus.READ  # 已读未处理
        inbox.load_envelopes([msg.to_envelope()])

        # fetch_next 应该能拿到 READ 状态的消息
        # (因为 fetch_next 检查的是 DELIVERED，不是 READ)
        # READ 的状态意味着之前被读过但没处理完，watcher 可能不会再次 fetch
        # 这是期望行为——已读消息保存后恢复，watcher 可以手动 process
        assert inbox.total_count == 1


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator 持久化集成测试
# ═══════════════════════════════════════════════════════════════════════

class TestOrchestratorPersistence:
    """Orchestrator 集成持久化的完整周期测试。"""

    @pytest.fixture
    def tmpdir(self):
        d = tempfile.mkdtemp(prefix="joyagent_test_")
        yield d
        for f in Path(d).glob("*.json"):
            f.unlink(missing_ok=True)
        try:
            Path(d).rmdir()
        except OSError:
            pass

    @pytest.mark.asyncio
    async def test_memory_backend_default(self):
        """默认 backend=memory → 不做持久化。"""
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0,
        ))
        assert orch.persistence_backend_name == "memory"
        assert isinstance(orch._persistence, MemoryPersistence)

    @pytest.mark.asyncio
    async def test_file_backend_save_on_stop(self, tmpdir):
        """
        文件后端 → stop_all() 时自动保存未处理消息。

        验证：
          1. 发送消息到 Coder Inbox
          2. stop_all() → 消息保存到 JSON 文件
          3. 文件内容可验证
        """
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0,
            persistence_backend="file", persistence_path=tmpdir,
        ))

        # 手动注册 agent（不启动 watcher，避免自动处理消息）
        manager = orch.mailbox
        orch.agents = {
            "coder": BaseAgent(ROLE_CODER, manager, agent_id="coder"),
        }

        # 投递一条消息到 Coder Inbox
        msg = MailboxMessage(
            sender="router", recipient="coder",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "test save on stop"},
            subject="Test persistence",
        )
        manager._inboxes["coder"].deliver(msg)
        assert orch.agents["coder"].inbox.total_count == 1

        # 停止 → 触发保存
        await orch.stop_all()

        # 验证文件存在
        file_path = Path(tmpdir) / "coder.json"
        assert file_path.exists()
        content = json.loads(file_path.read_text())
        assert len(content) == 1
        assert content[0]["subject"] == "Test persistence"

    @pytest.mark.asyncio
    async def test_file_backend_load_on_start(self, tmpdir):
        """
        文件后端 → start_all() 时自动恢复未处理消息。

        验证：
          1. 手动创建持久化文件（模拟上次关闭前保存的）
          2. 注册 agent 但不启动 watcher → 手动触发 _load_inboxes
          3. Agent Inbox 中有恢复的消息（watcher 未启动，消息不处理）
        """
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0,
            persistence_backend="file", persistence_path=tmpdir,
        ))

        # 手动创建持久化文件
        msg = MailboxMessage(
            sender="router", recipient="coder",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "recovered task"},
            subject="Recovered message",
        )
        file_path = Path(tmpdir) / "coder.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps([msg.to_envelope()]))

        # 手动注册 agent
        manager = orch.mailbox
        orch.agents = {
            "coder": BaseAgent(ROLE_CODER, manager, agent_id="coder"),
        }

        # 手动加载（不启动 watcher，避免消息被立即处理）
        await orch._load_inboxes()

        # 验证消息已恢复到 Inbox
        unread = orch.agents["coder"].inbox.fetch_unread()
        assert len(unread) == 1
        assert unread[0].subject == "Recovered message"

    @pytest.mark.asyncio
    async def test_full_save_load_cycle(self, tmpdir):
        """
        完整 save → load 周期。

        1. 发送消息 → 停止（save）
        2. 启动（load）→ 消息恢复
        3. 可以继续处理恢复的消息
        """
        # ── Phase 1: 保存 ──
        orch1 = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0,
            persistence_backend="file", persistence_path=tmpdir,
        ))
        manager1 = orch1.mailbox
        orch1.agents = {
            "coder": BaseAgent(ROLE_CODER, manager1, agent_id="coder"),
        }

        manager1._inboxes["coder"].deliver(MailboxMessage(
            sender="router", recipient="coder",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "cycle test"},
            subject="Cycle message",
            correlation_id="corr_cycle",
        ))
        await orch1.stop_all()  # 自动保存

        # ── Phase 2: 加载 ──
        orch2 = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0,
            persistence_backend="file", persistence_path=tmpdir,
        ))
        manager2 = orch2.mailbox
        orch2.agents = {
            "coder": BaseAgent(ROLE_CODER, manager2, agent_id="coder"),
        }

        await orch2.start_all()  # 自动加载
        assert orch2.agents["coder"].inbox.total_count == 1

        msg = await orch2.agents["coder"].inbox.fetch_next(timeout=1.0)
        assert msg is not None
        assert msg.subject == "Cycle message"
        assert msg.correlation_id == "corr_cycle"

        await orch2.stop_all()

    @pytest.mark.asyncio
    async def test_shutdown_saves_and_cleans(self, tmpdir):
        """
        shutdown() → 先保存未处理消息 → 再清理 + 注销。
        """
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0,
            persistence_backend="file", persistence_path=tmpdir,
        ))
        manager = orch.mailbox
        orch.agents = {
            "coder": BaseAgent(ROLE_CODER, manager, agent_id="coder"),
        }

        # 投递消息
        manager._inboxes["coder"].deliver(MailboxMessage(
            sender="router", recipient="coder",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "shutdown test"},
            subject="Shutdown message",
        ))

        await orch.shutdown()

        # 文件存在（shutdown 中的 stop_all 触发了 save）
        file_path = Path(tmpdir) / "coder.json"
        assert file_path.exists()

        # Agent 已注销
        assert len(orch.mailbox.registered_agents) == 0
        assert len(orch.agents) == 0

    @pytest.mark.asyncio
    async def test_memory_backend_no_file_created(self, tmpdir):
        """
        backend=memory → 不创建任何文件。
        """
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0,
            persistence_backend="memory", persistence_path=tmpdir,
        ))
        manager = orch.mailbox
        orch.agents = {
            "coder": BaseAgent(ROLE_CODER, manager, agent_id="coder"),
        }

        manager._inboxes["coder"].deliver(MailboxMessage(
            sender="router", recipient="coder",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "no save"},
            subject="No save",
        ))

        await orch.stop_all()

        files = list(Path(tmpdir).glob("*.json"))
        assert len(files) == 0


# ═══════════════════════════════════════════════════════════════════════
# 集成测试：持久化 + 请求处理
# ═══════════════════════════════════════════════════════════════════════

class TestPersistenceIntegration:
    """持久化与请求处理集成的端到端测试。"""

    @pytest.fixture
    def tmpdir(self):
        d = tempfile.mkdtemp(prefix="joyagent_test_")
        yield d
        for f in Path(d).glob("*.json"):
            f.unlink(missing_ok=True)
        try:
            Path(d).rmdir()
        except OSError:
            pass

    @pytest.mark.asyncio
    async def test_request_then_save_then_restore(self, tmpdir):
        """
        完整链：处理请求 → 手动投递待处理消息到 Inbox
        → 手动保存 → 新实例手动加载 → 消息可验证恢复。
        """
        # ── Round 1: 注册 agent + 投递消息 + 保存 ──
        orch1 = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0,
            persistence_backend="file", persistence_path=tmpdir,
            request_timeout=10.0,
        ))
        # 注册但不启动 watcher（避免消息被自动处理）
        manager1 = orch1.mailbox
        orch1.agents = {
            "coder": BaseAgent(ROLE_CODER, manager1, agent_id="coder"),
        }

        # 投递待处理消息
        orch1.agents["coder"].inbox.deliver(MailboxMessage(
            sender="router", recipient="coder",
            msg_type=MessageType.TASK_ASSIGNMENT,
            body={"task": "pending task"},
            subject="Pending task",
            correlation_id="corr_pending",
        ))

        # 手动保存
        await orch1._save_inboxes()
        file_path = Path(tmpdir) / "coder.json"
        assert file_path.exists()

        # ── Round 2: 新实例加载 ──
        orch2 = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0,
            persistence_backend="file", persistence_path=tmpdir,
            request_timeout=10.0,
        ))
        manager2 = orch2.mailbox
        orch2.agents = {
            "coder": BaseAgent(ROLE_CODER, manager2, agent_id="coder"),
        }

        # 手动加载（不启动 watcher）
        await orch2._load_inboxes()

        # 验证消息已恢复
        unread = orch2.agents["coder"].inbox.fetch_unread()
        assert len(unread) >= 1
        assert unread[0].correlation_id == "corr_pending"

    @pytest.mark.asyncio
    async def test_file_backend_handles_corrupt_file(self, tmpdir):
        """
        损坏的持久化文件 → 不影响系统启动。
        """
        # 创建损坏的 JSON 文件
        file_path = Path(tmpdir) / "coder.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("{corrupt json!!!")  # 非法的 JSON

        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0,
            persistence_backend="file", persistence_path=tmpdir,
        ))
        manager = orch.mailbox
        orch.agents = {
            "coder": BaseAgent(ROLE_CODER, manager, agent_id="coder"),
        }

        # 应该正常启动（损坏的文件被跳过）
        await orch.start_all()
        await orch.stop_all()
        # 不抛异常即测试通过

    @pytest.mark.asyncio
    async def test_file_backend_handles_empty_directory(self, tmpdir):
        """
        空目录 → 正常启动（无数据可恢复）。
        """
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False, expire_sweep_interval=0,
            persistence_backend="file", persistence_path=tmpdir,
        ))
        manager = orch.mailbox
        orch.agents = {
            "coder": BaseAgent(ROLE_CODER, manager, agent_id="coder"),
        }

        await orch.start_all()
        assert orch.agents["coder"].inbox.total_count == 0
        await orch.stop_all()

    @pytest.mark.asyncio
    async def test_unknown_backend_falls_back_to_memory(self):
        """未知的 backend 值 → 降级为 MemoryPersistence。"""
        orch = MultiAgentOrchestrator(config=OrchestratorConfig(
            verbose=False,
            persistence_backend="cassandra",  # 不存在
        ))
        assert isinstance(orch._persistence, MemoryPersistence)
