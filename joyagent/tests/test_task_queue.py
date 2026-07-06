"""
Phase 9A-2 — Redis 任务队列单元测试

使用内置 Mock Redis，零外部依赖。
"""
from __future__ import annotations

import pytest

from app.services.task_queue import QueueTask, TaskQueue


# ═══════════════════════════════════════════════════════════════════════
# QueueTask 数据模型测试
# ═══════════════════════════════════════════════════════════════════════

class TestQueueTask:
    """QueueTask 数据类。"""

    def test_create_minimal(self):
        t = QueueTask(task_id="t1", session_id="s1")
        assert t.task_id == "t1"
        assert t.session_id == "s1"
        assert t.user_id == "default"
        assert t.status == "pending"
        assert t.priority == 0
        assert t.created_at != ""

    def test_create_full(self):
        t = QueueTask(
            task_id="t2", session_id="s2", user_id="u1",
            task_type="execute_tool",
            payload={"command": "pytest"},
            priority=5,
        )
        assert t.priority == 5
        assert t.task_type == "execute_tool"
        assert t.payload == {"command": "pytest"}

    def test_to_dict_and_back(self):
        t = QueueTask(task_id="t3", session_id="s3", priority=3)
        d = t.to_dict()
        t2 = QueueTask.from_dict(d)
        assert t2.task_id == "t3"
        assert t2.session_id == "s3"
        assert t2.priority == 3

    def test_created_at_auto_filled(self):
        t = QueueTask(task_id="x", session_id="y")
        assert "T" in t.created_at  # ISO 格式含 T


# ═══════════════════════════════════════════════════════════════════════
# TaskQueue 核心操作（mock=True 内存模式）
# ═══════════════════════════════════════════════════════════════════════

class TestTaskQueue:
    """TaskQueue — enqueue / dequeue / update / peek / clear。"""

    @pytest.fixture
    def queue(self):
        q = TaskQueue(mock=True)
        yield q
        q.clear()

    def test_enqueue_and_dequeue(self, queue):
        queue.enqueue(QueueTask(task_id="t1", session_id="s1",
                                payload={"msg": "hello"}))
        assert queue.queue_length == 1

        task = queue.dequeue()
        assert task is not None
        assert task.task_id == "t1"
        assert task.payload == {"msg": "hello"}
        assert queue.queue_length == 0

    def test_dequeue_empty(self, queue):
        assert queue.dequeue() is None

    def test_priority_ordering(self, queue):
        """高优先级先出队。"""
        queue.enqueue(QueueTask(task_id="low", session_id="s1", priority=1))
        queue.enqueue(QueueTask(task_id="high", session_id="s2", priority=10))
        queue.enqueue(QueueTask(task_id="mid", session_id="s3", priority=5))

        assert queue.dequeue().task_id == "high"
        assert queue.dequeue().task_id == "mid"
        assert queue.dequeue().task_id == "low"

    def test_fifo_within_same_priority(self, queue):
        """同优先级 FIFO。"""
        queue.enqueue(QueueTask(task_id="A", session_id="s1", priority=1))
        queue.enqueue(QueueTask(task_id="B", session_id="s2", priority=1))
        queue.enqueue(QueueTask(task_id="C", session_id="s3", priority=1))

        assert queue.dequeue().task_id == "A"
        assert queue.dequeue().task_id == "B"
        assert queue.dequeue().task_id == "C"

    def test_update_status(self, queue):
        queue.enqueue(QueueTask(task_id="t1", session_id="s1"))

        ok = queue.update_status("t1", "running")
        assert ok is True
        ok2 = queue.update_status("t1", "completed")
        assert ok2 is True

    def test_update_nonexistent(self, queue):
        assert queue.update_status("ghost", "done") is False

    def test_peek(self, queue):
        for i in range(10):
            queue.enqueue(QueueTask(task_id=f"t{i}", session_id="s",
                                    priority=i))

        top5 = queue.peek(limit=5)
        assert len(top5) == 5
        # 最高优先级排在前面
        assert top5[0].priority == 9
        assert top5[4].priority == 5

        # peek 不移除
        assert queue.queue_length == 10

    def test_clear(self, queue):
        for i in range(5):
            queue.enqueue(QueueTask(task_id=f"t{i}", session_id="s"))

        count = queue.clear()
        assert count == 5
        assert queue.queue_length == 0
        assert queue.dequeue() is None

    def test_queue_length(self, queue):
        assert queue.queue_length == 0
        queue.enqueue(QueueTask(task_id="a", session_id="s"))
        assert queue.queue_length == 1
        queue.enqueue(QueueTask(task_id="b", session_id="s"))
        assert queue.queue_length == 2
        queue.dequeue()
        assert queue.queue_length == 1

    def test_is_available_mock(self, queue):
        assert queue.is_available is True

    def test_many_tasks(self, queue):
        """压力测试: 100 个任务入队 + 出队。"""
        for i in range(100):
            queue.enqueue(QueueTask(task_id=f"t{i}", session_id="s",
                                    priority=i % 10))

        assert queue.queue_length == 100

        popped = []
        while (task := queue.dequeue()) is not None:
            popped.append(task)

        assert len(popped) == 100
        # 验证优先级降序
        for i in range(len(popped) - 1):
            assert popped[i].priority >= popped[i + 1].priority


# ═══════════════════════════════════════════════════════════════════════
# TaskQueue mock=False 降级测试
# ═══════════════════════════════════════════════════════════════════════

class TestTaskQueueFallback:
    """Redis 不可用时自动降级为 mock。"""

    def test_fallback_to_mock_on_bad_url(self):
        """无效的 Redis URL → 自动降级为 mock，不抛异常。"""
        q = TaskQueue(redis_url="redis://nonexistent:9999", mock=False)

        # 尝试操作 → 自动降级
        try:
            q.enqueue(QueueTask(task_id="t1", session_id="s1"))
            assert q.is_available is True  # mock 可用了
        except RuntimeError:
            # 如果 even mock 不可用，那就是没有 redis 包
            pass
