"""
Phase 9A-2 — Redis 任务队列

基于 Redis Sorted Set 实现优先级任务队列，用于多用户并发场景下的
请求排队和负载削峰。

设计决策：
  - Redis ZSET（Sorted Set）：score = priority，value = task_id
  - 出队时 zpopmax：先处理高优先级任务
  - 同优先级内 FIFO（通过 score 微调实现）
  - Task 详情存储在单独的 String key 中（带 TTL 自动过期）
  - Redis 不可用时优雅降级（返回 None 而非抛异常）

面试要点：
  Q: "为什么用 Redis 而不是 RabbitMQ？"
  A: "MVP 阶段 Redis 零额外运维——已在项目中用于 Phase 7 的
      RedisPersistence，复用同一个 Redis 实例。Sorted Set 原生支持
      优先级排序。缺点是缺少 ACK（消息可能丢失），生产环境可升级
      到 RabbitMQ 或 Celery。Trade-off 明确。"

使用方式：
  from app.services.task_queue import TaskQueue, QueueTask

  queue = TaskQueue()
  await queue.enqueue(QueueTask(
      task_id="task-1", session_id="sess-1", user_id="u1",
      task_type="llm_chat", payload={"message": "Build API"},
      priority=5,
  ))
  task = await queue.dequeue()  # 取出最高优先级任务
  await queue.update_status(task.task_id, "completed")
"""

from __future__ import annotations

import datetime
import json
import time
from dataclasses import dataclass, field
from typing import Any


# ═══════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class QueueTask:
    """
    Redis 队列中的任务。

    Attributes:
        task_id:    唯一标识（UUID）
        session_id: 所属 Session
        user_id:    用户标识
        task_type:  任务类型（llm_chat / execute_tool / run_tests）
        payload:    任务内容（dict）
        status:     pending → running → completed | failed
        priority:   优先级（越高越优先，0 最低）
        created_at: ISO 时间戳
    """
    task_id: str
    session_id: str
    user_id: str = "default"
    task_type: str = "llm_chat"
    payload: dict = field(default_factory=dict)
    status: str = "pending"
    priority: int = 0
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> QueueTask:
        return cls(**data)


# ═══════════════════════════════════════════════════════════════════════
# TaskQueue
# ═══════════════════════════════════════════════════════════════════════

class TaskQueue:
    """
    基于 Redis Sorted Set 的优先级任务队列。

    设计原理：
      - 入队: ZADD queue_key {priority + micro_offset} task_id
              + SET task:{task_id} json_data EX ttl
      - 出队: ZPOPMAX queue_key → 获取 task_id →
              GET + DELETE task:{task_id} → 解析 QueueTask
      - 同优先级 FIFO：score = priority + (1 - timestamp / 1e10)

    初始化时传 `mock=True` 可在无 Redis 环境下使用内存模拟版本，
    方便本地开发和单元测试。
    """

    # Redis key 命名空间
    _QUEUE_KEY = "joyagent:task_queue"
    _TASK_PREFIX = "joyagent:task:"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        mock: bool = False,
    ):
        """
        Args:
            redis_url: Redis 连接 URL（如 redis://localhost:6379）
            mock:      True = 使用内存模拟（无需 Redis），适合测试
        """
        self._redis_url = redis_url
        self._mock = mock
        self._client: Any = None  # redis.Redis

    # ── 连接管理 ──────────────────────────────────────────

    def _get_client(self):
        """获取或创建 Redis 客户端（懒加载 + 单例缓存）。"""
        if self._mock:
            if self._client is None:
                self._client = _MockRedis()
            return self._client

        if self._client is not None:
            return self._client

        try:
            import redis
            self._client = redis.Redis.from_url(
                self._redis_url,
                socket_connect_timeout=3,
                decode_responses=True,
            )
            # 快速 ping 验证连接
            self._client.ping()
        except ImportError:
            raise RuntimeError(
                "redis package is required for TaskQueue. "
                "Install: pip install redis>=8.0.1"
            )
        except Exception as e:
            print(f"  [task_queue] Redis unavailable: {e} — "
                  f"falling back to mock", flush=True)
            self._mock = True
            self._client = _MockRedis()
            return self._client

        return self._client

    # ── 队列操作 ──────────────────────────────────────────

    def enqueue(self, task: QueueTask) -> str:
        """
        入队任务。

        任务详情存为 String key（带 TTL），任务 ID 入 ZSET。

        Args:
            task: QueueTask 实例

        Returns:
            task_id（与传入的一致）

        Raises:
            RuntimeError: Redis 不可用
        """
        client = self._get_client()
        task.status = "pending"

        # score = priority + 微偏移（保证同优先级 FIFO）
        micro = 1.0 - (time.time() / 1e10)
        score = float(task.priority) + micro

        # 1. 存储任务详情（String key，1 小时过期）
        client.set(
            f"{self._TASK_PREFIX}{task.task_id}",
            json.dumps(task.to_dict()),
            ex=3600,
        )

        # 2. 入队（ZSET）
        client.zadd(self._QUEUE_KEY, {task.task_id: score})

        return task.task_id

    def dequeue(self) -> QueueTask | None:
        """
        出队最高优先级任务。

        原子操作：ZPOPMAX 弹出最高分 → 读取详情 → 删除 key。
        队列为空时返回 None。

        Returns:
            QueueTask | None
        """
        client = self._get_client()

        # ZPOPMAX 弹出最高分（原子操作）
        result = client.zpopmax(self._QUEUE_KEY, count=1)
        if not result:
            return None

        task_id = result[0][0]  # [("task_id", score), ...]

        # 读取任务详情
        task_key = f"{self._TASK_PREFIX}{task_id}"
        raw = client.get(task_key)
        if raw is None:
            return None

        # 删除 key（已出队）
        client.delete(task_key)

        return QueueTask.from_dict(json.loads(raw))

    def update_status(self, task_id: str, status: str) -> bool:
        """
        更新任务状态（不改变队列位置）。

        Args:
            task_id: 任务 ID
            status:  新状态（running / completed / failed）

        Returns:
            True 更新成功，False 任务不存在
        """
        client = self._get_client()
        task_key = f"{self._TASK_PREFIX}{task_id}"
        raw = client.get(task_key)
        if raw is None:
            return False

        task_data = json.loads(raw)
        task_data["status"] = status
        client.set(task_key, json.dumps(task_data), keepttl=True)
        return True

    def peek(self, limit: int = 5) -> list[QueueTask]:
        """
        查看队列中前 N 个任务（不移出队列）。

        Args:
            limit: 查看数量

        Returns:
            list[QueueTask]: 按优先级降序
        """
        client = self._get_client()
        # ZREVRANGE: 按 score 降序
        task_ids = client.zrevrange(self._QUEUE_KEY, 0, limit - 1)
        tasks = []
        for tid in task_ids:
            raw = client.get(f"{self._TASK_PREFIX}{tid}")
            if raw:
                tasks.append(QueueTask.from_dict(json.loads(raw)))
        return tasks

    @property
    def queue_length(self) -> int:
        """当前队列中的任务数。"""
        client = self._get_client()
        return client.zcard(self._QUEUE_KEY)

    @property
    def is_available(self) -> bool:
        """Redis 是否可用。"""
        try:
            client = self._get_client()
            if self._mock:
                return True
            client.ping()
            return True
        except Exception:
            return False

    def clear(self) -> int:
        """清空队列，返回清除的任务数。"""
        client = self._get_client()
        count = client.zcard(self._QUEUE_KEY)
        task_ids = client.zrange(self._QUEUE_KEY, 0, -1)
        client.delete(self._QUEUE_KEY)
        for tid in task_ids:
            client.delete(f"{self._TASK_PREFIX}{tid}")
        return count


# ═══════════════════════════════════════════════════════════════════════
# Mock Redis（用于无 Redis 环境下的测试和本地开发）
# ═══════════════════════════════════════════════════════════════════════

class _MockRedis:
    """
    内存模拟 Redis 客户端，实现 TaskQueue 所需的 4 个命令：
      zadd, zpopmax, zcard, zrange, zrevrange, set, get, delete, ping
    """

    def __init__(self):
        self._data: dict[str, str] = {}          # key → JSON string
        self._expiry: dict[str, float] = {}       # key → expire timestamp
        self._zset: dict[str, float] = {}         # task_id → score

    def ping(self) -> bool:
        return True

    def set(self, key: str, value: str, ex: int | None = None,
            keepttl: bool = False) -> bool:
        self._data[key] = value
        if ex is not None:
            self._expiry[key] = time.time() + ex
        return True

    def get(self, key: str) -> str | None:
        # 检查过期
        if key in self._expiry and time.time() > self._expiry[key]:
            self._data.pop(key, None)
            self._expiry.pop(key, None)
            return None
        return self._data.get(key)

    def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in self._data:
                del self._data[k]
                count += 1
            self._expiry.pop(k, None)
            # 也清理 ZSET（_QUEUE_KEY 对应的数据）
            if k == "joyagent:task_queue":
                count += len(self._zset)
                self._zset.clear()
        return count

    def zadd(self, key: str, mapping: dict[str, float]) -> int:
        added = 0
        for task_id, score in mapping.items():
            if task_id not in self._zset:
                added += 1
            self._zset[task_id] = score
        return added

    def zpopmax(self, key: str, count: int = 1) -> list[tuple[str, float]]:
        if not self._zset:
            return []
        # 按 score 降序排序
        sorted_items = sorted(self._zset.items(), key=lambda x: -x[1])
        popped = sorted_items[:count]
        for task_id, _ in popped:
            del self._zset[task_id]
        return [(tid, score) for tid, score in popped]  # type: ignore

    def zcard(self, key: str) -> int:
        return len(self._zset)

    def zrange(self, key: str, start: int, end: int) -> list[str]:
        sorted_items = sorted(self._zset.items(), key=lambda x: x[1])
        return [tid for tid, _ in sorted_items[start:end + 1]]

    def zrevrange(self, key: str, start: int, end: int) -> list[str]:
        sorted_items = sorted(self._zset.items(), key=lambda x: -x[1])
        return [tid for tid, _ in sorted_items[start:end + 1]]
