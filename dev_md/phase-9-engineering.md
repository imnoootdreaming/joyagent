# Phase 9：工程化

> **返回主索引：** [0-1 coding.md](0-1%20coding.md)
> **上一阶段：** [Phase 8: MCP Plugin System](phase-8-mcp.md)

---

## 一、目标与定位

### 目标
将项目从 Demo 升级为**可多用户并发访问的 Agent 平台**：PostgreSQL 持久化、Redis 任务队列、WebSocket 实时日志、Human-in-the-Loop 权限审批、Docker Compose 一键部署。

### 范围调整 ⚠️

原文档将所有工程化内容放在一个 Phase，实际工作量过大。**调整为两个子阶段：**

- **9A（核心，必做）：** PostgreSQL + Redis 数据层、WebSocket 实时日志、Human-in-the-Loop
- **9B（可选，有精力再做）：** CI/CD、GitHub Actions、Benchmark 系统

### 在整体架构中的位置
Phase 9 为前面所有 Phase 补充**生产级基础设施**，使项目从"能跑"变为"可靠"。

---

## 二、前置依赖

| 依赖 | 用途 |
|------|------|
| Phase 1-8 的功能代码 | 被工程化的主体 |
| sqlalchemy + asyncpg | PostgreSQL ORM + 异步驱动 |
| alembic | 数据库迁移 |
| redis | 任务队列 + 状态缓存 |
| websockets | WebSocket 支持 |
| docker-compose | 一键部署编排 |

```bash
uv add sqlalchemy asyncpg alembic redis websockets
```

---

## 三、目录结构

```text
app/
├── api/
│   ├── agent.py              # REST API（已有，重构）
│   ├── ws.py                 # WebSocket 端点（新增）
│   └── dependencies.py       # FastAPI 依赖注入（新增）
│
├── core/
│   ├── config.py             # 全局配置（增强）
│   ├── security.py           # 权限系统（新增）
│   └── permissions.py        # Human-in-the-Loop 审批（新增）
│
├── db/
│   ├── __init__.py
│   ├── session.py            # 数据库连接管理
│   ├── models/
│   │   ├── __init__.py
│   │   ├── session.py        # Session 模型
│   │   └── task_log.py       # Task Log 模型
│   └── repositories/
│       ├── session_repo.py   # Session CRUD
│       └── task_log_repo.py  # Task Log CRUD
│
├── services/
│   ├── llm_service.py        # LLM Service（已有）
│   ├── task_queue.py         # Redis 任务队列（新增）
│   └── log_streamer.py       # WebSocket 日志推送（新增）
│
└── main.py                   # FastAPI 入口（增强）
```

---

## 四、核心数据模型 / Schema 定义

### 4.1 PostgreSQL 数据模型

```python
# db/models/session.py
from sqlalchemy import Column, String, Text, DateTime, JSON, Integer
from sqlalchemy.orm import declarative_base
import datetime

Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(String(36), primary_key=True)          # UUID
    user_id = Column(String(36), index=True)           # 用户 ID
    title = Column(String(255))
    status = Column(String(20), default="active")      # active | completed | failed
    
    # Agent 状态快照（JSON 序列化 AgentState）
    agent_state = Column(JSON, nullable=True)
    
    # 统计
    total_messages = Column(Integer, default=0)
    total_tool_calls = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class TaskLog(Base):
    __tablename__ = "task_logs"
    
    id = Column(String(36), primary_key=True)
    session_id = Column(String(36), index=True)
    
    event_type = Column(String(50))  # "llm_call" | "tool_call" | "tool_result" | "error" | "user_approval"
    event_data = Column(JSON)        # 事件详细数据
    
    # 工具调用相关
    tool_name = Column(String(100), nullable=True)
    tool_args = Column(JSON, nullable=True)
    tool_result = Column(Text, nullable=True)
    tool_success = Column(Integer, nullable=True)
    
    # Token 统计
    tokens_used = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
```

### 4.2 Human-in-the-Loop 权限模型

```python
from enum import Enum
from dataclasses import dataclass

class PermissionLevel(Enum):
    AUTO = "auto"              # 自动执行（read_file, git_status）
    CONFIRM = "confirm"        # 需要用户确认（write_file, execute_shell）
    DENY = "deny"              # 禁止（rm -rf, sudo, curl 等危险操作）

@dataclass
class PermissionRule:
    """单个权限规则"""
    tool_name: str
    level: PermissionLevel
    condition: str | None = None  # 可选的条件表达式（如 "file_path.endswith('.env')" → DENY）

# 默认权限规则表
DEFAULT_PERMISSIONS = [
    PermissionRule("read_file", PermissionLevel.AUTO),
    PermissionRule("git_status", PermissionLevel.AUTO),
    PermissionRule("git_diff", PermissionLevel.AUTO),
    PermissionRule("code_search", PermissionLevel.AUTO),
    PermissionRule("ast_analyze", PermissionLevel.AUTO),
    PermissionRule("pytest_run", PermissionLevel.AUTO),
    
    PermissionRule("write_file", PermissionLevel.CONFIRM),
    PermissionRule("execute_shell", PermissionLevel.CONFIRM),
    PermissionRule("apply_patch", PermissionLevel.CONFIRM),
    PermissionRule("git_commit", PermissionLevel.CONFIRM),
    
    # 危险命令模式（条件拒绝）
    PermissionRule("execute_shell", PermissionLevel.DENY, 
                   condition="command_contains('rm -rf') or command_contains('sudo')"),
]

@dataclass
class ApprovalRequest:
    """发给用户的审批请求"""
    request_id: str
    tool_name: str
    tool_args: dict
    risk_level: str             # "low" | "medium" | "high"
    reason: str                 # 为什么需要审批
    timestamp: str
    
@dataclass
class ApprovalResponse:
    """用户的审批结果"""
    request_id: str
    approved: bool
    approved_by: str
    comment: str | None = None
```

### 4.3 任务队列模型

```python
@dataclass
class QueueTask:
    """Redis 任务队列中的任务"""
    task_id: str
    session_id: str
    user_id: str
    task_type: str              # "llm_chat" | "execute_tool" | "run_tests"
    payload: dict
    status: str = "pending"     # pending → running → completed | failed
    priority: int = 0           # 越高越优先
    created_at: str = ""
```

---

## 五、详细开发清单（含 HOW）

### 9A-1：PostgreSQL 数据层（1.5 小时）

```python
# db/session.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/joyagent")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
```

**Repository Pattern 实现：**
```python
# db/repositories/session_repo.py
class SessionRepository:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create(self, session: Session) -> Session:
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return session
    
    async def get(self, session_id: str) -> Session | None:
        return await self.db.get(Session, session_id)
    
    async def update_state(self, session_id: str, agent_state: dict) -> None:
        session = await self.get(session_id)
        if session:
            session.agent_state = agent_state
            await self.db.commit()
    
    async def list_by_user(self, user_id: str, limit: int = 20) -> list[Session]:
        result = await self.db.execute(
            select(Session)
            .where(Session.user_id == user_id)
            .order_by(Session.updated_at.desc())
            .limit(limit)
        )
        return result.scalars().all()
```

### 9A-2：Redis 任务队列（1 小时）

```python
# services/task_queue.py
import json
import redis.asyncio as redis

class TaskQueue:
    """基于 Redis 的任务队列"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.Queue_KEY = "joyagent:task_queue"
        self.TASK_PREFIX = "joyagent:task:"
    
    async def enqueue(self, task: QueueTask, priority: int = 0) -> str:
        """入队任务"""
        task.status = "pending"
        task.created_at = datetime.datetime.utcnow().isoformat()
        
        # 存储任务详情
        await self.redis.set(
            f"{self.TASK_PREFIX}{task.task_id}",
            json.dumps(task.__dict__),
            ex=3600,  # 1 小时过期
        )
        
        # 入队（使用 sorted set 实现优先级队列）
        await self.redis.zadd(self.QUEUE_KEY, {task.task_id: priority})
        
        return task.task_id
    
    async def dequeue(self) -> QueueTask | None:
        """出队最高优先级任务"""
        # 原子操作：弹出最高分
        result = await self.redis.zpopmax(self.QUEUE_KEY, count=1)
        if not result:
            return None
        
        task_id = result[0][0]
        task_data = await self.redis.get(f"{self.TASK_PREFIX}{task_id}")
        if not task_data:
            return None
        
        return QueueTask(**json.loads(task_data))
    
    async def update_status(self, task_id: str, status: str) -> None:
        """更新任务状态"""
        task_data = await self.redis.get(f"{self.TASK_PREFIX}{task_id}")
        if task_data:
            task = json.loads(task_data)
            task["status"] = status
            await self.redis.set(
                f"{self.TASK_PREFIX}{task_id}",
                json.dumps(task),
                keepttl=True,
            )
    
    async def get_queue_length(self) -> int:
        return await self.redis.zcard(self.QUEUE_KEY)
```

**面试要点：Redis 做任务队列的利弊**
- ✅ 优点：轻量、快速、无额外依赖
- ❌ 缺点：无 ACK 机制（消息可能丢失）、不支持延迟消息、无死信队列
- 🎯 面试话术："MVP 阶段用 Redis 快速验证，生产环境可升级到 RabbitMQ 或 Celery"

### 9A-3：WebSocket 实时日志（1 小时）

```python
# api/ws.py
from fastapi import WebSocket, WebSocketDisconnect

class LogStreamer:
    """WebSocket 日志推送管理器"""
    
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}  # session_id → [ws]
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)
    
    async def broadcast(self, session_id: str, event: dict):
        """向某个 session 的所有 WebSocket 连接推送事件"""
        if session_id not in self.active_connections:
            return
        
        dead_connections = []
        for ws in self.active_connections[session_id]:
            try:
                await ws.send_json(event)
            except Exception:
                dead_connections.append(ws)
        
        for ws in dead_connections:
            self.active_connections[session_id].remove(ws)

log_streamer = LogStreamer()

# FastAPI WebSocket 端点
@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await log_streamer.connect(websocket, session_id)
    try:
        while True:
            # 保持连接，等待客户端消息（如心跳）
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        log_streamer.disconnect(websocket, session_id)
```

**事件格式：**
```json
{
  "type": "llm_call" | "tool_call" | "tool_result" | "error" | "progress" | "approval_request",
  "timestamp": "2025-01-01T00:00:00Z",
  "data": { ... }
}
```

### 9A-4：Human-in-the-Loop 权限审批（1.5 小时）⭐ 核心

```python
# core/permissions.py
class PermissionManager:
    """Human-in-the-Loop 权限管理"""
    
    def __init__(self, rules: list[PermissionRule] = None):
        self.rules = rules or DEFAULT_PERMISSIONS
    
    def check(self, tool_name: str, tool_args: dict) -> PermissionLevel:
        """检查工具是否需要审批"""
        for rule in self.rules:
            if rule.tool_name != tool_name:
                continue
            
            # 检查条件（如果有）
            if rule.condition:
                if self._evaluate_condition(rule.condition, tool_args):
                    return rule.level
            else:
                return rule.level
        
        # 默认：未知工具需要确认
        return PermissionLevel.CONFIRM
    
    async def request_approval(self, approval: ApprovalRequest, 
                               session_id: str) -> ApprovalResponse:
        """
        向用户请求审批。
        通过 WebSocket 推送审批请求，等待用户响应。
        """
        # 推送到 WebSocket
        await log_streamer.broadcast(session_id, {
            "type": "approval_request",
            "data": approval.__dict__,
        })
        
        # 等待用户响应（带超时）
        try:
            response = await self._wait_for_response(approval.request_id, timeout=60)
            return response
        except TimeoutError:
            return ApprovalResponse(
                request_id=approval.request_id,
                approved=False,
                comment="Timeout: No response from user",
            )
    
    def _evaluate_condition(self, condition: str, args: dict) -> bool:
        """评估权限规则的条件表达式（安全沙箱环境中执行）"""
        # 简化版：字符串匹配
        if "rm -rf" in condition:
            return "rm -rf" in str(args.get("command", ""))
        if "sudo" in condition:
            return "sudo" in str(args.get("command", ""))
        if ".env" in condition:
            return ".env" in str(args.get("path", ""))
        return False
```

**审批流程：**
```text
Agent 要执行工具
    │
    ▼
PermissionManager.check(tool_name, args)
    │
    ├── AUTO ──▶ 直接执行
    │
    ├── CONFIRM ──▶ WebSocket 推审批请求 ──▶ 用户 Approve/Deny
    │                                               │
    │                                     ┌─────────┴─────────┐
    │                                   Approve            Deny
    │                                     │                  │
    │                                     ▼                  ▼
    │                                  执行工具         返回 "User denied"
    │
    └── DENY ──▶ 直接拒绝，返回 "Operation not allowed"
```

### 9A-5：集成到 Agent Runtime（1 小时）
- 在 ReAct Loop 中插入权限检查（Phase 1 §5 Step 5 预留的 `is_dangerous` 钩子）
- 在每次工具调用前后记录 TaskLog 到 PostgreSQL
- 在 Agent 执行过程中通过 WebSocket 推送实时事件
- Session 启动/结束时持久化到 PostgreSQL

### 9B-1：Docker Compose（30 分钟）

```yaml
# docker-compose.yml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://joyagent:joyagent@db:5432/joyagent
      - REDIS_URL=redis://redis:6379
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL}
    depends_on:
      - db
      - redis
    volumes:
      - ./workspace:/workspace
      - /var/run/docker.sock:/var/run/docker.sock  # Docker-in-Docker

  db:
    image: postgres:16
    environment:
      POSTGRES_USER: joyagent
      POSTGRES_PASSWORD: joyagent
      POSTGRES_DB: joyagent
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redisdata:/data

volumes:
  pgdata:
  redisdata:
```

### 9B-2：Benchmark 评估系统 + Bad Case 回归（1 小时）

**设计目标：** 建立自动化的 Agent 能力评估体系，确保每次 Prompt/Skill 变更都有量化数据支撑，防止回归。

```python
# ─── Benchmark 任务定义 ──────────────────────────

# 自建 10 个编程任务作为 Benchmark（覆盖典型场景）
BENCHMARK_TASKS = [
    {
        "id": "calc-01",
        "task": "创建 calculator.py，包含 add/subtract/multiply/divide",
        "category": "algorithm",
        "expected_files": ["calculator.py"],
        "test": "pytest test_calculator.py --json-report",
    },
    {
        "id": "crud-01",
        "task": "创建 User CRUD API，包含 create/read/update/delete 四个端点",
        "category": "crud",
        "expected_files": ["app.py", "models.py", "requirements.txt"],
        "test": "pytest test_api.py --json-report",
    },
    {
        "id": "file-01",
        "task": "读取 data.csv 文件，计算每一列的均值和中位数，结果写入 summary.json",
        "category": "file_op",
        "expected_files": ["process.py", "summary.json"],
        "test": "pytest test_process.py --json-report",
    },
    # ... 共 10 个任务（覆盖 crud / algorithm / file_op / api_dev 四类场景）
]


# ─── Prompt 版本管理 ──────────────────────────────

@dataclass
class PromptVersion:
    """Prompt 版本记录"""
    version: str                  # "v1.0.0"
    system_prompt_hash: str       # SHA256 of system prompt
    skill_prompts_hash: dict      # {skill_name: sha256}
    benchmark_pass_rate: float    # Benchmark 完成率
    created_at: str


class PromptVersionManager:
    """Prompt 版本管理器。
    
    核心原则：每次 Prompt 变更必须跑 Benchmark 并记录完成率。
    新版本完成率低于旧版本 → 阻止上线。
    """
    
    def __init__(self):
        self.versions: list[PromptVersion] = []
        self._min_pass_rate = 0.60  # 最低通过阈值
    
    def record_version(self, system_prompt: str, 
                       skill_prompts: dict[str, str],
                       benchmark_result: dict) -> PromptVersion:
        """记录新的 Prompt 版本及其 Benchmark 结果"""
        import hashlib
        
        version_num = len(self.versions) + 1
        v = PromptVersion(
            version=f"v{version_num}.0.0",
            system_prompt_hash=hashlib.sha256(system_prompt.encode()).hexdigest()[:12],
            skill_prompts_hash={
                name: hashlib.sha256(p.encode()).hexdigest()[:12]
                for name, p in skill_prompts.items()
            },
            benchmark_pass_rate=benchmark_result["pass_rate"],
            created_at=datetime.datetime.utcnow().isoformat(),
        )
        self.versions.append(v)
        return v
    
    def check_regression(self, current_v: PromptVersion, 
                         previous_v: PromptVersion) -> bool:
        """检查是否发生回归（新版本完成率显著低于旧版本）"""
        drop = previous_v.benchmark_pass_rate - current_v.benchmark_pass_rate
        return drop > 0.10  # 完成率下降 >10% 视为回归
    
    def can_deploy(self, v: PromptVersion) -> tuple[bool, str]:
        """判断此版本能否上线"""
        if v.benchmark_pass_rate < self._min_pass_rate:
            return False, f"完成率 {v.benchmark_pass_rate:.1%} 低于阈值 {self._min_pass_rate:.1%}"
        if len(self.versions) >= 2:
            prev = self.versions[-2]
            if self.check_regression(v, prev):
                return False, f"完成率从 {prev.benchmark_pass_rate:.1%} 降至 {v.benchmark_pass_rate:.1%}（回归）"
        return True, "OK"


# ─── Bad Case 回放测试 ────────────────────────────

@dataclass
class RegressionTest:
    """单个回归测试 = 曾失败的历史 Bad Case"""
    case_id: str
    task_description: str
    task_category: str
    original_error: str           # 原始失败原因
    fix_prompt_version: str       # 修复后的 Prompt 版本
    expected_pass: bool = True    # 期望在新 Prompt 下通过


class BadCaseRegressionRunner:
    """Bad Case 回放测试执行器。
    
    流程：
    1. 从 BadCaseAnalyzer（Phase 5）加载所有已收集的 Bad Case
    2. 用新 Prompt 版本重新执行每个 Bad Case 对应的任务
    3. 对比新旧结果：修复了哪些？回归了哪些？
    4. 输出回归测试报告
    """
    
    def __init__(self, agent, benchmark_runner, 
                 bad_case_analyzer: "BadCaseAnalyzer"):
        self.agent = agent
        self.benchmark = benchmark_runner
        self.bad_cases = bad_case_analyzer
    
    async def run_regression(self, new_prompt_version: str) -> dict:
        """对新 Prompt 版本执行回放测试"""
        results = {
            "version": new_prompt_version,
            "total": 0,
            "fixed": 0,          # 以前失败、现在通过
            "still_failed": 0,   # 以前失败、现在仍失败
            "regression": 0,     # 以前通过、现在失败（新引入的回归）
            "details": [],
        }
        
        for case in self.bad_cases.cases:
            results["total"] += 1
            
            # 重新执行任务
            result = await self.agent.chat(case.task_description)
            passed = self._verify_result(result, case)
            
            detail = {
                "case_id": case.final_state,
                "task": case.task_description[:80],
                "original_error": case.error_categories,
                "new_result": "passed" if passed else "failed",
            }
            results["details"].append(detail)
            
            if passed:
                results["fixed"] += 1       # ✅ 修复成功
            else:
                results["still_failed"] += 1  # ❌ 仍未修复
        
        # 检查是否有新引入的回归（对 Benchmark 任务集中之前通过的做对比）
        # ... 
        
        results["fix_rate"] = results["fixed"] / results["total"] if results["total"] else 0
        return results
    
    def _verify_result(self, result: dict, case: BadCase) -> bool:
        """验证单个任务是否通过"""
        return result.get("test_passed", False)
    
    def generate_regression_report(self, results: dict) -> str:
        """生成回归测试报告"""
        report = f"""
╔══════════════════════════════════════════════════════════╗
║           Bad Case 回归测试报告                            ║
╠══════════════════════════════════════════════════════════╣
║  Prompt 版本: {results['version']:<44s}║
║  总回放数: {results['total']:>4d}                                         ║
║  修复成功: {results['fixed']:>4d}  ({results.get('fix_rate', 0)*100:.0f}%)                                  ║
║  仍未修复: {results['still_failed']:>4d}                                         ║
║  新回归:   {results['regression']:>4d}                                         ║
╚══════════════════════════════════════════════════════════╝
"""
        return report


# ─── 完整评估流水线 ────────────────────────────────

class AgentEvaluationPipeline:
    """Agent 评估流水线：Benchmark → Bad Case 回归 → 上线判断"""
    
    def __init__(self, agent, benchmark_tasks, bad_case_analyzer,
                 prompt_manager: PromptVersionManager):
        self.agent = agent
        self.tasks = benchmark_tasks
        self.bad_case_runner = BadCaseRegressionRunner(
            agent, None, bad_case_analyzer
        )
        self.prompt_mgr = prompt_manager
    
    async def evaluate(self, system_prompt: str,
                       skill_prompts: dict[str, str]) -> dict:
        """
        完整评估流水线：
        1. 运行 Benchmark → 获取 pass_rate
        2. 运行 Bad Case 回归 → 检查修复率 + 是否有回归
        3. 判断是否可以上线
        """
        # Step 1: Benchmark
        benchmark_result = await self._run_benchmark()
        
        # Step 2: 记录 Prompt 版本
        version = self.prompt_mgr.record_version(
            system_prompt, skill_prompts, benchmark_result
        )
        
        # Step 3: Bad Case 回归
        regression_result = await self.bad_case_runner.run_regression(
            version.version
        )
        
        # Step 4: 上线判断
        can_deploy, reason = self.prompt_mgr.can_deploy(version)
        
        return {
            "version": version.version,
            "benchmark": benchmark_result,
            "regression": regression_result,
            "can_deploy": can_deploy,
            "reason": reason,
        }
    
    async def _run_benchmark(self) -> dict:
        """执行 Benchmark"""
        results = []
        for task in self.tasks:
            result = await self.agent.chat(task["task"])
            passed = self._verify_task(task, result)
            results.append({"task_id": task["id"], "passed": passed})
        
        pass_rate = sum(1 for r in results if r["passed"]) / len(results)
        return {"results": results, "pass_rate": pass_rate, "total": len(results)}
    
    def _verify_task(self, task: dict, result: dict) -> bool:
        """验证单个 Benchmark 任务"""
        return result.get("test_passed", False)


# 评估流水线总结
#
# ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────┐
# │ 改 Prompt   │───▶│ Benchmark    │───▶│ Bad Case     │───▶│ 上线    │
# │ 或 Skill    │    │ 10 题评估    │    │ 回放测试      │    │ 判断    │
# └─────────────┘    └──────┬───────┘    └──────┬───────┘    └────┬────┘
#                          │                   │                 │
#                          ▼                   ▼                 ▼
#                    pass_rate ≥ 60%     fix_rate ≥ 30%     can_deploy?
#                           + 无回归           + 无新回归       + 记录版本
```

**面试要点：** 这条完整展示了"如何衡量 AI Agent 好不好"和"如何让它变好"——大厂面试官最看重的数据驱动思维。Benchmark → Bad Case → 回归 → 上线判断的四步流水线，是 Production AI 系统的标准工程范式。

---

## 六、关键代码模式与伪代码

### 6.1 集成权限检查后的 ReAct Loop

```python
async def agent_runtime_with_permissions(user_input, state, session_id):
    # Anthropic 原生格式：messages 是纯 dict 列表
    messages = list(state.messages)  # 浅拷贝
    messages.append({"role": "user", "content": user_input})
    perm_manager = PermissionManager()
    client = get_or_create_client()

    while state.iterations < state.max_iterations:
        response = client.messages.create(
            model=MODEL, system=SYSTEM_PROMPT,
            messages=messages, tools=TOOLS, max_tokens=4096,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return {"response": extract_text(response.content)}

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            # ⬇ Human-in-the-Loop 检查
            permission = perm_manager.check(block.name, block.input)

            if permission == PermissionLevel.DENY:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Operation denied for security reasons: {block.name}",
                })
                await log_streamer.broadcast(session_id, {
                    "type": "tool_denied",
                    "data": {"tool": block.name, "reason": "Security policy"}
                })
                continue

            elif permission == PermissionLevel.CONFIRM:
                approval = await perm_manager.request_approval(
                    ApprovalRequest(
                        request_id=str(uuid4()),
                        tool_name=block.name,
                        tool_args=block.input,
                        risk_level="medium",
                        reason=f"Tool '{block.name}' requires user confirmation",
                    ),
                    session_id,
                )
                if not approval.approved:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"User denied execution of {block.name}",
                    })
                    continue

            # 执行工具
            result = await tool_registry.execute(block.name, **block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result),
            })

            # 记录到 TaskLog（数据库审计）
            await task_log_repo.create(TaskLog(
                session_id=session_id,
                event_type="tool_call",
                tool_name=block.name,
                tool_args=block.input,
                tool_success=result.success if hasattr(result, 'success') else True,
            ))

            # WebSocket 实时推送
            await log_streamer.broadcast(session_id, {
                "type": "tool_result",
                "data": {"tool": block.name, "success": True}
            })

        messages.append({"role": "user", "content": tool_results})
```

---

## 七、完成标志

### 9A 完成
- [ ] PostgreSQL：Session 和 TaskLog 能正确 CRUD
- [ ] Redis 任务队列：任务能正常入队/出队/状态更新
- [ ] WebSocket：Agent 执行过程中前端能收到实时事件流
- [ ] Human-in-the-Loop：危险操作弹出审批，AUTO 操作直接执行
- [ ] Session 恢复：重启后能从 PostgreSQL 恢复 Agent 状态
- [ ] 多用户：两个用户同时请求不会互相干扰

### 9B 完成（可选）
- [ ] `docker compose up` 一键启动全栈
- [ ] Benchmark 能自动运行并输出完成率
- [ ] Prompt 版本管理：每次 Prompt 变更记录版本 + Benchmark 完成率
- [ ] Bad Case 回放测试：新 Prompt 在历史 Bad Case 上跑回归
- [ ] 上线判断：完成率低于阈值或发生回归时自动阻止上线

### 自测用例

```bash
# 测试 1：Session 持久化
curl -X POST /api/chat -d '{"message": "创建 test.txt", "session_id": "test-session-1"}'
# 重启服务
curl -X GET /api/sessions/test-session-1
# 期望：返回 session 信息，包含之前的对话

# 测试 2：WebSocket 实时日志
# 用 websocat 或浏览器连接 ws://localhost:8000/ws/test-session-1
# 然后发 POST /api/chat → WebSocket 收到 tool_call 事件

# 测试 3：Human-in-the-Loop
curl -X POST /api/chat -d '{"message": "删除所有 .py 文件", "session_id": "test-2"}'
# 期望：WebSocket 收到 approval_request，等待用户确认

# 测试 4：多用户并发
# 同时发起 3 个 POST /api/chat 请求，验证互不干扰
```

---

## 八、文档差距分析

### 原文档缺失的内容及补充

| # | 原文档问题 | 实际开发需要 | 本文档补充位置 |
|---|-----------|-------------|-------------|
| 1 | **Phase 9 内容过多** | 拆分为 9A（核心）和 9B（可选），优先保证数据层 + WebSocket + 权限 | §1 |
| 2 | **完全没有 Human-in-the-Loop** | Agent 安全的核心设计：权限分级（AUTO/CONFIRM/DENY）+ WebSocket 推送审批 + 超时处理 | §4.2, §5 (9A-4) |
| 3 | Redis 任务队列没说可靠性问题 | Redis List 做队列无 ACK；生产需 RabbitMQ。面试必须提及这个 trade-off | §5 (9A-2) |
| 4 | WebSocket 没说具体推送什么事件 | 需要定义事件 schema：llm_call / tool_call / tool_result / error / approval_request / progress | §5 (9A-3) |
| 5 | Session 存储没说怎么恢复 | Agent State 需要序列化为 JSON 存 PG，恢复时反序列化 + 从 ChromaDB 加载 Long-term Memory | §5 (9A-1) |
| 6 | 没说 Repository Pattern | 数据访问层抽象，使 Agent 不直接依赖 SQLAlchemy | §5 (9A-1) |
| 7 | CI/CD 过于笼统 | Docker Compose 一键部署就够了；GitHub Actions 是锦上添花 | §5 (9B-1) |
| 8 | 没有 Benchmark 评估 | 10 个编程任务的自动化评估对简历有巨大加分 | §5 (9B-2) |

### Human-in-the-Loop 设计的重要性

**这是面试中 Agent 安全的核心考点。** 必问："你的 Agent 怎么防止执行危险操作？"

**标准答案结构：**
1. **技术层隔离：** Docker Sandbox（Phase 5）— 进程/网络/文件系统三层隔离
2. **流程层审批：** Human-in-the-Loop（Phase 9）— 危险操作必须用户确认
3. **设计层分级：** 工具安全等级（Phase 2 `is_dangerous`）— AUTO / CONFIRM / DENY
4. **审计层追溯：** TaskLog（Phase 9）— 所有操作记录可审计

---

## 九、面试考点映射

| 面试题 | 答案要点 | 本文档参考 |
|--------|---------|-----------|
| **FastAPI 为什么适合 Agent 系统？** | 1) 原生 async/await 适合 LLM 的 I/O 密集型调用 2) WebSocket 内置支持 3) 自动 OpenAPI 文档 4) Pydantic 数据校验。 | §5 (9A-3) |
| **WebSocket 在项目中的作用？** | 实时推送 Agent 执行过程：LLM 推理进度、工具调用结果、错误信息、审批请求。让用户感知 Agent 在做什么，不是黑盒等待。 | §5 (9A-3) |
| **Redis 为什么适合任务队列？** | 快（内存）、简单（List/ZSet 即可实现）、生态成熟。但无 ACK——MVP 够用，生产可选 RabbitMQ 或 Celery。 | §5 (9A-2) |
| **Redis 与 RabbitMQ 的区别？** | Redis = 内存数据库，做队列靠 List/BLPOP，无 ACK，消息可能丢失。RabbitMQ = 专业消息队列，ACK + 死信 + 延迟 + 持久化。MVP 用 Redis，生产用 RabbitMQ。 | §5 (9A-2) |
| **Session 如何存储？** | PostgreSQL JSON 字段存储 AgentState（Plan、Messages Summary、Tool History）。启动时从 PG 恢复 State，同时从 ChromaDB 加载 Long-term Memory。 | §4.1, §5 (9A-1) |
| **如何实现任务恢复？** | Checkpoint（LangGraph）+ Session 持久化（PG）+ Memory 持久化（ChromaDB）。三层状态都可恢复。 | §5 (9A-1), Phase 3 |
| **如何扩展到多用户系统？** | 1) Session 隔离（每个用户独立 Session）2) Task Queue 排队 3) Docker Sandbox 每用户独立容器 4) PostgreSQL 多租户（user_id 过滤）。 | §4.1, §5 |
| **如何评估 Agent 的能力？** | 不靠人工感觉。三层评估体系：1) Benchmark（10 题自动化评估，覆盖 CRUD/算法/文件/API 四类场景）→ pass_rate；2) Bad Case 回归（历史失败案例回放）→ fix_rate；3) Prompt 版本对比 → 防止回归。数据驱动，可重复，可追溯。 | §5 (9B-2) |
| **怎么防止 Prompt 改坏？** | Prompt 版本管理 + 自动回归检查：每次变更记录 version + prompt_hash + pass_rate；新版本完成率下降 >10% 自动阻止上线；Bad Case 回放确保已修复的不再坏。 | §5 (9B-2) |
