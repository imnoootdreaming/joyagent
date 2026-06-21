# Phase 6：Memory System

> **返回主索引：** [0-1 coding.md](0-1%20coding.md)
> **上一阶段：** [Phase 5: Docker Sandbox + Auto Testing](phase-5-sandbox-testing.md)
> **下一阶段：** [Phase 7: Multi-Agent](phase-7-multi-agent.md)

---

## 一、目标与定位

### 目标
构建三级记忆系统，支持 Agent 执行长时间、跨会话的任务，并能够从历史经验中学习。

### 三级记忆架构

```
┌─────────────────────────────────────────────────────┐
│                  Memory System                        │
│                                                       │
│  ┌─────────────────┐  ┌────────────┐  ┌───────────┐ │
│  │ Short-term       │  │ Long-term  │  │ Reflection│ │
│  │ Memory           │  │ Memory     │  │ Memory    │ │
│  │                  │  │            │  │           │ │
│  │ 滑动窗口         │  │ ChromaDB   │  │ 错误嵌入  │ │
│  │ + 摘要压缩       │  │ 向量检索    │  │ 经验召回  │ │
│  │                  │  │            │  │           │ │
│  │ 范围：当前会话   │  │ 范围：跨会话│  │ 范围：长期 │ │
│  │ 容量：~8K tokens │  │ 容量：50K+  │  │ 容量：10K+ │ │
│  └─────────────────┘  └────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────┘
```

### 在整体架构中的位置
Phase 1-5 的 Agent 每次对话都是"失忆"的——重启后完全忘记之前做过什么。Phase 6 让 Agent 拥有持续的记忆能力。

### 本 Phase 不做什么
- ❌ 不做多用户记忆隔离（Phase 9）
- ❌ 不做记忆的复杂推理（如知识图谱）

---

## 二、前置依赖

| 依赖 | 用途 |
|------|------|
| Phase 5 完成 | Sandbox + Testing |
| chromadb | 向量数据库 |
| sentence-transformers | Embedding 生成（本地，免费） |
| tiktoken | Token 计数 |
| Redis | Long-term Memory 缓存层 |

```bash
uv add chromadb tiktoken redis sentence-transformers
```

---

## 三、目录结构

```text
app/
├── memory/
│   ├── __init__.py
│   ├── short_term.py        # Short-term Memory：滑动窗口 + 摘要
│   ├── long_term.py         # Long-term Memory：ChromaDB 向量存储
│   ├── reflection.py        # Reflection Memory：错误经验
│   ├── summary.py           # Context Compression：上下文压缩
│   ├── embeddings.py        # Embedding 服务封装
│   └── token_manager.py     # Token 计数 + Context Window 管理
```

---

## 四、核心数据模型 / Schema 定义

### 4.1 Short-term Memory

```python
from dataclasses import dataclass, field

@dataclass
class ShortTermMemory:
    """
    当前会话的工作记忆。
    策略：滑动窗口（最近 N 条消息）+ 早期对话摘要。
    """
    messages: list[dict] = field(default_factory=list)  # Anthropic 原生 dict 格式
    summary: str = ""                          # 早期对话的摘要
    max_messages: int = 50                     # 最多保留的消息数
    summary_trigger_tokens: int = 6000         # 达到此 Token 数触发摘要压缩
    
    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        # 超过限制 → 触发压缩
        if self._count_tokens() > self.summary_trigger_tokens:
            self._compress()
    
    def get_context(self) -> list[dict]:
        """返回给 LLM 的上下文：摘要 + 最近的对话（Anthropic dict 格式）

        注意：摘要不放在 system 参数中，而是作为 user 消息注入 ——
        这样保持了 messages 列表的自包含性，便于 LangGraph 的 add_messages reducer。
        真正的 system prompt 由 Agent 层单独传入 client.messages.create(system=...)。
        """
        result = []
        if self.summary:
            result.append({"role": "user", "content": f"[Previous context summary]: {self.summary}"})
        result.extend(self.messages[-self.max_messages:])
        return result
    
    def _compress(self):
        """将前半部分消息压缩为摘要"""
        half = len(self.messages) // 2
        old_messages = self.messages[:half]
        self.messages = self.messages[half:]
        # 调用 LLM 生成摘要（见 summary.py）
        self.summary = generate_summary(old_messages, self.summary)
    
    def _count_tokens(self) -> int:
        return count_tokens(self.messages)
```

### 4.2 Long-term Memory

```python
from dataclasses import dataclass

@dataclass
class MemoryEntry:
    """存入 Long-term Memory 的记录"""
    id: str
    content: str               # 原始内容
    embedding: list[float]     # 向量表示
    memory_type: str           # "code_snippet" | "task_result" | "conversation" | "error"
    metadata: dict             # {file_path, task_id, timestamp, tags}
    created_at: str

@dataclass
class MemoryQueryResult:
    """记忆检索结果"""
    entry: MemoryEntry
    similarity_score: float     # 0.0 ~ 1.0
```

### 4.3 Reflection Memory

```python
@dataclass
class ErrorExperience:
    """一次错误经验的记录"""
    id: str
    error_type: str            # "SyntaxError" | "ImportError" | ...
    error_message: str
    file_path: str
    context_snippet: str       # 出错代码片段
    fix_description: str       # 如何修复的
    fix_success: bool          # 修复是否成功
    embedding: list[float]     # 错误场景的嵌入
    timestamp: str

@dataclass
class ReflectionResult:
    """反思记忆的检索结果"""
    similar_errors: list[ErrorExperience]
    suggestion: str            # LLM 根据历史经验生成的建议
```

---

## 五、详细开发清单（含 HOW）

### Step 1：Token Manager（30 分钟）

**`memory/token_manager.py`：**
```python
import tiktoken

class TokenManager:
    """Token 计数和 Context Window 管理"""
    
    def __init__(self, model: str = "gpt-4o"):
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # 模型的 Context Window 大小
        self.CONTEXT_LIMITS = {
            "gpt-4o": 128_000,
            "gpt-4o-mini": 128_000,
            "claude-sonnet-4-6": 200_000,
            "claude-opus-4-8": 200_000,
        }
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    def count_messages(self, messages: list) -> int:
        """估算消息列表的总 Token 数"""
        total = 0
        for msg in messages:
            total += self.count_tokens(str(msg.content))
            total += 4  # 每条消息的格式开销（粗略估算）
        return total
    
    def get_remaining_budget(self, messages: list, model: str) -> int:
        """还剩多少 Token 预算"""
        limit = self.CONTEXT_LIMITS.get(model, 128_000)
        used = self.count_messages(messages)
        return max(0, limit - used)
    
    def should_compress(self, messages: list, model: str, 
                        reserve_for_response: int = 4096) -> bool:
        """是否需要压缩上下文"""
        limit = self.CONTEXT_LIMITS.get(model, 128_000)
        used = self.count_messages(messages)
        return (used + reserve_for_response) > (limit * 0.8)  # 80% 阈值
```

### Step 2：Embedding Service（30 分钟）

**`memory/embeddings.py`：**
```python
from chromadb.utils import embedding_functions

class EmbeddingService:
    """Embedding 生成服务"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # 使用 sentence-transformers（本地，免费）
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
    
    def embed(self, text: str) -> list[float]:
        return self.ef(text)
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.ef(texts)
```

**模型选型说明：**
- `all-MiniLM-L6-v2`：轻量（80MB），本地运行，适合 MVP
- `text-embedding-3-small`（OpenAI）：质量更好但需 API 费用
- 面试可说："MVP 用本地模型快速迭代；生产可切换到更强的 embedding 模型"

### Step 3：Short-term Memory（1 小时）

- 按 §4.1 实现滑动窗口 + 摘要压缩
- 关键逻辑：`add_message()` 后自动检查 Token 数，超阈值触发压缩
- 压缩策略：保留最新 50% 消息，前半部分用 LLM 生成摘要（调用 Phase 1 的 LLM Service）

### Step 4：Long-term Memory — ChromaDB（1.5 小时）

**`memory/long_term.py`：**
```python
import chromadb
from chromadb.config import Settings

class LongTermMemory:
    """基于 ChromaDB 的长期记忆"""
    
    COLLECTIONS = {
        "code": "存储代码片段和文件内容",
        "conversation": "存储对话摘要",
        "task": "存储任务执行记录",
    }
    
    def __init__(self, persist_dir: str = "./data/chroma"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedding_service = EmbeddingService()
        self._ensure_collections()
    
    def _ensure_collections(self):
        """确保集合存在"""
        self.collections = {}
        for name in self.COLLECTIONS:
            self.collections[name] = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_service.ef,
            )
    
    async def store(self, entry: MemoryEntry) -> str:
        """存储一条记忆"""
        collection = self.collections.get(entry.memory_type, self.collections["code"])
        
        collection.add(
            ids=[entry.id],
            documents=[entry.content],
            metadatas=[entry.metadata],
            embeddings=[entry.embedding],
        )
        return entry.id
    
    async def search(self, query: str, memory_type: str = "code", 
                     top_k: int = 5) -> list[MemoryQueryResult]:
        """语义搜索记忆"""
        collection = self.collections.get(memory_type)
        if not collection:
            return []
        
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        
        return [
            MemoryQueryResult(
                entry=MemoryEntry(
                    id=results["ids"][0][i],
                    content=results["documents"][0][i],
                    embedding=[],  # 不返回 embedding 节省带宽
                    memory_type=memory_type,
                    metadata=results["metadatas"][0][i] or {},
                    created_at="",
                ),
                similarity_score=1 - results["distances"][0][i],  # distance → similarity
            )
            for i in range(len(results["ids"][0]))
        ]
    
    async def forget(self, memory_id: str, memory_type: str) -> None:
        """删除记忆（用于隐私合规）"""
        collection = self.collections.get(memory_type)
        if collection:
            collection.delete(ids=[memory_id])
```

### Step 5：Reflection Memory（1 小时）

**`memory/reflection.py`：**
```python
class ReflectionMemory:
    """
    反思记忆：记录错误和修复经验。
    当 Agent 遇到新错误时，先检索历史中类似错误，参考过去的修复方案。
    """
    
    def __init__(self, long_term: LongTermMemory):
        self.ltm = long_term
    
    async def record_error(self, error: ErrorExperience) -> None:
        """记录一次错误经验"""
        embedding = EmbeddingService().embed(
            f"{error.error_type}: {error.error_message}\n{error.context_snippet}"
        )
        error.embedding = embedding
        
        await self.ltm.store(MemoryEntry(
            id=f"error_{error.id}",
            content=f"Error: {error.error_type}: {error.error_message}\n"
                    f"Fix: {error.fix_description}\n"
                    f"Success: {error.fix_success}",
            embedding=embedding,
            memory_type="task",
            metadata={
                "type": "error_experience",
                "error_type": error.error_type,
                "file_path": error.file_path,
                "fix_success": str(error.fix_success),
            },
            created_at=error.timestamp,
        ))
    
    async def find_similar_errors(self, current_error: str, 
                                   top_k: int = 3) -> ReflectionResult:
        """找到历史上类似的错误及修复方案"""
        results = await self.ltm.search(
            query=current_error,
            memory_type="task",
            top_k=top_k,
        )
        
        # 过滤出错误经验
        error_results = [
            r for r in results 
            if r.entry.metadata.get("type") == "error_experience"
        ]
        
        # 如果找到相似错误，让 LLM 基于历史经验生成建议
        if error_results:
            suggestion = await self._generate_suggestion(current_error, error_results)
        else:
            suggestion = "No similar errors found in history."
        
        return ReflectionResult(
            similar_errors=error_results,
            suggestion=suggestion,
        )
```

### Step 6：接入 Agent Runtime（1 小时）
- 在每次对话开始时，从 Long-term Memory 检索相关代码和任务上下文
- 在每次工具调用后，判断是否值得存入 Long-term Memory
- 在 Fix Loop（Phase 5）中，每次修复后记录错误经验到 Reflection Memory
- 在 Short-term Memory 触发压缩时，将压缩后的摘要存入 Long-term Memory

---

## 六、关键代码模式与伪代码

### 6.1 上下文压缩策略

```python
async def compress_context(messages: list, existing_summary: str, 
                           token_budget: int) -> tuple[list, str]:
    """
    压缩对话上下文。
    
    策略：
    1. 保留 system prompt（不动）
    2. 保留最近 N 条消息（保证最近的上下文完整）
    3. 中间的消息用 LLM 生成递进式摘要
    4. 工具调用结果截断（保留关键输出，丢弃冗长内容）
    """
    
    # 分层处理（Anthropic 格式：system 是独立参数，不在 messages 列表中）
    recent_msgs = messages[-20:]  # 最近 20 条
    middle_msgs = messages[:-20]  # 旧消息（用于压缩）
    
    # 对中间层生成摘要
    if middle_msgs:
        summary_prompt = f"""
        Existing summary: {existing_summary}
        
        New messages to summarize:
        {format_messages(middle_msgs)}
        
        Write a concise summary that includes:
        1. Key decisions made
        2. Important file changes
        3. Errors encountered and fixes
        4. Current task state
        """
        
        # Anthropic 原生调用（不含工具——摘要不触发工具调用）
        response = client.messages.create(
            model=MODEL,
            system="You are a conversation summarizer. Be concise.",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=2048,
        )
        new_summary = extract_text(response.content)
    else:
        new_summary = existing_summary
    
    # 返回压缩后的上下文（Anthropic dict 格式）
    compressed = []
    if new_summary:
        compressed.append({"role": "user", "content": f"[Context]: {new_summary}"})
    compressed.extend(recent_msgs)
    
    return compressed, new_summary
```

### 6.2 记忆检索流程

```python
async def retrieve_relevant_context(query: str, ltm: LongTermMemory) -> str:
    """
    在开始新任务前，检索相关历史上下文。
    这使 Agent 能"记住"之前做过的事。
    """
    # 搜索相关代码
    code_results = await ltm.search(query, "code", top_k=3)
    # 搜索相关对话
    conv_results = await ltm.search(query, "conversation", top_k=2)
    # 搜索相关错误经验
    error_results = await ltm.search(query, "task", top_k=3)
    
    # 构建上下文注入
    context = "## Relevant Historical Context\n\n"
    
    if code_results:
        context += "### Related code from past sessions:\n"
        for r in code_results:
            context += f"- {r.entry.metadata.get('file_path', 'unknown')}: {r.entry.content[:200]}...\n"
    
    if error_results:
        context += "\n### Past errors & fixes:\n"
        for r in error_results:
            if r.entry.metadata.get("type") == "error_experience":
                context += f"- {r.entry.content[:200]}...\n"
    
    return context
```

---

## 七、完成标志

### 基本完成
- [ ] Short-term Memory：消息超过 Token 阈值自动压缩为摘要
- [ ] Long-term Memory：代码片段和对话摘要可存入 ChromaDB 并能语义检索
- [ ] Reflection Memory：错误经验自动记录，下次类似错误能被检索到
- [ ] Agent 重启后能通过 Long-term Memory 恢复任务上下文
- [ ] Token 管理：Context Window 使用率保持在 80% 以下

### 自测用例

```bash
# 测试 1：跨会话记忆
# 会话 1：创建一个名为 "user_manager.py" 的文件
curl -X POST /api/chat -d '{"message": "创建 user_manager.py，包含 User 类"}'

# 重启服务

# 会话 2：Agent 应该能"记住"
curl -X POST /api/chat -d '{"message": "给 user_manager.py 的 User 类添加 email 属性"}'
# 期望：Agent 从 Long-term Memory 中检索到 user_manager.py 的内容

# 测试 2：Reflection Memory
# 故意触发错误并修复，然后触发类似错误
curl -X POST /api/chat -d '{
  "message": "创建 broken.py，内容 import nonexistent_module，然后修复它"
}'
# 期望：错误经验被记录；下次类似 ImportError 能被检索到
```

---

## 八、文档差距分析

### 原文档缺失的内容及补充

| # | 原文档问题 | 实际开发需要 | 本文档补充位置 |
|---|-----------|-------------|-------------|
| 1 | **完全没提到向量数据库** | Long-term Memory 需要语义检索（按意思搜，不是按关键词搜），必须用向量存储。ChromaDB 是最轻量的选择。 | §4.2, §5 Step 4 |
| 2 | "Short-term Memory = Message History" | 不只是存消息，还需要：1) 滑动窗口裁剪 2) Token 计数 3) 触发摘要压缩 4) 摘要与原始消息的混合 | §4.1 |
| 3 | "Long-term Memory = Redis" | Redis 只适合做缓存/队列，不适合做记忆的语义检索。需要 ChromaDB 做向量存储 + Redis 做热缓存 | §4.2 |
| 4 | **没有 Token 管理** | 这是 Agent 系统的核心挑战：不管理 Context Window 的话，长对话会超出限制导致崩溃 | §5 Step 1, §6.1 |
| 5 | "Context Compression" 没有具体策略 | 需要分层压缩：System Prompt 不动 + 前 50% 消息摘要化 + 后 50% 保留原文 | §6.1 |
| 6 | Reflection Memory 没说怎么检索相似错误 | 错误经验需要 embedding → 存入向量存储 → 新错误来时向量检索 → LLM 基于历史经验生成建议 | §5 Step 5 |
| 7 | 没有提到 ChromaDB 集合设计 | 需要分 collection（code/conversation/task），否则不同类型的记忆混在一起检索精度差 | §5 Step 4 |

### ChromaDB vs Milvus vs pgvector 选型说明

| 方案 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| **ChromaDB**（本项目选择） | pip install 即用，零配置 | 单机，不适合分布式 | MVP + Demo |
| pgvector | 与 PostgreSQL 共存，减少组件 | 需额外配置扩展 | 有 PG 的中型项目 |
| Milvus | 分布式，性能最强 | 运维复杂，资源消耗大 | 生产级亿级向量 |

**面试表述：** "MVP 阶段用 ChromaDB 快速迭代，生产环境可切换到 pgvector（与现有 PostgreSQL 共存，减少运维组件）或 Milvus。"

---

## 九、面试考点映射

| 面试题 | 答案要点 | 本文档参考 |
|--------|---------|-----------|
| **Short-term vs Long-term Memory 区别？** | Short = 当前会话，滑动窗口 + 摘要，容量 ~8K tokens。Long = 跨会话持久化，向量检索，容量 50K+。前者解决"当前对话太长"，后者解决"重启后完全失忆"。 | §1, §4 |
| **Reflection Memory 如何设计？** | 记录 (错误类型, 错误信息, 上下文, 修复方案, 是否成功) 五元组。Embedding 后存入向量存储。新错误来时向量检索最相似的历史错误，LLM 参考历史修复方案。 | §4.3, §5 Step 5 |
| **Context Compression 如何实现？** | 分层策略：System Prompt 不动；前半部分消息用 LLM 生成增量摘要（"递进式摘要"）；后半部分保留原文保证最近上下文完整性。触发条件：Token 使用率 > 80%。 | §6.1 |
| **为什么需要 Summary Memory？** | 1) 突破 Context Window 限制 2) 减少 Token 消耗（摘要比原文短 5-10x）3) 保持对话连续性（摘要桥接前后文）。 | §4.1 |
| **向量数据库在记忆系统中的作用？** | 让"按意思搜索"成为可能——Agent 能检索"类似的错误"、"相关代码"、"关联任务"，而不是仅靠关键词匹配。这是 Long-term Memory 能实际工作的基础。 | §4.2, §5 Step 4 |
