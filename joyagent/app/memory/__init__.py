"""
Phase 6: Memory System — 三级记忆架构，让 Agent 拥有持续的记忆能力。

在 Phase 1-5 中，Agent 每次对话都是"失忆"的——重启后完全忘记之前做过什么。
Phase 6 引入三级记忆系统，让 Agent 能够：
  1. 在当前会话内管理对话上下文（Short-term Memory）
  2. 跨会话检索历史知识（Long-term Memory via ChromaDB）
  3. 从错误经验中学习，避免重复犯错（Reflection Memory）

三级记忆架构：
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

模块结构：
  token_manager.py  — Step 1: Token 计数 + Context Window 管理
  embeddings.py     — Step 2: Embedding 向量生成服务
  short_term.py     — Step 3: Short-term Memory（滑动窗口 + 摘要压缩）
  summary.py        — Step 3: Context Compression 上下文压缩器
  long_term.py      — Step 4: Long-term Memory（ChromaDB 向量存储 + 检索）
  reflection.py     — Step 5: Reflection Memory（错误经验 + 相似召回）

数据模型：
  ShortTermMemory     — 当前会话工作记忆
  MemoryEntry         — Long-term Memory 存储记录
  MemoryQueryResult   — 记忆检索结果
  ErrorExperience     — 错误经验记录
  ReflectionResult    — 反思检索结果

使用示例：
  from app.memory.token_manager import TokenManager
  from app.memory.short_term import ShortTermMemory

  tm = TokenManager()
  stm = ShortTermMemory(max_messages=50)

  for msg in agent_messages:
      stm.add_message(msg)
      if tm.should_compress(stm.messages, model="claude-sonnet-4-6"):
          stm.compress()
"""
