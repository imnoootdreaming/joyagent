"""
Phase 6 Step ？: Remember Tool — Agent 可调用的长期记忆工具。

Agent 通过此工具将信息持久化到 ChromaDB，也在需要时检索历史记忆。
解决了之前 Agent 试图通过 execute_shell 在 sandbox 里跑 Python
来操作 ChromaDB 的问题——现在有了正规通道。

使用方式：
  remember.save(content="用户偏好每行注释", memory_type="preference")
  remember.search(query="代码风格", top_k=5)
"""

from app.tools.base import BaseTool, ToolResult
from app.memory.long_term import LongTermMemory, MemoryEntry, get_long_term_memory
from app.memory.embeddings import get_embedding_service


class RememberTool(BaseTool):
    """保存信息到长期记忆，或从长期记忆中检索信息。"""

    name = "remember"
    description = (
        "Save information to long-term memory or search existing memories. "
        "Use 'action=save' to remember something (code snippet, user preference, "
        "task result, conversation summary). "
        "Use 'action=search' to find relevant past memories by semantic similarity. "
        "Long-term memory persists across sessions — what you save now "
        "will be available in future conversations."
    )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save", "search"],
                    "description": (
                        "'save' — persist information to long-term memory.\n"
                        "'search' — find relevant past memories."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "For action='save': the text to remember (required).\n"
                        "For action='search': the query to search for (required)."
                    ),
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["code", "task", "conversation", "preference"],
                    "description": (
                        "Category of the memory:\n"
                        "  'code' — code snippets, patterns, fixes\n"
                        "  'task' — completed tasks, commands, workflows\n"
                        "  'conversation' — dialogue history, decisions\n"
                        "  'preference' — user preferences, coding style, conventions"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (for action='search'). Default 5.",
                },
                "title": {
                    "type": "string",
                    "description": "Optional short title/label for the memory (e.g. 'user code style v1').",
                },
            },
            "required": ["action", "content"],
        }

    def __init__(self):
        self._ltm = get_long_term_memory()
        self._emb = get_embedding_service()

    async def execute(self, action: str, content: str,
                      memory_type: str = "task", top_k: int = 5,
                      title: str = "", **kwargs) -> ToolResult:
        if action == "save":
            return await self._save(content, memory_type, title)
        elif action == "search":
            return await self._search(content, memory_type, top_k)
        else:
            return ToolResult(success=False, message=f"Unknown action: {action}")

    async def _save(self, content: str, memory_type: str,
                    title: str) -> ToolResult:
        try:
            embedding = self._emb.embed(content)
            entry = MemoryEntry(
                id=title or f"mem_{hash(content) % 10**8:x}",
                content=content,
                embedding=embedding,
                memory_type=memory_type,
                metadata={
                    "stored_by": "agent_tool",
                    "memory_type": memory_type,
                },
            )
            await self._ltm.store(entry)
            return ToolResult(
                success=True,
                message=(
                    f"Successfully saved to long-term memory.\n"
                    f"  ID: {entry.id}\n"
                    f"  Type: {memory_type}\n"
                    f"  Content length: {len(content)} chars\n"
                    f"  Total entries: {self._ltm.count()}\n"
                    f"  Storage: {self._ltm.persist_dir}"
                ),
                metadata={
                    "id": entry.id,
                    "memory_type": memory_type,
                    "total_entries": self._ltm.count(),
                    "persist_dir": self._ltm.persist_dir,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Failed to save to memory: {e}",
                error=str(e),
            )

    async def _search(self, query: str, memory_type: str,
                      top_k: int) -> ToolResult:
        try:
            results = await self._ltm.search(
                query=query,
                memory_type=memory_type or None,
                top_k=top_k,
            )
            if not results:
                return ToolResult(
                    success=True,
                    message=f"No matching memories found for query: {query}",
                )

            lines = [f"Found {len(results)} relevant memories:\n"]
            for i, r in enumerate(results, 1):
                score = getattr(r, 'similarity_score', 0)
                entry = r.entry if hasattr(r, 'entry') else r
                c = (entry.content if hasattr(entry, 'content') else str(entry))
                mtype = getattr(entry, 'memory_type', 'unknown')
                lines.append(
                    f"── [{i}] score={score:.3f} type={mtype} ──\n"
                    f"{c[:500]}"
                )
                if len(c) > 500:
                    lines.append(f"\n  ... (+{len(c)-500} more chars)")

            return ToolResult(
                success=True,
                message="\n".join(lines),
                metadata={"count": len(results)},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Failed to search memory: {e}",
                error=str(e),
            )

    @property
    def is_dangerous(self) -> bool:
        return False
