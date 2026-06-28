from __future__ import annotations
"""
Phase 6 Step 5: Reflection Memory — 错误经验记录 + 历史相似错误召回。

ReflectionMemory 是三级记忆系统中"从错误中学习"的模块。它把 Agent 的每一次
错误和修复过程记录为结构化的 ErrorExperience，存入 Long-term Memory 的向量
存储中。当 Agent 遇到新错误时，先检索历史上最相似的错误经验，让 LLM 参考
过去的修复方案来生成建议——实现"不重复犯同样的错"。

运作流程：
  1. Agent 遇到错误（在 Fix Loop 中）
  2. 调用 find_similar_errors() → 向量检索最相似的历史错误
  3. 如果有相似错误 → LLM 基于历史经验生成修复建议
  4. Agent 尝试修复 → 记录修复结果
  5. 调用 record_error() 存入经验库（无论成功或失败，都有价值）

与 Phase 5 Fix Loop 的集成点：
  FixLoop.fix_and_retest()
    ├── ErrorParser.parse()          → 解析错误
    ├── ReflectionMemory.find_similar_errors()  → 查找历史相似错误 ← 本模块
    ├── LLM fix generation           → 生成修复方案
    ├── PatchApplier.apply()         → 应用修复
    └── ReflectionMemory.record_error()         → 记录本次经验 ← 本模块

错误经验的"五元组"：
  (error_type, error_message, context_snippet, fix_description, fix_success)
  这五个维度完整描述了一次错误的"症状→根因→修复→结果"全链路。

使用方式：
  from app.memory.reflection import (
      ReflectionMemory, ErrorExperience, ReflectionResult,
      get_reflection_memory,
  )
  from app.memory.long_term import get_long_term_memory

  ltm = get_long_term_memory()
  rm = get_reflection_memory(ltm)

  # 遇到新错误时
  result = await rm.find_similar_errors(
      "ImportError: No module named 'requests'",
      file_path="main.py",
  )
  if result.similar_errors:
      print(f"Found {len(result.similar_errors)} similar past errors")
      print(f"LLM suggests: {result.suggestion}")

  # 修复完成后
  await rm.record_error(ErrorExperience(
      id="err_20260628_001",
      error_type="ImportError",
      error_message="No module named 'requests'",
      file_path="main.py",
      context_snippet="import requests\n...",
      fix_description="Ran 'pip install requests' and verified import",
      fix_success=True,
  ))
"""

# ── Python 标准库 ──
import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Optional

# ── 项目内导入 ──
from app.memory.long_term import (
    LongTermMemory,
    MemoryEntry,
    MemoryQueryResult,
    get_long_term_memory,
)
from app.memory.embeddings import EmbeddingService, get_embedding_service
from app.service.llm_service import get_or_create_client
from app.core.config import Config


# ═══════════════════════════════════════════════════════════════════════════════
# 内部工具
# ═══════════════════════════════════════════════════════════════════════════════

def _emb_empty(emb) -> bool:
    """
    安全判断 embedding 是否为空（兼容 numpy array + Python list + None）。

    sentence-transformers 返回 numpy.ndarray，直接 `if not emb` 在 numpy
    数组上有歧义，会抛出 ValueError。用 len() 判断才是安全的。
    """
    if emb is None:
        return True
    try:
        return len(emb) == 0
    except TypeError:
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ErrorExperience:
    """
    一次错误经验的完整记录（"五元组"）。

    记录从错误发生到修复完成的全链路信息，便于将来遇到类似错误时
    快速找到有效的修复方案。

    字段说明：
      error_type      — Python 异常类型字符串（"SyntaxError" | "ImportError" | ...）
                        映射关系见 app/sandbox/error_parser.py 的 _ERROR_CATEGORIES
      error_message   — 原始错误消息（如 "No module named 'requests'"）
      file_path       — 出错文件路径
      context_snippet — 出错代码片段（前后各 3-5 行，帮助理解上下文）
      fix_description — 修复描述（如何修复的，方便人类阅读和 LLM 参考）
      fix_success     — 修复是否最终成功（True = 经验可信度高，False = 仍有参考价值）
      embedding       — 错误场景的向量表示（用于相似检索；可为空，自动计算）
      timestamp       — ISO 8601 时间戳
    """
    id: str = ""
    error_type: str = "UnknownError"
    error_message: str = ""
    file_path: str = ""
    context_snippet: str = ""
    fix_description: str = ""
    fix_success: bool = False
    embedding: list[float] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = f"err_{uuid.uuid4().hex[:10]}"
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    def build_search_text(self) -> str:
        """
        构建用于向量检索的文本——合并错误类型、消息和上下文。

        这就是"错误签名"——同类错误（即使变量名不同）会产生相似的向量。
        例如 "ImportError: No module named 'requests'" 和
            "ImportError: No module named 'numpy'" 在向量空间中会很接近。
        """
        parts = [
            f"{self.error_type}: {self.error_message}",
        ]
        if self.context_snippet:
            parts.append(self.context_snippet[:500])
        return "\n".join(parts)


@dataclass
class ReflectionResult:
    """
    反思记忆的检索结果。

    similar_errors — 历史上类似错误的列表（按相似度降序）
    suggestion     — LLM 基于历史经验生成的修复建议文本
                     如果没有找到相似错误，此项为 "No similar errors found in history."
    """
    similar_errors: list[ErrorExperience] = field(default_factory=list)
    suggestion: str = ""

    @property
    def has_relevant_history(self) -> bool:
        """是否找到了相关的历史经验。"""
        return len(self.similar_errors) > 0

    @property
    def successful_fix_available(self) -> bool:
        """历史中是否有成功的修复方案。"""
        return any(e.fix_success for e in self.similar_errors)


@dataclass
class ErrorStatsSummary:
    """
    错误经验统计摘要。

    用于诊断和分析：哪些类型的错误最频繁？修复成功率如何？
    """
    total_errors: int = 0
    by_type: dict[str, int] = field(default_factory=dict)       # {"ImportError": 5, ...}
    success_rate: float = 0.0                                     # 0.0 ~ 1.0
    most_common_error: str = ""                                   # 最常见的错误类型
    total_successful_fixes: int = 0
    total_failed_fixes: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# ReflectionMemory — 错误经验学习引擎
# ═══════════════════════════════════════════════════════════════════════════════

class ReflectionMemory:
    """
    反思记忆：记录错误和修复经验，从历史中学习。

    职责：
      1. 记录错误经验 → 存入 Long-term Memory（向量存储）
      2. 相似错误检索 → 向量搜索最相似的历史错误
      3. LLM 建议生成 → 基于历史经验为新错误生成修复建议
      4. 统计与诊断 → 错误类型分布、修复成功率

    为什么记录失败的修复也有价值？
      - 失败的修复告诉 Agent "这条路走不通"
      - 避免陷入同样的无效修复循环
      - 在 Fix Loop 的停滞检测之外提供第二层防护

    面试表述：
      "Reflection Memory 让 Agent 具备经验学习能力——不是简单的 rule-based
       错误处理，而是基于向量语义相似度找到历史上最类似的错误场景，
       让 LLM 参考过去的修复方案（无论成功或失败）来生成建议。
       本质是 RAG + Few-shot Learning 的应用。"
    """

    # ── LLM 建议生成的 System Prompt ──
    SUGGESTION_SYSTEM_PROMPT = (
        "You are an expert debugging assistant. Given a current error and "
        "a list of similar past errors with their fixes, generate a concise, "
        "actionable suggestion for fixing the current error.\n\n"
        "Rules:\n"
        "1. Reference specific past fixes that worked (✅)\n"
        "2. Warn against approaches that failed (⚠️)\n"
        "3. Be specific — mention exact commands, file edits, or dependency changes\n"
        "4. Keep it under 300 words\n"
        "5. Output ONLY the suggestion — no preamble, no meta-commentary"
    )

    # 建议生成最大 token 数
    SUGGESTION_MAX_TOKENS = 2048

    def __init__(
        self,
        long_term: Optional[LongTermMemory] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        初始化 Reflection Memory。

        Args:
            long_term:        LongTermMemory 实例。None 时使用全局单例。
            embedding_service: EmbeddingService 实例。None 时使用全局单例。
        """
        self.ltm = long_term or get_long_term_memory()
        self._embedding_service = embedding_service or get_embedding_service()

        # ── 统计 ──
        self._lock = threading.Lock()
        self._record_count = 0
        self._search_count = 0

    # ── 核心：记录错误经验 ─────────────────────────────────────────

    async def record_error(self, error: ErrorExperience) -> str:
        """
        记录一次错误经验到 Long-term Memory。

        自动计算错误场景的向量嵌入，构建结构化的 MemoryEntry，
        存入 "task" 集合（带 type="error_experience" 标签）。

        Args:
            error: ErrorExperience 实例

        Returns:
            存储的记忆 ID（"error_{error.id}"）

        Example:
            await rm.record_error(ErrorExperience(
                error_type="ImportError",
                error_message="No module named 'requests'",
                file_path="main.py",
                context_snippet="import requests",
                fix_description="pip install requests",
                fix_success=True,
            ))
        """
        # ── 自动计算 embedding（兼容 numpy array） ──
        if _emb_empty(error.embedding):
            search_text = error.build_search_text()
            error.embedding = self._embedding_service.embed(search_text)

        # ── 构建记忆内容 ──
        status_icon = "✅" if error.fix_success else "⚠️"
        content = (
            f"{status_icon} **{error.error_type}** in `{error.file_path}`\n\n"
            f"Error: {error.error_message}\n\n"
            f"Context:\n```\n{error.context_snippet or '(no context)'}\n```\n\n"
            f"Fix: {error.fix_description or '(no fix recorded)'}\n\n"
            f"Success: {error.fix_success}"
        )

        # ── 构建元数据 ──
        metadata = {
            "type": "error_experience",
            "error_type": error.error_type,
            "file_path": error.file_path,
            "fix_success": str(error.fix_success),
            "created_at": error.timestamp,
        }
        # 可选的额外标签
        if error.fix_description:
            metadata["has_fix"] = "True"

        # ── 存入 Long-term Memory ──
        entry_id = f"error_{error.id}"
        await self.ltm.store(MemoryEntry(
            id=entry_id,
            content=content,
            embedding=error.embedding,
            memory_type="task",
            metadata=metadata,
            created_at=error.timestamp,
        ))

        with self._lock:
            self._record_count += 1

        return entry_id

    async def record_error_batch(self, errors: list[ErrorExperience]) -> list[str]:
        """
        批量记录错误经验（比逐条 record_error 快）。

        Args:
            errors: ErrorExperience 列表

        Returns:
            所有存储的记忆 ID 列表
        """
        if not errors:
            return []

        entries: list[MemoryEntry] = []
        for error in errors:
            if _emb_empty(error.embedding):
                error.embedding = self._embedding_service.embed(
                    error.build_search_text()
                )

            status_icon = "✅" if error.fix_success else "⚠️"
            entries.append(MemoryEntry(
                id=f"error_{error.id}",
                content=(
                    f"{status_icon} **{error.error_type}** in `{error.file_path}`\n\n"
                    f"Error: {error.error_message}\n\n"
                    f"Fix: {error.fix_description or '(no fix)'}\n\n"
                    f"Success: {error.fix_success}"
                ),
                embedding=error.embedding,
                memory_type="task",
                metadata={
                    "type": "error_experience",
                    "error_type": error.error_type,
                    "file_path": error.file_path,
                    "fix_success": str(error.fix_success),
                    "created_at": error.timestamp,
                },
                created_at=error.timestamp,
            ))

        ids = await self.ltm.store_batch(entries)

        with self._lock:
            self._record_count += len(errors)

        return ids

    # ── 核心：相似错误检索 ─────────────────────────────────────────

    async def find_similar_errors(
        self,
        current_error: str,
        file_path: str = "",
        top_k: int = 5,
        model: str = None,
    ) -> ReflectionResult:
        """
        找到历史上类似的错误及修复方案（完整流程）。

        步骤：
          1. 在 "task" 集合中向量检索最相似的条目
          2. 过滤出 type="error_experience" 的记录
          3. 如果有相似错误 → LLM 基于历史经验生成修复建议
          4. 如果没有 → 返回空列表 + 提示文本

        Args:
            current_error: 当前错误的描述文本（原始 traceback 或错误摘要）
            file_path:     出错的文件路径（帮助 LLM 结合文件上下文生成建议）
            top_k:         返回最多多少条相似错误
            model:         用于生成建议的 LLM 模型

        Returns:
            ReflectionResult — 相似错误列表 + LLM 建议

        Example:
            result = await rm.find_similar_errors(
                current_error="ImportError: No module named 'requests'",
                file_path="main.py",
            )
            if result.has_relevant_history:
                print(f"Found {len(result.similar_errors)} similar past errors")
                print(f"Suggestion: {result.suggestion}")
            # → "Based on 2 similar ImportErrors, try: pip install requests"
        """
        # ── 1. 向量检索 ──
        results = await self.ltm.search(
            query=current_error,
            memory_type="task",
            top_k=top_k,
        )

        # ── 2. 过滤出错误经验 ──
        error_results: list[MemoryQueryResult] = []
        for r in results:
            if r.entry.metadata.get("type") == "error_experience":
                error_results.append(r)

        with self._lock:
            self._search_count += 1

        # ── 3. 转换为 ErrorExperience 列表 ──
        similar_errors = _convert_memory_results_to_errors(error_results)

        # ── 4. 生成建议 ──
        if similar_errors:
            suggestion = await self._generate_suggestion(
                current_error=current_error,
                file_path=file_path,
                similar_errors=similar_errors,
                model=model,
            )
        else:
            suggestion = "No similar errors found in history."

        return ReflectionResult(
            similar_errors=similar_errors,
            suggestion=suggestion,
        )

    async def _generate_suggestion(
        self,
        current_error: str,
        file_path: str,
        similar_errors: list[ErrorExperience],
        model: str = None,
    ) -> str:
        """
        调用 LLM 基于历史经验生成修复建议。

        Prompt 设计：
          - 列出历史上每个相似错误（类型、消息、修复方案、成功与否）
          - 成功的修复用 ✅ 标记，失败的用 ⚠️ 标记
          - 要求 LLM 综合考量所有历史经验，给出最可能有效的建议

        Args:
            current_error:  当前错误的描述
            file_path:      出错文件路径
            similar_errors: 历史上相似的错误经验
            model:          LLM 模型

        Returns:
            LLM 生成的修复建议文本

        Raises:
            RuntimeError: LLM 调用失败时降级回退为模板化建议
        """
        model = model or Config.DEFAULT_MODEL

        # ── 构建历史经验参考文本 ──
        history_lines = []
        for i, err in enumerate(similar_errors, 1):
            status = "✅ SUCCESSFUL" if err.fix_success else "⚠️ FAILED"
            history_lines.append(
                f"### Case {i} [{status}]\n"
                f"- Error: {err.error_type}: {err.error_message}\n"
                f"- File: {err.file_path}\n"
                f"- Fix: {err.fix_description or '(no fix description)'}\n"
            )

        history_text = "\n".join(history_lines)

        # ── 构建 Prompt ──
        prompt = (
            f"## Current Error\n\n"
            f"```\n{current_error[:1000]}\n```\n"
        )
        if file_path:
            prompt += f"File: `{file_path}`\n\n"

        prompt += (
            f"## Similar Past Errors ({len(similar_errors)} found)\n\n"
            f"{history_text}\n\n"
            f"Based on the past experiences above, generate a specific, "
            f"actionable suggestion for fixing the current error. "
            f"Prioritize approaches that worked in the past (✅). "
            f"Avoid approaches that failed (⚠️). "
            f"If the past cases are truly similar, recommend the most "
            f"effective fix with any necessary adaptations for the current context."
        )

        # ── 调用 LLM ──
        try:
            client = get_or_create_client(model)
            response = client.messages.create(
                model=model,
                system=self.SUGGESTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.SUGGESTION_MAX_TOKENS,
            )
        except Exception as e:
            # 降级：LLM 不可用时返回基于统计的模板化建议
            return _fallback_suggestion(current_error, similar_errors, str(e))

        # ── 提取文本 ──
        text_parts = []
        for block in response.content:
            if hasattr(block, 'type'):
                if block.type == "text":
                    text_parts.append(getattr(block, 'text', ''))
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            else:
                text_parts.append(str(block))

        return "".join(text_parts).strip()

    # ── 便捷方法 ─────────────────────────────────────────────────

    async def record_successful_fix(
        self,
        error_type: str,
        error_message: str,
        file_path: str,
        context_snippet: str,
        fix_description: str,
    ) -> str:
        """
        快捷记录一次成功的修复（fix_success=True 自动设置）。

        适合在 Fix Loop 的"修复成功"分支中直接调用。

        Returns:
            存储的记忆 ID
        """
        return await self.record_error(ErrorExperience(
            error_type=error_type,
            error_message=error_message,
            file_path=file_path,
            context_snippet=context_snippet,
            fix_description=fix_description,
            fix_success=True,
        ))

    async def record_failed_fix(
        self,
        error_type: str,
        error_message: str,
        file_path: str,
        context_snippet: str,
        fix_description: str,
    ) -> str:
        """
        快捷记录一次失败的修复（fix_success=False 自动设置）。

        失败的修复也有价值——告诉 Agent 哪些方法不奏效。
        """
        return await self.record_error(ErrorExperience(
            error_type=error_type,
            error_message=error_message,
            file_path=file_path,
            context_snippet=context_snippet,
            fix_description=fix_description,
            fix_success=False,
        ))

    async def get_fix_suggestion_for_parse_error(
        self,
        error_text: str,
        file_path: str = "",
    ) -> str:
        """
        一步到位的便捷方法：给定错误文本，直接返回修复建议。

        内部调用 find_similar_errors() → 返回 suggestion 文本。
        适合在 Fix Loop 中快速集成，无需关心 ReflectionResult 结构。

        Args:
            error_text: 错误描述文本
            file_path:  出错文件

        Returns:
            修复建议文本
        """
        result = await self.find_similar_errors(error_text, file_path)
        return result.suggestion

    # ── 统计与诊断 ───────────────────────────────────────────────

    async def get_error_stats(self) -> ErrorStatsSummary:
        """
        获取错误经验的统计摘要。

        从 Long-term Memory 的 "task" 集合中分析所有 error_experience 记录。

        Returns:
            ErrorStatsSummary 包含按类型分布、成功率等统计信息

        注意：此方法需要遍历所有 error_experience 条目，
        在数据量很大时（>10K）可能较慢。建议用于诊断面板而非热路径。
        """
        from app.memory.long_term import MemoryQueryResult

        # 使用通用 query 获取尽可能多的错误经验
        results = await self.ltm.search(
            query="error experience",
            memory_type="task",
            top_k=1000,  # 取最近 1000 条
        )

        by_type: dict[str, int] = {}
        total = 0
        success_count = 0

        for r in results:
            if r.entry.metadata.get("type") != "error_experience":
                continue
            total += 1
            error_type = r.entry.metadata.get("error_type", "UnknownError")
            by_type[error_type] = by_type.get(error_type, 0) + 1
            if r.entry.metadata.get("fix_success") == "True":
                success_count += 1

        most_common = ""
        if by_type:
            most_common = max(by_type, key=by_type.get)  # type: ignore[arg-type]

        return ErrorStatsSummary(
            total_errors=total,
            by_type=by_type,
            success_rate=round(success_count / total, 3) if total > 0 else 0.0,
            most_common_error=most_common,
            total_successful_fixes=success_count,
            total_failed_fixes=total - success_count,
        )

    # ── 指标 ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """获取运行时统计。"""
        return {
            "total_errors_recorded": self._record_count,
            "total_searches": self._search_count,
            "ltm_task_collection_size": self.ltm.count("task"),
        }

    # ── 重置 ─────────────────────────────────────────────────────

    def reset_stats(self) -> None:
        """重置运行时统计（不影响已存储的错误经验数据）。"""
        with self._lock:
            self._record_count = 0
            self._search_count = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _convert_memory_results_to_errors(
    results: list[MemoryQueryResult],
) -> list[ErrorExperience]:
    """
    将 MemoryQueryResult 列表转换为 ErrorExperience 列表。

    从 MemoryEntry 的 metadata 和 content 中重建 ErrorExperience 的字段。
    """
    errors = []
    for r in results:
        meta = r.entry.metadata
        errors.append(ErrorExperience(
            id=r.entry.id.replace("error_", "", 1),
            error_type=meta.get("error_type", "UnknownError"),
            error_message=_extract_error_message(r.entry.content),
            file_path=meta.get("file_path", ""),
            context_snippet=_extract_context_snippet(r.entry.content),
            fix_description=_extract_fix_description(r.entry.content),
            fix_success=meta.get("fix_success", "False") == "True",
            timestamp=meta.get("created_at", ""),
        ))
    return errors


def _extract_error_message(content: str) -> str:
    """从 MemoryEntry.content 中提取 Error: 行。"""
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Error: "):
            return stripped[len("Error: "):]
    return content[:200]


def _extract_context_snippet(content: str) -> str:
    """从 MemoryEntry.content 中提取 Context: 代码块。"""
    in_context = False
    lines = []
    for line in content.split("\n"):
        if line.strip().startswith("Context:"):
            in_context = True
            continue
        if in_context:
            if line.strip().startswith("```"):
                if lines:
                    break
                continue
            lines.append(line)
    return "\n".join(lines).strip()


def _extract_fix_description(content: str) -> str:
    """从 MemoryEntry.content 中提取 Fix: 行。"""
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Fix: "):
            return stripped[len("Fix: "):]
    return ""


def _fallback_suggestion(
    current_error: str,
    similar_errors: list[ErrorExperience],
    llm_error: str,
) -> str:
    """
    LLM 不可用时的降级建议生成（基于统计的模板化建议）。

    不依赖 LLM，纯规则分析历史经验，输出简单的统计建议。
    """
    lines = [
        f"[Fallback suggestion — LLM unavailable: {llm_error}]",
        "",
    ]

    # 统计成功/失败的修复
    successful = [e for e in similar_errors if e.fix_success]
    failed = [e for e in similar_errors if not e.fix_success]

    if successful:
        lines.append(f"## {len(successful)} successful fixes found for similar errors:")
        for e in successful[:3]:
            lines.append(f"- ✅ [{e.error_type}] {e.fix_description[:200]}")

    if failed:
        lines.append(f"\n## {len(failed)} failed attempts (avoid these):")
        for e in failed[:2]:
            lines.append(f"- ⚠️ [{e.error_type}] {e.fix_description[:200]}")

    if not successful and not failed:
        lines.append("No actionable history found.")

    lines.append(
        "\nBased on the most common successful fix above, "
        "adapt it to the current error context."
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 全局单例
# ═══════════════════════════════════════════════════════════════════════════════

_reflection_memory: Optional[ReflectionMemory] = None
_rm_lock = threading.Lock()


def get_reflection_memory(
    long_term: Optional[LongTermMemory] = None,
    embedding_service: Optional[EmbeddingService] = None,
) -> ReflectionMemory:
    """
    获取全局 ReflectionMemory 单例（线程安全懒加载）。

    首次调用时创建实例，后续调用返回同一实例。
    所有模块（Fix Loop、Agent Runtime）共享同一个 Reflection Memory。

    Args:
        long_term:        LongTermMemory 实例（仅首次调用时生效）
        embedding_service: EmbeddingService 实例（仅首次调用时生效）

    Returns:
        全局单例 ReflectionMemory

    Example:
        from app.memory.reflection import get_reflection_memory

        rm = get_reflection_memory()
        result = await rm.find_similar_errors("ImportError: ...")
    """
    global _reflection_memory

    if _reflection_memory is not None:
        return _reflection_memory

    with _rm_lock:
        if _reflection_memory is not None:
            return _reflection_memory

        print(
            "[reflection_memory] Initializing global singleton..."
        )
        _reflection_memory = ReflectionMemory(
            long_term=long_term,
            embedding_service=embedding_service,
        )
        return _reflection_memory


def reset_reflection_memory() -> None:
    """
    重置全局单例（用于测试）。

    注意：不会删除已存储的错误经验数据（数据在 Long-term Memory 中持久化）。
    """
    global _reflection_memory
    with _rm_lock:
        if _reflection_memory is not None:
            print("[reflection_memory] Resetting singleton")
            _reflection_memory = None
