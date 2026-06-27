"""
Phase 4 Step 4: Diff Generator — 生成 unified diff 格式的代码差异。

DiffGenerator 是 Coding Agent 的"变更表达层"——LLM 生成修改后的代码后，
DiffGenerator 计算原始代码与新代码之间的精确差异，输出标准 unified diff 格式。
后续由 PatchApplier (Step 5) 将 diff 应用到实际文件。

为什么需要 Diff 而不是直接 write_file 覆盖？
  1. 精准：只改变更的行，不重写整个文件（100 行改 2 行 → 只输出 2 行的 diff）
  2. 可审计：diff 本身就是变更记录，可以 review / 回滚
  3. LLM 友好：unified diff 格式是 LLM 训练数据中大量存在的格式
  4. 多文件批处理：一次生成多个文件的 diff，统一应用

diff 格式说明 (unified diff)：
  --- a/file.py         原始文件（前缀 a/）
  +++ b/file.py         新文件（前缀 b/）
  @@ -10,6 +10,8 @@     变更位置：原文件第 10 行起 6 行 → 新文件第 10 行起 8 行
    unchanged line       上下文行（未变）
  -removed line          被删除的行
  +added line            被添加的行
    unchanged line       上下文行
"""

# ── Python 标准库 ──
import difflib                         # 核心：unified_diff 生成
from dataclasses import dataclass, field  # 数据类装饰器


# ═══════════════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiffHunk:
    """
    unified diff 中的单个"块"（hunk）。

    一个 diff 通常包含 1~N 个 hunk——每个 hunk 表示文件中一处不连续的变更。
    hunk 之间以 `@@ -a,b +c,d @@` 头部行分隔。

    Example:
      @@ -10,6 +10,8 @@ def chat(request):
       unchanged
      -old line
      +new line
       unchanged

    → header="@@ -10,6 +10,8 @@ def chat(request: ChatRequest):"
      old_start=10, old_count=6, new_start=10, new_count=8
      lines_added=1, lines_removed=1
    """
    header: str                        # @@ -start,count +start,count @@ context
    old_start: int                     # 原文件起始行号
    old_count: int                     # 原文件行数（含上下文）
    new_start: int                     # 新文件起始行号
    new_count: int                     # 新文件行数（含上下文）
    lines: list[str]                   # hunk 包含的所有行（含上下文 + add/del）
    lines_added: int                   # 本 hunk 新增行数（+ 开头）
    lines_removed: int                 # 本 hunk 删除行数（- 开头）


@dataclass
class DiffResult:
    """
    单个文件的 Diff 生成结果。

    这是 DiffGenerator.generate() 的返回类型，包含：
      - 原始/修改后的完整内容（用于校验）
      - unified diff 文本（用于 patch 命令或 LLM 消费）
      - 变更统计（hunk 数、增删行数）
      - 可干净应用判断

    字段：
      file_path         — 被修改的文件路径
      original_content  — 修改前的完整文件内容
      modified_content  — 修改后的完整文件内容（LLM 生成）
      diff_text         — unified diff 格式的差异文本
      hunks             — 解析后的 hunk 列表
      lines_added       — 新增行总数
      lines_removed     — 删除行总数
      is_empty          — 是否无变更（两版本完全相同）
    """
    file_path: str                     # 文件相对路径
    original_content: str              # 原始内容（完整）
    modified_content: str              # 修改后内容（完整）
    diff_text: str                     # unified diff 文本
    hunks: list[DiffHunk]              # 解析后的 hunk 列表
    lines_added: int = 0               # 总新增行数
    lines_removed: int = 0             # 总删除行数
    is_empty: bool = False             # 是否无变更

    def can_apply_cleanly(self) -> bool:
        """
        检查 diff 是否"看起来可以干净应用"。

        判断条件：
          - diff 非空（至少有一行差异）
          - 有实际的代码变更（不仅仅是空白变更）
          - hunk 头部格式正确
        """
        if self.is_empty:
            return False
        if not self.diff_text.strip():
            return False
        # 至少有 1 处增/删
        if self.lines_added == 0 and self.lines_removed == 0:
            return False
        return True

    def format_for_llm(self, max_diff_lines: int = 100) -> str:
        """
        将 diff 格式化为 LLM 可消费的紧凑文本。

        输出格式：
          ## Diff: app/api/agent.py (2 hunks, +5, -3)
          ```diff
          @@ -359,7 +359,7 @@ async def chat(request: ChatRequest):
           ...
          ```

        Args:
            max_diff_lines: diff 文本最多输出多少行（防止撑爆 LLM 上下文）
        """
        if self.is_empty:
            return f"## Diff: {self.file_path} (no changes)"

        header = (
            f"## Diff: {self.file_path} "
            f"({len(self.hunks)} hunk(s), "
            f"+{self.lines_added}, -{self.lines_removed})"
        )

        # 截断过长的 diff
        diff_lines = self.diff_text.split("\n")
        if len(diff_lines) > max_diff_lines:
            truncated = "\n".join(diff_lines[:max_diff_lines])
            truncated += f"\n... ({len(diff_lines) - max_diff_lines} more lines)"
        else:
            truncated = self.diff_text

        return f"{header}\n```diff\n{truncated}\n```"

    def format_summary(self) -> str:
        """
        单行变更摘要，适合日志或批量显示。

        输出格式：app/api/agent.py: 2 hunks, +5, -3
        """
        if self.is_empty:
            return f"{self.file_path}: no changes"
        return (
            f"{self.file_path}: "
            f"{len(self.hunks)} hunk(s), "
            f"+{self.lines_added}, -{self.lines_removed}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DiffGenerator
# ═══════════════════════════════════════════════════════════════════════════════

class DiffGenerator:
    """
    unified diff 生成器。

    职责：
      1. 计算原始代码与新代码的差异（difflib.unified_diff）
      2. 解析 hunk 结构（@@ 头部行 → DiffHunk）
      3. 统计变更（增/删行数）
      4. 检测空 diff（两版本完全相同）
      5. 可选预处理：统一行尾、去除末尾空白

    使用方式：
      generator = DiffGenerator()
      result = generator.generate(
          file_path="app/api/agent.py",
          original="async def chat(request):\n    ...",
          modified="async def chat(request, timeout=30):\n    ...",
      )
      print(result.format_for_llm())
      print(f"Can apply: {result.can_apply_cleanly()}")

    面试要点：
      - difflib.unified_diff 是 Python 标准库，零依赖
      - keepends=True 保留行尾换行符，避免 diff 出现空行错位
      - fromfile/tofile 前缀 a/ b/ 是 git diff 的约定
    """

    def __init__(self, context_lines: int = 3):
        """
        Args:
            context_lines: hunk 中变更行前后保留多少行上下文（默认 3，与 git diff 一致）
                          增大 → 更多上下文，减小 → 更紧凑的 diff
        """
        self.context_lines = context_lines

    def generate(
        self,
        file_path: str,
        original: str,
        modified: str,
    ) -> DiffResult:
        """
        生成 unified diff。

        核心流程：
          1. 分割行（保留 keepends=True 维持行尾信息）
          2. 调用 difflib.unified_diff 生成 diff 行列表
          3. 解析 hunk 结构
          4. 统计变更

        Args:
            file_path: 文件路径（出现在 ---/+++ 头部）
            original:  原始文件内容（完整字符串）
            modified:  修改后的文件内容（LLM 生成的新版本）

        Returns:
            DiffResult: 包含 diff 文本 + hunk 解析 + 变更统计
        """
        # ── 0. 空内容保护 ──
        # 确保字符串不为 None（避免 splitlines 崩溃）
        original = original or ""
        modified = modified or ""

        # ── 1. 分割为行列表 ───────────────────────────────────
        # keepends=True: 保留每行末尾的 \n（difflib 要求）
        # 这确保 diff 输出中的行与原始文件行完全对应
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        # ── 2. 生成 unified diff ──────────────────────────────
        # difflib.unified_diff 返回一个迭代器，每次 yield 一行
        # 参数说明：
        #   fromfile="a/path" — 原始文件标签（git diff 约定用 a/ 前缀）
        #   tofile="b/path"   — 新文件标签
        #   n=context_lines   — 上下文行数
        diff_iterator = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",       # git 风格前缀
            tofile=f"b/{file_path}",
            n=self.context_lines,            # 上下文行数（默认 3）
        )

        diff_lines = list(diff_iterator)     # 转为列表（便于后续解析）
        diff_text = "".join(diff_lines)      # 拼接为完整字符串

        # ── 3. 检测空 diff ───────────────────────────────────
        # 如果 diff_text 只有 --- 和 +++ 两行（无 @@ 头部），
        # 说明原始和新版本完全相同
        if self._is_empty_diff(diff_text):
            return DiffResult(
                file_path=file_path,
                original_content=original,
                modified_content=modified,
                diff_text=diff_text,
                hunks=[],
                lines_added=0,
                lines_removed=0,
                is_empty=True,
            )

        # ── 4. 解析 hunk 结构 ────────────────────────────────
        hunks = self._parse_hunks(diff_lines)

        # ── 5. 统计变更 ──────────────────────────────────────
        total_added = sum(h.lines_added for h in hunks)
        total_removed = sum(h.lines_removed for h in hunks)

        return DiffResult(
            file_path=file_path,
            original_content=original,
            modified_content=modified,
            diff_text=diff_text,
            hunks=hunks,
            lines_added=total_added,
            lines_removed=total_removed,
        )

    def generate_batch(
        self,
        changes: list[tuple[str, str, str]],
    ) -> list[DiffResult]:
        """
        批量生成多个文件的 diff。

        适合 Agent 一次修改多个文件后统一生成所有 diff。

        Args:
            changes: [(file_path, original_content, modified_content), ...]

        Returns:
            list[DiffResult] — 与输入顺序一致的 diff 列表

        Example:
            changes = [
                ("app/api/agent.py", old_api, new_api),
                ("app/agent/agent.py", old_agent, new_agent),
            ]
            results = generator.generate_batch(changes)
            for r in results:
                print(r.format_summary())
        """
        results = []
        for file_path, original, modified in changes:
            results.append(
                self.generate(file_path, original, modified)
            )
        return results

    def format_batch_for_llm(
        self,
        results: list[DiffResult],
        max_per_file: int = 80,
    ) -> str:
        """
        将多个文件的 diff 合并为一份 LLM 友好的文本。

        用于一次性向 LLM 展示所有文件的变更，减少对话轮次。

        output:
          # Multi-File Diff (3 files, 5 hunks, +12, -4)

          ## Diff: app/api/agent.py (2 hunks, +5, -3)
          ```diff
          ...
          ```

          ## Diff: app/agent/agent.py (1 hunk, +3, -1)
          ```diff
          ...
          ```
        """
        if not results:
            return "# Multi-File Diff (no changes)"

        total_hunks = sum(len(r.hunks) for r in results)
        total_added = sum(r.lines_added for r in results)
        total_removed = sum(r.lines_removed for r in results)

        header = (
            f"# Multi-File Diff "
            f"({len(results)} files, {total_hunks} hunks, "
            f"+{total_added}, -{total_removed})"
        )

        parts = [header, ""]
        for r in results:
            parts.append(r.format_for_llm(max_diff_lines=max_per_file))
            parts.append("")              # 空行分隔

        return "\n".join(parts)

    # ── 私有辅助方法 ──────────────────────────────────────────

    @staticmethod
    def _is_empty_diff(diff_text: str) -> bool:
        """
        判断 diff 是否表示"无任何变更"。

        空 diff 的特征：
          - 只有 0-2 行（--- 和 +++ 头部）
          - 或者根本没有 diff 输出
          - 没有 @@ 开头的 hunk 头部行

        正常的 diff 至少包含：
          --- a/file.py
          +++ b/file.py
          @@ -10,6 +10,8 @@ ...   ← 至少一个 hunk 头部
          ...
        """
        if not diff_text.strip():
            return True
        # 检查是否包含 hunk 头部（@@ ... @@）
        return "@@" not in diff_text

    @staticmethod
    def _parse_hunks(diff_lines: list[str]) -> list[DiffHunk]:
        """
        从 diff 行列表中解析出结构化 hunk。

        unified diff 格式：
          @@ -start,count +start,count @@ context
          -line
          +line
           context line
          @@ -start2,count2 +start2,count2 @@ context  ← 下一个 hunk

        hunk 头部正则: @@ -(d+),?(d*) +(d+),?(d*) @@(.*)
        """
        hunks: list[DiffHunk] = []
        current_header = ""              # 当前 hunk 的 @@ 头部
        current_lines: list[str] = []    # 当前 hunk 的内容行
        current_added = 0               # 当前 hunk 新增行计数
        current_removed = 0             # 当前 hunk 删除行计数
        old_start = new_start = 0       # 行号（从头部解析）
        old_count = new_count = 0

        for line in diff_lines:
            # ── hunk 头部行 ──
            if line.startswith("@@"):
                # 如果已经收集了一个 hunk，先保存
                if current_lines:
                    hunks.append(DiffHunk(
                        header=current_header,
                        old_start=old_start,
                        old_count=old_count,
                        new_start=new_start,
                        new_count=new_count,
                        lines=current_lines[:],     # 浅拷贝
                        lines_added=current_added,
                        lines_removed=current_removed,
                    ))

                # 重置为新 hunk
                current_header = line.strip()
                current_lines = [line]            # hunk 头部也是 diff 的一部分
                current_added = 0
                current_removed = 0

                # 解析行号信息: @@ -3,7 +3,8 @@ context
                #   3 = 原起始行, 7 = 原行数, 3 = 新起始行, 8 = 新行数
                import re
                match = re.match(
                    r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',
                    line,
                )
                if match:
                    old_start = int(match.group(1))
                    # 原行数可能省略（默认 1）
                    old_count = int(match.group(2)) if match.group(2) else 1
                    new_start = int(match.group(3))
                    new_count = int(match.group(4)) if match.group(4) else 1

            elif current_lines:
                # ── hunk 内容行 ──
                current_lines.append(line)

                # 统计增/删
                if line.startswith("+") and not line.startswith("+++"):
                    current_added += 1           # 新增行（排除 +++ 文件头）
                elif line.startswith("-") and not line.startswith("---"):
                    current_removed += 1         # 删除行（排除 --- 文件头）

        # ── 保存最后一个 hunk ──
        if current_lines:
            hunks.append(DiffHunk(
                header=current_header,
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                lines=current_lines[:],
                lines_added=current_added,
                lines_removed=current_removed,
            ))

        return hunks
