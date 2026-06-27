"""
Phase 4 Step 6: generate_diff tool — 代码差异生成工具。

将 DiffGenerator 包装为 BaseTool，让 LLM Agent 可以生成 unified diff。
Agent 给出修改后的代码，工具计算与原始文件的差异。
"""

from app.tools.base import BaseTool, ToolResult         # 工具基类和返回格式
from app.coding.diff_generator import DiffGenerator     # unified diff 生成器


class GenerateDiffTool(BaseTool):
    """
    Diff 生成工具 —— 比较两个代码版本生成 unified diff。

    通常在 Agent 生成修改后的代码之后调用：
      1. Agent 用 read_file 读原始代码
      2. Agent 生成修改后的代码
      3. Agent 调用 generate_diff(original=..., modified=...) 生成 diff
      4. Agent 展示 diff 给用户确认
      5. Agent 调用 apply_patch 应用 diff
    """

    name = "generate_diff"

    description = (
        "Generate a unified diff between original and modified code. "
        "Provide 'path' (file path), 'original' (current code), and 'modified' (new code). "
        "Returns the diff in standard unified format that can be applied with apply_patch."
    )

    @property
    def is_dangerous(self) -> bool:
        return False                     # 只计算差异，不修改文件

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path that is being modified (appears in diff header).",
                },
                "original": {
                    "type": "string",
                    "description": "The original/current file content.",
                },
                "modified": {
                    "type": "string",
                    "description": "The new/modified file content (LLM-generated).",
                },
            },
            "required": ["path", "original", "modified"],
        }

    async def execute(self, path: str, original: str, modified: str,
                      **kwargs) -> ToolResult:
        try:
            generator = DiffGenerator()
            result = generator.generate(path, original, modified)

            if result.is_empty:
                return ToolResult(success=False, message="", error="No changes detected (original == modified).")

            # ── 格式化 diff 为 LLM 友好输出 ──
            formatted = result.format_for_llm()
            return ToolResult(
                success=True,
                message=formatted,
                metadata={
                    "hunks": len(result.hunks),
                    "lines_added": result.lines_added,
                    "lines_removed": result.lines_removed,
                    "can_apply": result.can_apply_cleanly(),
                },
            )
        except Exception as e:
            return ToolResult(success=False, message="", error=str(e))
