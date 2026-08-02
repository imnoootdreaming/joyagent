"""
可插拔 Skill 加载器。

Skill 是「Prompt + 工具约束 + 流程约束」的封装单元。SkillLoader 负责：
  1. 注册内置 Skill（代内置码）
  2. 从外部目录加载自定义 Skill（.py / .json）
  3. 根据用户请求文本自动匹配触发的 Skill
  4. 将匹配到的 Skill 渲染为可追加到 System Prompt 的文本
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Skill:
    """一个可插拔技能单元。"""

    name: str
    description: str = ""
    triggers: list[str] = field(default_factory=list)
    system_prompt: str = ""
    tools: list[str] = field(default_factory=list)

    def render(self) -> str:
        """渲染该 Skill 追加到 System Prompt 的文本片段。"""
        lines = [f"## Skill: {self.name}", self.description]
        if self.tools:
            lines.append("可用工具约束: " + ", ".join(self.tools))
        if self.system_prompt:
            lines.append(self.system_prompt)
        return "\n".join([ln for ln in lines if ln])


class SkillLoader:
    """Skill 注册中心 + 匹配器。"""

    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._register_builtins()

    # ── 注册 ──────────────────────────────────────────────
    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def register_builtin_skills(self, skills: list[Skill]) -> None:
        for s in skills:
            self.register(s)

    def load_from_dir(self, directory: str) -> int:
        """从目录加载 .json 形式的 Skill 定义，返回加载数量。"""
        if not os.path.isdir(directory):
            return 0
        count = 0
        for fn in os.listdir(directory):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(directory, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.register(Skill(**data))
                count += 1
            except Exception:
                continue
        return count

    # ── 匹配 ──────────────────────────────────────────────
    def match(self, user_request: str) -> list[Skill]:
        """根据用户请求文本，返回命中的 Skill 列表（去重，保持注册顺序）。"""
        text = (user_request or "").lower()
        matched: list[Skill] = []
        for skill in self._skills.values():
            if any(t.lower() in text for t in skill.triggers):
                matched.append(skill)
        return matched

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._skills.keys())

    # ── 渲染 ──────────────────────────────────────────────
    def render_for(self, skills: list[Skill]) -> str:
        if not skills:
            return ""
        blocks = ["\n\n# Active Skills", ""]
        for s in skills:
            blocks.append(s.render())
            blocks.append("")
        return "\n".join(blocks)

    # ── 内置 Skill ────────────────────────────────────────
    def _register_builtins(self) -> None:
        # 文献整理 Skill：对接 Zotero MCP，自动化文献检索与归类
        self.register(Skill(
            name="literature_curation",
            description=(
                "文献整理技能：调用 Zotero MCP 检索、归类与整理实验室文献库，"
                "支持按集合拉取条目、读取 PDF 注解、新增条目并打标签。"
            ),
            triggers=["文献", "论文", "zotero", "参考文献", "整理文献", "文献库"],
            tools=["zotero__search_items", "zotero__get_item", "zotero__add_item",
                   "zotero__add_tags", "zotero__get_pdf_annotations"],
            system_prompt=(
                "当用户要求整理文献时，优先使用 Zotero MCP 工具："
                "先用 search_items 按集合/关键词检索，再 get_item 读取元数据，"
                "用 get_pdf_annotations 汇总批注，最后 add_tags / add_item 归档。"
                "输出前给出结构化清单（标题、作者、年份、标签、链接）。"
            ),
        ))


# 全局单例
skill_loader = SkillLoader()
