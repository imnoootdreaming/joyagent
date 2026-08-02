"""
可插拔 Skill 机制 (Pluggable Skill Mechanism)

设计目标：
  将「特定科研任务的完整 Prompt + 工具约束 + 流程约束」封装为独立 Skill 单元，
  运行时按场景动态挂载到 System Prompt，使同一套智能体能快速适配不同子任务。

一个 Skill 包含：
  - name         唯一标识
  - description  一句话说明（用于自动选择）
  - triggers     触发关键词（用户请求命中时自动启用）
  - system_prompt 追加到 Agent System Prompt 的段落
  - tools        该 Skill 启用的工具/能力约束（可选）

见 app/skills/loader.py 的 SkillLoader，以及 app/agent/prompt.py 的
build_system_prompt(skills=...) 动态组装入口。
"""
from app.skills.loader import Skill, SkillLoader

__all__ = ["Skill", "SkillLoader"]
