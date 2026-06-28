"""
Phase 2: ToolRegistry — 工具注册中心。

单例模式，全局唯一。管理工具的注册、发现、执行和 Hook 生命周期。
Agent 通过 registry 获取工具 Schema（传给 Anthropic API）并执行工具调用。
"""
from __future__ import annotations

import time                          # 工具执行耗时计算
from app.tools.base import BaseTool, ToolResult  # 工具基类和统一返回格式
from app.tools.hooks import ToolHook  # Hook 协议（Phase 2 Step 8）


class ToolRegistry:
    """
    工具注册中心 —— 单例模式，全局唯一。

    职责：
      1. 注册/发现工具（register_tool / get_tool / list_tools）
      2. 生成 Anthropic 原生工具 Schema（get_tool_schemas）
      3. 统一执行入口（execute），在生命周期中插入 Hook 调用
      4. 管理 Hook 链（register_hook / remove_hook）
    """

    def __init__(self):
        # 工具字典：key=工具名(str), value=工具实例(BaseTool)
        self._tools: dict[str, BaseTool] = {}
        # Hook 列表：按注册顺序依次调用（Phase 2 Step 8）
        self._hooks: list[ToolHook] = []

    # ── 工具注册/查询 ─────────────────────────────────────

    def register_tool(self, tool: BaseTool) -> ToolResult:
        """
        注册一个工具实例到注册中心。

        同名工具重复注册 → 返回失败（防止意外覆盖已注册工具）。
        """
        if tool.name in self._tools:
            return ToolResult(
                success=False,
                message="",
                error=f"Tool: '{tool.name}' is already registered."
            )
        self._tools[tool.name] = tool     # 存入字典
        return ToolResult(
            success=True,
            message=f"Tool: '{tool.name}' registered successfully.",
            error=None,
        )

    def get_tool(self, tool_name: str) -> BaseTool | None:
        """按名称获取工具实例，不存在返回 None。"""
        return self._tools.get(tool_name, None)

    def list_tools(self) -> list[str]:
        """列出所有已注册工具的名称。"""
        return list(self._tools.keys())

    # ── Schema 生成 ────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict]:
        """
        生成所有工具的 Anthropic 原生格式 Schema 列表。

        直接传给 client.messages.create(tools=...):
          response = client.messages.create(
              ...,
              tools=tool_registry.get_tool_schemas(),
          )
        """
        return [tool.to_schema() for tool in self._tools.values()]

    def get_dangerous_tools(self) -> list[str]:
        """获取所有标记为危险的工具名称列表。"""
        return [tool.name for tool in self._tools.values() if tool.is_dangerous]

    # ── Hook 管理（Phase 2 Step 8） ────────────────────────

    def register_hook(self, hook: ToolHook) -> None:
        """
        注册一个 ToolHook 到执行链。

        Hook 按注册顺序依次调用：
          register_hook(A) → register_hook(B)
          执行时：A → B
        """
        self._hooks.append(hook)

    def remove_hook(self, hook: ToolHook) -> None:
        """
        移除一个已注册的 ToolHook。

        注意：按对象引用匹配，不是按类型。
        如果同一个 Hook 实例注册了多次，只移除第一次出现的。
        """
        if hook in self._hooks:
            self._hooks.remove(hook)

    # ── 统一执行入口（含 Hook 生命周期） ───────────────────

    async def execute(self, name: str, **kwargs) -> ToolResult:
        """
        根据工具名称执行工具调用。

        这是 Agent 调用工具的唯一入口，内部集成了完整的 Hook 生命周期：

          on_pre_execute  → 执行前检查（可阻止）
          tool.execute()  → 实际执行
          on_post_execute → 执行后处理（统计/日志）
          on_error        → 异常处理（记录/降级）

        参数映射：
          Agent 调用 tool_registry.execute("read_file", path="/a/b.txt")
          → Registry 查找 name="read_file" 的 BaseTool
          → 调用 tool.execute(path="/a/b.txt")
        """
        # 1. 查找工具
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                message="",
                error=f"Unknown tool: {name}",
            )

        start_time = time.time()       # 记录开始时间（用于计算耗时）

        try:
            # ── Hook Phase 1: pre_execute ──
            for hook in self._hooks:
                override = await hook.on_pre_execute(name, kwargs)
                if override is not None:
                    # Hook 返回了替代结果 → 跳过实际执行
                    # 支持两种返回格式：dict（转为 ToolResult）或 ToolResult 实例
                    if isinstance(override, dict):
                        return ToolResult(**override)
                    return override     # 假设是 ToolResult 实例

            # ── 实际执行 ──
            result = await tool.execute(**kwargs)
            elapsed_ms = (time.time() - start_time) * 1000  # 转换为毫秒

            # ── Hook Phase 2: post_execute ──
            for hook in self._hooks:
                result = await hook.on_post_execute(
                    name, kwargs, result, elapsed_ms
                )

            return result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000

            # ── Hook Phase 3: on_error ──
            swallowed = False            # 标记是否有 Hook 吞掉了异常
            for hook in self._hooks:
                if await hook.on_error(name, kwargs, e):
                    swallowed = True     # 至少一个 Hook 决定降级
            if swallowed:
                # 异常被吞掉 → 返回失败结果（避免 Agent 循环崩溃）
                return ToolResult(
                    success=False,
                    message="",
                    error=f"Error swallowed by hook: {str(e)}",
                )

            # 无 Hook 吞异常 → 正常返回错误
            return ToolResult(
                success=False,
                message="",
                error=str(e),
            )


# ── 全局单例 ──
# Agent、API、Hook 均通过此实例访问注册中心
tool_registry = ToolRegistry()
