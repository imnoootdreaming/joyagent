"""
Phase 2 Step 8: Tool Hook 中间件 + 统计收集器。

ToolHook 在工具执行生命周期中提供三个拦截点：
  - on_pre_execute:  执行前（可阻止执行、修改参数）
  - on_post_execute: 执行后（可修改结果、记录日志）
  - on_error:        执行异常（记录错误、决定是否降级）

ToolStatsCollector: 内置统计收集器，实现 Agent 行为可观测性。
  - 维度: 调用次数 / 成功率 / P50/P95/P99 耗时
  - 输出: 周期性控制台日志 + get_stats() API 查询
"""

import time                          # 计算工具执行耗时（elapsed_ms）
from abc import ABC, abstractmethod  # Hook 抽象基类
from dataclasses import dataclass, field  # 数据类装饰器
from typing import Any               # 泛型类型标注


# ═══════════════════════════════════════════════════════════════
# Hook 协议
# ═══════════════════════════════════════════════════════════════

class ToolHook(ABC):
    """
    工具执行生命周期 Hook 抽象基类。

    子类只需覆盖关心的拦截点，不需要全部实现。
    ToolRegistry 在 execute() 的对应阶段依次调用所有已注册 Hook。

    三个拦截点：
      on_pre_execute  → 执行前（可阻止执行）
      on_post_execute → 执行后（可记录/修改结果）
      on_error        → 异常时（可记录/吞掉异常）
    """

    async def on_pre_execute(self, tool_name: str, kwargs: dict) -> dict | None:
        """
        工具执行前回调。

        Args:
            tool_name: 工具名称（如 "read_file"）
            kwargs:    LLM 传入的工具参数（如 {"path": "/a/b.txt"}）

        Returns:
            None → 继续正常执行
            dict → 跳过实际执行，将 dict 作为 ToolResult(**dict) 直接返回
        """
        return None                    # 默认：不干预，正常执行

    async def on_post_execute(self, tool_name: str, kwargs: dict,
                              result: Any, elapsed_ms: float) -> Any:
        """
        工具执行成功后回调。

        Args:
            tool_name:  工具名称
            kwargs:     工具参数
            result:     工具执行结果（ToolResult）
            elapsed_ms: 工具执行耗时（毫秒）

        Returns:
            修改后的 result（原样返回则不做修改）
        """
        return result                  # 默认：不修改结果

    async def on_error(self, tool_name: str, kwargs: dict,
                       error: Exception) -> bool:
        """
        工具执行异常时回调。

        Args:
            tool_name: 工具名称
            kwargs:    工具参数
            error:     捕获到的异常对象

        Returns:
            True  → 吞掉异常（降级处理，返回带 error 的 ToolResult）
            False → 继续向外抛出异常
        """
        return False                   # 默认：不吞异常，继续传播


# ═══════════════════════════════════════════════════════════════
# 单次调用记录
# ═══════════════════════════════════════════════════════════════

@dataclass
class ToolCallRecord:
    """
    单次工具调用的完整记录。

    用于统计收集器内部存储，支持后续聚合分析。
    field(default_factory=...) 确保每条记录自动打时间戳。
    """
    tool_name: str                   # 工具名称（如 "execute_shell"）
    success: bool                    # 是否执行成功
    elapsed_ms: float                # 执行耗时（毫秒），异常时为 0
    error: str | None = None         # 错误信息（成功时为 None）
    timestamp: float = field(        # 调用时间戳（epoch 秒）
        default_factory=time.time    # 每次创建自动填充当前时间
    )


# ═══════════════════════════════════════════════════════════════
# 统计收集器
# ═══════════════════════════════════════════════════════════════

class ToolStatsCollector(ToolHook):
    """
    工具调用统计收集器 — 实现 Agent 行为可观测性。

    统计维度：
      - 调用次数 (total) / 成功次数 (success) / 失败次数 (failed)
      - 成功率 (success_rate)
      - P50 / P95 / P99 耗时 (latency_p50_ms / p95 / p99)

    输出模式：
      1. 自动输出：每 log_interval 次调用后在控制台打印统计摘要
      2. API 查询：get_stats(tool_name) / get_all_stats() 供外部调用

    使用方式：
      collector = ToolStatsCollector(log_interval=10)
      tool_registry.register_hook(collector)
      # ... Agent 工作 ...
      print(collector.get_all_stats())
    """

    def __init__(self, log_interval: int = 10):
        """
        Args:
            log_interval: 每多少次工具调用后自动输出统计摘要（默认 10）
        """
        self.log_interval = log_interval     # 自动输出间隔
        self.records: list[ToolCallRecord] = []  # 全部调用记录（按时间排序）
        # 按工具名分组存储，key=tool_name, value=list[ToolCallRecord]
        self._by_tool: dict[str, list[ToolCallRecord]] = {}

    # ── Hook 实现 ──────────────────────────────────────────

    async def on_post_execute(self, tool_name: str, kwargs: dict,
                              result: Any, elapsed_ms: float) -> Any:
        """
        工具执行成功后记录统计。

        在 on_post_execute 中记录而非 on_pre_execute：
          - 此时已知 success/elapsed_ms 等完整信息
          - 可以准确判断工具是否成功
        """
        # 创建记录：从 result 中提取 success 字段
        record = ToolCallRecord(
            tool_name=tool_name,
            success=getattr(result, 'success', True),  # 默认 True（兼容非 ToolResult）
            elapsed_ms=elapsed_ms,
        )
        # 写入全量列表
        self.records.append(record)
        # 写入分组字典（自动创建不存在的 key）
        self._by_tool.setdefault(tool_name, []).append(record)

        # 达到间隔阈值 → 自动输出统计摘要
        if len(self.records) % self.log_interval == 0:
            self._log_summary()

        return result                    # 不修改结果，原样返回

    async def on_error(self, tool_name: str, kwargs: dict,
                       error: Exception) -> bool:
        """
        工具执行异常时记录失败统计。

        与 post_execute 的差异：
          - success 强制设为 False
          - elapsed_ms 为 0（异常通常很快发生，不计时）
          - 记录 error 消息供后续排查
        """
        record = ToolCallRecord(
            tool_name=tool_name,
            success=False,               # 异常 → 失败
            elapsed_ms=0,                # 异常不计耗时
            error=str(error),            # 保留错误信息
        )
        self.records.append(record)
        self._by_tool.setdefault(tool_name, []).append(record)

        return False                     # 不吞异常，继续传播

    # ── 统计查询 API ───────────────────────────────────────

    def get_stats(self, tool_name: str = None) -> dict:
        """
        获取统计摘要。

        Args:
            tool_name: 工具名 → 返回该工具的统计；None → 返回全局统计

        Returns:
            dict with keys: tool_name, total, success, failed,
                           success_rate, latency_p50_ms, p95, p99
            无记录时返回 {"tool_name": ..., "total": 0}
        """
        # 选择数据源：指定工具 → 分组数据；否则 → 全量数据
        records = self._by_tool.get(tool_name) if tool_name else self.records
        if not records:
            return {"tool_name": tool_name, "total": 0}

        # 统计成功次数
        success_count = sum(1 for r in records if r.success)
        # 提取成功调用耗时并排序（用于计算百分位）
        latencies = sorted(
            r.elapsed_ms for r in records if r.elapsed_ms > 0
        )

        def percentile(p: float) -> float:
            """
            计算耗时百分位。

            例如 percentile(0.95) = 第 95 百分位耗时
            空列表 → 返回 0
            """
            if not latencies:
                return 0
            # 百分位索引 = 列表长度 × 百分比（向下取整，不超过列表末尾）
            idx = min(int(len(latencies) * p), len(latencies) - 1)
            return latencies[idx]

        return {
            "tool_name": tool_name or "all",
            "total": len(records),                         # 总调用次数
            "success": success_count,                      # 成功次数
            "failed": len(records) - success_count,        # 失败次数
            "success_rate": f"{success_count / len(records) * 100:.1f}%",  # 成功率
            "latency_p50_ms": percentile(0.50),            # P50 耗时
            "latency_p95_ms": percentile(0.95),            # P95 耗时
            "latency_p99_ms": percentile(0.99),            # P99 耗时
        }

    def get_all_stats(self) -> dict:
        """
        获取所有工具的统计摘要。

        Returns:
            {"read_file": {...}, "execute_shell": {...}, ...}
            每个 value 格式同 get_stats() 返回值
        """
        return {name: self.get_stats(name) for name in self._by_tool}

    # ── 内部辅助 ───────────────────────────────────────────

    def _log_summary(self):
        """
        在控制台输出格式化统计摘要表格。

        输出时机：每 log_interval 次工具调用后自动触发。
        包含：每个工具的调用次数、成功率、P50/P95 耗时。
        """
        stats = self.get_all_stats()
        # 分隔线
        print(f"\n{'=' * 68}")
        print(f"  Tool Call Statistics (total calls: {len(self.records)})")
        print(f"{'=' * 68}")
        # 按成功率降序排列工具
        for name, s in sorted(
            stats.items(),
            key=lambda kv: float(kv[1].get('success_rate', '0').rstrip('%')),
            reverse=True,              # 成功率高的排前面
        ):
            print(
                f"  {name:25s} "       # 工具名（左对齐，占 25 列）
                f"| total={s['total']:>4d} "        # 总次数（右对齐 4 位）
                f"| success={s['success_rate']:>6s} " # 成功率（右对齐 6 位）
                f"| p50={s['latency_p50_ms']:>7.1f}ms "  # P50（右对齐 7 位，1 位小数）
                f"| p95={s['latency_p95_ms']:>7.1f}ms"   # P95
            )
        print(f"{'=' * 68}\n")
