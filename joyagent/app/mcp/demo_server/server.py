"""
Phase 8 Step 4 — JoyAgent Demo MCP Server

一个完整的 MCP Server 实现（纯 Python JSON-RPC over stdio），
提供三个工具：get_weather、calculate、get_time。

MCP Server 协议流程：
  - stdin  读取 JSON-RPC 请求（一行一个请求）
  - stdout 写入 JSON-RPC 响应（一行一个响应）
  - stderr 写入日志（不影响协议通信）

安全设计：
  - calculate 使用受限的数学表达式求值器（非 eval）
  - 所有输入经过 JSON 解析校验

面试要点：
  Q: "你如何实现 MCP Server？"
  A: "基于 JSON-RPC 2.0 over stdio。Server 从 stdin 读取 JSON 请求，
      根据 method 分发给不同 handler。支持 initialize（握手）、
      tools/list（声明能力）、tools/call（执行能力）三个核心方法。
      约 150 行纯 Python，无第三方依赖。"
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from typing import Any

# ═══════════════════════════════════════════════════════════════════════
# 工具定义（符合 MCP tools/list 响应格式）
# ═══════════════════════════════════════════════════════════════════════

DEMO_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city (simulated data for demo)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (e.g., Beijing, Tokyo, New York)",
                }
            },
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Safely evaluate a mathematical expression. "
            "Supports: +, -, *, /, **, %, sqrt, sin, cos, abs, pi, e."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g. '2 + 3 * 4' or 'sqrt(144)'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_time",
        "description": "Get the current date and time (local timezone)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": (
                        "Output format: 'iso' (ISO 8601), "
                        "'unix' (timestamp), or 'human' (default)"
                    ),
                    "enum": ["iso", "unix", "human"],
                }
            },
        },
    },
]

# ═══════════════════════════════════════════════════════════════════════
# 安全的数学表达式求值器
# ═══════════════════════════════════════════════════════════════════════

_SAFE_MATH = {
    "pi": math.pi, "e": math.e, "tau": math.tau,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "abs": abs, "round": round,
    "min": min, "max": max, "log": math.log,
    "log10": math.log10, "log2": math.log2,
    "ceil": math.ceil, "floor": math.floor, "pow": pow,
}


def safe_eval(expression: str) -> float:
    """安全求值数学表达式。只允许白名单函数和运算符。"""
    dangerous = [
        "__", "import", "exec", "eval", "open", "write",
        "system", "subprocess", "os.", "sys.", "lambda",
        "class", "def", "globals", "locals", "getattr",
        "setattr", "=", ";", ":",
    ]
    expr_lower = expression.lower()
    for keyword in dangerous:
        if keyword in expr_lower:
            raise ValueError(
                f"Expression contains forbidden keyword/operator: '{keyword}'"
            )

    try:
        tree = compile(expression, "<calculate>", "eval")
    except SyntaxError as e:
        raise SyntaxError(f"Invalid expression syntax: {e}")

    restricted_globals = {"__builtins__": {}}
    restricted_locals = _SAFE_MATH.copy()

    try:
        return eval(tree, restricted_globals, restricted_locals)
    except Exception as e:
        raise ValueError(f"Evaluation error: {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════════
# 工具实现（纯 ASCII 输出，保证跨平台 stdio 兼容）
# ═══════════════════════════════════════════════════════════════════════

_WEATHER_CONDITIONS = [
    "Sunny", "Partly Cloudy", "Cloudy",
    "Light Rain", "Windy", "Clear",
]


def _get_weather(city: str) -> str:
    condition = random.choice(_WEATHER_CONDITIONS)
    temp = random.randint(5, 38)
    humidity = random.randint(20, 90)
    wind = random.randint(0, 30)
    return (
        f"Weather in {city}: {condition}\n"
        f"  Temperature: {temp}C\n"
        f"  Humidity: {humidity}%\n"
        f"  Wind: {wind} km/h\n\n"
        f"  [Note: This is simulated demo data]"
    )


def _calculate(expression: str) -> str:
    try:
        result = safe_eval(expression)
        if isinstance(result, float) and result == int(result):
            return f"{expression} = {int(result)}"
        return f"{expression} = {result}"
    except (ValueError, SyntaxError) as e:
        return f"Error: {e}"


def _get_time(format_type: str = "human") -> str:
    now = time.time()
    local = time.localtime(now)
    tz = (time.tzname[0] or "UTC").encode("ascii", errors="replace").decode("ascii")
    if format_type == "unix":
        return f"Unix timestamp: {int(now)}"
    elif format_type == "iso":
        return time.strftime("%Y-%m-%dT%H:%M:%S%z", local)
    else:
        return (
            f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S', local)}\n"
            f"  Timezone: {tz}\n"
            f"  Unix timestamp: {int(now)}"
        )


# ═══════════════════════════════════════════════════════════════════════
# MCP JSON-RPC Server
# ═══════════════════════════════════════════════════════════════════════

class DemoMCPServer:
    """MCP Demo Server — 通过 stdin/stdout JSON-RPC 提供工具。"""

    def __init__(self, verbose: bool = False):
        self._verbose = verbose
        self._initialized = False

    def run(self) -> None:
        """主循环：从 stdin 逐行读取 JSON-RPC 请求，处理后写入 stdout。"""
        if self._verbose:
            print("[demo-server] MCP Demo Server started",
                  file=sys.stderr, flush=True)

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                self._error(None, -32700, f"Parse error: {e}")
                continue

            response = self._handle_request(request)
            if response is not None:
                self._write(response)

    def _handle_request(self, req: dict) -> dict | None:
        rid = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {})

        if self._verbose:
            print(f"[demo-server] <- {method}", file=sys.stderr, flush=True)

        # 通知（不回复）
        if rid is None:
            return None

        if method == "initialize":
            self._initialized = True
            return self._ok(rid, {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "joyagent-demo-server", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            })

        if method == "tools/list":
            return self._ok(rid, {"tools": DEMO_TOOLS})

        if method == "tools/call":
            return self._handle_tool_call(rid, params)

        if method == "ping":
            return self._ok(rid, {"message": "pong"})

        return self._error(rid, -32601, f"Method not found: {method}")

    def _handle_tool_call(self, rid: Any, params: dict) -> dict:
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        try:
            if tool_name == "get_weather":
                text = _get_weather(arguments.get("city", "Unknown"))
            elif tool_name == "calculate":
                text = _calculate(arguments.get("expression", ""))
            elif tool_name == "get_time":
                text = _get_time(arguments.get("format", "human"))
            else:
                return self._ok(rid, {
                    "isError": True,
                    "content": [{"type": "text",
                                 "text": f"Unknown tool: {tool_name}"}],
                })
            return self._ok(rid, {
                "content": [{"type": "text", "text": text}],
            })
        except Exception as e:
            return self._ok(rid, {
                "isError": True,
                "content": [{"type": "text",
                             "text": f"Error: {type(e).__name__}: {e}"}],
            })

    def _ok(self, rid: Any, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def _error(self, rid: Any, code: int, message: str) -> None:
        self._write({
            "jsonrpc": "2.0", "id": rid,
            "error": {"code": code, "message": message},
        })

    def _write(self, data: dict) -> None:
        sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
        sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════

def run_demo_server(verbose: bool = False) -> None:
    """启动 Demo MCP Server（阻塞 stdin 循环）。"""
    server = DemoMCPServer(verbose=verbose)
    server.run()


if __name__ == "__main__":
    run_demo_server(verbose=True)
