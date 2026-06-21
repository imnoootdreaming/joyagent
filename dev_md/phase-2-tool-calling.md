# Phase 2：Tool Calling Framework

> **返回主索引：** [0-1 coding.md](0-1%20coding.md)
> **上一阶段：** [Phase 1: Agent MVP](phase-1-mvp.md)
> **下一阶段：** [Phase 3: LangGraph Workflow](phase-3-langgraph-workflow.md)

---

## 一、目标与定位

### 目标
建立统一的工具抽象层（BaseTool + Tool Registry），新增 Shell 执行、Git 操作工具，使 Agent 能执行命令和管理版本。

### 在整体架构中的位置
本 Phase 把 Phase 1 的"散装工具"升级为**可注册、可发现、可扩展的工具框架**。这是后续所有工具（Docker、MCP、Browser）的基础。

```
Phase 1:   file_read.py  +  file_write.py  (散装)
                ↓
Phase 2:   BaseTool → ToolRegistry → Shell / Git / File 系列 (统一框架)
```

### 本 Phase 不做什么
- ❌ 不做 Shell 安全沙箱（Phase 5）
- ❌ 不做 MCP 协议接入（Phase 8）
- ❌ 不做工具执行日志持久化（Phase 9）

---

## 二、前置依赖

| 依赖 | 用途 |
|------|------|
| Phase 1 完成 | Agent 基础框架 |
| gitpython | Git 操作封装（可选，也可用 subprocess） |
| Python subprocess | Shell 执行 |

```bash
uv add gitpython
```

---

## 三、目录结构

```text
app/
├── tools/
│   ├── __init__.py
│   ├── base.py              # BaseTool 抽象类
│   ├── registry.py          # ToolRegistry 注册中心
│   ├── schemas.py           # 工具 Schema 定义集中管理
│   │
│   ├── file/
│   │   ├── read.py          # 文件读取（从 Phase 1 迁移）
│   │   └── write.py         # 文件写入（从 Phase 1 迁移）
│   │
│   ├── shell/
│   │   └── execute.py       # Shell 命令执行（新增）
│   │
│   └── git/
│       ├── status.py        # Git 状态查询（新增）
│       └── diff.py          # Git 差异查看（新增）
│
├── agent/
│   └── agent.py             # Agent 改为使用 ToolRegistry（修改）
```

---

## 四、核心数据模型 / Schema 定义

### 4.1 BaseTool 抽象类

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class ToolResult:
    """所有工具的执行结果统一格式"""
    success: bool
    output: str            # 给 LLM 看的文本
    error: str | None = None
    metadata: dict | None = None  # 额外信息（如耗时、文件大小）

class BaseTool(ABC):
    """所有工具的基类。每个工具 = name + description + parameters + execute()"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称，LLM 通过此名称调用工具"""
        ...
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述，LLM 据此判断何时使用此工具"""
        ...
    
    @property
    @abstractmethod
    def parameters(self) -> dict:
        """工具参数，JSON Schema 格式"""
        ...
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """执行工具逻辑"""
        ...
    
    @property
    def is_dangerous(self) -> bool:
        """是否需要用户确认才能执行。默认 False，Shell/Git/Write 覆盖为 True"""
        return False
    
    def to_schema(self) -> dict:
        """转为 Anthropic Messages API 原生工具格式

        Anthropic 格式：{"name": ..., "description": ..., "input_schema": {...}}
        无需 OpenAI Function Calling 的 "type": "function" 包装层。
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }
```

### 4.2 ToolRegistry

```python
class ToolRegistry:
    """工具注册中心。单例模式，全局唯一。"""
    
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        """注册一个工具。同名工具会覆盖旧工具。"""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)
    
    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())
    
    def get_tool_schemas(self) -> list[dict]:
        """获取所有工具的 Anthropic 原生 Schema，直接传给 client.messages.create(tools=...)"""
        return [t.to_schema() for t in self._tools.values()]
    
    def get_dangerous_tools(self) -> list[str]:
        """获取所有危险工具名称"""
        return [t.name for t in self._tools.values() if t.is_dangerous]
    
    async def execute(self, name: str, **kwargs) -> ToolResult:
        """根据名称执行工具"""
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, output="", error=f"Unknown tool: {name}")
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

# 全局单例
tool_registry = ToolRegistry()
```

### 4.3 具体工具实现示例

**`app/tools/shell/execute.py`：**
```python
import subprocess
import asyncio
from app.tools.base import BaseTool, ToolResult

class ShellExecuteTool(BaseTool):
    name = "execute_shell"
    description = "Execute a shell command and return its output"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the command (optional)"
                }
            },
            "required": ["command"]
        }
    
    @property
    def is_dangerous(self) -> bool:
        return True  # Shell 命令必须用户确认
    
    async def execute(self, command: str, working_dir: str = None, **kwargs) -> ToolResult:
        # ⚠️ Phase 2 不做安全限制，Phase 5 加 Docker Sandbox
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir,
            )
            stdout, stderr = await process.communicate()
            
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += "\n[STDERR]\n" + stderr.decode("utf-8", errors="replace")
            
            return ToolResult(
                success=process.returncode == 0,
                output=output[:5000],  # 截断
                metadata={"exit_code": process.returncode}
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
```

**`app/tools/git/status.py`：**
```python
from app.tools.base import BaseTool, ToolResult
import subprocess

class GitStatusTool(BaseTool):
    name = "git_status"
    description = "Show the working tree status (git status)"
    
    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}
    
    @property
    def is_dangerous(self) -> bool:
        return False  # git status 只读，安全
    
    async def execute(self, **kwargs) -> ToolResult:
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                capture_output=True, text=True, timeout=10
            )
            return ToolResult(
                success=True,
                output=result.stdout or "Working tree clean",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
```

---

## 五、详细开发清单（含 HOW）

### Step 1：定义 BaseTool 与 ToolResult（30 分钟）
- 按 §4.1 的结构实现 `base.py`
- 关键设计点：`is_dangerous` 属性（为 Human-in-the-Loop 预留）、`to_schema()` 转换方法（Anthropic 原生 `input_schema` 格式）、`ToolResult` 统一返回格式

### Step 2：实现 ToolRegistry（30 分钟）
- 按 §4.2 实现 `registry.py`
- 全局单例 `tool_registry`
- 提供 `get_tool_schemas()` 方法，供 Agent 传入 `client.messages.create(tools=...)`

### Step 3：迁移 + 重构现有工具（30 分钟）
- 将 Phase 1 的 `file_read.py` / `file_write.py` 改造为继承 `BaseTool`
- 注册到 `tool_registry`

### Step 4：实现 Shell 工具（30 分钟）
- 按 §4.3 实现 `shell/execute.py`
- ⚠️ **Phase 2 不实现安全限制**——这是设计债务，在 Phase 5 偿还
- 用 `asyncio.create_subprocess_shell` 实现异步执行
- 设置 30 秒超时

### Step 5：实现 Git 工具（30 分钟）
- `git/status.py`：封装 `git status --short`
- `git/diff.py`：封装 `git diff`
- 后续可扩展：`git log`、`git branch`、`git commit`（需用户确认）

### Step 6：修改 Agent 使用 ToolRegistry（30 分钟）
- 将 Phase 1 的硬编码工具列表替换为 `tool_registry.get_tool_schemas()`
- 将工具执行逻辑替换为 `tool_registry.execute(block.name, **block.input)`
- 添加 `is_dangerous` 检查（Phase 2 暂时只打日志，Phase 9 接入用户审批）

### Step 7：工具注册入口（15 分钟）

**`app/tools/__init__.py`：**
```python
from app.tools.registry import tool_registry
from app.tools.file.read import FileReadTool
from app.tools.file.write import FileWriteTool
from app.tools.shell.execute import ShellExecuteTool
from app.tools.git.status import GitStatusTool
from app.tools.git.diff import GitDiffTool

def register_all_tools():
    tool_registry.register(FileReadTool())
    tool_registry.register(FileWriteTool())
    tool_registry.register(ShellExecuteTool())
    tool_registry.register(GitStatusTool())
    tool_registry.register(GitDiffTool())

# 在 app/main.py 启动时调用 register_all_tools()
```

---

## 六、关键代码模式与伪代码

### 修改后的 Agent Runtime（使用 ToolRegistry + Anthropic 原生模式）

```python
class Agent:
    def __init__(self):
        self.client = get_or_create_client()
        # Phase 2: 通过 ToolRegistry 获取工具列表，直接传给 Anthropic SDK
        # 无需 bind_tools() —— Anthropic SDK 原生支持

    async def chat(self, user_message: str) -> dict:
        messages = [{"role": "user", "content": user_message}]

        while iterations < max_iterations:
            response = self.client.messages.create(
                model=MODEL, system=SYSTEM_PROMPT,
                messages=messages,
                tools=tool_registry.get_tool_schemas(),  # ⬅ Phase 2
                max_tokens=4096,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return {"response": extract_text(response.content)}

            # ⬇ Phase 1: 硬编码 TOOL_HANDLERS
            # handler = TOOL_HANDLERS.get(block.name)

            # ⬇ Phase 2: 统一通过 Registry
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_result = await tool_registry.execute(
                    block.name, **block.input  # Anthropic 用 .input 而非 ["args"]
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_result.output if tool_result.success
                               else f"Error: {tool_result.error}",
                })
                # is_dangerous 检查
                if tool_registry.get(block.name).is_dangerous:
                    print(f"  \033[33m[dangerous] {block.name} executed\033[0m")

            messages.append({"role": "user", "content": tool_results})
```

---

## 七、完成标志

### 基本完成
- [ ] `BaseTool` 抽象类和 `ToolRegistry` 正常工作
- [ ] Agent 可以执行 Shell 命令（如 `ls`、`python --version`）
- [ ] Agent 可以查看 Git 状态（`git status`）和差异（`git diff`）
- [ ] `is_dangerous` 属性正确标记（Shell/Write = True, Read/Status = False）

### 自测用例

```bash
# 测试 1：Shell 工具
curl -X POST /api/chat -d '{"message": "列出当前目录的所有 Python 文件"}'
# 期望：Agent 调用 execute_shell(command="ls *.py")

# 测试 2：Git 工具
curl -X POST /api/chat -d '{"message": "查看当前 git 仓库的状态"}'
# 期望：Agent 调用 git_status

# 测试 3：多工具组合
curl -X POST /api/chat -d '{"message": "找到所有 Python 文件，把每个文件名写入 file_list.txt"}'
# 期望：先 execute_shell(ls) → read_file → write_file
```

---

## 八、文档差距分析

### 原文档缺失的内容及补充

| # | 原文档问题 | 实际开发需要 | 本文档补充位置 |
|---|-----------|-------------|-------------|
| 1 | `BaseTool` 只说名字，没定义字段 | 必须定义 `name` / `description` / `parameters` / `execute` / `is_dangerous` / `to_schema`（Anthropic 原生格式） | §4.1 |
| 2 | `Tool Registry` 没说怎么注册和发现 | 需要 register/get/list/get_tool_schemas 等完整 API | §4.2 |
| 3 | 没说工具 Schema 格式 | 工具 Schema 必须符合 Anthropic Messages API 的 `input_schema` JSON Schema 规范，无需 `"type": "function"` 包装 | §4.3 |
| 4 | Shell 执行没提安全问题 | 需要标记 `is_dangerous`，Phase 2 先打日志，Phase 5 加 Docker 沙箱 | §4.3, §5 Step 4 |
| 5 | 没说 `ToolResult` 统一返回格式 | Agent 需要统一处理成功/失败/错误，否则解析逻辑分散 | §4.1 |
| 6 | Git 工具只列了 status 和 diff | commit/push 等写入操作暂不实现（设计选择需说明） | §5 Step 5 |
| 7 | 没说工具注册的入口和初始化时机 | 需要 `register_all_tools()` 在 FastAPI 启动时注册 | §5 Step 7 |

### 本 Phase 新增债务

| 债务 | 偿还 Phase |
|------|-----------|
| Shell 执行无安全限制（任意命令、无超时强制、无资源限制） | Phase 5 |
| `is_dangerous` 标记了但未接入审批流 | Phase 9 |
| 工具执行超时控制不完善 | Phase 5 |
| 工具执行结果无持久化日志 | Phase 9 |
| 没有工具调用重试机制 | Phase 9 |

---

## 九、面试考点映射

| 面试题 | 答案要点 | 本文档参考 |
|--------|---------|-----------|
| **Anthropic Tool Use 的实现原理？** | 模型通过 `stop_reason == "tool_use"` 标识需要调用工具，`response.content` 中混有 `text` 和 `tool_use` 两种 block 类型。应用层遍历 content 检查 `block.type == "tool_use"`，执行 handler，结果以 `{"type": "tool_result", "tool_use_id": ..., "content": ...}` 作为 user 消息追加。 | §4.1, §6 |
| **为什么需要 ToolRegistry？** | 解耦工具定义和执行；支持动态注册/卸载；统一 Schema 生成（Anthropic 原生 `input_schema` 格式）；集中管理工具元数据（is_dangerous 等）。 | §4.2 |
| **Anthropic 工具格式 vs OpenAI 格式？** | Anthropic: `{"name": "...", "description": "...", "input_schema": {...}}`。OpenAI: `{"type": "function", "function": {"name": "...", "parameters": {...}}}`。Anthropic 少一层 `"type": "function"` 包装，更简洁。 | §4.1 |
| **is_dangerous 的设计意义？** | 安全控制：不是所有工具都应该自动执行。Shell 命令、文件写入需要用户确认。这是 Human-in-the-Loop 的基础设施。 | §4.1, §4.3 |
| **为什么 ToolResult 要统一格式？** | Agent 需要一致的方式处理成功/失败/错误。统一格式让 Agent 可以做结构化判断（"工具失败了，我能重试吗？"）。 | §4.1 |
| **Git 工具为什么不直接给 commit 权限？** | 渐进式安全：只读工具（status/diff）自动执行，写入工具（commit/push）需用户明确确认。避免 Agent 自动提交错误代码。 | §5 Step 5 |
