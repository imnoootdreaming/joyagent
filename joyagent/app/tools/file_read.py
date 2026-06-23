# ── 标准库导入 ──
import os                          # 文件路径存在性检查和文件读取

# ── 项目内导入 ──
from app.tools.base import BaseTool, ToolResult  # 工具基类和统一返回格式


class FileReadTool(BaseTool):
    """
    Phase 2: 文件读取工具，继承 BaseTool 统一抽象。
    只读操作，is_dangerous 保持默认 False（无需用户确认即可执行）。
    """

    # ─── 工具标识（类属性，覆盖抽象 property） ───
    # LLM 通过此名称调用工具，如 {"name": "read_file", "input": {"path": "/a/b.txt"}}
    name = "read_file"

    # LLM 据此判断"什么时候该用这个工具"——描述越精确，LLM 调度越准确
    description = (
        "Read the contents of a file at the given path. "
        "Returns the file content as text, truncated to 10000 characters if too long."
    )

    # ─── JSON Schema 定义（@property 覆盖抽象方法） ───
    @property
    def input_schema(self) -> dict:
        """
        定义工具的输入参数 JSON Schema。
        Anthropic API 要求 input_schema 字段（非 OpenAI 的 parameters）。
        此处只有一个必填参数 path（str）。
        """
        return {
            "type": "object",                         # JSON Schema 根类型必须为 "object"
            "properties": {                           # 每个参数都在 properties 下声明
                "path": {
                    "type": "string",                 # 参数类型：字符串
                    "description": "File path to read",  # 参数描述，帮助 LLM 正确传参
                }
            },
            "required": ["path"],                     # 必填参数列表：path 不可缺
        }

    # ─── 核心执行逻辑 ───
    async def execute(self, path: str, **kwargs) -> ToolResult:
        """
        执行文件读取。
        因为 BaseTool.execute 签名是 (**kwargs)，ToolRegistry 会把 LLM 传来的
        {"path": "/foo/bar.txt"} 解包成 path="/foo/bar.txt" 传进来。

        ⚠️ Phase 2 不做路径安全校验（目录穿越、越权读取），Phase 5 用 Docker 沙箱偿还。
        """
        # 1. 检查文件是否存在——不存在直接返回失败结果
        if not os.path.exists(path):
            return ToolResult(
                success=False,                        # 标记为失败
                message=f"Error: File '{path}' not found.",  # LLM 看到的错误文本
                error=f"FileNotFound: {path}",        # 结构化错误信息（供日志/Hook 使用）
            )

        # 2. 读取文件内容
        try:
            with open(path, "r", encoding="utf-8") as f:
                # 使用 with 确保文件句柄自动关闭
                content = f.read()

            # 3. 截断过长内容——避免超出 LLM 上下文窗口
            truncated = content[:10000]               # 最多返回前 10000 个字符
            was_truncated = len(content) > 10000      # 标记是否被截断

            return ToolResult(
                success=True,                         # 标记为成功
                message=truncated,                    # 文件内容作为 LLM 可读的 message
                metadata={                            # 额外元数据
                    "file_path": path,                #   - 文件路径
                    "file_size": len(content),        #   - 原始文件大小（字节数）
                    "truncated": was_truncated,       #   - 是否被截断
                    "returned_chars": len(truncated), #   - 实际返回的字符数
                },
            )
        except PermissionError:
            # 没有权限读取——这是系统级错误，不是文件不存在
            return ToolResult(
                success=False,
                message=f"Error: Permission denied for '{path}'.",
                error=f"PermissionError: {path}",
            )
        except UnicodeDecodeError:
            # 文件不是 UTF-8 文本（可能是二进制文件）
            return ToolResult(
                success=False,
                message=f"Error: Cannot read '{path}' as UTF-8 text. It may be a binary file.",
                error=f"UnicodeDecodeError: {path}",
            )
        except Exception as e:
            # 兜底：捕获所有其他异常，防止工具崩溃影响 Agent 循环
            return ToolResult(
                success=False,
                message=f"Error: Failed to read '{path}': {e}",
                error=str(e),
            )
