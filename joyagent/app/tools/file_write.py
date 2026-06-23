# ── 标准库导入 ──
import os                          # 创建父目录（os.makedirs）

# ── 项目内导入 ──
from app.tools.base import BaseTool, ToolResult  # 工具基类和统一返回格式


class FileWriteTool(BaseTool):
    """
    Phase 2: 文件写入工具，继承 BaseTool 统一抽象。
    写入操作 is_dangerous = True（需要用户确认），Phase 2 先打日志，Phase 9 接入审批流。
    """

    # ─── 工具标识（类属性，覆盖抽象 property） ───
    name = "write_file"                            # LLM 调用的工具名

    description = (
        "Write content to a file at the given path. "
        "Creates parent directories automatically if they don't exist."
    )

    # ─── JSON Schema 定义 ───
    @property
    def input_schema(self) -> dict:
        """
        输入参数：path（文件路径，必填）和 content（写入内容，必填）。
        """
        return {
            "type": "object",                       # JSON Schema 根类型
            "properties": {
                "path": {
                    "type": "string",               # 参数类型
                    "description": "File path to write to",  # 帮助 LLM 理解参数含义
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],        # 两个参数都必填
        }

    # ─── 安全标记 ───
    @property
    def is_dangerous(self) -> bool:
        """
        文件写入会修改磁盘内容，标记为危险操作。
        Phase 2: ToolRegistry.execute() 会检查此标记并在控制台打黄色警告。
        Phase 9: 接入 Human-in-the-Loop，弹出确认对话框。
        """
        return True

    # ─── 核心执行逻辑 ───
    async def execute(self, path: str, content: str, **kwargs) -> ToolResult:
        """
        将 content 写入 path 指定的文件。
        自动创建父目录；写入成功后返回字节数。

        ⚠️ Phase 2 不做路径安全校验（可写任意路径），Phase 5 用 Docker 沙箱限制写入范围。
        """
        try:
            # 1. 确保父目录存在——如果 path 是 "a/b/c.txt"，则创建 a/b/
            parent_dir = os.path.dirname(path) or "."  # "" 表示当前目录，兜底为 "."
            os.makedirs(parent_dir, exist_ok=True)     # exist_ok=True 避免并发竞争报错

            # 2. 写入文件
            with open(path, "w", encoding="utf-8") as f:
                # "w" 模式会覆盖已有文件；with 保证写完自动关闭
                bytes_written = f.write(content)       # write() 返回写入的字符数

            return ToolResult(
                success=True,                          # 写入成功
                message=f"Successfully wrote {bytes_written} characters to {path}",
                metadata={                             # 额外信息供 Hook 统计
                    "file_path": path,                 #   - 写入路径
                    "bytes_written": bytes_written,    #   - 写入字节数
                    "parent_dir": parent_dir,          #   - 父目录
                },
            )

        except PermissionError:
            # 没有写权限——路径不可写
            return ToolResult(
                success=False,
                message=f"Error: Permission denied for '{path}'.",
                error=f"PermissionError: {path}",
            )
        except IsADirectoryError:
            # path 指向一个目录而非文件
            return ToolResult(
                success=False,
                message=f"Error: '{path}' is a directory, not a file.",
                error=f"IsADirectoryError: {path}",
            )
        except Exception as e:
            # 兜底：捕获所有其他异常
            return ToolResult(
                success=False,
                message=f"Error: Failed to write to '{path}': {e}",
                error=str(e),
            )
