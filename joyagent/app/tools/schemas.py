# Anthropic 原生工具格式：{"name": ..., "description": ..., "input_schema": {...}}
# 无需 OpenAI Function Calling 的 "type": "function" 包装层
from app.tools.file_read import read_file
from app.tools.file_write import write_file

READ_FILE_TOOL = {
    "name": "read_file",
    "description": "Read the contents of a file at the given path",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read"}
        },
        "required": ["path"]
    }
}

WRITE_FILE_TOOL = {
    "name": "write_file",
    "description": "Write content to a file. Creates parent directories if needed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string", "description": "File path to write to"
                },
            "content": {
                "type": "string", "description": "Content to write"
                }
        },
        "required": ["path", "content"]
    }
}

TOOLS = [READ_FILE_TOOL, WRITE_FILE_TOOL]

# 工具执行映射：tool_name -> handler function
TOOL_HANDLERS = {
    "read_file": read_file,
    "write_file": write_file,
}
