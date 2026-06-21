import os

def read_file(path: str) -> str:
    """读取文件内容。注意：本 Phase 不做路径安全校验（Phase 5 补充）。"""
    if not os.path.exists(path):
        return f"Error: File '{path}' not found."
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content[:10000]  # 截断过长文件