import os

def read_file(path: str) -> str:
    """读取文件内容"""
    if not os.path.exists(path):
        return f"Error: File '{path}' not found"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content