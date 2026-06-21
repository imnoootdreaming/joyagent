import os
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量

class Config:
    """
    读取配置文件 API KEY 信息
    """
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "DeepSeek-v4-pro[1m]")
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", 15))
