import os
from anthropic import Anthropic
from app.core.config import Config


def get_client(model_name: str = None) -> Anthropic:
    """创建 Anthropic 客户端 —— 按协议族分流

    设计思路：
    - Claude（原生 Anthropic API）和 DeepSeek（Anthropic 兼容端点）都使用
      同一个 `anthropic` 包，仅通过 base_url 区分目标地址
    - 一行注册：Anthropic(base_url=...) 即可，无需工厂模式
    - API Key 通过 ANTHROPIC_API_KEY 环境变量；DeepSeek 兼容端点
      通过 ANTHROPIC_BASE_URL 切换目标地址
    """
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    return Anthropic(base_url=base_url)


# 全局客户端实例（按需创建，支持不同的 base_url）
_client_cache: dict[str, Anthropic] = {}


def get_or_create_client(model_name: str = None) -> Anthropic:
    """获取或创建客户端（带缓存，按协议族分流）

    DeepSeek — 使用其 Anthropic 兼容端点
    Claude — 使用原生 API（或不设置 base_url 时的默认端点）
    OpenAI — 保留分支位置，后续按需补充（OpenAI SDK）
    """
    model_name = model_name or Config.DEFAULT_MODEL
    name_lower = model_name.lower()

    if "deepseek" in name_lower:
        cache_key = "deepseek"
        if cache_key not in _client_cache:
            _client_cache[cache_key] = Anthropic(
                base_url="https://api.deepseek.com/anthropic"
            )
        return _client_cache[cache_key]

    # Claude 原生 API（或其他不设 base_url 的端点）
    cache_key = "default"
    if cache_key not in _client_cache:
        _client_cache[cache_key] = get_client(model_name)
    return _client_cache[cache_key]
