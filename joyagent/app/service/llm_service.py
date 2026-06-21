from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from joyagent.app.core.config import Config


def get_llm(model_name: str = None, tools: list[dict] = None):
    """LLM 工厂函数

    设计思路：
    - DeepSeek 提供了 Anthropic Messages API 兼容端点，和 Claude 共用
      ChatAnthropic，仅通过 base_url 区分目标地址
    """
    model_name = model_name or Config.DEFAULT_MODEL
    name_lower = model_name.lower()

    # Anthropic 协议系列：Claude（原生） + DeepSeek（Anthropic 兼容端点）
    if "claude" in name_lower or "deepseek" in name_lower:
        kwargs = dict(
            model=model_name,
            temperature=0.3,
            anthropic_api_key=Config.ANTHROPIC_API_KEY,
            max_tokens=4096,
        )
        if "deepseek" in name_lower:
            kwargs["base_url"] = "https://api.deepseek.com/anthropic"
        llm = ChatAnthropic(**kwargs)
    else:
        llm = ChatOpenAI(
            model=model_name,
            api_key=Config.ANTHROPIC_API_KEY,
            temperature=0.3,
        )

    if tools:
        llm = llm.bind_tools(tools)
    return llm