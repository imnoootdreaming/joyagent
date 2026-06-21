from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from joyagent.app.core.config import Config

def get_llm(model_name: str = None,
            tools: list[dict] = None):
    """LLM 工厂函数，根据模型名称以及 tools 列表返回对应的 ChatModel"""
    model_name = model_name or Config.DEFAULT_MODEL
    # TODO - llm 模型选择 if-else 逻辑优化
    if "claude" in model_name.lower():
        llm = ChatAnthropic(model=model_name, 
                            temperature=0.3,
                            anthropic_api_key=Config.ANTHROPIC_API_KEY,
                            max_tokens=4096
                            )
    elif "deepseek" in model_name.lower():
        llm = ChatOpenAI(model_name=model_name,
                         temperature=0.3,
                         openai_api_key=Config.ANTHROPIC_API_KEY,
                         base_url="https://api.deepseek.com",
                         )
    else:
        llm = ChatOpenAI(
            model=model_name,
            api_key=Config.ANTHROPIC_API_KEY,
            temperature=0.3,
        )
    
    if tools:
        llm.bind_tools(tools)
    return llm