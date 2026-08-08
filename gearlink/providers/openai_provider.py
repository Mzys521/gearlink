import os
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

# 支持两种运行方式：作为包导入（from providers.openai_provider import ...）
# 或直接在 providers 目录下以脚本方式运行
try:
    from base import ModelProvider, ModelResponse, ToolCall
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from base import ModelProvider, ModelResponse, ToolCall

DEFAULT_API_KEY = "sk-8216784bec9747ea9e77ce1069f1718f"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


class OpenAIProvider(ModelProvider):
    """OpenAI 兼容接口的模型提供者（适用于 OpenAI / DeepSeek 等）"""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None, base_url: Optional[str] = None,) -> None:
        self.model = model
        self.client = OpenAI(
            api_key=api_key or DEFAULT_API_KEY,
            base_url=base_url or DEFAULT_BASE_URL,
        )

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, ) -> ModelResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=False,
        )
        message = response.choices[0].message
        return ModelResponse(
            content=message.content,
            tool_calls=[
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in (message.tool_calls or [])
            ],
        )
