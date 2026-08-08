import os
from typing import Any

from openai import OpenAI

from gearlink.exceptions import ProviderError
from gearlink.providers.base import ModelProvider, ModelResponse, ToolCall

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


class OpenAIProvider(ModelProvider):
    """OpenAI 兼容接口的模型提供者（适用于 OpenAI / DeepSeek 等）"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """初始化 OpenAI 兼容提供者。

        各参数未传入时依次回退到环境变量，再到内置默认值：
        api_key <- DEEPSEEK_API_KEY；model <- DEEPSEEK_MODEL；base_url <- DEEPSEEK_BASE_URL。

        Args:
            model: 模型名称，默认使用 DeepSeek flash 模型。
            api_key: API 密钥；未传入时从环境变量 DEEPSEEK_API_KEY 读取。
            base_url: 服务地址，默认指向 DeepSeek 官方地址。

        Raises:
            ValueError: 未传入 api_key 且环境变量 DEEPSEEK_API_KEY 缺失时抛出。
        """
        if api_key is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("缺少 API key：请传入 api_key 参数，或设置环境变量 DEEPSEEK_API_KEY")
        self.model = model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        """根据消息列表发起一次对话。

        Args:
            messages: OpenAI 消息格式的对话历史。
            tools: 可用工具的 schema 列表；None 表示不启用工具调用。

        Returns:
            ModelResponse: 统一响应结构，含文本内容与工具调用请求。

        Raises:
            ProviderError: 底层服务调用失败（网络、鉴权、限流等）。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                stream=False,
            )
        except Exception as e:
            # 包装第三方异常，保留原始异常链
            raise ProviderError(f"模型 {self.model} 调用失败: {e}") from e
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
