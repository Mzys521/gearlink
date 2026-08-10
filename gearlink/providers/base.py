from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """模型返回的一次工具调用"""

    id: str
    name: str
    arguments: str  # JSON 字符串


@dataclass
class ModelResponse:
    """模型响应的统一表示"""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class StreamChunk:
    """流式响应中的单个事件。

    Attributes:
        delta: 本次事件的增量文本；工具调用累积阶段为空字符串。
        response: 本次对话的完整响应（累积后的文本与工具调用）；
            仅终止事件非 None，提供者可据此判断流已结束。
    """

    delta: str = ""
    response: ModelResponse | None = None


class ModelProvider(ABC):
    """模型提供者抽象接口：屏蔽具体模型服务的调用细节"""

    @abstractmethod
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

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamChunk]:
        """流式发起一次对话，按模型输出片段逐步产出增量事件。

        默认实现回退到非流式 `chat()`：一次性产出全文片段与终止事件；
        具体提供者可覆写以获得真正的逐 token 流式输出。实现者必须保证
        最后一个事件携带完整 `response`。

        Args:
            messages: OpenAI 消息格式的对话历史。
            tools: 可用工具的 schema 列表；None 表示不启用工具调用。

        Yields:
            StreamChunk: 增量事件；终止事件的 response 为完整响应。

        Raises:
            ProviderError: 底层服务调用失败（网络、鉴权、限流等）。
        """
        response = self.chat(messages=messages, tools=tools)
        yield StreamChunk(delta=response.content or "", response=response)
