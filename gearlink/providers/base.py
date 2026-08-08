from abc import ABC, abstractmethod
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
