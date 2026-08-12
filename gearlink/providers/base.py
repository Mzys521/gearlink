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

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典，与 from_dict 保证往返一致。"""
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        """从字典反序列化。

        Args:
            data: to_dict 产出的字典，须包含 id / name / arguments 字段。

        Returns:
            ToolCall 实例。

        Raises:
            KeyError: 缺少必需字段时抛出。
        """
        return cls(id=data["id"], name=data["name"], arguments=data["arguments"])


@dataclass
class TokenUsage:
    """单次模型调用的 token 用量（可观测性，开发方向 §5.1）。

    Attributes:
        input_tokens: 输入（prompt）消耗的 token 数。
        output_tokens: 输出（completion）消耗的 token 数。
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """输入与输出 token 总数。"""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """两个用量相加，便于聚合统计。"""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典，与 from_dict 保证往返一致。"""
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenUsage":
        """从字典反序列化。

        Args:
            data: to_dict 产出的字典。

        Returns:
            TokenUsage 实例。
        """
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
        )


@dataclass
class ModelResponse:
    """模型响应的统一表示"""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None  # 本次调用的 token 用量；提供者不支持时为 None

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（含嵌套的 tool_calls），与 from_dict 保证往返一致。"""
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "usage": self.usage.to_dict() if self.usage is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelResponse":
        """从字典反序列化。

        Args:
            data: to_dict 产出的字典（兼容缺少 tool_calls / usage 字段的数据）。

        Returns:
            ModelResponse 实例。
        """
        usage_data = data.get("usage")
        return cls(
            content=data.get("content"),
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            usage=TokenUsage.from_dict(usage_data) if usage_data is not None else None,
        )


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
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """根据消息列表发起一次对话。

        Args:
            messages: OpenAI 消息格式的对话历史。
            tools: 可用工具的 schema 列表；None 表示不启用工具调用。
            response_format: 结构化输出格式（如 ``{"type": "json_object"}``）；
                None 表示不约束输出格式。不支持结构化输出的实现可忽略本参数
                （开发方向 §4.3，纯新增可选参数）。

        Returns:
            ModelResponse: 统一响应结构，含文本内容与工具调用请求。

        Raises:
            ProviderError: 底层服务调用失败（网络、鉴权、限流等）；
                瞬时故障建议以 ``retryable=True`` 标记，供调用方决策重试。
        """

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Iterator[StreamChunk]:
        """流式发起一次对话，按模型输出片段逐步产出增量事件。

        默认实现回退到非流式 `chat()`：一次性产出全文片段与终止事件；
        具体提供者可覆写以获得真正的逐 token 流式输出。实现者必须保证
        最后一个事件携带完整 `response`。

        Args:
            messages: OpenAI 消息格式的对话历史。
            tools: 可用工具的 schema 列表；None 表示不启用工具调用。
            response_format: 结构化输出格式；None 表示不约束。默认回退实现
                不透传本参数（兼容未支持该参数的旧实现）；需要结构化流式输出
                的提供者应自行覆写 `chat_stream`。

        Yields:
            StreamChunk: 增量事件；终止事件的 response 为完整响应。

        Raises:
            ProviderError: 底层服务调用失败（网络、鉴权、限流等）。
        """
        response = self.chat(messages=messages, tools=tools)
        yield StreamChunk(delta=response.content or "", response=response)
