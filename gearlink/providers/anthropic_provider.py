"""Anthropic 模型提供者：适配 Anthropic Messages API（开发方向 §4.2）。

Anthropic 的消息与工具调用格式与 OpenAI 差异较大，本模块负责双向归一化：

- 请求侧：OpenAI 消息格式 → Messages API（system 消息提取为独立参数，
  tool 消息转为 ``tool_result`` 块，assistant 的 ``tool_calls`` 转为
  ``tool_use`` 块）；
- 响应侧：``text`` / ``tool_use`` 内容块 → 统一的 ``ModelResponse``。

依赖 ``anthropic`` SDK（可选依赖）：``pip install gearlink[anthropic]``。
"""

import json
import logging
import os
from typing import Any

from gearlink.exceptions import ProviderError
from gearlink.providers.base import ModelProvider, ModelResponse, ToolCall

#: 默认模型
DEFAULT_MODEL = "claude-sonnet-4-20250514"

#: Messages API 必须显式给出单次生成的最大 token 数
DEFAULT_MAX_TOKENS = 4096

logger = logging.getLogger(__name__)


def _convert_messages(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """把 OpenAI 消息格式转换为 Anthropic Messages API 格式。

    Args:
        messages: OpenAI 消息格式的对话历史。

    Returns:
        (system 文本, 转换后的消息列表) 二元组；无 system 消息时前者为 None。
        连续的 tool 消息会合并进同一个 user 消息（API 要求角色交替）。
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            if message.get("content"):
                system_parts.append(message["content"])
        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id", ""),
                "content": message.get("content") or "",
            }
            # 连续的工具结果合并进同一个 user 消息，保持角色交替
            if (
                converted
                and converted[-1]["role"] == "user"
                and isinstance(converted[-1]["content"], list)
            ):
                converted[-1]["content"].append(block)
            else:
                converted.append({"role": "user", "content": [block]})
        elif role == "assistant":
            content: list[dict[str, Any]] = []
            if message.get("content"):
                content.append({"type": "text", "text": message["content"]})
            for tool_call in message.get("tool_calls") or []:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "input": json.loads(tool_call.get("arguments") or "{}"),
                    }
                )
            converted.append({"role": "assistant", "content": content or ""})
        else:
            converted.append({"role": role, "content": message.get("content") or ""})
    system = "\n\n".join(system_parts) or None
    return system, converted


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 OpenAI 工具 schema 转换为 Anthropic 工具格式。"""
    return [
        {
            "name": tool["function"]["name"],
            "description": tool["function"].get("description", ""),
            "input_schema": tool["function"].get("parameters", {"type": "object"}),
        }
        for tool in tools
    ]


def _normalize_response(response: Any) -> ModelResponse:
    """把 Anthropic 响应内容块归一化为统一的 ModelResponse。"""
    text_parts = [block.text for block in response.content if block.type == "text"]
    tool_calls = [
        ToolCall(
            id=block.id,
            name=block.name,
            arguments=json.dumps(block.input, ensure_ascii=False),
        )
        for block in response.content
        if block.type == "tool_use"
    ]
    return ModelResponse(content="".join(text_parts) or None, tool_calls=tool_calls)


class AnthropicProvider(ModelProvider):
    """Anthropic Claude 模型提供者。"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        """初始化 Anthropic 提供者。

        各参数未传入时回退到环境变量，再到内置默认值：
        api_key <- ANTHROPIC_API_KEY；model <- ANTHROPIC_MODEL；
        base_url <- ANTHROPIC_BASE_URL。

        Args:
            model: 模型名称，默认 ``claude-sonnet-4-20250514``。
            api_key: API 密钥；未传入时从环境变量 ANTHROPIC_API_KEY 读取。
            base_url: 服务地址；None 表示使用 SDK 默认地址。
            max_tokens: 单次生成的最大 token 数（Messages API 必填项）。

        Raises:
            ValueError: 未安装 anthropic SDK，或缺少 API key 时抛出。
        """
        try:
            import anthropic
        except ImportError as e:
            raise ValueError(
                "AnthropicProvider 需要 anthropic 依赖：pip install gearlink[anthropic]"
            ) from e
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("缺少 API key：请传入 api_key 参数，或设置环境变量 ANTHROPIC_API_KEY")
        self._anthropic = anthropic
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
        self.max_tokens = max_tokens
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        elif os.environ.get("ANTHROPIC_BASE_URL"):
            client_kwargs["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
        self.client = anthropic.Anthropic(**client_kwargs)
        logger.debug(
            "AnthropicProvider 初始化: model=%s, max_tokens=%d", self.model, self.max_tokens
        )

    def _is_retryable(self, error: Exception) -> bool:
        """判断 anthropic SDK 异常是否为可重试的瞬时故障（网络/限流/服务端错误）。"""
        anthropic = self._anthropic
        if isinstance(
            error,
            (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError),
        ):
            return True
        if isinstance(error, anthropic.APIStatusError) and error.status_code >= 500:
            return True
        return False

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """根据消息列表发起一次对话（自动完成 OpenAI → Anthropic 格式归一化）。

        Args:
            messages: OpenAI 消息格式的对话历史。
            tools: 可用工具的 schema 列表（OpenAI 格式）；None 表示不启用工具调用。
            response_format: 结构化输出格式；Anthropic Messages API 无对应的
                原生参数，本提供者忽略该参数（开发规范 §5.2 允许的可选忽略）。

        Returns:
            ModelResponse: 统一响应结构，含文本内容与工具调用请求。

        Raises:
            ProviderError: 底层服务调用失败（网络、鉴权、限流等），
                瞬时故障（网络/限流/服务端错误）标记 retryable=True。
        """
        system, converted_messages = _convert_messages(messages)
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": converted_messages,
        }
        if system is not None:
            create_kwargs["system"] = system
        if tools:
            create_kwargs["tools"] = _convert_tools(tools)
        logger.debug(
            "模型调用: model=%s, messages=%d, tools=%d",
            self.model,
            len(converted_messages),
            len(tools or []),
        )
        try:
            response = self.client.messages.create(**create_kwargs)
        except Exception as e:
            retryable = self._is_retryable(e)
            logger.warning("模型 %s 调用失败（retryable=%s）: %s", self.model, retryable, e)
            # 包装第三方异常，保留原始异常链；标记可重试性供调用方决策
            raise ProviderError(f"模型 {self.model} 调用失败: {e}", retryable=retryable) from e
        return _normalize_response(response)
