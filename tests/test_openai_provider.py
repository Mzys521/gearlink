"""OpenAIProvider 测试：mock 外部服务，不进行真实调用。"""

from unittest.mock import MagicMock, patch

import openai
import pytest

from gearlink.exceptions import ProviderError
from gearlink.providers.base import ModelProvider, ModelResponse
from gearlink.providers.openai_provider import OpenAIProvider


def make_stream_chunk(content=None, tool_calls=None):
    """构造一个 OpenAI 流式响应 chunk 的 MagicMock"""
    delta = MagicMock(content=content, tool_calls=tool_calls)
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=delta)]
    return chunk


def test_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        OpenAIProvider(api_key=None)


def test_provider_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    with patch("gearlink.providers.openai_provider.OpenAI"):
        provider = OpenAIProvider()
    assert provider.client is not None


def test_provider_reads_model_and_base_url_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.com/v1")
    with patch("gearlink.providers.openai_provider.OpenAI") as fake_openai:
        provider = OpenAIProvider()
    assert provider.model == "deepseek-chat"
    fake_openai.assert_called_once_with(api_key="sk-test", base_url="https://example.com/v1")


def test_chat_returns_model_response():
    # 注意：MagicMock 的构造参数 name 是保留参数，需通过赋值设置属性
    fake_function = MagicMock(arguments="{}")
    fake_function.name = "get_current_time"
    fake_tool_call = MagicMock(id="call_1", function=fake_function)
    fake_message = MagicMock(content="现在 12 点", tool_calls=[fake_tool_call])
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=fake_message)]

    provider = OpenAIProvider(api_key="sk-test")
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = fake_response

    response = provider.chat(messages=[{"role": "user", "content": "几点了"}])

    assert response.content == "现在 12 点"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "get_current_time"


def test_chat_wraps_service_error():
    provider = OpenAIProvider(api_key="sk-test")
    provider.client = MagicMock()
    provider.client.chat.completions.create.side_effect = RuntimeError("network down")

    with pytest.raises(ProviderError) as exc_info:
        provider.chat(messages=[{"role": "user", "content": "你好"}])
    assert exc_info.value.__cause__ is not None


def test_chat_stream_yields_content_deltas_and_final_response():
    provider = OpenAIProvider(api_key="sk-test")
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = iter(
        [
            make_stream_chunk(content="你好"),
            make_stream_chunk(content="，世界"),
            MagicMock(choices=[]),
        ]
    )

    events = list(provider.chat_stream(messages=[{"role": "user", "content": "你好"}]))

    # 文本增量逐片段产出，终止事件携带完整响应
    assert [e.delta for e in events if e.delta] == ["你好", "，世界"]
    final = events[-1]
    assert final.response is not None
    assert final.response.content == "你好，世界"
    assert final.response.tool_calls == []
    provider.client.chat.completions.create.assert_called_once()
    assert provider.client.chat.completions.create.call_args.kwargs["stream"] is True


def test_chat_stream_accumulates_tool_call_deltas():
    # 工具调用按 index 分片返回：id 与 name 在前，arguments 分两片
    fake_function_head = MagicMock(arguments="")
    fake_function_head.name = "get_current"
    fake_function_tail = MagicMock(arguments="{}")
    fake_function_tail.name = None
    head = make_stream_chunk(
        tool_calls=[MagicMock(index=0, id="call_1", function=fake_function_head)]
    )
    tail = make_stream_chunk(tool_calls=[MagicMock(index=0, id=None, function=fake_function_tail)])
    # 注意：MagicMock 的 name 是保留参数，function.name 需通过赋值设置
    fake_function_head.name = "get_current_time"

    provider = OpenAIProvider(api_key="sk-test")
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = iter([head, tail])

    events = list(provider.chat_stream(messages=[{"role": "user", "content": "几点了"}]))

    final = events[-1]
    assert final.response is not None
    assert final.response.content is None
    assert len(final.response.tool_calls) == 1
    tool_call = final.response.tool_calls[0]
    assert (tool_call.id, tool_call.name, tool_call.arguments) == (
        "call_1",
        "get_current_time",
        "{}",
    )


def test_chat_stream_wraps_service_error():
    provider = OpenAIProvider(api_key="sk-test")
    provider.client = MagicMock()
    provider.client.chat.completions.create.side_effect = RuntimeError("network down")

    with pytest.raises(ProviderError) as exc_info:
        list(provider.chat_stream(messages=[{"role": "user", "content": "你好"}]))
    assert exc_info.value.__cause__ is not None


def test_default_chat_stream_falls_back_to_chat():
    # 未覆写 chat_stream 的提供者（只实现了 chat）也能通过基类默认实现获得流式接口
    class ChatOnlyProvider(ModelProvider):
        def chat(self, messages, tools=None):
            return ModelResponse(content="完整回答")

    events = list(ChatOnlyProvider().chat_stream(messages=[{"role": "user", "content": "你好"}]))

    assert events[0].delta == "完整回答"
    assert events[-1].response == ModelResponse(content="完整回答")


# ------------------- 可重试标记（retryable，开发方向 §4.3） -------------------


def test_chat_marks_network_error_retryable():
    provider = OpenAIProvider(api_key="sk-test")
    provider.client = MagicMock()
    provider.client.chat.completions.create.side_effect = openai.APIConnectionError(
        request=MagicMock()
    )

    with pytest.raises(ProviderError) as exc_info:
        provider.chat(messages=[{"role": "user", "content": "你好"}])
    assert exc_info.value.retryable is True


def test_chat_marks_rate_limit_retryable():
    provider = OpenAIProvider(api_key="sk-test")
    provider.client = MagicMock()
    provider.client.chat.completions.create.side_effect = openai.RateLimitError(
        "限流", response=MagicMock(status_code=429), body=None
    )

    with pytest.raises(ProviderError) as exc_info:
        provider.chat(messages=[{"role": "user", "content": "你好"}])
    assert exc_info.value.retryable is True


def test_chat_marks_auth_error_not_retryable():
    provider = OpenAIProvider(api_key="sk-test")
    provider.client = MagicMock()
    provider.client.chat.completions.create.side_effect = openai.AuthenticationError(
        "鉴权失败", response=MagicMock(status_code=401), body=None
    )

    with pytest.raises(ProviderError) as exc_info:
        provider.chat(messages=[{"role": "user", "content": "你好"}])
    assert exc_info.value.retryable is False


def test_chat_passes_response_format_when_set():
    fake_message = MagicMock(content='{"ok": true}', tool_calls=None)
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=fake_message)]

    provider = OpenAIProvider(api_key="sk-test")
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = fake_response

    provider.chat(
        messages=[{"role": "user", "content": "输出 JSON"}], response_format={"type": "json_object"}
    )

    kwargs = provider.client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


def test_chat_omits_response_format_by_default():
    fake_message = MagicMock(content="普通回答", tool_calls=None)
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=fake_message)]

    provider = OpenAIProvider(api_key="sk-test")
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = fake_response

    provider.chat(messages=[{"role": "user", "content": "你好"}])

    kwargs = provider.client.chat.completions.create.call_args.kwargs
    assert "response_format" not in kwargs  # 默认不传，行为等价现状
