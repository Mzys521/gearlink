"""AnthropicProvider 测试：mock 外部服务（注入伪造 anthropic 模块），不进行真实调用。"""

import sys
import types
from unittest.mock import MagicMock

import pytest

from gearlink.exceptions import ProviderError
from gearlink.providers.anthropic_provider import AnthropicProvider


def make_fake_anthropic() -> types.ModuleType:
    """构造伪造的 anthropic 模块：含异常体系与 Anthropic 客户端类。"""
    mod = types.ModuleType("anthropic")

    class APIConnectionError(Exception):
        pass

    class APIStatusError(Exception):
        def __init__(self, message: str = "", *, status_code: int = 500) -> None:
            super().__init__(message)
            self.status_code = status_code

    class RateLimitError(Exception):
        pass

    class InternalServerError(Exception):
        pass

    mod.APIConnectionError = APIConnectionError
    mod.APIStatusError = APIStatusError
    mod.RateLimitError = RateLimitError
    mod.InternalServerError = InternalServerError
    mod.Anthropic = MagicMock()
    return mod


@pytest.fixture
def fake_anthropic(monkeypatch):
    mod = make_fake_anthropic()
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return mod


@pytest.fixture
def provider(fake_anthropic, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    return AnthropicProvider()


def make_response(*blocks):
    response = MagicMock()
    response.content = list(blocks)
    return response


def make_text_block(text: str):
    return MagicMock(type="text", text=text)


def make_tool_use_block(block_id: str, name: str, input_data: dict):
    # 注意：MagicMock 的构造参数 name 是保留参数，需通过赋值设置属性
    block = MagicMock(type="tool_use", id=block_id)
    block.name = name
    block.input = input_data
    return block


def test_missing_sdk_raises_value_error(monkeypatch):
    # sys.modules 中置 None 会让 import 抛 ImportError，模拟未安装
    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    with pytest.raises(ValueError, match="anthropic"):
        AnthropicProvider()


def test_provider_requires_api_key(fake_anthropic, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider()


def test_provider_reads_env_overrides(fake_anthropic, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com")

    provider = AnthropicProvider()

    assert provider.model == "claude-haiku"
    fake_anthropic.Anthropic.assert_called_once_with(
        api_key="sk-ant-test", base_url="https://proxy.example.com"
    )


def test_chat_converts_system_user_and_tools(provider):
    provider.client.messages.create.return_value = make_response(make_text_block("好的"))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "获取时间",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    provider.chat(
        messages=[
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "几点了"},
        ],
        tools=tools,
    )

    kwargs = provider.client.messages.create.call_args.kwargs
    # system 消息被提取为独立参数
    assert kwargs["system"] == "你是助手"
    assert kwargs["messages"] == [{"role": "user", "content": "几点了"}]
    assert kwargs["max_tokens"] > 0
    # 工具 schema 被转换为 Anthropic 格式
    assert kwargs["tools"] == [
        {
            "name": "get_time",
            "description": "获取时间",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


def test_chat_converts_tool_roundtrip_messages(provider):
    provider.client.messages.create.return_value = make_response(make_text_block("12 点"))

    provider.chat(
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "tu_1", "name": "get_time", "arguments": '{"tz": "UTC"}'}],
            },
            {"role": "tool", "tool_call_id": "tu_1", "content": "12:00"},
            {"role": "tool", "tool_call_id": "tu_2", "content": "晴"},
        ]
    )

    kwargs = provider.client.messages.create.call_args.kwargs
    messages = kwargs["messages"]
    # assistant 的 tool_calls 转为 tool_use 块
    assert messages[0] == {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "tu_1", "name": "get_time", "input": {"tz": "UTC"}}],
    }
    # 连续的 tool 消息合并进同一个 user 消息（API 要求角色交替）
    assert messages[1]["role"] == "user"
    assert [b["tool_use_id"] for b in messages[1]["content"]] == ["tu_1", "tu_2"]


def test_chat_normalizes_text_and_tool_use_response(provider):
    provider.client.messages.create.return_value = make_response(
        make_text_block("我来查一下"),
        make_tool_use_block("tu_9", "get_time", {"tz": "UTC"}),
    )

    response = provider.chat(messages=[{"role": "user", "content": "几点了"}])

    assert response.content == "我来查一下"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "tu_9"
    assert response.tool_calls[0].name == "get_time"
    assert response.tool_calls[0].arguments == '{"tz": "UTC"}'


def test_chat_wraps_error_with_retryable_flag(provider, fake_anthropic):
    provider.client.messages.create.side_effect = fake_anthropic.RateLimitError("限流")

    with pytest.raises(ProviderError) as exc_info:
        provider.chat(messages=[{"role": "user", "content": "你好"}])

    assert exc_info.value.retryable is True


def test_chat_wraps_unknown_error_as_not_retryable(provider):
    provider.client.messages.create.side_effect = RuntimeError("boom")

    with pytest.raises(ProviderError) as exc_info:
        provider.chat(messages=[{"role": "user", "content": "你好"}])

    assert exc_info.value.retryable is False


def test_chat_stream_falls_back_to_chat(provider):
    provider.client.messages.create.return_value = make_response(make_text_block("完整回复"))

    chunks = list(provider.chat_stream(messages=[{"role": "user", "content": "你好"}]))

    assert chunks[-1].response.content == "完整回复"
