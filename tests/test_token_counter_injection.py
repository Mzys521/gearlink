"""TokenCounter 注入、精确截断与 Agent / Memory 集成契约。"""

import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from gearlink import (
    HeuristicTokenCounter,
    TiktokenTokenCounter,
    TokenCounter,
    truncate_text,
)
from gearlink.core.agent import MAX_TOOL_RESULT_TOKENS, PlanExecuteAgent, ReactAgent
from gearlink.core.events import ToolCallEndEvent
from gearlink.core.memory import MemoryManager, ShortTermMemory
from gearlink.core.tool import ToolRegistry
from gearlink.providers.base import ModelProvider, ModelResponse, ToolCall


_TRUNCATION_SUFFIX = "\n...(工具结果过长，已截断)"


class CharacterTokenCounter:
    """测试用计数器：每个 Python 字符计一个 token。"""

    def count_text(self, text: str) -> int:
        return len(text)

    def count_message(self, message: dict[str, Any]) -> int:
        content = message.get("content")
        if content is None:
            content_text = ""
        elif isinstance(content, str):
            content_text = content
        else:
            content_text = json.dumps(content, ensure_ascii=False)
        tokens = self.count_text(content_text)
        if message.get("tool_calls"):
            tokens += self.count_text(json.dumps(message["tool_calls"], ensure_ascii=False))
        return tokens

    def truncate_text(self, text: str, max_tokens: int) -> str:
        return text[:max_tokens]


class SequenceProvider(ModelProvider):
    """按顺序返回预设响应，避免测试访问真实模型。"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.calls = 0

    def chat(self, messages, tools=None, response_format=None) -> ModelResponse:
        response = self.responses[self.calls]
        self.calls += 1
        return response


class ByteEncoding:
    """测试用 encoding：把每个 UTF-8 字节视为一个 token。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.disallowed_special_values: list[tuple[Any, ...]] = []

    def encode(self, text: str, *, disallowed_special=()) -> list[int]:
        self.disallowed_special_values.append(tuple(disallowed_special))
        return list(text.encode("utf-8"))

    def decode_single_token_bytes(self, token: int) -> bytes:
        return bytes([token])


def test_token_counter_protocol_accepts_structural_implementation():
    counter = CharacterTokenCounter()

    assert isinstance(counter, TokenCounter)
    assert not isinstance(object(), TokenCounter)


@pytest.mark.parametrize(
    ("text", "budget", "suffix", "expected"),
    [
        ("汉" * 10, 6, "[截断]", "汉" * 3 + "[截断]"),
        ("abcdefghijklmnop", 3, "..", "abcdefgh.."),
        ("🙂" * 20, 4, "[截]", "🙂" * 8 + "[截]"),
    ],
)
def test_heuristic_truncation_enforces_exact_budget_including_suffix(
    text: str, budget: int, suffix: str, expected: str
):
    counter = HeuristicTokenCounter()

    result = truncate_text(text, budget, token_counter=counter, suffix=suffix)

    assert result == expected
    assert result.endswith(suffix)
    assert counter.count_text(result) == budget


def test_truncate_text_keeps_text_and_omits_suffix_when_within_budget():
    counter = HeuristicTokenCounter()

    result = truncate_text("你好", 2, token_counter=counter, suffix="...[截断]")

    assert result == "你好"


def test_tiktoken_counter_selects_encoding_name_and_model(monkeypatch):
    calls: list[tuple[str, str]] = []
    encodings: dict[str, ByteEncoding] = {}

    def get_encoding(name: str) -> ByteEncoding:
        calls.append(("encoding_name", name))
        return encodings.setdefault(f"encoding:{name}", ByteEncoding(name))

    def encoding_for_model(model: str) -> ByteEncoding:
        calls.append(("model", model))
        return encodings.setdefault(f"model:{model}", ByteEncoding(model))

    fake_tiktoken = SimpleNamespace(
        get_encoding=get_encoding,
        encoding_for_model=encoding_for_model,
    )
    monkeypatch.setitem(sys.modules, "tiktoken", fake_tiktoken)

    by_name = TiktokenTokenCounter(encoding_name="unit-test-encoding")
    by_model = TiktokenTokenCounter(model="unit-test-model")

    assert calls == [
        ("encoding_name", "unit-test-encoding"),
        ("model", "unit-test-model"),
    ]
    assert by_name.count_text("你好") == len("你好".encode())
    assert by_model.count_text("abc") == 3
    assert by_name.encoding.disallowed_special_values[-1] == ()
    assert by_model.encoding.disallowed_special_values[-1] == ()


def test_tiktoken_counter_rejects_ambiguous_configuration():
    with pytest.raises(ValueError, match="不能同时指定"):
        TiktokenTokenCounter(model="unit-model", encoding_name="unit-encoding")


def test_tiktoken_counter_truncates_only_to_valid_utf8(monkeypatch):
    encoding = ByteEncoding("utf8-bytes")
    fake_tiktoken = SimpleNamespace(
        get_encoding=lambda name: encoding,
        encoding_for_model=lambda model: encoding,
    )
    monkeypatch.setitem(sys.modules, "tiktoken", fake_tiktoken)
    counter = TiktokenTokenCounter(encoding_name="utf8-bytes")

    # "A你B" 的 UTF-8 字节是 1 + 3 + 1。2 token 会落在“你”的中间，
    # 必须丢弃不完整尾字节；4 token 则刚好包含完整的“你”。
    cut_inside_character = counter.truncate_text("A你B", 2)
    cut_after_character = counter.truncate_text("A你B", 4)

    assert cut_inside_character == "A"
    assert cut_after_character == "A你"
    assert "\ufffd" not in cut_inside_character
    cut_inside_character.encode("utf-8")
    cut_after_character.encode("utf-8")


def test_short_term_memory_uses_injected_token_counter():
    counter = CharacterTokenCounter()
    memory = ShortTermMemory(max_tokens=5, max_message=None, token_counter=counter)

    memory.add_message({"role": "user", "content": "abcd"})
    memory.add_message({"role": "assistant", "content": "wxyz"})

    assert memory.token_counter is counter
    assert memory.get_messages() == [{"role": "assistant", "content": "wxyz"}]


def test_memory_manager_uses_injected_token_counter_for_context_budget():
    counter = CharacterTokenCounter()
    short_term = ShortTermMemory(max_message=None)
    manager = MemoryManager(
        short_term=short_term,
        max_context_tokens=5,
        token_counter=counter,
    )
    manager.add_message({"role": "user", "content": "abcd"})
    manager.add_message({"role": "assistant", "content": "wxyz"})

    messages = manager.build_context("unused")

    assert manager.token_counter is counter
    assert messages == [{"role": "assistant", "content": "wxyz"}]
    assert len(short_term.get_messages()) == 2


def test_react_agent_tool_result_respects_hard_token_limit():
    counter = CharacterTokenCounter()
    registry = ToolRegistry()
    registry.register_tool(
        "oversized_result",
        lambda: "中" * (MAX_TOOL_RESULT_TOKENS + 100),
        {
            "description": "返回超过 token 预算的结果",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    )
    provider = SequenceProvider(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="call_1", name="oversized_result", arguments="{}")]
            ),
            ModelResponse(content="完成"),
        ]
    )
    agent = ReactAgent(
        provider=provider,
        tool_registry=registry,
        token_counter=counter,
    )

    events = list(agent.run_events("运行工具"))
    tool_end = next(event for event in events if isinstance(event, ToolCallEndEvent))
    tool_message = next(
        message for message in agent.memory.get_messages() if message.get("role") == "tool"
    )

    assert tool_end.truncated is True
    assert tool_end.result == tool_message["content"]
    assert tool_end.result.endswith(_TRUNCATION_SUFFIX)
    assert counter.count_text(tool_end.result) == MAX_TOOL_RESULT_TOKENS


def test_plan_execute_agent_forwards_token_counter_to_executor():
    counter = CharacterTokenCounter()
    agent = PlanExecuteAgent(provider=SequenceProvider([]), token_counter=counter)

    assert agent.executor.token_counter is counter
    assert agent.executor.memory.token_counter is counter
