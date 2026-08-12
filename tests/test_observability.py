"""可观测性测试：token 用量、事件落盘与回放（开发方向 §5.1，外部服务全部 mock）。"""

import json
from unittest.mock import MagicMock

import pytest

from gearlink import (
    FinalAnswerEvent,
    JsonlEventSink,
    ModelMessageEvent,
    ReactAgent,
    StepStartEvent,
    TokenUsage,
    UsageTracker,
    jsonl_hook,
    load_jsonl_events,
)
from gearlink.providers.base import ModelProvider, ModelResponse, ToolCall
from gearlink.providers.openai_provider import OpenAIProvider
from gearlink.utils.usage import UsageRecord


class FakeProvider(ModelProvider):
    """返回带 usage 的预设响应。"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.calls = 0

    def chat(self, messages, tools=None) -> ModelResponse:
        response = self.responses[self.calls]
        self.calls += 1
        return response


# ==================== TokenUsage / ModelResponse.usage ====================


def test_token_usage_add_and_total():
    usage = TokenUsage(input_tokens=10, output_tokens=5) + TokenUsage(3, 2)

    assert usage.input_tokens == 13
    assert usage.output_tokens == 7
    assert usage.total_tokens == 20


def test_token_usage_dict_roundtrip():
    usage = TokenUsage(input_tokens=10, output_tokens=5)

    assert TokenUsage.from_dict(usage.to_dict()) == usage


def test_model_response_usage_roundtrip():
    response = ModelResponse(
        content="你好",
        tool_calls=[ToolCall(id="t1", name="f", arguments="{}")],
        usage=TokenUsage(input_tokens=8, output_tokens=4),
    )

    assert ModelResponse.from_dict(response.to_dict()) == response


def test_model_response_without_usage_stays_none():
    # 验收：默认值等价现状——不带 usage 的响应序列化后仍为 None
    response = ModelResponse(content="你好")

    assert response.to_dict()["usage"] is None
    assert ModelResponse.from_dict(response.to_dict()).usage is None


def test_openai_provider_populates_usage():
    fake_message = MagicMock(content="回复", tool_calls=[])
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=fake_message)]
    fake_response.usage = MagicMock(prompt_tokens=12, completion_tokens=7)

    provider = OpenAIProvider(api_key="sk-test")
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = fake_response

    response = provider.chat(messages=[{"role": "user", "content": "你好"}])

    assert response.usage == TokenUsage(input_tokens=12, output_tokens=7)


def test_openai_provider_usage_absent_is_none():
    fake_message = MagicMock(content="回复", tool_calls=[])
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=fake_message)]
    fake_response.usage = None

    provider = OpenAIProvider(api_key="sk-test")
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = fake_response

    response = provider.chat(messages=[{"role": "user", "content": "你好"}])

    assert response.usage is None


# ==================== 事件落盘与回放 ====================


def test_event_to_dict_is_json_serializable():
    event = ModelMessageEvent(
        seq=3,
        iteration=1,
        content="回复",
        tool_calls=[ToolCall(id="t1", name="f", arguments="{}")],
        usage=TokenUsage(input_tokens=5, output_tokens=2),
    )

    data = json.loads(json.dumps(event.to_dict(), ensure_ascii=False))

    assert data["type"] == "model_message"
    assert data["seq"] == 3
    assert data["timestamp"] > 0
    assert data["usage"] == {"input_tokens": 5, "output_tokens": 2}


def test_jsonl_sink_write_and_replay(tmp_path):
    path = tmp_path / "events.jsonl"
    with JsonlEventSink(path) as sink:
        sink.write(StepStartEvent(seq=1, iteration=0))
        sink.write(FinalAnswerEvent(seq=2, iteration=0, content="答案"))

    events = load_jsonl_events(path)

    assert [e["type"] for e in events] == ["step_start", "final_answer"]
    assert [e["seq"] for e in events] == [1, 2]
    assert events[1]["content"] == "答案"


def test_jsonl_hook_with_agent_records_full_run(tmp_path):
    provider = FakeProvider([ModelResponse(content="你好！", usage=TokenUsage(6, 3))])
    path = tmp_path / "run.jsonl"
    with JsonlEventSink(path) as sink:
        agent = ReactAgent(provider=provider, hooks=[jsonl_hook(sink)])
        assert agent.run("你好") == "你好！"

    events = load_jsonl_events(path)
    types = [e["type"] for e in events]

    assert "step_start" in types and "final_answer" in types
    # usage 经 ModelMessageEvent 落盘（可观测性验收：ModelResponse.usage 生效）
    model_event = next(e for e in events if e["type"] == "model_message")
    assert model_event["usage"] == {"input_tokens": 6, "output_tokens": 3}


# ==================== UsageTracker ====================


def test_usage_tracker_aggregates_by_label():
    tracker = UsageTracker()
    tracker.add(TokenUsage(10, 5), label="model-a")
    tracker.add(TokenUsage(2, 1), label="model-a")
    tracker.add(TokenUsage(7, 3), label="model-b")
    tracker.add(None, label="model-a")  # None 忽略

    assert tracker.records["model-a"] == UsageRecord(input_tokens=12, output_tokens=6, calls=2)
    assert tracker.total() == TokenUsage(input_tokens=19, output_tokens=9)


def test_usage_tracker_estimates_cost():
    tracker = UsageTracker()
    tracker.add(TokenUsage(2000, 1000), label="model-a")

    # 单价按每千 token 计：2000/1000*0.1 + 1000/1000*0.2 = 0.4
    cost = tracker.estimate_cost({"model-a": (0.1, 0.2)})

    assert cost == pytest.approx(0.4)


def test_usage_tracker_dict_roundtrip():
    tracker = UsageTracker()
    tracker.add(TokenUsage(10, 5), label="model-a")

    assert UsageTracker.from_dict(tracker.to_dict()) == tracker
