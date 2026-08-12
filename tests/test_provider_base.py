"""providers/base.py 数据结构测试：to_dict / from_dict 往返一致性。"""

import pytest

from gearlink.providers.base import ModelResponse, ToolCall


def test_tool_call_serialization_round_trip():
    call = ToolCall(id="call_1", name="get_current_time", arguments='{"tz": "UTC"}')
    assert ToolCall.from_dict(call.to_dict()) == call


def test_tool_call_from_dict_missing_field_raises():
    with pytest.raises(KeyError):
        ToolCall.from_dict({"id": "call_1", "name": "get_current_time"})


def test_model_response_round_trip_without_tool_calls():
    response = ModelResponse(content="你好")
    assert ModelResponse.from_dict(response.to_dict()) == response


def test_model_response_round_trip_with_tool_calls():
    response = ModelResponse(
        content=None,
        tool_calls=[
            ToolCall(id="call_1", name="add", arguments='{"a": 1, "b": 2}'),
            ToolCall(id="call_2", name="get_current_time", arguments="{}"),
        ],
    )
    restored = ModelResponse.from_dict(response.to_dict())
    assert restored == response
    assert restored.tool_calls[0].name == "add"


def test_model_response_from_dict_tolerates_missing_tool_calls():
    restored = ModelResponse.from_dict({"content": "只有文本"})
    assert restored.content == "只有文本"
    assert restored.tool_calls == []
