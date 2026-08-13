"""全局日志开关测试：验证 enable_logging / disable_logging 的开关、级别与幂等行为。"""

import logging

import pytest

from gearlink import disable_logging, enable_logging
from gearlink.core.agent import ReactAgent
from gearlink.providers.base import ModelProvider, ModelResponse, ToolCall
from gearlink.utils.logging import _root_logger


@pytest.fixture(autouse=True)
def _reset_logging_switch():
    """每个用例前后重置全局开关，避免用例间状态污染。"""
    disable_logging()
    yield
    disable_logging()


class SequenceProvider(ModelProvider):
    """按序返回预设响应的测试用提供者"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = responses
        self._calls = 0

    def chat(self, messages, tools=None) -> ModelResponse:
        response = self._responses[self._calls]
        self._calls += 1
        return response


def _agent_with_tool_call() -> ReactAgent:
    """构造一个会触发 [工具调用] INFO 日志（get_current_time）的 Agent。"""
    tool_response = ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="get_current_time", arguments="{}")],
    )
    final_response = ModelResponse(content="现在是中午。")
    return ReactAgent(provider=SequenceProvider([tool_response, final_response]))


def test_enable_logging_outputs_tool_calls(capsys):
    """开启开关后，ReAct 工具调用日志应输出到 stderr"""
    enable_logging()

    assert _agent_with_tool_call().run("现在几点了？") == "现在是中午。"

    captured = capsys.readouterr().err
    assert "[工具调用] get_current_time" in captured


def test_disable_logging_silences_output(capsys):
    """关闭开关后，内部日志不再输出"""
    enable_logging()
    _agent_with_tool_call().run("现在几点了？")
    capsys.readouterr()  # 清空开启期间的输出

    disable_logging()
    _agent_with_tool_call().run("现在几点了？")

    captured = capsys.readouterr().err
    assert "[工具调用]" not in captured


def test_enable_logging_respects_level(capsys):
    """enable_logging 的级别应生效：WARNING 级别下 INFO 日志被过滤"""
    enable_logging(logging.WARNING)
    _agent_with_tool_call().run("现在几点了？")

    assert "[工具调用]" not in capsys.readouterr().err


def test_enable_logging_sets_effective_level():
    """开启开关后，gearlink 命名空间的有效级别应为所配级别"""
    enable_logging(logging.DEBUG)

    root = _root_logger()
    assert root.level == logging.DEBUG
    assert root.isEnabledFor(logging.DEBUG)


def test_enable_logging_is_idempotent():
    """重复开启不应重复添加 handler"""
    enable_logging()
    # 记录首次开启后的 handler 数量（pytest 9.x 会向 propagate=False 的
    # logger 注入 LogCaptureHandler，故不能断言绝对值 1）
    n = len(_root_logger().handlers)
    enable_logging()
    enable_logging()

    # 重复调用不应新增任何 handler
    assert len(_root_logger().handlers) == n


def test_exports_from_top_level():
    """开关应从顶层包 gearlink 显式导出"""
    import gearlink

    assert "enable_logging" in gearlink.__all__
    assert "disable_logging" in gearlink.__all__
