"""Orchestrator 多 Agent 协作测试：主管-工人编排（开发方向 §5.3）。"""

import pytest

from gearlink.core.agent import ReactAgent
from gearlink.core.events import (
    AgentHandoffEvent,
    FinalAnswerEvent,
    SubtaskEndEvent,
    TeamPlanGeneratedEvent,
)
from gearlink.core.orchestrator import Orchestrator, _parse_assignments
from gearlink.exceptions import GearLinkError
from gearlink.providers.base import ModelProvider, ModelResponse


class FakeProvider(ModelProvider):
    """按序返回预设响应的测试用提供者"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.calls = 0

    def chat(self, messages, tools=None, response_format=None) -> ModelResponse:
        response = self.responses[self.calls]
        self.calls += 1
        return response


def make_agent(answer_or_provider):
    """构造 ReactAgent：传字符串时直接返回固定答案，传 provider 时原样使用"""
    if isinstance(answer_or_provider, str):
        provider = FakeProvider([ModelResponse(content=answer_or_provider)])
    else:
        provider = answer_or_provider
    return ReactAgent(provider=provider)


def make_orchestrator(supervisor_responses, workers, **kwargs):
    """构造编排器：主管按脚本响应（首个为分派输出，后续为汇总输出）"""
    return Orchestrator(
        supervisor=make_agent(FakeProvider(supervisor_responses)),
        workers={name: make_agent(answer) for name, answer in workers.items()},
        **kwargs,
    )


def test_orchestrator_dispatches_and_synthesizes():
    """主管分派两个子任务给不同工人，结果经汇总整合"""
    orchestrator = make_orchestrator(
        [
            ModelResponse(
                content='[{"worker": "researcher", "task": "调研 A"}, '
                '{"worker": "writer", "task": "撰写 B"}]'
            ),
            ModelResponse(content="最终答案"),
        ],
        {"researcher": "结果 A", "writer": "结果 B"},
    )

    assert orchestrator.run("完成调研与撰写") == "最终答案"
    assert orchestrator.provider.calls == 2  # 分派 + 汇总


def test_orchestrator_event_sequence():
    """事件流：分派清单 → 逐个派单/工人事件/完成 → 最终答案，seq 全局递增"""
    orchestrator = make_orchestrator(
        [
            ModelResponse(
                content='[{"worker": "a", "task": "子任务甲"}, {"worker": "b", "task": "子任务乙"}]'
            ),
            ModelResponse(content="汇总答案"),
        ],
        {"a": "结果甲", "b": "结果乙"},
    )

    events = list(orchestrator.run_events("任务"))

    assert isinstance(events[0], TeamPlanGeneratedEvent)
    assert events[0].assignments == [
        {"worker": "a", "task": "子任务甲"},
        {"worker": "b", "task": "子任务乙"},
    ]

    handoffs = [e for e in events if isinstance(e, AgentHandoffEvent)]
    assert [(e.index, e.worker, e.task) for e in handoffs] == [
        (0, "a", "子任务甲"),
        (1, "b", "子任务乙"),
    ]

    subtask_ends = [e for e in events if isinstance(e, SubtaskEndEvent)]
    assert [(e.worker, e.result) for e in subtask_ends] == [("a", "结果甲"), ("b", "结果乙")]

    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].content == "汇总答案"
    assert [e.seq for e in events] == list(range(len(events)))


def test_orchestrator_single_subtask_passthrough():
    """单子任务时直接透传工人结果，不调用汇总器"""
    orchestrator = make_orchestrator(
        [ModelResponse(content='[{"worker": "solo", "task": "唯一子任务"}]')],
        {"solo": "直接结果"},
    )

    assert orchestrator.run("任务") == "直接结果"
    assert orchestrator.provider.calls == 1  # 仅分派，无汇总调用


def test_orchestrator_invalid_dispatch_falls_back_to_all_workers():
    """主管输出无法解析时退化为把原任务派给全部工人"""
    orchestrator = make_orchestrator(
        [ModelResponse(content="这不是合法 JSON"), ModelResponse(content="兜底汇总")],
        {"a": "结果甲", "b": "结果乙"},
    )

    events = list(orchestrator.run_events("原始任务"))

    plan = events[0]
    assert isinstance(plan, TeamPlanGeneratedEvent)
    assert plan.assignments == [
        {"worker": "a", "task": "原始任务"},
        {"worker": "b", "task": "原始任务"},
    ]
    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].content == "兜底汇总"


def test_orchestrator_unknown_worker_falls_back():
    """主管指派未登记工人时退化为全员兜底分派"""
    orchestrator = make_orchestrator(
        [
            ModelResponse(content='[{"worker": "ghost", "task": "幽灵任务"}]'),
            ModelResponse(content="兜底答案"),
        ],
        {"a": "结果甲", "b": "结果乙"},
    )

    events = list(orchestrator.run_events("任务"))

    assert isinstance(events[0], TeamPlanGeneratedEvent)
    assert [a["worker"] for a in events[0].assignments] == ["a", "b"]


def test_orchestrator_parallel_execution():
    """并行模式下结果与事件顺序仍按分派清单确定"""
    orchestrator = make_orchestrator(
        [
            ModelResponse(content='[{"worker": "a", "task": "甲"}, {"worker": "b", "task": "乙"}]'),
            ModelResponse(content="并行汇总"),
        ],
        {"a": "结果甲", "b": "结果乙"},
        parallel=True,
    )

    events = list(orchestrator.run_events("任务"))

    handoffs = [e for e in events if isinstance(e, AgentHandoffEvent)]
    # 并行模式先产出全部派单事件，再按序产出工人事件与完成事件
    assert [e.index for e in handoffs] == [0, 1]
    subtask_ends = [e for e in events if isinstance(e, SubtaskEndEvent)]
    assert [(e.worker, e.result) for e in subtask_ends] == [("a", "结果甲"), ("b", "结果乙")]
    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].content == "并行汇总"
    assert [e.seq for e in events] == list(range(len(events)))


def test_orchestrator_empty_workers_raises():
    with pytest.raises(GearLinkError):
        Orchestrator(supervisor=make_agent("占位"), workers={})


def test_orchestrator_shared_hooks_observe_worker_events():
    """编排层与工人共享回调列表：钩子能观察到工人内部事件"""
    seen: list[str] = []
    orchestrator = make_orchestrator(
        [
            ModelResponse(content='[{"worker": "a", "task": "甲"}, {"worker": "b", "task": "乙"}]'),
            ModelResponse(content="汇总"),
        ],
        {"a": "结果甲", "b": "结果乙"},
        hooks=[lambda event: seen.append(event.type)],
    )

    orchestrator.run("任务")

    assert "team_plan_generated" in seen
    assert "agent_handoff" in seen
    assert "model_message" in seen  # 工人内部事件经共享钩子可见
    assert "final_answer" in seen


def test_parse_assignments_strips_code_fence():
    """分派解析容错：剥离 markdown 代码围栏"""
    text = '```json\n[{"worker": "a", "task": "子任务"}]\n```'
    assert _parse_assignments(text) == [{"worker": "a", "task": "子任务"}]


def test_parse_assignments_rejects_invalid():
    assert _parse_assignments("随便说点什么") is None
    assert _parse_assignments("[]") is None
    assert _parse_assignments('[{"worker": "a"}]') is None  # 缺 task
    assert _parse_assignments('[{"worker": "", "task": "x"}]') is None  # worker 为空


# ------------------- response_format 透传（开发方向 §6.5） -------------------


def test_orchestrator_dispatch_uses_response_format():
    """_dispatch 调用主管模型时传入 response_format={"type": "json_object"}。"""

    class ResponseFormatCapturingProvider(ModelProvider):
        def __init__(self, response):
            self.response = response
            self.received_response_format = []

        def chat(self, messages, tools=None, response_format=None):
            self.received_response_format.append(response_format)
            return self.response

    provider = ResponseFormatCapturingProvider(
        ModelResponse(content='[{"worker": "a", "task": "子任务"}]')
    )
    worker_provider = FakeProvider([ModelResponse(content="结果")])
    orchestrator = Orchestrator(
        supervisor=ReactAgent(provider=provider),
        workers={"a": ReactAgent(provider=worker_provider)},
    )

    orchestrator.run("任务")

    assert provider.received_response_format[0] == {"type": "json_object"}
