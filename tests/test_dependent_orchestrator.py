"""DependentOrchestrator 依赖编排测试：主管-工人 + 工人间依赖（开发方向 §6.8）。"""

import pytest

from gearlink.core.agent import Agent, ReactAgent
from gearlink.core.events import (
    AgentHandoffEvent,
    FinalAnswerEvent,
    LoopAbortEvent,
    SubtaskEndEvent,
    TeamPlanGeneratedEvent,
    TextDeltaEvent,
)
from gearlink.core.orchestrator import DependentOrchestrator, _detect_dependency_cycle
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


class CapturingProvider(ModelProvider):
    """记录收到的请求消息并返回预设响应的测试用提供者"""

    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.calls = 0
        self.received_messages: list[list[dict]] = []

    def chat(self, messages, tools=None, response_format=None) -> ModelResponse:
        self.calls += 1
        # 拷贝快照：ReactAgent 之后会向同一列表追加 assistant 消息
        self.received_messages.append([dict(m) for m in messages])
        return self.response


def make_agent(answer_or_provider):
    """构造 ReactAgent：传字符串时直接返回固定答案，传 provider 时原样使用"""
    if isinstance(answer_or_provider, str):
        provider = FakeProvider([ModelResponse(content=answer_or_provider)])
    else:
        provider = answer_or_provider
    return ReactAgent(provider=provider)


def make_dependent(supervisor_responses, workers, dependencies=None, **kwargs):
    """构造依赖编排器：主管按脚本响应（首个为分派输出，后续为汇总输出）"""
    return DependentOrchestrator(
        supervisor=make_agent(FakeProvider(supervisor_responses)),
        workers=workers,
        dependencies=dependencies,
        **kwargs,
    )


def test_dependent_injects_upstream_result_into_downstream_task():
    """下游工人的任务文本自动追加上游结果报告段落"""
    writer_provider = CapturingProvider(ModelResponse(content="新闻稿"))
    orchestrator = make_dependent(
        [
            ModelResponse(
                content='[{"worker": "researcher", "task": "整理资料"}, '
                '{"worker": "writer", "task": "撰写新闻"}]'
            ),
            ModelResponse(content="最终答案"),
        ],
        {
            "researcher": make_agent("资料集"),
            "writer": ReactAgent(provider=writer_provider),
        },
        dependencies={"writer": ["researcher"]},
    )

    assert orchestrator.run("写新闻") == "最终答案"

    task_text = writer_provider.received_messages[0][-1]["content"]
    assert "撰写新闻" in task_text
    assert "[上游结果]" in task_text
    assert "- researcher: 资料集" in task_text


def test_dependent_reorders_dispatched_assignments_topologically():
    """分派清单乱序（下游在前）时仍按拓扑序执行"""
    orchestrator = make_dependent(
        [
            ModelResponse(
                content='[{"worker": "writer", "task": "撰写"}, '
                '{"worker": "researcher", "task": "调研"}]'
            ),
            ModelResponse(content="最终答案"),
        ],
        {"writer": make_agent("文稿"), "researcher": make_agent("资料")},
        dependencies={"writer": ["researcher"]},
    )

    events = list(orchestrator.run_events("任务"))

    # 执行序：researcher（分派下标 1）先完成，writer（分派下标 0）后完成
    subtask_ends = [e for e in events if isinstance(e, SubtaskEndEvent)]
    assert [(e.worker, e.index) for e in subtask_ends] == [("researcher", 1), ("writer", 0)]

    # writer 的派单事件携带注入后的任务（含 researcher 结果）
    writer_handoff = next(
        e for e in events if isinstance(e, AgentHandoffEvent) and e.worker == "writer"
    )
    assert "[上游结果]" in writer_handoff.task
    assert "- researcher: 资料" in writer_handoff.task


def test_dependent_diamond_injects_multiple_upstream_results():
    """菱形依赖：下游同时注入两个上游工人的结果"""
    orchestrator = make_dependent(
        [
            ModelResponse(
                content='[{"worker": "a", "task": "甲"}, {"worker": "b", "task": "乙"}, '
                '{"worker": "c", "task": "丙"}]'
            ),
            ModelResponse(content="最终答案"),
        ],
        {"a": make_agent("结果甲"), "b": make_agent("结果乙"), "c": make_agent("结果丙")},
        dependencies={"c": ["a", "b"]},
    )

    events = list(orchestrator.run_events("任务"))

    subtask_ends = [e for e in events if isinstance(e, SubtaskEndEvent)]
    assert [e.worker for e in subtask_ends] == ["a", "b", "c"]
    # c 在 a、b 完成后才执行
    assert subtask_ends[2].seq > subtask_ends[0].seq
    assert subtask_ends[2].seq > subtask_ends[1].seq


def test_dependent_parallel_layers_runs_parallel_within_layer():
    """parallel=True 时同层并行、跨层串行，事件顺序仍按分派清单确定"""
    orchestrator = make_dependent(
        [
            ModelResponse(
                content='[{"worker": "a", "task": "甲"}, {"worker": "b", "task": "乙"}, '
                '{"worker": "c", "task": "丙"}]'
            ),
            ModelResponse(content="并行汇总"),
        ],
        {"a": make_agent("结果甲"), "b": make_agent("结果乙"), "c": make_agent("结果丙")},
        dependencies={"c": ["a", "b"]},
        parallel=True,
    )

    events = list(orchestrator.run_events("任务"))

    subtask_ends = [e for e in events if isinstance(e, SubtaskEndEvent)]
    assert [e.worker for e in subtask_ends] == ["a", "b", "c"]
    handoffs = [e for e in events if isinstance(e, AgentHandoffEvent)]
    assert [e.worker for e in handoffs] == ["a", "b", "c"]
    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].content == "并行汇总"
    assert [e.seq for e in events] == list(range(len(events)))


def test_dependent_cycle_raises():
    """工人依赖图存在循环时构造抛 GearLinkError"""
    with pytest.raises(GearLinkError, match="循环"):
        DependentOrchestrator(
            supervisor=make_agent("占位"),
            workers={"a": make_agent("甲"), "b": make_agent("乙")},
            dependencies={"a": ["b"], "b": ["a"]},
        )


def test_dependent_unknown_worker_raises():
    """dependencies 引用未登记工人时构造抛 GearLinkError"""
    with pytest.raises(GearLinkError, match="未登记"):
        DependentOrchestrator(
            supervisor=make_agent("占位"),
            workers={"a": make_agent("甲")},
            dependencies={"a": ["ghost"]},
        )
    with pytest.raises(GearLinkError, match="未登记"):
        DependentOrchestrator(
            supervisor=make_agent("占位"),
            workers={"a": make_agent("甲")},
            dependencies={"ghost": ["a"]},
        )


def test_dependent_upstream_not_dispatched_ignores_dependency():
    """依赖的上游工人未被分派时依赖静默忽略，任务不注入"""
    writer_provider = CapturingProvider(ModelResponse(content="文稿"))
    orchestrator = make_dependent(
        [ModelResponse(content='[{"worker": "writer", "task": "撰写"}]')],
        {"writer": ReactAgent(provider=writer_provider), "researcher": make_agent("资料")},
        dependencies={"writer": ["researcher"]},
    )

    assert orchestrator.run("任务") == "文稿"  # 单任务透传

    task_text = writer_provider.received_messages[0][-1]["content"]
    assert task_text == "撰写"


def test_dependent_unconverged_upstream_injects_fallback_text():
    """上游工人未收敛时兜底文案注入，下游照常执行"""

    class EmptyWorker(Agent):
        """不产出任何事件的工人（未收敛场景）"""

        def run_events(self, user_input, *, stream=False):
            yield from ()

    writer_provider = CapturingProvider(ModelResponse(content="文稿"))
    orchestrator = make_dependent(
        [
            ModelResponse(
                content='[{"worker": "researcher", "task": "调研"}, '
                '{"worker": "writer", "task": "撰写"}]'
            ),
            ModelResponse(content="最终答案"),
        ],
        {
            "researcher": EmptyWorker(provider=FakeProvider([])),
            "writer": ReactAgent(provider=writer_provider),
        },
        dependencies={"writer": ["researcher"]},
    )

    assert orchestrator.run("任务") == "最终答案"

    task_text = writer_provider.received_messages[0][-1]["content"]
    assert "[上游结果]" in task_text
    assert "达到最大推理轮数" in task_text


def test_dependent_loop_abort_upstream_injects_reason():
    """上游工人以 LoopAbortEvent 中止时其 reason 注入下游"""

    class AbortingWorker(Agent):
        """产出 LoopAbortEvent 的工人（循环中止场景）"""

        def run_events(self, user_input, *, stream=False):
            yield LoopAbortEvent(reason="上游中止：资料不足")

    writer_provider = CapturingProvider(ModelResponse(content="文稿"))
    orchestrator = make_dependent(
        [
            ModelResponse(
                content='[{"worker": "researcher", "task": "调研"}, '
                '{"worker": "writer", "task": "撰写"}]'
            ),
            ModelResponse(content="最终答案"),
        ],
        {
            "researcher": AbortingWorker(provider=FakeProvider([])),
            "writer": ReactAgent(provider=writer_provider),
        },
        dependencies={"writer": ["researcher"]},
    )

    assert orchestrator.run("任务") == "最终答案"

    task_text = writer_provider.received_messages[0][-1]["content"]
    assert "上游中止：资料不足" in task_text


def test_dependent_single_subtask_passthrough():
    """单子任务直接透传工人结果，不调用汇总器"""
    orchestrator = make_dependent(
        [ModelResponse(content='[{"worker": "solo", "task": "唯一子任务"}]')],
        {"solo": make_agent("直接结果")},
    )

    assert orchestrator.run("任务") == "直接结果"
    assert orchestrator.provider.calls == 1  # 仅分派，无汇总调用


def test_dependent_none_dependencies_matches_orchestrator():
    """dependencies=None 时行为与 Orchestrator 一致"""
    orchestrator = make_dependent(
        [
            ModelResponse(
                content='[{"worker": "a", "task": "子任务甲"}, {"worker": "b", "task": "子任务乙"}]'
            ),
            ModelResponse(content="汇总答案"),
        ],
        {"a": make_agent("结果甲"), "b": make_agent("结果乙")},
    )

    events = list(orchestrator.run_events("任务"))

    assert isinstance(events[0], TeamPlanGeneratedEvent)
    assert events[0].dependencies is None
    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].content == "汇总答案"
    assert [e.seq for e in events] == list(range(len(events)))


def test_dependent_event_sequence():
    """事件流：分派清单（含依赖）→ 按拓扑序派单/工人事件/完成 → 最终答案，seq 全局递增"""
    orchestrator = make_dependent(
        [
            ModelResponse(content='[{"worker": "a", "task": "甲"}, {"worker": "b", "task": "乙"}]'),
            ModelResponse(content="汇总答案"),
        ],
        {"a": make_agent("结果甲"), "b": make_agent("结果乙")},
        dependencies={"b": ["a"]},
    )

    events = list(orchestrator.run_events("任务"))

    assert isinstance(events[0], TeamPlanGeneratedEvent)
    assert events[0].assignments == [
        {"worker": "a", "task": "甲"},
        {"worker": "b", "task": "乙"},
    ]
    assert events[0].dependencies == {"b": ["a"]}

    handoffs = [e for e in events if isinstance(e, AgentHandoffEvent)]
    assert [(e.worker, e.task) for e in handoffs] == [
        ("a", "甲"),
        ("b", "乙\n\n[上游结果]\n- a: 结果甲"),
    ]

    subtask_ends = [e for e in events if isinstance(e, SubtaskEndEvent)]
    assert [(e.worker, e.result) for e in subtask_ends] == [("a", "结果甲"), ("b", "结果乙")]

    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].content == "汇总答案"
    assert [e.seq for e in events] == list(range(len(events)))


def test_dependent_shared_hooks_observe_worker_events():
    """编排层与工人共享回调列表：钩子能观察到工人内部事件"""
    seen: list[str] = []
    orchestrator = make_dependent(
        [
            ModelResponse(content='[{"worker": "a", "task": "甲"}, {"worker": "b", "task": "乙"}]'),
            ModelResponse(content="汇总"),
        ],
        {"a": make_agent("结果甲"), "b": make_agent("结果乙")},
        dependencies={"b": ["a"]},
        hooks=[lambda event: seen.append(event.type)],
    )

    orchestrator.run("任务")

    assert "team_plan_generated" in seen
    assert "agent_handoff" in seen
    assert "model_message" in seen  # 工人内部事件经共享钩子可见
    assert "final_answer" in seen


def test_dependent_same_worker_multiple_tasks_inject_all_results():
    """同一上游工人的多个任务结果全部注入下游"""
    writer_provider = CapturingProvider(ModelResponse(content="文稿"))
    researcher = FakeProvider([ModelResponse(content="资料一"), ModelResponse(content="资料二")])
    orchestrator = make_dependent(
        [
            ModelResponse(
                content='[{"worker": "researcher", "task": "调研一"}, '
                '{"worker": "researcher", "task": "调研二"}, '
                '{"worker": "writer", "task": "撰写"}]'
            ),
            ModelResponse(content="最终答案"),
        ],
        {
            "researcher": ReactAgent(provider=researcher),
            "writer": ReactAgent(provider=writer_provider),
        },
        dependencies={"writer": ["researcher"]},
    )

    assert orchestrator.run("任务") == "最终答案"

    task_text = writer_provider.received_messages[0][-1]["content"]
    assert "- researcher: 资料一" in task_text
    assert "- researcher: 资料二" in task_text


def test_dependent_stream_emits_text_delta_before_final_answer():
    """stream=True 且多子任务时先产出合成文本增量，再产出最终答案"""
    orchestrator = make_dependent(
        [
            ModelResponse(content='[{"worker": "a", "task": "甲"}, {"worker": "b", "task": "乙"}]'),
            ModelResponse(content="流式汇总"),
            ModelResponse(content='[{"worker": "a", "task": "甲"}, {"worker": "b", "task": "乙"}]'),
            ModelResponse(content="流式汇总"),
        ],
        {
            "a": make_agent(FakeProvider([ModelResponse(content="结果甲")] * 2)),
            "b": make_agent(FakeProvider([ModelResponse(content="结果乙")] * 2)),
        },
        dependencies={"b": ["a"]},
    )

    events = list(orchestrator.run_events("任务", stream=True))

    # 工人流式文本增量 + 多子任务的合成文本增量（汇总器）都会流出
    deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert [d.delta for d in deltas] == ["结果甲", "结果乙", "流式汇总"]
    assert events.index(deltas[-1]) < events.index(events[-1])
    assert isinstance(events[-1], FinalAnswerEvent)

    # run_stream 只转发文本增量
    assert list(orchestrator.run_stream("任务")) == ["结果甲", "结果乙", "流式汇总"]


def test_detect_dependency_cycle():
    """环检测辅助函数：无环返回 None，有环返回环上 worker 列表"""
    assert _detect_dependency_cycle({}) is None
    assert _detect_dependency_cycle({"a": ["b"], "b": []}) is None
    cycle = _detect_dependency_cycle({"a": ["b"], "b": ["a"]})
    assert cycle is not None and cycle[0] == cycle[-1]
