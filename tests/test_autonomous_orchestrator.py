"""AutonomousOrchestrator 模型自主编排测试：主管自主产出 DAG 计划（节点 + 依赖边）。"""

import pytest

from gearlink.core.agent import ReactAgent
from gearlink.core.events import (
    AgentHandoffEvent,
    FinalAnswerEvent,
    SubtaskEndEvent,
    TeamPlanGeneratedEvent,
    TextDeltaEvent,
)
from gearlink.core.orchestrator import AutonomousOrchestrator, _parse_graph_plan
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

    def chat(self, messages, tools=None, response_format=None):
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


def make_autonomous(supervisor_responses, workers, **kwargs):
    """构造自主编排器：主管按脚本响应（首个为规划输出，后续为汇总输出）"""
    return AutonomousOrchestrator(
        supervisor=make_agent(FakeProvider(supervisor_responses)),
        workers=workers,
        **kwargs,
    )


DIAMOND_PLAN = (
    '{"nodes": [{"id": "n1", "worker": "a", "task": "甲"}, '
    '{"id": "n2", "worker": "b", "task": "乙"}, '
    '{"id": "n3", "worker": "c", "task": "丙"}], '
    '"edges": [{"from": "n1", "to": "n3"}, {"from": "n2", "to": "n3"}]}'
)


# ------------------- 计划解析与校验 -------------------


def test_parse_graph_plan_valid():
    """合法 DAG 计划：节点与边原样解析"""
    nodes, edges = _parse_graph_plan(DIAMOND_PLAN, {"a", "b", "c"})
    assert [n["id"] for n in nodes] == ["n1", "n2", "n3"]
    assert [n["worker"] for n in nodes] == ["a", "b", "c"]
    assert edges == [("n1", "n3"), ("n2", "n3")]


def test_parse_graph_plan_array_compat_serial_chain():
    """兼容旧分派格式：纯数组按顺序解释为串行链"""
    text = '[{"worker": "a", "task": "甲"}, {"worker": "b", "task": "乙"}]'
    nodes, edges = _parse_graph_plan(text, {"a", "b"})
    assert [n["id"] for n in nodes] == ["0", "1"]
    assert edges == [("0", "1")]


def test_parse_graph_plan_strips_code_fence():
    """解析容错：剥离 markdown 代码围栏"""
    text = "```json\n" + DIAMOND_PLAN + "\n```"
    assert _parse_graph_plan(text, {"a", "b", "c"}) is not None


def test_parse_graph_plan_empty_edges_is_all_parallel():
    """edges 为空数组：全部节点互不依赖"""
    text = '{"nodes": [{"id": "n1", "worker": "a", "task": "甲"}], "edges": []}'
    nodes, edges = _parse_graph_plan(text, {"a"})
    assert nodes and edges == []


def test_parse_graph_plan_rejects_invalid():
    """非法计划一律返回 None：非 JSON / 空节点 / 未登记工人 / 缺失引用 / 自环 / 重复 ID"""
    roster = {"a", "b"}
    assert _parse_graph_plan("随便说点什么", roster) is None
    assert _parse_graph_plan('{"nodes": [], "edges": []}', roster) is None
    assert (
        _parse_graph_plan('{"nodes": [{"id": "n1", "worker": "ghost", "task": "x"}]}', roster)
        is None
    )
    assert (
        _parse_graph_plan(
            '{"nodes": [{"id": "n1", "worker": "a", "task": "x"}], '
            '"edges": [{"from": "n1", "to": "ghost"}]}',
            roster,
        )
        is None
    )
    assert (
        _parse_graph_plan(
            '{"nodes": [{"id": "n1", "worker": "a", "task": "x"}], '
            '"edges": [{"from": "n1", "to": "n1"}]}',
            roster,
        )
        is None
    )
    assert (
        _parse_graph_plan(
            '{"nodes": [{"id": "n1", "worker": "a", "task": "x"}, '
            '{"id": "n1", "worker": "b", "task": "y"}], "edges": []}',
            roster,
        )
        is None
    )
    assert (
        _parse_graph_plan('{"nodes": [{"id": "n1", "worker": "a"}], "edges": []}', roster) is None
    )  # 缺 task


def test_parse_graph_plan_rejects_cycle():
    """依赖边成环时返回 None"""
    text = (
        '{"nodes": [{"id": "n1", "worker": "a", "task": "甲"}, '
        '{"id": "n2", "worker": "b", "task": "乙"}], '
        '"edges": [{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n1"}]}'
    )
    assert _parse_graph_plan(text, {"a", "b"}) is None


# ------------------- DAG 执行与消息同步 -------------------


def test_autonomous_diamond_injects_all_upstream_before_downstream():
    """菱形依赖：下游在全部上游完成后执行，且注入双方结果"""
    writer_provider = CapturingProvider(ModelResponse(content="合成稿"))
    orchestrator = make_autonomous(
        [ModelResponse(content=DIAMOND_PLAN), ModelResponse(content="最终答案")],
        {
            "a": make_agent("结果甲"),
            "b": make_agent("结果乙"),
            "c": ReactAgent(provider=writer_provider),
        },
    )

    events = list(orchestrator.run_events("任务"))

    subtask_ends = [e for e in events if isinstance(e, SubtaskEndEvent)]
    assert [e.worker for e in subtask_ends] == ["a", "b", "c"]
    # c（第 1 层）在 a、b（第 0 层）全部完成后才执行
    assert subtask_ends[2].seq > subtask_ends[0].seq
    assert subtask_ends[2].seq > subtask_ends[1].seq

    task_text = writer_provider.received_messages[0][-1]["content"]
    assert "丙" in task_text
    assert "[上游结果]" in task_text
    assert "- a: 结果甲" in task_text
    assert "- b: 结果乙" in task_text


def test_autonomous_plan_event_carries_graph_and_parallel_groups():
    """TeamPlanGeneratedEvent 携带依赖图与拓扑分层（层内下标按分派序）"""
    orchestrator = make_autonomous(
        [ModelResponse(content=DIAMOND_PLAN), ModelResponse(content="汇总")],
        {"a": make_agent("结果甲"), "b": make_agent("结果乙"), "c": make_agent("结果丙")},
    )

    events = list(orchestrator.run_events("任务"))

    plan = events[0]
    assert isinstance(plan, TeamPlanGeneratedEvent)
    assert plan.assignments == [
        {"worker": "a", "task": "甲"},
        {"worker": "b", "task": "乙"},
        {"worker": "c", "task": "丙"},
    ]
    assert plan.graph == {
        "nodes": [
            {"id": "n1", "worker": "a", "task": "甲"},
            {"id": "n2", "worker": "b", "task": "乙"},
            {"id": "n3", "worker": "c", "task": "丙"},
        ],
        "edges": [{"from": "n1", "to": "n3"}, {"from": "n2", "to": "n3"}],
    }
    assert plan.parallel_groups == [[0, 1], [2]]

    # 派单/完成事件携带分层与消息同步来源
    handoffs = [e for e in events if isinstance(e, AgentHandoffEvent)]
    assert [(e.layer, e.upstream) for e in handoffs] == [(0, []), (0, []), (1, [0, 1])]
    ends = [e for e in events if isinstance(e, SubtaskEndEvent)]
    assert [e.layer for e in ends] == [0, 0, 1]

    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].content == "汇总"
    assert [e.seq for e in events] == list(range(len(events)))


def test_autonomous_serial_chain_executes_topologically():
    """串行链：乱序节点仍按依赖拓扑序执行"""
    plan = (
        '{"nodes": [{"id": "w", "worker": "writer", "task": "撰写"}, '
        '{"id": "r", "worker": "researcher", "task": "调研"}], '
        '"edges": [{"from": "r", "to": "w"}]}'
    )
    orchestrator = make_autonomous(
        [ModelResponse(content=plan), ModelResponse(content="最终答案")],
        {"writer": make_agent("文稿"), "researcher": make_agent("资料")},
        parallel=False,
    )

    events = list(orchestrator.run_events("任务"))

    # 执行序：researcher（分派下标 1）先完成，writer（分派下标 0）后完成
    subtask_ends = [e for e in events if isinstance(e, SubtaskEndEvent)]
    assert [(e.worker, e.index) for e in subtask_ends] == [("researcher", 1), ("writer", 0)]

    writer_handoff = next(
        e for e in events if isinstance(e, AgentHandoffEvent) and e.worker == "writer"
    )
    assert "[上游结果]" in writer_handoff.task
    assert "- researcher: 资料" in writer_handoff.task


def test_autonomous_same_worker_multiple_nodes_run_in_order():
    """同一工人的多个节点按声明顺序串行执行（隐式链式边）"""
    researcher = FakeProvider([ModelResponse(content="资料一"), ModelResponse(content="资料二")])
    orchestrator = make_autonomous(
        [
            ModelResponse(
                content='{"nodes": [{"id": "n1", "worker": "r", "task": "调研一"}, '
                '{"id": "n2", "worker": "r", "task": "调研二"}], "edges": []}'
            ),
            ModelResponse(content="汇总"),
        ],
        {"r": ReactAgent(provider=researcher)},
    )

    events = list(orchestrator.run_events("任务"))

    subtask_ends = [e for e in events if isinstance(e, SubtaskEndEvent)]
    assert [(e.index, e.result, e.layer) for e in subtask_ends] == [
        (0, "资料一", 0),
        (1, "资料二", 1),
    ]


def test_autonomous_independent_nodes_parallel_within_layer():
    """parallel=True（默认）时同层并行，事件顺序仍按分派清单确定"""
    orchestrator = make_autonomous(
        [
            ModelResponse(
                content='{"nodes": [{"id": "n1", "worker": "a", "task": "甲"}, '
                '{"id": "n2", "worker": "b", "task": "乙"}], "edges": []}'
            ),
            ModelResponse(content="并行汇总"),
        ],
        {"a": make_agent("结果甲"), "b": make_agent("结果乙")},
    )

    events = list(orchestrator.run_events("任务"))

    # 并行模式先产出全部派单事件，再按序产出工人事件与完成事件
    handoffs = [e for e in events if isinstance(e, AgentHandoffEvent)]
    assert [e.index for e in handoffs] == [0, 1]
    subtask_ends = [e for e in events if isinstance(e, SubtaskEndEvent)]
    assert [(e.worker, e.result) for e in subtask_ends] == [("a", "结果甲"), ("b", "结果乙")]
    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].content == "并行汇总"
    assert [e.seq for e in events] == list(range(len(events)))


# ------------------- 兜底与降级 -------------------


def test_autonomous_invalid_plan_falls_back_to_all_workers():
    """规划输出无法解析时退化为全员兜底分派（无图/分层字段）"""
    orchestrator = make_autonomous(
        [ModelResponse(content="这不是合法 JSON"), ModelResponse(content="兜底汇总")],
        {"a": make_agent("结果甲"), "b": make_agent("结果乙")},
    )

    events = list(orchestrator.run_events("原始任务"))

    plan = events[0]
    assert isinstance(plan, TeamPlanGeneratedEvent)
    assert plan.assignments == [
        {"worker": "a", "task": "原始任务"},
        {"worker": "b", "task": "原始任务"},
    ]
    assert plan.graph is None
    assert plan.parallel_groups is None
    handoffs = [e for e in events if isinstance(e, AgentHandoffEvent)]
    assert [(e.layer, e.upstream) for e in handoffs] == [(None, None), (None, None)]
    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].content == "兜底汇总"


def test_autonomous_cycle_plan_falls_back():
    """依赖边成环的计划降级为全员兜底分派"""
    plan = (
        '{"nodes": [{"id": "n1", "worker": "a", "task": "甲"}, '
        '{"id": "n2", "worker": "b", "task": "乙"}], '
        '"edges": [{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n1"}]}'
    )
    orchestrator = make_autonomous(
        [ModelResponse(content=plan), ModelResponse(content="兜底答案")],
        {"a": make_agent("结果甲"), "b": make_agent("结果乙")},
    )

    events = list(orchestrator.run_events("任务"))

    assert isinstance(events[0], TeamPlanGeneratedEvent)
    assert events[0].graph is None
    assert [a["worker"] for a in events[0].assignments] == ["a", "b"]
    assert events[-1].content == "兜底答案"


def test_autonomous_unknown_worker_plan_falls_back():
    """计划引用未登记工人时降级为全员兜底分派"""
    orchestrator = make_autonomous(
        [
            ModelResponse(
                content='{"nodes": [{"id": "n1", "worker": "ghost", "task": "幽灵任务"}], '
                '"edges": []}'
            ),
            ModelResponse(content="兜底答案"),
        ],
        {"a": make_agent("结果甲"), "b": make_agent("结果乙")},
    )

    events = list(orchestrator.run_events("任务"))

    assert isinstance(events[0], TeamPlanGeneratedEvent)
    assert events[0].graph is None
    assert [a["worker"] for a in events[0].assignments] == ["a", "b"]


# ------------------- 通用语义（与 Orchestrator 一致） -------------------


def test_autonomous_single_subtask_passthrough():
    """单子任务直接透传工人结果，不调用汇总器"""
    orchestrator = make_autonomous(
        [
            ModelResponse(
                content='{"nodes": [{"id": "n1", "worker": "solo", "task": "唯一子任务"}], '
                '"edges": []}'
            )
        ],
        {"solo": make_agent("直接结果")},
    )

    assert orchestrator.run("任务") == "直接结果"
    assert orchestrator.provider.calls == 1  # 仅规划，无汇总调用


def test_autonomous_stream_emits_text_delta_before_final_answer():
    """stream=True 且多子任务时先产出合成文本增量，再产出最终答案"""
    orchestrator = make_autonomous(
        [
            ModelResponse(content=DIAMOND_PLAN),
            ModelResponse(content="流式汇总"),
        ],
        {
            "a": make_agent("结果甲"),
            "b": make_agent("结果乙"),
            "c": make_agent("结果丙"),
        },
        parallel=False,
    )

    events = list(orchestrator.run_events("任务", stream=True))

    # 工人流式文本增量 + 多子任务的合成文本增量（汇总器）都会流出
    deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert [d.delta for d in deltas] == ["结果甲", "结果乙", "结果丙", "流式汇总"]
    assert events.index(deltas[-1]) < events.index(events[-1])
    assert isinstance(events[-1], FinalAnswerEvent)


def test_autonomous_shared_hooks_observe_worker_events():
    """编排层与工人共享回调列表：钩子能观察到工人内部事件"""
    seen: list[str] = []
    orchestrator = make_autonomous(
        [ModelResponse(content=DIAMOND_PLAN), ModelResponse(content="汇总")],
        {"a": make_agent("结果甲"), "b": make_agent("结果乙"), "c": make_agent("结果丙")},
        hooks=[lambda event: seen.append(event.type)],
    )

    orchestrator.run("任务")

    assert "team_plan_generated" in seen
    assert "agent_handoff" in seen
    assert "model_message" in seen  # 工人内部事件经共享钩子可见
    assert "final_answer" in seen


def test_autonomous_empty_workers_raises():
    with pytest.raises(GearLinkError):
        AutonomousOrchestrator(supervisor=make_agent("占位"), workers={})
