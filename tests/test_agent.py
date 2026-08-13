"""ReactAgent / PlanExecuteAgent 测试：使用 FakeProvider 替代真实模型服务。"""

from pathlib import Path

import pytest

from gearlink.core.agent import (
    MAX_ITERATIONS,
    SYSTEM_PROMPT,
    PlanExecuteAgent,
    ReactAgent,
    _extract_json,
)
from gearlink.core.events import FinalAnswerEvent, PlanGeneratedEvent
from gearlink.core.memory import LongTermMemory, MemoryManager, ShortTermMemory
from gearlink.core.tool import ToolRegistry
from gearlink.exceptions import ProviderError
from gearlink.providers.base import ModelProvider, ModelResponse, StreamChunk, ToolCall
from gearlink.skills import Skill, SkillLoader, SkillRegistry


class FakeCollection:
    """模拟 chromadb Collection 的最小子集"""

    def __init__(self) -> None:
        self.records: dict[str, tuple[str, dict]] = {}  # id -> (document, metadata)

    def add(self, ids, documents, metadatas) -> None:
        for record_id, doc, meta in zip(ids, documents, metadatas):
            self.records[record_id] = (doc, meta)

    def get(self, include=None):
        ids = list(self.records)
        return {
            "ids": ids,
            "documents": [self.records[i][0] for i in ids],
            "metadatas": [self.records[i][1] for i in ids],
        }

    def query(self, query_texts, n_results):
        items = list(self.records.items())[:n_results]
        return {
            "ids": [[record_id for record_id, _ in items]],
            "documents": [[doc for _, (doc, _) in items]],
            "metadatas": [[meta for _, (_, meta) in items]],
            "distances": [[float(i) for i in range(len(items))]],
        }

    def delete(self, ids) -> None:
        for record_id in ids:
            self.records.pop(record_id, None)


class FakeClient:
    def get_or_create_collection(self, name):
        return FakeCollection()


class FakeProvider(ModelProvider):
    """按序返回预设响应的测试用提供者"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.calls = 0

    def chat(self, messages, tools=None, response_format=None) -> ModelResponse:
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_agent_returns_final_answer():
    provider = FakeProvider([ModelResponse(content="你好！")])
    agent = ReactAgent(provider=provider)

    assert agent.run("你好") == "你好！"


def test_agent_runs_tool_then_answers():
    tool_response = ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="get_current_time", arguments="{}")],
    )
    final_response = ModelResponse(content="现在是中午。")
    agent = ReactAgent(provider=FakeProvider([tool_response, final_response]))

    assert agent.run("现在几点了？") == "现在是中午。"


def test_agent_recovers_from_unknown_tool():
    tool_response = ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="no_such_tool", arguments="{}")],
    )
    final_response = ModelResponse(content="抱歉，工具不可用。")
    agent = ReactAgent(provider=FakeProvider([tool_response, final_response]))

    # 工具失败是可恢复信号，不应中断循环
    assert agent.run("帮我查一下") == "抱歉，工具不可用。"


def test_agent_default_memory_includes_system_prompt():
    agent = ReactAgent(provider=FakeProvider([ModelResponse(content="你好！")]))

    messages = agent.memory.get_messages()
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_agent_accepts_injected_memory():
    memory = ShortTermMemory(max_message=5)
    memory.add_message({"role": "system", "content": "自定义提示"})
    agent = ReactAgent(provider=FakeProvider([ModelResponse(content="你好！")]), memory=memory)

    assert agent.memory is memory
    assert agent.run("你好") == "你好！"


def test_agent_with_memory_manager_builds_context():
    """注入 MemoryManager 时应通过 build_context 组装请求消息"""

    class CapturingProvider(FakeProvider):
        def chat(self, messages, tools=None):
            self.last_messages = messages
            return super().chat(messages, tools)

    manager = MemoryManager(short_term=ShortTermMemory(max_message=20))
    agent = ReactAgent(
        provider=CapturingProvider([ModelResponse(content="你好！")]), memory=manager
    )

    assert agent.run("你好") == "你好！"
    # 短期记忆未预置 system 提示（由调用方自行决定），仅含用户消息
    assert agent.provider.last_messages == [{"role": "user", "content": "你好"}]


def test_agent_truncates_oversized_tool_result():
    from gearlink.core.tool import register_tool

    def huge_result():
        return "x" * 20000

    register_tool(
        "test_huge_result_tool",
        huge_result,
        {
            "description": "返回超大结果",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    )
    tool_response = ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="test_huge_result_tool", arguments="{}")],
    )
    final_response = ModelResponse(content="完成。")
    agent = ReactAgent(provider=FakeProvider([tool_response, final_response]))

    assert agent.run("运行工具") == "完成。"

    tool_messages = [m for m in agent.memory.get_messages() if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "(工具结果过长，已截断)" in tool_messages[0]["content"]
    assert len(tool_messages[0]["content"]) < 20000


def test_agent_returns_fallback_after_max_iterations():
    """模型连续请求工具直至达到迭代上限时，应返回兜底文案而非死循环"""
    tool_response = ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="get_current_time", arguments="{}")],
    )
    agent = ReactAgent(provider=FakeProvider([tool_response] * MAX_ITERATIONS))

    result = agent.run("反复调用工具")

    assert result == "已达到最大推理轮数，无法得出最终答案。"


def test_agent_retrieves_long_term_on_every_iteration_when_enabled():
    """开启 retrieve_every_iteration 时，非首轮迭代仍注入长期记忆检索结果"""

    class CapturingProvider(FakeProvider):
        def chat(self, messages, tools=None):
            self.last_messages = messages
            return super().chat(messages, tools)

    tool_response = ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="no_such_tool", arguments="{}")],
    )
    final_response = ModelResponse(content="完成了")

    def make_agent(retrieve_every_iteration):
        long_term = LongTermMemory(vector_db=FakeClient(), collection_name="test")
        long_term.add_message({"role": "user", "content": "我喜欢喝红茶"})
        manager = MemoryManager(
            short_term=ShortTermMemory(max_message=20),
            long_term=long_term,
        )
        provider = CapturingProvider([tool_response, final_response])
        return ReactAgent(
            provider=provider,
            memory=manager,
            retrieve_every_iteration=retrieve_every_iteration,
        ), provider

    # 开启：第 2 次 chat 的消息仍含检索注入的 system 内容
    enabled_agent, enabled_provider = make_agent(retrieve_every_iteration=True)
    assert enabled_agent.run("查询我的偏好") == "完成了"
    second_messages = enabled_provider.last_messages
    assert second_messages[0]["role"] == "system"
    assert "红茶" in second_messages[0]["content"]

    # 未开启：第 2 次 chat 不携带检索注入，消息以用户输入开头
    disabled_agent, disabled_provider = make_agent(retrieve_every_iteration=False)
    assert disabled_agent.run("查询我的偏好") == "完成了"
    second_messages = disabled_provider.last_messages
    assert second_messages[0]["role"] == "user"


def test_agent_default_prompt_lists_registered_skills():
    """注入技能注册表时，默认系统提示应包含可用技能列表与 load_skill 引导"""
    registry = SkillRegistry()
    registry.register(Skill(name="demo-skill", description="演示技能", path=Path(".")))

    agent = ReactAgent(
        provider=FakeProvider([ModelResponse(content="你好！")]), skill_registry=registry
    )

    system_prompt = agent.memory.get_messages()[0]["content"]
    assert system_prompt.startswith(SYSTEM_PROMPT)
    assert "demo-skill: 演示技能" in system_prompt
    assert "load_skill" in system_prompt


def test_agent_without_registry_keeps_plain_system_prompt():
    """未注入注册表时，默认系统提示不包含技能引导（不诱导调用不存在的工具）"""
    agent = ReactAgent(provider=FakeProvider([ModelResponse(content="你好！")]))

    assert agent.memory.get_messages()[0]["content"] == SYSTEM_PROMPT


def test_load_skill_tool_resolves_injected_registry(tmp_path):
    """Agent 注入注册表后，load_skill 工具可按名加载技能完整指令"""
    from gearlink.core.tool import call_tool
    import gearlink.tools.load_skill  # noqa: F401  # 显式导入，触发工具注册

    skill_dir = tmp_path / "demo-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: 演示技能\n---\n\n# 示例指令\n", encoding="utf-8"
    )
    registry = SkillRegistry()
    for skill in SkillLoader.discover_from_directory(tmp_path):
        registry.register(skill)

    # 构造 Agent 时会经 set_skill_registry 登记注册表
    ReactAgent(provider=FakeProvider([ModelResponse(content="你好！")]), skill_registry=registry)

    result = call_tool("load_skill", {"skill_name": "demo-skill"})

    assert result["skill_name"] == "demo-skill"
    assert result["instructions"].startswith("# 示例指令")


class FakeStreamingProvider(ModelProvider):
    """按序把预设响应的文本切成片段流式产出的测试用提供者"""

    def __init__(self, responses: list[ModelResponse], chunk_size: int = 2) -> None:
        self.responses = responses
        self.chunk_size = chunk_size
        self.calls = 0

    def chat(self, messages, tools=None) -> ModelResponse:
        raise AssertionError("流式测试不应调用非流式 chat")

    def chat_stream(self, messages, tools=None):
        response = self.responses[self.calls]
        self.calls += 1
        content = response.content or ""
        for i in range(0, len(content), self.chunk_size):
            yield StreamChunk(delta=content[i : i + self.chunk_size])
        yield StreamChunk(response=response)


def test_run_stream_yields_content_deltas():
    provider = FakeStreamingProvider([ModelResponse(content="你好世界，很高兴见到你")])
    agent = ReactAgent(provider=provider)

    deltas = list(agent.run_stream("你好"))

    # 逐片段产出且拼接后为完整答案
    assert len(deltas) > 1
    assert "".join(deltas) == "你好世界，很高兴见到你"
    assert agent.memory.get_messages()[-1] == {
        "role": "assistant",
        "content": "你好世界，很高兴见到你",
    }


def test_run_stream_executes_tool_calls_and_continues():
    tool_response = ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="get_current_time", arguments="{}")],
    )
    final_response = ModelResponse(content="现在是中午。")
    agent = ReactAgent(provider=FakeStreamingProvider([tool_response, final_response]))

    deltas = list(agent.run_stream("现在几点了？"))

    # 工具调用阶段无文本流出，仅流出最终答案
    assert "".join(deltas) == "现在是中午。"
    # 工具执行结果已写回记忆
    roles = [m["role"] for m in agent.memory.get_messages()]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


def test_run_stream_falls_back_for_non_streaming_provider():
    # FakeProvider 未覆写 chat_stream，应经基类默认实现回退到 chat
    agent = ReactAgent(provider=FakeProvider([ModelResponse(content="完整回答")]))

    deltas = list(agent.run_stream("你好"))

    assert "".join(deltas) == "完整回答"


def test_run_stream_yields_fallback_on_max_iterations():
    tool_response = ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="get_current_time", arguments="{}")],
    )
    agent = ReactAgent(provider=FakeStreamingProvider([tool_response] * MAX_ITERATIONS))

    assert "".join(agent.run_stream("循环调用")) == "已达到最大推理轮数，无法得出最终答案。"


def test_run_events_emits_full_sequence():
    """run_events 应按序产出 ReAct 循环事件，run() 等价于消费事件流"""
    tool_response = ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="get_current_time", arguments="{}")],
    )
    final_response = ModelResponse(content="现在是中午。")
    agent = ReactAgent(provider=FakeProvider([tool_response, final_response]))

    events = list(agent.run_events("现在几点了？"))

    types = [e.type for e in events]
    assert types == [
        "step_start",
        "model_message",
        "tool_call_start",
        "tool_call_end",
        "step_start",
        "model_message",
        "final_answer",
    ]
    # 序号全局递增
    assert [e.seq for e in events] == list(range(len(events)))
    # 收敛事件携带最终答案
    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].content == "现在是中午。"
    # 工具事件携带工具名与执行结果（无错误）
    assert events[2].name == "get_current_time" and events[2].arguments == "{}"
    assert events[3].name == "get_current_time"
    assert events[3].error is None


def test_run_events_hooks_can_observe_and_replace():
    """hooks 回调可观察每个事件，也可返回替换事件修改内容"""
    seen: list[str] = []

    def replace_final(event):
        if isinstance(event, FinalAnswerEvent):
            return FinalAnswerEvent(iteration=event.iteration, content="被回调替换的答案")
        seen.append(event.type)
        return None

    agent = ReactAgent(
        provider=FakeProvider([ModelResponse(content="你好！")]), hooks=[replace_final]
    )

    assert agent.run("你好") == "被回调替换的答案"
    # 回调观察到全部非收敛事件，替换事件未再进入回调
    assert seen == ["step_start", "model_message"]


def test_run_stream_emits_tool_events():
    """run_stream 内部同样经事件流执行，工具调用事件可被回调观察"""
    tool_events: list[str] = []

    def collect(event):
        if event.type in ("tool_call_start", "tool_call_end"):
            tool_events.append(event.type)
        return None

    tool_response = ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="get_current_time", arguments="{}")],
    )
    agent = ReactAgent(
        provider=FakeStreamingProvider([tool_response, ModelResponse(content="现在是中午。")]),
        hooks=[collect],
    )

    assert "".join(agent.run_stream("现在几点了？")) == "现在是中午。"
    assert tool_events == ["tool_call_start", "tool_call_end"]


class PlanStreamProvider(ModelProvider):
    """规划器/整合器走非流式 chat、执行器走流式 chat_stream 的测试用提供者"""

    def __init__(self, responses: list[ModelResponse], chunk_size: int = 2) -> None:
        self.responses = responses
        self.chunk_size = chunk_size
        self.calls = 0

    def chat(self, messages, tools=None, response_format=None) -> ModelResponse:
        response = self.responses[self.calls]
        self.calls += 1
        return response

    def chat_stream(self, messages, tools=None):
        response = self.responses[self.calls]
        self.calls += 1
        content = response.content or ""
        for i in range(0, len(content), self.chunk_size):
            yield StreamChunk(delta=content[i : i + self.chunk_size])
        yield StreamChunk(response=response)


def test_plan_execute_runs_steps_and_synthesizes():
    """多步骤任务：规划 → 逐步执行 → 整合出最终答案"""
    provider = FakeProvider(
        [
            ModelResponse(content='["查询当前时间", "生成问候语"]'),
            ModelResponse(content="现在是中午。"),
            ModelResponse(content="你好，注意休息。"),
            ModelResponse(content="现在是中午，祝你午后愉快。"),
        ]
    )
    agent = PlanExecuteAgent(provider=provider)

    assert agent.run("生成问候") == "现在是中午，祝你午后愉快。"
    # 规划器 + 两步执行器 + 整合器共 4 次模型调用
    assert provider.calls == 4


def test_plan_execute_falls_back_to_single_step_on_parse_failure():
    """规划器输出无法解析时，退化为单步骤直接执行原任务"""
    provider = FakeProvider(
        [
            ModelResponse(content="抱歉，我直接回答：你好！"),
            ModelResponse(content="直接回答的结果"),
        ]
    )
    agent = PlanExecuteAgent(provider=provider)

    assert agent.run("你好") == "直接回答的结果"
    # 规划（失败）+ 单步骤执行，共 2 次调用
    assert provider.calls == 2


def test_plan_execute_handles_markdown_fenced_plan():
    """规划器输出带 markdown 代码围栏时，仍能正确解析步骤"""
    provider = FakeProvider(
        [
            ModelResponse(content='```json\n["步骤一"]\n```'),
            ModelResponse(content="结果一"),
        ]
    )
    agent = PlanExecuteAgent(provider=provider)

    assert agent.run("任务") == "结果一"
    assert provider.calls == 2


def test_plan_execute_truncates_plan_to_max_steps():
    """规划步骤超出 max_steps 时截断，截断后单步骤不触发整合器"""
    provider = FakeProvider(
        [
            ModelResponse(content='["步骤一", "步骤二", "步骤三"]'),
            ModelResponse(content="结果一"),
        ]
    )
    agent = PlanExecuteAgent(provider=provider, max_steps=1)

    assert agent.run("任务") == "结果一"
    # 规划 + 1 步执行（截断到 max_steps），共 2 次调用
    assert provider.calls == 2


def test_plan_execute_events_sequence_and_seq_monotonic():
    """run_events 事件序列：规划事件 + 执行器事件 + 最终答案，seq 全局递增"""
    provider = FakeProvider(
        [
            ModelResponse(content='["步骤一", "步骤二"]'),
            ModelResponse(content="结果一"),
            ModelResponse(content="结果二"),
            ModelResponse(content="最终答案"),
        ]
    )
    agent = PlanExecuteAgent(provider=provider)

    events = list(agent.run_events("任务"))
    types = [e.type for e in events]

    assert types == [
        "plan_generated",
        "plan_step_start",
        "step_start",  # 执行器 ReAct 循环
        "model_message",
        "final_answer",  # 步骤一的结果
        "plan_step_end",
        "plan_step_start",
        "step_start",
        "model_message",
        "final_answer",  # 步骤二的结果
        "plan_step_end",
        "final_answer",  # 整合后的最终答案
    ]
    # 执行器事件经转发重新编号，整个事件流 seq 全局递增
    assert [e.seq for e in events] == list(range(len(events)))
    # 规划事件携带步骤清单，最终事件为整合答案
    assert isinstance(events[0], PlanGeneratedEvent)
    assert events[0].steps == ["步骤一", "步骤二"]
    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].content == "最终答案"


def test_plan_execute_hooks_observe_plan_and_executor_events():
    """hooks 经 add_hook 同步注册，可观察到规划事件与执行器工具事件"""
    seen: list[str] = []

    def collect(event):
        seen.append(event.type)
        return None

    planner = ModelResponse(content='["查询时间"]')
    tool_response = ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="get_current_time", arguments="{}")],
    )
    provider = FakeProvider([planner, tool_response, ModelResponse(content="现在是中午。")])
    agent = PlanExecuteAgent(provider=provider, hooks=[collect])

    assert agent.run("任务") == "现在是中午。"
    assert "plan_generated" in seen
    assert "tool_call_start" in seen
    assert "tool_call_end" in seen


def test_plan_execute_propagates_provider_error():
    """规划器调用失败时，ProviderError 应向上传播（不静默吞错）"""

    class RaisingProvider(ModelProvider):
        def chat(self, messages, tools=None, response_format=None):
            raise ProviderError("模拟服务不可用")

    agent = PlanExecuteAgent(provider=RaisingProvider())

    with pytest.raises(ProviderError):
        agent.run("任务")


def test_plan_execute_run_stream_yields_step_text():
    """流式模式：单步骤任务的文本增量经事件流逐段流出"""
    provider = PlanStreamProvider(
        [ModelResponse(content='["直接回答"]'), ModelResponse(content="你好，我是助手。")]
    )
    agent = PlanExecuteAgent(provider=provider)

    assert "".join(agent.run_stream("你好")) == "你好，我是助手。"


def test_plan_execute_run_stream_synthesizes_multi_step():
    """流式模式：多步骤任务的文本 = 各步流式文本 + 整合答案增量"""
    provider = PlanStreamProvider(
        [
            ModelResponse(content='["步骤一", "步骤二"]'),
            ModelResponse(content="结果一"),
            ModelResponse(content="结果二"),
            ModelResponse(content="整合答案"),
        ]
    )
    agent = PlanExecuteAgent(provider=provider)

    assert "".join(agent.run_stream("任务")) == "结果一结果二整合答案"


# ------------------- 模型调用重试（max_retries，开发方向 §4.3） -------------------


class FlakyProvider(ModelProvider):
    """前 fail_times 次调用抛出指定异常，之后返回预设响应的测试用提供者"""

    def __init__(self, responses, error, fail_times):
        self.responses = responses
        self.error = error
        self.fail_times = fail_times
        self.calls = 0

    def chat(self, messages, tools=None, response_format=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return self.responses[self.calls - 1 - self.fail_times]


@pytest.fixture
def no_sleep(monkeypatch):
    """屏蔽重试退避等待，避免测试变慢"""
    sleeps = []
    monkeypatch.setattr("gearlink.core.agent.time.sleep", lambda s: sleeps.append(s))
    return sleeps


def test_max_retries_recovers_from_retryable_error(no_sleep):
    provider = FlakyProvider(
        [ModelResponse(content="恢复了")],
        ProviderError("限流", retryable=True),
        fail_times=2,
    )
    agent = ReactAgent(provider=provider, max_retries=2)

    assert agent.run("你好") == "恢复了"
    assert provider.calls == 3
    assert no_sleep == [1.0, 2.0]  # 指数退避


def test_max_retries_default_zero_no_retry():
    provider = FlakyProvider(
        [ModelResponse(content="不会到达")],
        ProviderError("限流", retryable=True),
        fail_times=1,
    )
    agent = ReactAgent(provider=provider)  # 默认 max_retries=0，等价现状

    with pytest.raises(ProviderError):
        agent.run("你好")
    assert provider.calls == 1


def test_non_retryable_error_not_retried(no_sleep):
    provider = FlakyProvider(
        [ModelResponse(content="不会到达")],
        ProviderError("鉴权失败", retryable=False),
        fail_times=1,
    )
    agent = ReactAgent(provider=provider, max_retries=3)

    with pytest.raises(ProviderError, match="鉴权失败"):
        agent.run("你好")
    assert provider.calls == 1
    assert no_sleep == []


def test_retry_exhausted_raises_last_error(no_sleep):
    provider = FlakyProvider(
        [ModelResponse(content="不会到达")],
        ProviderError("持续限流", retryable=True),
        fail_times=5,
    )
    agent = ReactAgent(provider=provider, max_retries=2)

    with pytest.raises(ProviderError, match="持续限流"):
        agent.run("你好")
    assert provider.calls == 3  # 首次 + 2 次重试


# ------------------- 工具白名单（tools，开发方向 §4.4） -------------------


class ToolsCapturingProvider(ModelProvider):
    """记录每次调用收到的 tools 参数并返回预设响应"""

    def __init__(self, response):
        self.response = response
        self.received_tools = []

    def chat(self, messages, tools=None, response_format=None):
        self.received_tools.append(tools)
        return self.response


def test_tools_whitelist_filters_schemas():
    from gearlink.core.tool import TOOL_SCHEMAS

    provider = ToolsCapturingProvider(ModelResponse(content="答"))
    agent = ReactAgent(provider=provider, tools=["get_current_time"])
    agent.run("几点了")

    names = [schema["function"]["name"] for schema in provider.received_tools[0]]
    assert names == ["get_current_time"]
    assert len(TOOL_SCHEMAS) > 1  # 全量注册表不受影响


def test_tools_none_passes_all_schemas():
    from gearlink.core.tool import TOOL_SCHEMAS

    provider = ToolsCapturingProvider(ModelResponse(content="答"))
    agent = ReactAgent(provider=provider)  # 默认 None = 全量（现状）
    agent.run("几点了")

    assert provider.received_tools[0] == TOOL_SCHEMAS


# ------------------- 并行工具执行（parallel_tool_calls，开发方向 §4.4） -------------------


def _register_parallel_tools():
    from gearlink.core.tool import TOOL_REGISTRY, register_tool

    schema = {
        "description": "测试用工具",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    if "parallel_a_test" not in TOOL_REGISTRY:
        register_tool("parallel_a_test", lambda: "A", schema)
        register_tool("parallel_b_test", lambda: "B", schema)


def test_parallel_tool_calls_same_result_as_serial():
    _register_parallel_tools()
    tool_response = ModelResponse(
        tool_calls=[
            ToolCall(id="call_a", name="parallel_a_test", arguments="{}"),
            ToolCall(id="call_b", name="parallel_b_test", arguments="{}"),
        ]
    )
    final_response = ModelResponse(content="完成")

    serial_agent = ReactAgent(provider=FakeProvider([tool_response, final_response]))
    parallel_agent = ReactAgent(
        provider=FakeProvider([tool_response, final_response]), parallel_tool_calls=True
    )

    assert serial_agent.run("任务") == parallel_agent.run("任务") == "完成"

    # 并行分支的结果写回记忆顺序与串行一致（assistant 消息后按调用顺序）
    serial_tool_msgs = [m for m in serial_agent.memory.get_messages() if m.get("role") == "tool"]
    parallel_tool_msgs = [
        m for m in parallel_agent.memory.get_messages() if m.get("role") == "tool"
    ]
    assert parallel_tool_msgs == serial_tool_msgs
    assert [m["tool_call_id"] for m in parallel_tool_msgs] == ["call_a", "call_b"]


def test_parallel_tool_calls_event_order_deterministic():
    _register_parallel_tools()
    tool_response = ModelResponse(
        tool_calls=[
            ToolCall(id="call_a", name="parallel_a_test", arguments="{}"),
            ToolCall(id="call_b", name="parallel_b_test", arguments="{}"),
        ]
    )
    agent = ReactAgent(
        provider=FakeProvider([tool_response, ModelResponse(content="完成")]),
        parallel_tool_calls=True,
    )

    types_seq = [e.type for e in agent.run_events("任务")]
    assert types_seq == [
        "step_start",
        "model_message",
        "tool_call_start",
        "tool_call_start",
        "tool_call_end",
        "tool_call_end",
        "step_start",
        "model_message",
        "final_answer",
    ]


# ------------------- ToolRegistry 实例隔离（开发方向 §6.4） -------------------


def test_react_agent_with_tool_registry_isolation():
    """两个 Agent 持有不同的 ToolRegistry 实例，各自只能调用自己注册的工具。"""
    registry_a = ToolRegistry()
    registry_b = ToolRegistry()

    schema = {
        "description": "隔离工具",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    registry_a.register_tool("iso_tool_a", lambda: "from_a", schema)
    registry_b.register_tool("iso_tool_b", lambda: "from_b", schema)

    # Agent A 调用自己的工具成功
    tool_response_a = ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="iso_tool_a", arguments="{}")],
    )
    final_response_a = ModelResponse(content="A 完成")
    agent_a = ReactAgent(
        provider=FakeProvider([tool_response_a, final_response_a]),
        tool_registry=registry_a,
    )
    assert agent_a.run("任务") == "A 完成"

    # Agent B 调用自己的工具成功
    tool_response_b = ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="iso_tool_b", arguments="{}")],
    )
    final_response_b = ModelResponse(content="B 完成")
    agent_b = ReactAgent(
        provider=FakeProvider([tool_response_b, final_response_b]),
        tool_registry=registry_b,
    )
    assert agent_b.run("任务") == "B 完成"

    # 验证隔离：Agent A 无法调用 B 的工具（工具调用失败为可恢复信号）
    wrong_response = ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="iso_tool_b", arguments="{}")],
    )
    final_wrong = ModelResponse(content="工具不可用")
    agent_a_wrong = ReactAgent(
        provider=FakeProvider([wrong_response, final_wrong]),
        tool_registry=registry_a,
    )
    assert agent_a_wrong.run("任务") == "工具不可用"
    tool_msgs = [m for m in agent_a_wrong.memory.get_messages() if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "未知工具" in tool_msgs[0]["content"]


# ------------------- _extract_json 工具函数（开发方向 §6.5） -------------------


def test_extract_json_strips_markdown_fences():
    """_extract_json 剥离 markdown 代码围栏后解析 JSON。"""
    text = '```json\n["步骤一", "步骤二"]\n```'
    assert _extract_json(text) == ["步骤一", "步骤二"]


def test_extract_json_extracts_from_surrounding_text():
    """_extract_json 从包含说明文字的文本中提取 JSON 数组。"""
    text = '以下是步骤：\n["查询", "执行"]\n以上是步骤列表。'
    assert _extract_json(text) == ["查询", "执行"]


def test_extract_json_returns_none_on_invalid():
    """_extract_json 对非 JSON 文本返回 None。"""
    assert _extract_json("这只是普通文本，没有 JSON") is None
