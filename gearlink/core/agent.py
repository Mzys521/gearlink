"""Agent 编排策略实现：Agent 抽象契约 + ReAct 与规划-执行两种编排策略。

装配期注入 `SkillRegistry` 后，ReAct 执行器会在默认系统提示中动态列出可用技能
（L1 元数据），模型可通过已注册的 `load_skill` 工具按名获取完整技能指令（L2）
并严格遵循执行；注册表经 `core.tool.set_skill_registry` 登记，供 `load_skill`
工具解析。

事件流：`Agent.run_events` 是各编排循环的唯一实现，逐步产出 `core.events.AgentEvent`
子类事件；`run` / `run_stream` 是事件流的通用消费方式（基类统一实现）。事件回调
经构造参数 `hooks` 或 `add_hook` 注入，在每个事件产出时获得观察/干预机会
（on_step 语义）。

策略实现：
- `ReactAgent`：ReAct 循环（推理→行动→观察），单 Agent 单轮往复；
- `PlanExecuteAgent`：规划-执行（先规划后执行），内部复用 `ReactAgent` 作为步骤执行器。
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from gearlink.core.events import (
    AgentEvent,
    FinalAnswerEvent,
    HookFn,
    LoopAbortEvent,
    ModelMessageEvent,
    PlanGeneratedEvent,
    PlanStepEndEvent,
    PlanStepStartEvent,
    StepStartEvent,
    TextDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from gearlink.core.memory import Memory, MemoryManager, ShortTermMemory
from gearlink.core.tool import ToolRegistry, _default_registry, set_skill_registry
from gearlink.exceptions import GearLinkError, ProviderError
from gearlink.providers.base import ModelProvider, ModelResponse, ToolCall
from gearlink.skills import SkillRegistry
from gearlink.utils.token_count import estimate_tokens

# ------------------- 共享常量 -------------------

#: 内置系统提示：引导 Agent 优先使用工具回答实时问题
SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "当需要实时信息（如当前时间）时，请调用可用的工具，而不是凭空回答。"
)

#: 注入技能注册表后追加的提示：告知模型可用技能列表的用途与加载方式
_SKILL_HINT_PROMPT = (
    "当遇到需要特定专业知识（如代码审查、文档编写等）的任务时，你可以调用 `load_skill` "
    "工具加载相应的技能指令，然后严格遵循这些指令执行。\n"
    "当前可用技能："
)

# ReAct 循环最大迭代次数，防止死循环
MAX_ITERATIONS = 10

#: 工具结果写入记忆前的 token 估算上限，超出时截断防止撑爆上下文
MAX_TOOL_RESULT_TOKENS = 2000

#: 达到最大迭代次数时的兜底文案（run / run_events / run_stream 共用）
_MAX_ITERATIONS_FALLBACK = "已达到最大推理轮数，无法得出最终答案。"

#: 模型调用重试的指数退避基数（秒）：第 N 次重试前等待 base * 2**N 秒
_RETRY_BACKOFF_BASE = 1.0

#: 模块级日志器：记录 ReAct 循环中的工具调用等过程信息
logger = logging.getLogger(__name__)


class Agent(ABC):
    """Agent 编排策略抽象：统一 Agent 的对外调用契约。

    所有 Agent 实现共享同一事件流契约：`run_events` 是各自编排循环的唯一实现
    （子类必须提供），`run` / `run_stream` 是事件流的两种通用消费方式（基类实现）；
    事件回调经构造参数 `hooks` 或 `add_hook` 注入，在每个事件产出时获得
    观察/干预机会（on_step 语义）。

    子类实现要求：
    - `run_events` 产出的事件须携带全局递增的 `seq`（经 `_emit` 分配）；
    - 循环收敛时产出 `FinalAnswerEvent`，循环中止时产出 `LoopAbortEvent`；
    - stream=True 时文本类内容以 `TextDeltaEvent` 产出。
    """

    _hooks: list[HookFn]

    def __init__(self, provider: ModelProvider, hooks: list[HookFn] | None = None) -> None:
        """初始化 Agent 公共状态。

        Args:
            provider: 模型提供者实例。
            hooks: 事件回调列表（on_step 语义）；每个事件产出时按序调用，
                可返回替换事件（None 表示不修改）。默认空列表，等价于无回调。
        """
        self.provider = provider
        self._hooks: list[HookFn] = list(hooks or [])

    def add_hook(self, hook: HookFn) -> None:
        """注册一个事件回调（on_step 语义）。

        回调接收每个已产出的事件，可返回替换事件；返回 None 表示不修改。
        回调应保持观察/干预语义：可记录日志、校验内容，或替换事件内容。
        命名回调（如 on_tool_call）可基于本回调按事件类型过滤薄封装。

        Args:
            hook: 事件回调函数。
        """
        self._hooks.append(hook)

    def _emit(self, event: AgentEvent, seq: int) -> AgentEvent:
        """对事件施加全部回调并分配序号/时间戳后返回（产出点由子类 yield）。

        Args:
            event: 待产出的事件。
            seq: 该事件应分配的全局序号。

        Returns:
            经回调处理（可能被替换）后的事件。
        """
        event.seq = seq
        event.timestamp = time.time()
        for hook in self._hooks:
            result = hook(event)
            if result is not None:
                event = result
        # 回调可能替换事件对象，替换后重新登记序号与时间戳
        event.seq = seq
        event.timestamp = time.time()
        return event

    def _emit_event(self, event: AgentEvent, seq: int) -> Generator[AgentEvent, None, int]:
        """经回调处理后产出单个事件，并返回下一可用序号（供 yield from 使用）。

        Args:
            event: 待产出的事件。
            seq: 该事件应分配的全局序号。

        Yields:
            经 _emit 处理（可能被回调替换）后的事件。

        Returns:
            下一可用序号（产出事件的 seq + 1，替换事件同样适用）。
        """
        event = self._emit(event, seq)
        yield event
        return event.seq + 1

    def run(self, user_input: str) -> str:
        """执行一次用户请求，运行本 Agent 的编排循环直到得到最终答案。

        实现为 `run_events` 事件流的消费者：收敛事件（FinalAnswerEvent）给出答案，
        循环中止事件（LoopAbortEvent）给出兜底文案。

        Args:
            user_input: 用户的输入文本。

        Returns:
            Agent 给出的最终答案文本；未收敛时返回兜底文案。

        Raises:
            ProviderError: 模型服务调用失败，由调用方决定重试策略。
        """
        answer: str | None = None
        for event in self.run_events(user_input):
            if isinstance(event, FinalAnswerEvent):
                answer = event.content
        return answer if answer is not None else _MAX_ITERATIONS_FALLBACK

    def run_stream(self, user_input: str) -> Iterator[str]:
        """执行一次用户请求，以流式逐步产出模型文本。

        与 `run` 共享编排循环（run_events(stream=True)），区别在于只转发
        TextDeltaEvent 的文本增量；工具调用阶段自动执行并继续循环；循环中止时
        产出兜底文案。

        Args:
            user_input: 用户的输入文本。

        Yields:
            模型输出的文本增量片段；全部片段拼接即为最终答案。

        Raises:
            ProviderError: 模型服务调用失败，由调用方决定重试策略。
            GearLinkError: 提供者的流式响应缺少携带完整响应的终止事件。
        """
        for event in self.run_events(user_input, stream=True):
            if isinstance(event, TextDeltaEvent):
                yield event.delta
            elif isinstance(event, LoopAbortEvent):
                yield event.reason

    @abstractmethod
    def run_events(self, user_input: str, *, stream: bool = False) -> Iterator[AgentEvent]:
        """以事件流方式执行一次用户请求（编排循环的唯一实现）。

        Args:
            user_input: 用户的输入文本。
            stream: True 时使用 provider 的流式接口，文本以 TextDeltaEvent
                逐片段产出；False 时使用 chat，整段响应以 ModelMessageEvent 产出。

        Yields:
            AgentEvent: 编排循环过程中的事件序列。

        Raises:
            ProviderError: 模型服务调用失败，由调用方决定重试策略。
        """


# ------------------- ReAct 策略 -------------------


def _build_system_prompt(skill_registry: SkillRegistry | None = None) -> str:
    """组装系统提示：基础提示 + 可用技能列表（仅当注入非空注册表时）。

    Args:
        skill_registry: 技能注册表；None 或不含技能时返回基础提示。

    Returns:
        组装后的系统提示文本。
    """
    skills = skill_registry.list_all() if skill_registry is not None else []
    if not skills:
        return SYSTEM_PROMPT
    skill_lines = "\n".join(f"- {skill.name}: {skill.description}" for skill in skills)
    return f"{SYSTEM_PROMPT}\n{_SKILL_HINT_PROMPT}\n{skill_lines}"


class ReactAgent(Agent):
    """ReAct Agent：推理(Reason) -> 行动(Act) -> 观察(Observe) 循环"""

    def __init__(
        self,
        provider: ModelProvider,
        memory: Memory | MemoryManager | None = None,
        retrieve_every_iteration: bool = False,
        skill_registry: SkillRegistry | None = None,
        hooks: list[HookFn] | None = None,
        max_retries: int = 0,
        tools: list[str] | None = None,
        parallel_tool_calls: bool = False,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        """初始化 Agent。

        Args:
            provider: 模型提供者实例。
            memory: 记忆实现；None 时默认使用 ShortTermMemory(max_message=20)
                并附加内置 SYSTEM_PROMPT（注入技能时为含技能列表的组装提示）。
                传入 MemoryManager 时启用长期记忆检索注入。
            retrieve_every_iteration: 是否每轮 ReAct 迭代都携带查询注入长期记忆；
                False 表示仅首轮注入（现状，后续轮次上下文已在短期记忆中）。
                开启时依赖 build_context 的短期窗口去重避免重复注入。
            skill_registry: 技能注册表；非 None 时登记给 load_skill 工具使用，
                且仅在 memory 为 None 时把可用技能列表拼入默认系统提示。
                自行注入 memory 时须自行添加所需的系统提示消息。
            hooks: 事件回调列表（on_step 语义）；每个事件产出时按序调用，
                可返回替换事件（None 表示不修改）。默认空列表，等价于无回调。
            max_retries: 模型调用失败时的最大重试次数；0 表示不重试（现状，
                开发方向 §4.3）。仅对标记 retryable 的 ProviderError（网络/限流
                等瞬时故障）重试，鉴权等确定性错误直接抛出；指数退避等待。
            tools: 工具白名单（工具名列表）；None 表示全量可用（现状）。指定后
                仅把名单内工具的 schema 传给模型，降低误选与 token 开销
                （开发方向 §4.4）；工具调度本身不受限制。
            parallel_tool_calls: 同一轮多个工具调用是否并行执行；False 表示串行
                （现状）。开启时用线程池并行，事件产出顺序与结果写回记忆的
                顺序保持不变（仅要求工具自身线程安全）。
            tool_registry: 工具注册表实例（开发方向 §6.4）；None 时使用模块级
                默认注册表（向后兼容）。注入后 Agent 持有隔离的工具集与技能集，
                同一进程内多个 Agent 互不干扰。
        """
        super().__init__(provider=provider, hooks=hooks)
        self.retrieve_every_iteration = retrieve_every_iteration
        self.max_retries = max_retries
        self.tools = tools
        self.parallel_tool_calls = parallel_tool_calls
        self._tool_registry = tool_registry or _default_registry
        if skill_registry is not None:
            # 登记到本 Agent 的工具注册表（实例化路径），供 load_skill 工具解析
            self._tool_registry.set_skill_registry(skill_registry)
            # 同时登记到模块级默认注册表（向后兼容：未注入 tool_registry 时
            # load_skill 回退到全局 _skill_registry）
            if tool_registry is None:
                set_skill_registry(skill_registry)
        if memory is None:
            memory = ShortTermMemory(max_message=20)
            memory.add_message({"role": "system", "content": _build_system_prompt(skill_registry)})
        self.memory = memory
        logger.debug(
            "ReactAgent 初始化: max_retries=%d, parallel_tool_calls=%s, tools=%s, memory=%s",
            self.max_retries,
            self.parallel_tool_calls,
            self.tools,
            type(memory).__name__,
        )

    def _build_messages(self, query: str | None = None) -> list[dict[str, Any]]:
        """基于记忆组装本轮请求消息。

        统一走 `build_context`（开发方向 §6.4）：`Memory` 基类默认返回 `get_messages()`，
        `MemoryManager` 覆写为检索注入 + 预算裁剪，不再对 `MemoryManager` 做 `isinstance` 特判。

        Args:
            query: 语义检索查询；仅对支持检索注入的记忆生效。

        Returns:
            OpenAI 消息格式的请求消息列表。
        """
        return self.memory.build_context(query or "")

    def _available_tool_schemas(self) -> list[dict[str, Any]]:
        """返回本轮可用的工具 schema（按白名单过滤，开发方向 §4.4）。

        Returns:
            本 Agent 工具注册表中的全部 schema；配置了 tools 白名单时仅返回名单内的条目。
        """
        return self._tool_registry.get_schemas(self.tools)

    def run_events(self, user_input: str, *, stream: bool = False) -> Iterator[AgentEvent]:
        """执行一次用户请求，逐步产出 ReAct 循环的事件（循环的唯一实现）。

        事件产出点与循环步骤一一对应：StepStartEvent（每轮开始）、TextDeltaEvent
        （仅 stream=True，流式文本增量）、ModelMessageEvent（模型完整响应）、
        ToolCallStartEvent / ToolCallEndEvent（每个工具执行前后）、
        FinalAnswerEvent（循环收敛）、LoopAbortEvent（达到最大轮数中止）。
        消费方关闭生成器可中止循环（自带取消能力）。

        Args:
            user_input: 用户的输入文本。
            stream: True 时使用 provider 的流式接口 chat_stream，文本以
                TextDeltaEvent 逐片段产出；False 时使用 chat，整段响应以
                ModelMessageEvent 产出。

        Yields:
            AgentEvent: 循环过程中的事件序列。

        Raises:
            ProviderError: 模型服务调用失败，由调用方决定重试策略。
            GearLinkError: 提供者的流式响应缺少携带完整响应的终止事件。
        """
        # 将用户输入添加到记忆
        self.memory.add_message({"role": "user", "content": user_input})
        logger.info("ReAct 循环开始: stream=%s", stream)

        seq = 0
        for iteration in range(MAX_ITERATIONS):
            seq = yield from self._emit_event(StepStartEvent(iteration=iteration), seq)

            # 推理：默认仅首轮携带查询注入长期记忆；开启 retrieve_every_iteration 时每轮都注入
            query = user_input if (iteration == 0 or self.retrieve_every_iteration) else None
            messages = self._build_messages(query)
            logger.debug("第 %d 轮: 组装 %d 条请求消息", iteration, len(messages))

            response, seq = yield from self._call_model(
                messages, iteration=iteration, stream=stream, seq=seq
            )

            seq = yield from self._emit_event(
                ModelMessageEvent(
                    iteration=iteration,
                    content=response.content,
                    tool_calls=response.tool_calls,
                    usage=response.usage,
                ),
                seq,
            )

            # 如果模型没有调用工具，说明已经生成了最终答案，直接返回
            if not response.tool_calls:
                self.memory.add_message({"role": "assistant", "content": response.content})
                logger.info(
                    "ReAct 收敛: 第 %d 轮产出最终答案（%d 字）",
                    iteration,
                    len(response.content or ""),
                )
                yield from self._emit_event(
                    FinalAnswerEvent(iteration=iteration, content=response.content), seq
                )
                return

            # 行动 + 观察：记录工具调用并执行，将结果写回记忆后继续循环
            seq = yield from self._execute_tool_calls(response, iteration=iteration, seq=seq)

        # 达到最大迭代次数仍未得到答案
        logger.warning("ReAct 达到最大迭代次数 %d，中止循环", MAX_ITERATIONS)
        yield from self._emit_event(LoopAbortEvent(reason=_MAX_ITERATIONS_FALLBACK), seq)

    def _call_model(
        self,
        messages: list[dict[str, Any]],
        *,
        iteration: int,
        stream: bool,
        seq: int,
    ) -> Generator[AgentEvent, None, tuple[ModelResponse, int]]:
        """调用模型（含可选重试），流式时同步转出文本增量事件。

        重试策略（开发方向 §4.3）：仅对标记 retryable 的 ProviderError（网络/
        限流等瞬时故障）重试，指数退避等待；鉴权等确定性错误与已产出文本
        增量的流式调用不重试，直接向上抛出。

        Args:
            messages: 本轮请求消息。
            iteration: 当前 ReAct 轮次（事件携带）。
            stream: True 时使用 chat_stream 并产出 TextDeltaEvent。
            seq: 事件流当前序号。

        Yields:
            TextDeltaEvent: 仅 stream=True 时的文本增量。

        Returns:
            (模型响应, 更新后的事件序号) 二元组。

        Raises:
            ProviderError: 不可重试或重试耗尽后的模型调用失败。
            GearLinkError: 流式响应缺少携带完整响应的终止事件。
        """
        attempt = 0
        tool_schemas = self._available_tool_schemas()
        logger.debug(
            "调用模型: iteration=%d, stream=%s, messages=%d, tools=%d",
            iteration,
            stream,
            len(messages),
            len(tool_schemas),
        )
        while True:
            emitted_delta = False
            try:
                if stream:
                    # 流式推理：实时转出文本增量，终止事件携带累积后的完整响应
                    response: ModelResponse | None = None
                    for chunk in self.provider.chat_stream(messages=messages, tools=tool_schemas):
                        if chunk.delta:
                            emitted_delta = True
                            seq = yield from self._emit_event(
                                TextDeltaEvent(delta=chunk.delta, iteration=iteration), seq
                            )
                        if chunk.response is not None:
                            response = chunk.response
                    if response is None:
                        raise GearLinkError("提供者的流式响应缺少终止事件（未携带完整 response）")
                    return response, seq
                return self.provider.chat(messages=messages, tools=tool_schemas), seq
            except ProviderError as e:
                # 仅重试可重试错误（网络/限流）；鉴权等确定性错误直接抛出；
                # 已产出文本增量的流式调用不重试（避免重复输出）
                if not e.retryable or attempt >= self.max_retries or (stream and emitted_delta):
                    raise
                delay = _RETRY_BACKOFF_BASE * 2**attempt
                logger.warning(
                    "[重试] 模型调用失败（%s），%.1f 秒后重试（%d/%d）",
                    e,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(delay)
                attempt += 1

    def _execute_tool_calls(
        self, response: ModelResponse, *, iteration: int, seq: int
    ) -> Generator[AgentEvent, None, int]:
        """记录助手的工具调用消息，执行各工具并将结果写回记忆（事件流工具段）。

        Args:
            response: 含工具调用请求的模型响应。
            iteration: 当前 ReAct 轮次。
            seq: 事件流当前序号。

        Yields:
            ToolCallStartEvent / ToolCallEndEvent：每个工具执行前后的事件。

        Returns:
            更新后的事件流序号（seq + 已产出事件数），供 `yield from` 接收。
        """
        # 行动：模型决定调用工具，将工具调用记录到记忆
        logger.debug(
            "执行工具调用: iteration=%d, 工具数=%d, 并行=%s",
            iteration,
            len(response.tool_calls),
            self.parallel_tool_calls and len(response.tool_calls) > 1,
        )
        self.memory.add_message(
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in response.tool_calls
                ],
            }
        )

        # 观察：执行工具调用并把结果写回记忆（串行或并行，见 parallel_tool_calls）
        if self.parallel_tool_calls and len(response.tool_calls) > 1:
            # 并行分支（开发方向 §4.4）：按序产出全部 start 事件 → 线程池并行执行
            # → 按原顺序产出 end 事件并写回记忆，保证事件与写回顺序确定不变
            for tool_call in response.tool_calls:
                seq = yield from self._emit_event(
                    ToolCallStartEvent(
                        iteration=iteration,
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        arguments=tool_call.arguments or "{}",
                    ),
                    seq,
                )
            with ThreadPoolExecutor(max_workers=len(response.tool_calls)) as pool:
                outcomes = list(pool.map(self._run_single_tool, response.tool_calls))
            for tool_call, outcome in zip(response.tool_calls, outcomes):
                seq = yield from self._emit_event(
                    ToolCallEndEvent(
                        iteration=iteration,
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        result=outcome[0],
                        truncated=outcome[1],
                        error=outcome[2],
                    ),
                    seq,
                )
                self._write_tool_result(tool_call.id, outcome[0])
            return seq

        # 串行分支（现状）：逐个执行，start/end 事件交替产出
        for tool_call in response.tool_calls:
            seq = yield from self._emit_event(
                ToolCallStartEvent(
                    iteration=iteration,
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    arguments=tool_call.arguments or "{}",
                ),
                seq,
            )

            result_text, truncated, error = self._run_single_tool(tool_call)

            seq = yield from self._emit_event(
                ToolCallEndEvent(
                    iteration=iteration,
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result=result_text,
                    truncated=truncated,
                    error=error,
                ),
                seq,
            )

            self._write_tool_result(tool_call.id, result_text)

        return seq

    def _run_single_tool(self, tool_call: ToolCall) -> tuple[str, bool, str | None]:
        """执行单个工具调用（含参数解析与错误兜底，可被线程池并行调用）。

        Args:
            tool_call: 模型返回的工具调用请求。

        Returns:
            (结果文本, 是否截断, 错误信息) 三元组；错误信息非 None 表示
            可恢复失败（结果已序列化，由调用方写回记忆交给模型处理）。
        """
        # 解析参数（若解析失败则使用空字典）
        try:
            arguments = json.loads(tool_call.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
            logger.debug("工具 %s 参数解析失败，使用空参数", tool_call.name)

        # 执行工具（经本 Agent 的工具注册表调度，含 contextvars 上下文设置）
        try:
            result = self._tool_registry.call_tool(tool_call.name, arguments)
            # 将结果转为 JSON 字符串，以便统一存储
            result_text = json.dumps(result, ensure_ascii=False)
            # 超出预算时按 4 字符/token 的启发式反推字符上限截断
            truncated = estimate_tokens(result_text) > MAX_TOOL_RESULT_TOKENS
            if truncated:
                result_text = (
                    result_text[: MAX_TOOL_RESULT_TOKENS * 4] + "\n...(工具结果过长，已截断)"
                )
            error: str | None = None
        except GearLinkError as e:
            # 工具执行失败是可恢复信号：将错误信息作为结果写回，让模型自行处理
            result_text = f"工具调用失败: {e}"
            error = str(e)
            truncated = False

        logger.info("[工具调用] %s(%s) -> %s", tool_call.name, arguments, result_text)
        return result_text, truncated, error

    def _write_tool_result(self, tool_call_id: str, result_text: str) -> None:
        """将工具执行结果作为 tool 角色消息存入记忆。"""
        self.memory.add_message(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_text,
            }
        )


# ------------------- 规划-执行策略 -------------------

#: 规划器系统提示：要求把任务分解为 JSON 步骤数组
_PLANNER_SYSTEM_PROMPT = (
    "你是任务规划器。请把用户的任务分解为若干个有序、可独立执行的步骤。"
    "每个步骤是一条给执行 Agent 的清晰、自包含的中文指令，不要带序号前缀。"
    "只输出一个 JSON 字符串数组，不要输出任何其他内容。"
)

#: 整合器系统提示：把各步骤执行结果整合为直接可用的最终回答
_SYNTHESIZER_SYSTEM_PROMPT = (
    "你是结果整合器。基于各步骤的执行结果，为用户的任务给出最终回答。"
    "回答应直接、完整、简洁，不要提及步骤或执行结果等过程性内容。"
)


def _extract_json(text: str) -> Any | None:
    """从可能包含 markdown 围栏或说明文字的文本中提取 JSON（开发方向 §6.5）。

    解析顺序：
    1. 剥离 markdown 代码围栏后直接 `json.loads`；
    2. 从文本中提取最外层 ``[...]`` 或 ``{...}`` 片段后解析。

    Args:
        text: 模型原始输出文本。

    Returns:
        解析后的 Python 对象；无法解析时返回 None。
    """
    cleaned = text.strip()
    # 1. 剥离 markdown 代码围栏
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # 2. 提取最外层 [ ] 或 { } 片段
    for open_char, close_char in (("[", "]"), ("{", "}")):
        start = cleaned.find(open_char)
        end = cleaned.rfind(close_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _parse_steps(text: str) -> list[str] | None:
    """解析规划器输出的步骤列表（JSON 字符串数组）。

    容错：经 `_extract_json` 剥离围栏与提取 JSON 片段后解析；
    结果须为非空字符串列表。

    Args:
        text: 规划器原始输出文本。

    Returns:
        解析出的步骤列表；格式非法时返回 None（由调用方退化为单步骤执行）。
    """
    steps = _extract_json(text)
    if steps is None:
        return None
    if not isinstance(steps, list) or not steps:
        return None
    if not all(isinstance(step, str) and step.strip() for step in steps):
        return None
    return [step.strip() for step in steps]


class PlanExecuteAgent(Agent):
    """规划-执行（Plan-and-Execute）编排策略：先规划后执行。

    两阶段流程：规划器（纯 LLM 调用，不启用工具）把用户任务分解为有序步骤清单
    （JSON 数组，解析失败退化为单步骤直接执行原任务）；随后对每个步骤依次运行
    内部 `ReactAgent` 执行器子循环（复用工具 / 记忆 / 技能能力），收集各步结果；
    多步骤时最后经一次整合对话生成最终答案，单步骤时直接透传步骤结果。

    事件流：`PlanGeneratedEvent`（步骤清单）→ 每步 `PlanStepStartEvent` →
    执行器事件（StepStart / ModelMessage / ToolCall / FinalAnswer 等，经转发
    重新编号保证 `seq` 全局递增）→ `PlanStepEndEvent`；最后产出
    `FinalAnswerEvent`（多步骤且 stream=True 时，先产出合成文本的 TextDeltaEvent）。
    """

    def __init__(
        self,
        provider: ModelProvider,
        memory: Memory | MemoryManager | None = None,
        retrieve_every_iteration: bool = False,
        skill_registry: SkillRegistry | None = None,
        max_steps: int = 5,
        hooks: list[HookFn] | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        """初始化规划-执行 Agent。

        Args:
            provider: 模型提供者实例（规划器、执行器与整合器共用）。
            memory: 记忆实现；None 时执行器默认使用 ShortTermMemory(max_message=20)
                并附加内置 SYSTEM_PROMPT。执行器各步骤共享该记忆。
            retrieve_every_iteration: 是否每轮 ReAct 迭代都携带查询注入长期记忆，
                透传给内部执行器。
            skill_registry: 技能注册表；非 None 时登记给 load_skill 工具使用，
                并透传给内部执行器。
            max_steps: 规划步骤数量上限，规划结果超出时截断。默认 5。
            hooks: 事件回调列表（on_step 语义）；同时作用于本 Agent 与内部执行器。
            tool_registry: 工具注册表实例（开发方向 §6.4）；None 时使用模块级
                默认注册表（向后兼容）。透传给内部执行器。
        """
        super().__init__(provider=provider, hooks=hooks)
        self.max_steps = max_steps
        # 执行器复用 ReAct 子循环
        self.executor = ReactAgent(
            provider=provider,
            memory=memory,
            retrieve_every_iteration=retrieve_every_iteration,
            skill_registry=skill_registry,
            tool_registry=tool_registry,
        )
        # 与执行器共享同一回调列表：基类 add_hook 对二者事件流同时生效
        self.executor._hooks = self._hooks

    def run_events(self, user_input: str, *, stream: bool = False) -> Iterator[AgentEvent]:
        """执行一次用户请求，逐步产出规划-执行流程的事件（编排循环的唯一实现）。

        Args:
            user_input: 用户的输入文本。
            stream: True 时执行器以流式接口产出文本增量（TextDeltaEvent）；
                False 时整段响应以 ModelMessageEvent 产出。

        Yields:
            AgentEvent: 规划-执行流程的事件序列。

        Raises:
            ProviderError: 规划器 / 执行器 / 整合器的模型服务调用失败，
                由调用方决定重试策略。
        """
        # 1) 规划：把任务分解为步骤清单（解析失败退化为单步骤）
        plan = self._plan(user_input)
        logger.info("规划完成: %d 个步骤", len(plan))

        seq = 0
        seq = yield from self._emit_event(PlanGeneratedEvent(steps=plan), seq)

        # 2) 执行：逐步运行内部 ReAct 执行器，转发执行器事件并统一编号
        step_results: list[str] = []
        for index, step in enumerate(plan):
            logger.debug("执行步骤 %d/%d: %s", index + 1, len(plan), step)
            seq = yield from self._emit_event(PlanStepStartEvent(index=index, step=step), seq)

            step_answer: str | None = None
            for sub_event in self.executor.run_events(step, stream=stream):
                sub_event.seq = seq
                seq += 1
                if isinstance(sub_event, FinalAnswerEvent):
                    step_answer = sub_event.content
                yield sub_event

            step_results.append(
                step_answer if step_answer is not None else _MAX_ITERATIONS_FALLBACK
            )

            seq = yield from self._emit_event(
                PlanStepEndEvent(index=index, step=step, result=step_results[-1]), seq
            )
            logger.debug("步骤 %d/%d 完成（%d 字）", index + 1, len(plan), len(step_results[-1]))

        # 3) 整合：多步骤时汇总为最终答案，单步骤直接透传。
        # 单步骤时执行器已以 TextDeltaEvent 流出文本，不再重复产出增量。
        if len(plan) > 1:
            logger.info("整合 %d 个步骤结果为最终答案", len(plan))
            answer = self._synthesize(user_input, plan, step_results)
        else:
            answer = step_results[0]
        if stream and len(plan) > 1:
            seq = yield from self._emit_event(TextDeltaEvent(delta=answer), seq)
        yield from self._emit_event(FinalAnswerEvent(content=answer), seq)

    def _plan(self, user_input: str) -> list[str]:
        """规划：调用规划器把任务分解为步骤列表。

        Args:
            user_input: 用户的输入文本。

        Returns:
            有序步骤列表；输出无法解析时退化为 [user_input]（单步骤直接执行）。

        Raises:
            ProviderError: 规划器调用失败时向上传播。
        """
        response = self.provider.chat(
            messages=[
                {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            response_format={"type": "json_object"},
        )
        steps = _parse_steps(response.content or "")
        if not steps:
            logger.warning("规划器输出无法解析，退化为单步骤直接执行：%s", user_input)
            return [user_input]
        if len(steps) > self.max_steps:
            logger.info("规划步骤 %d 个，超出上限 %d，已截断", len(steps), self.max_steps)
            return steps[: self.max_steps]
        logger.debug("规划器产出 %d 个步骤", len(steps))
        return steps

    def _synthesize(self, user_input: str, plan: list[str], step_results: list[str]) -> str:
        """整合：基于各步骤执行结果生成最终答案。

        Args:
            user_input: 用户的输入文本（原始任务）。
            plan: 步骤清单。
            step_results: 各步骤的执行结果，与 plan 一一对应。

        Returns:
            整合后的最终答案文本。

        Raises:
            ProviderError: 整合器调用失败时向上传播。
        """
        logger.debug("整合器调用: 步骤数=%d", len(plan))
        report = "\n".join(
            f"{index + 1}. {step}\n   结果：{result}"
            for index, (step, result) in enumerate(zip(plan, step_results))
        )
        response = self.provider.chat(
            messages=[
                {"role": "system", "content": _SYNTHESIZER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"原始任务：{user_input}\n\n各步骤执行结果：\n{report}",
                },
            ]
        )
        answer = response.content or ""
        logger.debug("整合器产出最终答案（%d 字）", len(answer))
        return answer
