"""Agent 循环的中间事件模型与回调钩子类型。

事件流是 ReAct 循环的单一事实来源：`ReactAgent.run_events()` 逐步产出
`AgentEvent` 子类实例，`run()` / `run_stream()` 只是其两种消费方式；
外部经构造参数 `hooks` 或 `add_hook` 注册回调，在每个事件产出时获得
观察/干预机会（on_step 语义），命名回调（如 on_tool_call）可基于通用回调薄封装。
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from gearlink.providers.base import ToolCall

__all__ = [
    "AgentEvent",
    "StepStartEvent",
    "TextDeltaEvent",
    "ModelMessageEvent",
    "ToolCallStartEvent",
    "ToolCallEndEvent",
    "FinalAnswerEvent",
    "LoopAbortEvent",
    "PlanGeneratedEvent",
    "PlanStepStartEvent",
    "PlanStepEndEvent",
    "HookFn",
]


@dataclass
class AgentEvent:
    """Agent 循环产出的最小可观测单元（所有事件类型的基类）。

    Attributes:
        seq: 事件流内全局递增序号，保证消费方按序处理。
        iteration: 事件产生时所在的 ReAct 轮次（从 0 开始）。
        timestamp: 事件产生的时间戳（unix 秒），用于耗时统计。
        type: 事件类型标识，作为 JSON 序列化时的判别字段。
    """

    seq: int = 0
    iteration: int = 0
    timestamp: float = field(default_factory=time.time)
    type: str = "event"


@dataclass
class StepStartEvent(AgentEvent):
    """一轮 ReAct 迭代开始。"""

    type: str = "step_start"


@dataclass
class TextDeltaEvent(AgentEvent):
    """流式模式下的文本增量（消费方可实时拼装出助手输出）。

    Attributes:
        delta: 本次增量文本。
    """

    delta: str = ""
    type: str = "text_delta"


@dataclass
class ModelMessageEvent(AgentEvent):
    """模型对本轮的完整响应（含工具调用请求或纯文本）。

    Attributes:
        content: 模型回复的文本；调用工具时可能为 None。
        tool_calls: 模型请求执行的工具调用列表；空列表表示无工具调用。
    """

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    type: str = "model_message"


@dataclass
class ToolCallStartEvent(AgentEvent):
    """单个工具开始执行前。

    Attributes:
        tool_call_id: 工具调用的唯一 ID（与模型返回一致）。
        name: 工具名。
        arguments: 模型返回的原始 JSON 参数字符串，保留原样便于审计。
    """

    tool_call_id: str = ""
    name: str = ""
    arguments: str = ""
    type: str = "tool_call_start"


@dataclass
class ToolCallEndEvent(AgentEvent):
    """单个工具执行结束（无论成败）。

    Attributes:
        tool_call_id: 工具调用的唯一 ID（与模型返回一致）。
        name: 工具名。
        result: 序列化后的工具结果文本（失败时为错误描述）。
        truncated: 结果是否因超出 token 预算被截断。
        error: 非 None 表示可恢复失败（结果已写回记忆交给模型处理，不中断循环）。
    """

    tool_call_id: str = ""
    name: str = ""
    result: str = ""
    truncated: bool = False
    error: str | None = None
    type: str = "tool_call_end"


@dataclass
class FinalAnswerEvent(AgentEvent):
    """循环收敛，模型给出最终答案。

    Attributes:
        content: 最终答案文本。
    """

    content: str = ""
    type: str = "final_answer"


@dataclass
class LoopAbortEvent(AgentEvent):
    """循环被中止（达到最大轮数）。

    Attributes:
        reason: 中止原因文案（可作为兜底答案返回给用户）。
    """

    reason: str = ""
    type: str = "loop_abort"


@dataclass
class PlanGeneratedEvent(AgentEvent):
    """规划-执行 Agent：规划器产出的步骤清单。

    Attributes:
        steps: 有序步骤列表（每条为给执行 Agent 的指令文本）。
    """

    steps: list[str] = field(default_factory=list)
    type: str = "plan_generated"


@dataclass
class PlanStepStartEvent(AgentEvent):
    """规划-执行 Agent：开始执行某个步骤。

    Attributes:
        index: 步骤序号（从 0 开始，与步骤清单下标对应）。
        step: 步骤指令文本。
    """

    index: int = 0
    step: str = ""
    type: str = "plan_step_start"


@dataclass
class PlanStepEndEvent(AgentEvent):
    """规划-执行 Agent：某个步骤执行完成。

    Attributes:
        index: 步骤序号（从 0 开始，与步骤清单下标对应）。
        step: 步骤指令文本。
        result: 该步骤执行器给出的结果文本。
    """

    index: int = 0
    step: str = ""
    result: str = ""
    type: str = "plan_step_end"


#: 回调钩子签名：接收一个事件，可返回替换事件；返回 None 表示不修改。
#: 回调应保持观察/干预语义（如日志记录、内容校验），避免产生依赖调用顺序的副作用。
HookFn = Callable[[AgentEvent], AgentEvent | None]
