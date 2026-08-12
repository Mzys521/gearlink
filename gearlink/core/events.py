"""Agent 循环的中间事件模型与回调钩子类型。

事件流是 ReAct 循环的单一事实来源：`ReactAgent.run_events()` 逐步产出
`AgentEvent` 子类实例，`run()` / `run_stream()` 只是其两种消费方式；
外部经构造参数 `hooks` 或 `add_hook` 注册回调，在每个事件产出时获得
观察/干预机会（on_step 语义），命名回调（如 on_tool_call）可基于通用回调薄封装。
"""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gearlink.providers.base import TokenUsage, ToolCall

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
    "TeamPlanGeneratedEvent",
    "AgentHandoffEvent",
    "SubtaskEndEvent",
    "HookFn",
    "JsonlEventSink",
    "jsonl_hook",
    "load_jsonl_events",
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

    def to_dict(self) -> dict[str, Any]:
        """序列化为纯 JSON 可存字典（嵌套 dataclass 递归展开，含 seq / timestamp），
        供事件落盘与离线回放（开发方向 §5.1）。"""
        return asdict(self)


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
        usage: 本次模型调用的 token 用量；提供者未上报时为 None。
    """

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None
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


@dataclass
class TeamPlanGeneratedEvent(AgentEvent):
    """多 Agent 编排：主管产出的分派清单（开发方向 §5.3）。

    Attributes:
        assignments: 子任务分派列表，每项为 {"worker": 工人名, "task": 子任务指令}。
    """

    assignments: list[dict[str, str]] = field(default_factory=list)
    type: str = "team_plan_generated"


@dataclass
class AgentHandoffEvent(AgentEvent):
    """多 Agent 编排：主管把一个子任务移交给工人（开发方向 §5.3）。

    Attributes:
        index: 子任务序号（从 0 开始，与分派清单下标对应）。
        worker: 接收任务的工人名称。
        task: 派给工人的子任务指令文本。
    """

    index: int = 0
    worker: str = ""
    task: str = ""
    type: str = "agent_handoff"


@dataclass
class SubtaskEndEvent(AgentEvent):
    """多 Agent 编排：某个子任务执行完成（开发方向 §5.3）。

    Attributes:
        index: 子任务序号（从 0 开始，与分派清单下标对应）。
        worker: 执行该子任务的工人名称。
        result: 工人给出的结果文本。
    """

    index: int = 0
    worker: str = ""
    result: str = ""
    type: str = "subtask_end"


#: 回调钩子签名：接收一个事件，可返回替换事件；返回 None 表示不修改。
#: 回调应保持观察/干预语义（如日志记录、内容校验），避免产生依赖调用顺序的副作用。
HookFn = Callable[[AgentEvent], AgentEvent | None]


class JsonlEventSink:
    """事件落盘器：把事件流逐条序列化为 JSONL（开发方向 §5.1）。

    每行一个事件（含 ``seq`` / ``timestamp``），支持离线回放与调试；
    追加写入，多次运行可沉淀到同一文件。可作为上下文管理器使用。
    """

    def __init__(self, path: str | Path) -> None:
        """打开目标文件（追加模式）。

        Args:
            path: JSONL 文件路径；不存在时自动创建。
        """
        self.path = Path(path)
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, event: AgentEvent) -> None:
        """写入一个事件并立即刷盘（进程崩溃也不丢已产出事件）。"""
        self._file.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        """关闭底层文件（幂等）。"""
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "JsonlEventSink":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def jsonl_hook(sink: JsonlEventSink) -> HookFn:
    """构造把事件写入 JSONL sink 的回调钩子（纯观察，不修改事件）。

    Args:
        sink: 已打开的 :class:`JsonlEventSink` 实例。

    Returns:
        可直接传入 ``ReactAgent(hooks=...)`` / ``add_hook`` 的回调。
    """

    def hook(event: AgentEvent) -> None:
        sink.write(event)
        return None

    return hook


def load_jsonl_events(path: str | Path) -> list[dict[str, Any]]:
    """回放 JSONL 事件文件：逐行解析为字典列表（按写入顺序）。

    Args:
        path: :class:`JsonlEventSink` 写出的文件路径。

    Returns:
        事件字典列表；空行跳过。
    """
    events: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
