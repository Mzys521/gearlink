"""多 Agent 协作编排层：主管-工人模式的 Orchestrator（开发方向 §5.3）。

纯新增编排层：`Agent` 的单输入/单输出契约不变，Orchestrator 把主管 Agent 与
多个工人 Agent（各配独立工具/技能/记忆）组合为协作团队——主管把任务拆分派给
合适的工人，各工人独立完成子任务后由汇总提示整合为最终答案。

事件流：`TeamPlanGeneratedEvent`（主管分派清单）→ 每个子任务
`AgentHandoffEvent`（派单）→ 工人事件（转发并统一编号）→ `SubtaskEndEvent`；
最后产出 `FinalAnswerEvent`（多子任务且 stream=True 时先产出合成文本的
TextDeltaEvent）。

与 `PlanExecuteAgent` 的关系：PlanExecuteAgent 是单模型串行步骤，
Orchestrator 是多角色分工（可并行），二者互补。
"""

import logging
import queue
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from gearlink.core.agent import Agent, _extract_json
from gearlink.core.events import (
    AgentEvent,
    AgentHandoffEvent,
    FinalAnswerEvent,
    HookFn,
    LoopAbortEvent,
    SubtaskEndEvent,
    TeamPlanGeneratedEvent,
    TextDeltaEvent,
)
from gearlink.exceptions import GearLinkError

__all__ = [
    "Orchestrator",
]

#: 模块级日志器：记录任务分派、工人执行与结果整合过程
logger = logging.getLogger(__name__)

#: 达到最大轮数等场景的兜底文案（与 ReAct 循环保持一致）
_MAX_ITERATIONS_FALLBACK = "已达到最大推理轮数，无法得出最终答案。"

#: 主管分派器系统提示：要求把任务拆分为 JSON 子任务数组并指派给工人
_DISPATCHER_SYSTEM_PROMPT = (
    "你是多智能体团队的主管。请把用户的任务拆分为若干子任务，并指派给最合适的工人。"
    "每个子任务是一条给工人的清晰、自包含的中文指令，不要带序号前缀。"
    '只输出一个 JSON 数组，每个元素形如 {"worker": 工人名, "task": 子任务指令}；'
    "worker 必须是给定工人名单中的一个，不要输出任何其他内容。"
)

#: 汇总器系统提示：把各工人结果整合为直接可用的最终回答
_SYNTHESIZER_SYSTEM_PROMPT = (
    "你是结果汇总器。基于各工人的执行结果，为用户的任务给出最终回答。"
    "回答应直接、完整、简洁，不要提及工人或执行结果等过程性内容。"
)


def _parse_assignments(text: str) -> list[dict[str, str]] | None:
    """解析主管输出的分派清单（JSON 对象数组）。

    容错：经 `_extract_json` 剥离围栏与提取 JSON 片段后解析；
    每个元素须为含 ``worker`` / ``task`` 字符串键的对象。

    Args:
        text: 主管原始输出文本。

    Returns:
        解析出的分派列表；格式非法时返回 None（由调用方退化为全员兜底分派）。
    """
    assignments = _extract_json(text)
    if assignments is None:
        return None
    if not isinstance(assignments, list) or not assignments:
        return None
    parsed: list[dict[str, str]] = []
    for item in assignments:
        if not isinstance(item, dict):
            return None
        worker, task = item.get("worker"), item.get("task")
        if not isinstance(worker, str) or not worker.strip():
            return None
        if not isinstance(task, str) or not task.strip():
            return None
        parsed.append({"worker": worker.strip(), "task": task.strip()})
    return parsed


class Orchestrator(Agent):
    """多 Agent 协作编排器：主管-工人模式（开发方向 §5.3）。

    流程：主管（纯 LLM 调用，不启用工具）把用户任务拆分为子任务并指派给登记的
    工人（分派输出无法解析时退化为把原任务派给全部工人）；随后各工人 Agent
    独立完成子任务（串行或并行，经 ``parallel`` 控制）；最后经一次汇总对话
    整合为最终答案，单子任务时直接透传工人结果。

    工人各自持有独立的工具白名单/技能/记忆（在构造工人 Agent 时自行配置）；
    分派到未登记工人名的子任务以错误文案记录结果，不中断编排。

    事件流：`TeamPlanGeneratedEvent` → 每个子任务 `AgentHandoffEvent` → 工人事件
    （转发并重新编号保证 seq 全局递增）→ `SubtaskEndEvent`；最后产出
    `FinalAnswerEvent`。
    """

    def __init__(
        self,
        supervisor: Agent,
        workers: dict[str, Agent],
        parallel: bool = False,
        hooks: list[HookFn] | None = None,
    ) -> None:
        """初始化编排器。

        Args:
            supervisor: 主管 Agent，负责任务拆分与分派（其 provider 同时用于
                结果汇总）。
            workers: 工人名称 → 工人 Agent 的映射；每个工人独立配置自己的
                工具/技能/记忆。不能为空。
            parallel: True 时各工人子任务经线程池并行执行（要求工人自身及其
                工具线程安全）；False 表示串行（现状）。事件产出顺序均按分派
                清单确定性排序。
            hooks: 事件回调列表（on_step 语义）；同时作用于编排层与全部工人，
                经共享回调列表实现（与 PlanExecuteAgent 一致）。

        Raises:
            GearLinkError: workers 为空时抛出。
        """
        if not workers:
            raise GearLinkError("Orchestrator 至少需要一名工人（workers 不能为空）")
        super().__init__(provider=supervisor.provider, hooks=hooks)
        self.supervisor = supervisor
        self.workers = dict(workers)
        self.parallel = parallel
        # 与主管/工人共享同一回调列表：编排层与全部子 Agent 事件统一经回调
        self.supervisor._hooks = self._hooks
        for worker in self.workers.values():
            worker._hooks = self._hooks

    def run_events(self, user_input: str, *, stream: bool = False) -> Iterator[AgentEvent]:
        """执行一次用户请求，逐步产出多 Agent 协作流程的事件（编排的唯一实现）。

        Args:
            user_input: 用户的输入文本。
            stream: True 时工人以流式接口产出文本增量（TextDeltaEvent）；
                False 时整段响应以 ModelMessageEvent 产出。

        Yields:
            AgentEvent: 协作流程的事件序列。

        Raises:
            ProviderError: 主管 / 工人 / 汇总器的模型服务调用失败，
                由调用方决定重试策略。
        """
        # 1) 分派：主管把任务拆分并指派给工人（解析失败退化为全员兜底）
        assignments = self._dispatch(user_input)

        seq = 0
        seq = yield from self._emit_event(TeamPlanGeneratedEvent(assignments=assignments), seq)

        # 2) 执行：各工人独立完成子任务，转发事件并统一编号。
        # 串行：逐个「派单 → 实时转发事件 → 完成」；并行：先全部派单，
        # 线程池并行执行后按分派序产出事件与完成事件（与并行工具调用惯例一致）。
        results: list[str] = []
        if self.parallel and len(assignments) > 1:
            for index, assignment in enumerate(assignments):
                seq = yield from self._emit_event(
                    AgentHandoffEvent(
                        index=index, worker=assignment["worker"], task=assignment["task"]
                    ),
                    seq,
                )
            with ThreadPoolExecutor(max_workers=len(assignments)) as pool:
                outcomes = list(pool.map(lambda a: self._run_worker(a, stream), assignments))
            for index, assignment in enumerate(assignments):
                worker_events, result = outcomes[index]
                for sub_event in worker_events:
                    sub_event.seq = seq
                    seq += 1
                    yield sub_event
                results.append(result)
                seq = yield from self._emit_event(
                    SubtaskEndEvent(index=index, worker=assignment["worker"], result=result),
                    seq,
                )
        else:
            for index, assignment in enumerate(assignments):
                seq = yield from self._emit_event(
                    AgentHandoffEvent(
                        index=index, worker=assignment["worker"], task=assignment["task"]
                    ),
                    seq,
                )
                result: str | None = None
                worker = self.workers.get(assignment["worker"])
                if worker is None:
                    result = f"工人 {assignment['worker']} 未登记，子任务未执行。"
                else:
                    for sub_event in worker.run_events(assignment["task"], stream=stream):
                        sub_event.seq = seq
                        seq += 1
                        if isinstance(sub_event, FinalAnswerEvent):
                            result = sub_event.content
                        elif isinstance(sub_event, LoopAbortEvent):
                            result = sub_event.reason
                        yield sub_event
                if result is None:
                    result = _MAX_ITERATIONS_FALLBACK
                results.append(result)
                seq = yield from self._emit_event(
                    SubtaskEndEvent(index=index, worker=assignment["worker"], result=result),
                    seq,
                )

        # 3) 汇总：多子任务时整合为最终答案，单子任务直接透传。
        # 单子任务时工人已以 TextDeltaEvent 流出文本，不再重复产出增量。
        if len(assignments) > 1:
            answer = self._synthesize(user_input, assignments, results)
        else:
            answer = results[0]
        if stream and len(assignments) > 1:
            seq = yield from self._emit_event(TextDeltaEvent(delta=answer), seq)
        yield from self._emit_event(FinalAnswerEvent(content=answer), seq)

    def _dispatch(self, user_input: str) -> list[dict[str, str]]:
        """分派：主管把任务拆分为子任务并指派给工人。

        Args:
            user_input: 用户的输入文本。

        Returns:
            分派清单；主管输出无法解析或指派了未登记工人时，退化为把原任务
            派给全部工人（兜底保证编排可用）。

        Raises:
            ProviderError: 主管调用失败时向上传播。
        """
        roster = "\n".join(f"- {name}" for name in self.workers)
        response = self.provider.chat(
            messages=[
                {"role": "system", "content": _DISPATCHER_SYSTEM_PROMPT},
                {"role": "user", "content": f"工人名单：\n{roster}\n\n用户任务：{user_input}"},
            ],
            response_format={"type": "json_object"},
        )
        assignments = _parse_assignments(response.content or "")
        if assignments is None:
            logger.warning("主管分派输出无法解析，退化为全员兜底分派：%s", user_input)
            return [{"worker": name, "task": user_input} for name in self.workers]
        unknown = [a["worker"] for a in assignments if a["worker"] not in self.workers]
        if unknown:
            logger.warning("主管指派了未登记的工人 %s，退化为全员兜底分派", unknown)
            return [{"worker": name, "task": user_input} for name in self.workers]
        return assignments

    def _run_worker(self, assignment: dict[str, str], stream: bool) -> tuple[list[AgentEvent], str]:
        """运行单个子任务：收集工人事件流并提取最终答案（线程安全，可被线程池调用）。

        Args:
            assignment: 单个分派条目（worker / task）。
            stream: 工人是否使用流式接口。

        Returns:
            (工人产出的事件列表, 结果文本) 二元组；工人未收敛时结果为兜底文案。
        """
        worker = self.workers.get(assignment["worker"])
        if worker is None:
            return [], f"工人 {assignment['worker']} 未登记，子任务未执行。"

        # 工人事件流可能在线程中产出：经队列转交给主线程按序消费
        event_queue: queue.Queue[Any] = queue.Queue()

        def run_worker_loop() -> None:
            try:
                for event in worker.run_events(assignment["task"], stream=stream):
                    event_queue.put(event)
            except Exception as exc:  # noqa: BLE001 - 线程内异常须带回主线程
                event_queue.put(exc)
            finally:
                event_queue.put(None)

        thread = threading.Thread(target=run_worker_loop)
        thread.start()

        events: list[AgentEvent] = []
        answer: str | None = None
        while True:
            item = event_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                thread.join()
                raise item
            events.append(item)
            if isinstance(item, FinalAnswerEvent):
                answer = item.content
            elif isinstance(item, LoopAbortEvent):
                answer = item.reason
        thread.join()
        return events, answer if answer is not None else _MAX_ITERATIONS_FALLBACK

    def _synthesize(
        self, user_input: str, assignments: list[dict[str, str]], results: list[str]
    ) -> str:
        """汇总：基于各工人执行结果生成最终答案。

        Args:
            user_input: 用户的输入文本（原始任务）。
            assignments: 分派清单。
            results: 各工人的执行结果，与 assignments 一一对应。

        Returns:
            整合后的最终答案文本。

        Raises:
            ProviderError: 汇总器调用失败时向上传播。
        """
        report = "\n".join(
            f"{index + 1}. [{assignment['worker']}] {assignment['task']}\n   结果：{result}"
            for index, (assignment, result) in enumerate(zip(assignments, results))
        )
        response = self.provider.chat(
            messages=[
                {"role": "system", "content": _SYNTHESIZER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"原始任务：{user_input}\n\n各工人执行结果：\n{report}",
                },
            ]
        )
        return response.content or ""
