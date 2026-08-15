"""GearLink 模型自主编排示例：AutonomousOrchestrator 主管自主产出 DAG 计划

演示模型自主编排（纯新增，Orchestrator 子类，现有编排器行为不变）：

- 主管 Agent 每次请求自主决定子任务拆分、工人指派与串/并行策略，输出
  `{"nodes": [...], "edges": [...]}` 形式的有向无环图（DAG）编排计划；
- 计划编译为 Kahn 拓扑分层：无依赖的节点层内并行，有依赖的层间串行；
- 下游节点执行前，全部直接上游结果按依赖边聚合，以 `[上游结果]` 报告段落
  注入其任务文本（层屏障保证下游开始前上游结果已收集完毕）；
- 协作过程经事件流暴露：`TeamPlanGeneratedEvent`（携带 graph /
  parallel_groups）→ `AgentHandoffEvent`（携带 layer / upstream）→ 工人事件
  → `SubtaskEndEvent`（携带 layer）→ `FinalAnswerEvent`；
- 计划解析失败 / 引用未登记工人 / 出现环时，降级为全员兜底分派并记录日志。

本示例使用无网络的脚本 Provider，无需任何 API key 即可运行：
    python examples/autonomous_orchestrator_demo.py
"""

from typing import Any

from gearlink import (
    AgentHandoffEvent,
    AutonomousOrchestrator,
    FinalAnswerEvent,
    ModelProvider,
    ModelResponse,
    ReactAgent,
    SubtaskEndEvent,
    TeamPlanGeneratedEvent,
)


class ScriptedProvider(ModelProvider):
    """脚本提供者：按序返回预设回复，并记录收到的请求消息，用于演示。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0
        self.received: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        # 拷贝快照：ReactAgent 之后会向同一列表追加 assistant 消息
        self.received.append([dict(m) for m in messages])
        response = self.responses[self.calls]
        self.calls += 1
        return ModelResponse(content=response)


#: 主管自主产出的 DAG 计划：两个调研节点并行，撰写节点依赖两者
GRAPH_PLAN = (
    '{"nodes": ['
    '{"id": "market", "worker": "market_researcher", '
    '"task": "调研 GearLink 的市场定位与用户反馈"}, '
    '{"id": "tech", "worker": "tech_researcher", "task": "调研 GearLink 的技术特性与竞品对比"}, '
    '{"id": "report", "worker": "writer", "task": "综合两份调研撰写产品分析报告"}], '
    '"edges": [{"from": "market", "to": "report"}, {"from": "tech", "to": "report"}]}'
)

SUMMARY = "产品分析报告：GearLink 以轻量级事件流契约与可插拔生态见长，适合快速搭建多 Agent 应用。"


def main() -> None:
    # 主管：第一次调用产出 DAG 计划（JSON），第二次调用产出汇总答案；
    # 后两条为下方 run 一行式调用的重复规划与汇总（脚本提供者按序消耗）
    supervisor = ReactAgent(
        provider=ScriptedProvider([GRAPH_PLAN, SUMMARY, GRAPH_PLAN, SUMMARY])
    )

    # 工人：各自独立的 ReactAgent（真实场景可配不同 provider / 工具 / 记忆）
    workers = {
        "market_researcher": ReactAgent(
            provider=ScriptedProvider(["市场调研：目标用户为中小团队，核心诉求是低门槛编排。"] * 2)
        ),
        "tech_researcher": ReactAgent(
            provider=ScriptedProvider(["技术调研：事件流 + 拓扑分层调度，竞品多为主管轮询式。"] * 2)
        ),
        "writer": ReactAgent(
            provider=ScriptedProvider(["报告初稿：GearLink 低门槛 + 拓扑调度是核心差异化。"] * 2)
        ),
    }

    # 自主编排：parallel=True（默认）层内并行、层间串行；计划非法时自动降级兜底
    orchestrator = AutonomousOrchestrator(supervisor=supervisor, workers=workers)

    # 事件流观察协作全过程：依赖图、并行分组、分层与消息同步来源
    answer = None
    for event in orchestrator.run_events("帮我出一份 GearLink 的产品分析报告"):
        if isinstance(event, TeamPlanGeneratedEvent):
            print("[编排计划] 节点:", [(n["id"], n["worker"]) for n in event.graph["nodes"]])
            print("[依赖边]  ", event.graph["edges"])
            print("[并行分组]", event.parallel_groups)
        elif isinstance(event, AgentHandoffEvent):
            print(f"[派单 {event.index}] 层 {event.layer} 上游 {event.upstream} -> {event.worker}:")
            print(f"  {event.task}")
        elif isinstance(event, SubtaskEndEvent):
            print(f"[完成 {event.index}] 层 {event.layer} {event.worker}: {event.result}")
        elif isinstance(event, FinalAnswerEvent):
            answer = event.content

    print("\n最终答案:", answer)

    # 展示消息同步效果：writer 实际收到的任务文本含两个上游的结果段落
    writer_messages = workers["writer"].provider.received[0]
    print("\nwriter 实际收到的任务文本:\n", writer_messages[-1]["content"])

    # run 一行式调用：直接得到汇总后的答案
    print("\n一行式调用:", orchestrator.run("帮我出一份 GearLink 的产品分析报告"))


if __name__ == "__main__":
    main()
