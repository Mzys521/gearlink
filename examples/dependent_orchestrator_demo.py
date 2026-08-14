"""GearLink 依赖编排示例：DependentOrchestrator 工人间流水线协作（开发方向 §6.8）

演示工人间依赖编排（纯新增，Orchestrator 子类，现有 Orchestrator 行为不变）：

- 主管 Agent 把任务拆分为子任务并指派给登记的工人；
- `dependencies` 声明工人依赖：writer 依赖 researcher 的全部结果；
- 执行按依赖拓扑分层：researcher 先完成，其结果自动注入 writer 的任务文本
  （追加 `[上游结果]` 报告段落）；
- 协作过程经事件流暴露：`TeamPlanGeneratedEvent`（携带 dependencies）→
  `AgentHandoffEvent`（task 为注入后文本）→ 工人事件 → `SubtaskEndEvent` →
  `FinalAnswerEvent`。

本示例使用无网络的脚本 Provider，无需任何 API key 即可运行：
    python examples/dependent_orchestrator_demo.py
"""

from typing import Any

from gearlink import (
    AgentHandoffEvent,
    DependentOrchestrator,
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


def main() -> None:
    # 主管：第一次调用产出分派清单（JSON），第二次调用产出汇总答案；
    # 后两条为下方 run 一行式调用的重复分派与汇总（脚本提供者按序消耗）
    supervisor = ReactAgent(
        provider=ScriptedProvider(
            [
                '[{"worker": "researcher", "task": "检索并整理近一周 GearLink 相关的新闻素材"}, '
                '{"worker": "writer", "task": "基于整理好的素材撰写一篇新闻稿"}]',
                "GearLink 新闻：轻量级 Agent 框架发布 0.2.0，新增依赖编排能力。",
                '[{"worker": "researcher", "task": "检索并整理近一周 GearLink 相关的新闻素材"}, '
                '{"worker": "writer", "task": "基于整理好的素材撰写一篇新闻稿"}]',
                "GearLink 新闻：轻量级 Agent 框架发布 0.2.0，新增依赖编排能力。",
            ]
        )
    )

    # 工人：各自独立的 ReactAgent（真实场景可配不同 provider / 工具 / 记忆）
    workers = {
        "researcher": ReactAgent(
            provider=ScriptedProvider(
                [
                    "素材一：GearLink 0.2.0 发布，核心包 216 测试全绿。",
                    "素材一：GearLink 0.2.0 发布，核心包 216 测试全绿。",
                ]
            )
        ),
        "writer": ReactAgent(
            provider=ScriptedProvider(
                [
                    "新闻稿：GearLink 0.2.0 发布，新增工人依赖编排，支持流水线式多 Agent 协作。",
                    "新闻稿：GearLink 0.2.0 发布，新增工人依赖编排，支持流水线式多 Agent 协作。",
                ]
            )
        ),
    }

    # 声明依赖：writer 在 researcher 完成后执行，且自动注入 researcher 的结果
    orchestrator = DependentOrchestrator(
        supervisor=supervisor,
        workers=workers,
        dependencies={"writer": ["researcher"]},
        parallel=False,
    )

    # 事件流观察协作全过程：writer 的派单事件携带注入后的任务文本
    answer = None
    for event in orchestrator.run_events("帮我写一篇 GearLink 的近期新闻"):
        if isinstance(event, TeamPlanGeneratedEvent):
            print("[主管分派]", event.assignments)
            print("[依赖声明]", event.dependencies)
        elif isinstance(event, AgentHandoffEvent):
            print(f"[派单 {event.index}] -> {event.worker}: {event.task}")
        elif isinstance(event, SubtaskEndEvent):
            print(f"[完成 {event.index}] {event.worker}: {event.result}")
        else:
            answer = getattr(event, "content", None) or answer

    print("\n最终答案:", answer)

    # 展示注入效果：writer 实际收到的任务文本含上游结果段落
    writer_messages = workers["writer"].provider.received[0]
    print("\nwriter 实际收到的任务文本:\n", writer_messages[-1]["content"])

    # run 一行式调用：直接得到汇总后的答案
    print("\n一行式调用:", orchestrator.run("帮我写一篇 GearLink 的近期新闻"))


if __name__ == "__main__":
    main()
