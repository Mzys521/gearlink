"""GearLink 多 Agent 协作示例：Orchestrator 主管-工人编排（开发方向 §5.3）

演示多 Agent 协作编排层（纯新增，Agent 单输入/单输出契约不变）：

- 主管 Agent 把任务拆分为子任务并指派给登记的工人；
- 各工人是独立 `ReactAgent`（各配自己的工具/记忆，此处用无网络 Provider 演示）；
- 协作过程经事件流暴露：`TeamPlanGeneratedEvent` → `AgentHandoffEvent` →
  工人事件 → `SubtaskEndEvent` → `FinalAnswerEvent`。

本示例使用无网络的脚本 Provider，无需任何 API key 即可运行：
    python examples/orchestrator_demo.py
"""

from typing import Any

from gearlink import (
    AgentHandoffEvent,
    ModelProvider,
    ModelResponse,
    Orchestrator,
    ReactAgent,
    SubtaskEndEvent,
    TeamPlanGeneratedEvent,
)


class ScriptedProvider(ModelProvider):
    """脚本提供者：按序返回预设回复，用于演示与测试。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        response = self.responses[self.calls]
        self.calls += 1
        return ModelResponse(content=response)


def main() -> None:
    # 主管：第一次调用产出分派清单（JSON），第二次调用产出汇总答案；
    # 后两条为下方 run 一行式调用的重复分派与汇总（脚本提供者按序消耗）
    supervisor = ReactAgent(
        provider=ScriptedProvider(
            [
                '[{"worker": "researcher", "task": "调研 GearLink 的核心特性"}, '
                '{"worker": "writer", "task": "撰写一段产品介绍"}]',
                "GearLink 是一个轻量级 Agent 框架，支持三可插拔维度与技能体系。",
                '[{"worker": "researcher", "task": "调研 GearLink 的核心特性"}, '
                '{"worker": "writer", "task": "撰写一段产品介绍"}]',
                "GearLink 是一个轻量级 Agent 框架，支持三可插拔维度与技能体系。",
            ]
        )
    )

    # 工人：各自独立的 ReactAgent（真实场景可配不同 provider / 工具 / 记忆）；
    # 脚本按序消耗，两次调用各备两条回复
    workers = {
        "researcher": ReactAgent(
            provider=ScriptedProvider(
                [
                    "调研结果：三可插拔（模型/记忆/工具）+ 技能渐进披露。",
                    "调研结果：三可插拔（模型/记忆/工具）+ 技能渐进披露。",
                ]
            )
        ),
        "writer": ReactAgent(
            provider=ScriptedProvider(
                [
                    "产品文案：GearLink——把 Agent 装配成你需要的样子。",
                    "产品文案：GearLink——把 Agent 装配成你需要的样子。",
                ]
            )
        ),
    }

    orchestrator = Orchestrator(supervisor=supervisor, workers=workers, parallel=False)

    # 事件流观察协作全过程
    answer = None
    for event in orchestrator.run_events("为 GearLink 准备一份产品介绍"):
        if isinstance(event, TeamPlanGeneratedEvent):
            print("[主管分派]", event.assignments)
        elif isinstance(event, AgentHandoffEvent):
            print(f"[派单 {event.index}] -> {event.worker}: {event.task}")
        elif isinstance(event, SubtaskEndEvent):
            print(f"[完成 {event.index}] {event.worker}: {event.result}")
        else:
            answer = getattr(event, "content", None) or answer

    print("\n最终答案:", answer)

    # run 一行式调用：直接得到汇总后的答案
    print("一行式调用:", orchestrator.run("为 GearLink 准备一份产品介绍"))


if __name__ == "__main__":
    main()
