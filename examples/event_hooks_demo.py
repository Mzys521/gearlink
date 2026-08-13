"""GearLink 事件流与回调钩子示例

演示 `run_events` 全事件流消费与 `add_hook` 的 on_step 观察/替换语义；
脚本化 Provider 会先请求一次工具调用、再给出最终答案，便于观察完整事件序列。

本示例无需 API key 即可运行：
    python examples/event_hooks_demo.py
"""

from typing import Any

from gearlink import (
    AgentEvent,
    FinalAnswerEvent,
    ModelProvider,
    ModelResponse,
    ReactAgent,
    StepStartEvent,
    ToolCall,
    ToolCallEndEvent,
    ToolCallStartEvent,
)


class ScriptedProvider(ModelProvider):
    """脚本化提供者：第一轮请求调用 get_current_time 工具，收到工具结果后给出最终答案。"""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        if any(m.get("role") == "tool" for m in messages):
            return ModelResponse(content="已为你查询到当前时间（来自 get_current_time 工具）。")
        return ModelResponse(
            tool_calls=[ToolCall(id="call_1", name="get_current_time", arguments="{}")]
        )


def main() -> None:
    agent = ReactAgent(provider=ScriptedProvider())

    # 观察型钩子：打印每个事件（返回 None 表示不修改）
    def log_hook(event: AgentEvent) -> AgentEvent | None:
        print(f"  [hook] seq={event.seq} type={event.type}")
        return None

    # 干预型钩子：给最终答案追加后缀（返回替换事件）
    def stamp_hook(event: AgentEvent) -> AgentEvent | None:
        if isinstance(event, FinalAnswerEvent):
            event.content += "（经 stamp_hook 盖章）"
            return event
        return None

    agent.add_hook(log_hook)
    agent.add_hook(stamp_hook)

    print("事件流：")
    final = ""
    for event in agent.run_events("现在几点了？"):
        match event:
            case StepStartEvent():
                print(f"- 第 {event.iteration} 轮开始")
            case ToolCallStartEvent():
                print(f"- 工具开始执行: {event.name}")
            case ToolCallEndEvent():
                print(f"- 工具执行结束: {event.name} -> {event.result[:50]}")
            case FinalAnswerEvent():
                final = event.content
    print("\n最终答案:", final)


if __name__ == "__main__":
    main()
