"""GearLink 可观测性示例（开发方向 §5.1）

演示 token 用量透传、事件流 JSONL 落盘与离线回放、按标签聚合用量与成本估算：

- `ModelResponse.usage`（`TokenUsage`）由提供者上报，经 `ModelMessageEvent` 透传；
- `JsonlEventSink` + `jsonl_hook` 把事件流逐条落盘，`load_jsonl_events` 回放；
- `UsageTracker` 按标签聚合用量并估算成本。

本示例使用无网络的 ScriptProvider，无需任何 API key 即可运行：
    python examples/observability_demo.py
"""

import tempfile
from pathlib import Path
from typing import Any

from gearlink import (
    JsonlEventSink,
    ModelProvider,
    ModelResponse,
    ReactAgent,
    TokenUsage,
    UsageTracker,
    jsonl_hook,
    load_jsonl_events,
)


class ScriptProvider(ModelProvider):
    """脚本提供者：返回预设回复并上报 token 用量，用于演示与测试。"""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return ModelResponse(
            content=f"[ScriptProvider] 收到：{last_user}",
            usage=TokenUsage(input_tokens=120, output_tokens=45),
        )


def main() -> None:
    # 1) 事件落盘：jsonl_hook 把 run_events 的每个事件写入 JSONL 文件
    events_path = Path(tempfile.gettempdir()) / "gearlink_observability_demo.jsonl"
    events_path.unlink(missing_ok=True)
    with JsonlEventSink(events_path) as sink:
        agent = ReactAgent(provider=ScriptProvider(), hooks=[jsonl_hook(sink)])
        print("回答:", agent.run("今天天气怎么样？"))

    # 2) 离线回放：逐行读回事件（含 seq / timestamp / usage）
    events = load_jsonl_events(events_path)
    print(f"\n落盘事件共 {len(events)} 条：")
    for event in events:
        usage = event.get("usage")
        suffix = f"（usage={usage}）" if usage else ""
        print(f"  seq={event['seq']} type={event['type']}{suffix}")

    # 3) 用量聚合与成本估算：按标签（如模型名/会话名）累计
    tracker = UsageTracker()
    for event in events:
        if event.get("usage"):
            tracker.add(TokenUsage.from_dict(event["usage"]), label="deepseek-chat")
    print("\n聚合用量:", tracker.total().to_dict())

    prices = {"deepseek-chat": (0.002, 0.008)}  # 每千 token（输入, 输出）单价
    costs = tracker.estimate_cost(prices)
    print(f"估算成本: {costs}")

    events_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
