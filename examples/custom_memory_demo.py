"""GearLink 自定义 Memory 示例

演示记忆扩展点（四类扩展点之一）：继承 `Memory` 实现
`add_message` / `get_messages` / `clear` 三方法，并经 `ReactAgent(memory=...)` 注入。

本示例使用去重滑窗记忆 + 无网络的 EchoProvider，无需 API key 即可运行：
    python examples/custom_memory_demo.py
"""

from typing import Any

from gearlink import Memory, ModelProvider, ModelResponse, ReactAgent


class DedupWindowMemory(Memory):
    """去重滑窗记忆：内容相同的消息只保留一条，最多保存 max_message 条。"""

    def __init__(self, max_message: int = 10) -> None:
        self.max_message = max_message
        self.messages: list[dict[str, Any]] = []

    def add_message(self, message: dict[str, Any]) -> None:
        """添加一条消息；与既有消息内容完全相同时跳过（去重）。"""
        if any(m.get("content") == message.get("content") for m in self.messages):
            return
        self.messages.append(message)
        if len(self.messages) > self.max_message:
            self.messages = self.messages[-self.max_message :]

    def get_messages(self, limit: int | None = None) -> list[dict[str, Any]]:
        """按时间顺序返回消息；limit 为 None 时返回全部。"""
        return self.messages if limit is None else self.messages[-limit:]

    def clear(self) -> None:
        """清空所有消息。"""
        self.messages = []


class EchoProvider(ModelProvider):
    """回声提供者：把最后一条用户消息原样返回，用于演示与测试。"""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return ModelResponse(content=f"[Echo] {last_user}")


def main() -> None:
    memory = DedupWindowMemory(max_message=10)
    agent = ReactAgent(provider=EchoProvider(), memory=memory)

    agent.run("我喜欢喝奶茶")
    agent.run("我喜欢喝奶茶")  # 重复输入：user 消息被去重跳过
    agent.run("周末喜欢爬山")

    print("记忆中现有消息：")
    for m in memory.get_messages():
        print(f"  [{m['role']}] {m.get('content')}")

    memory.clear()
    print("清空后消息条数:", len(memory.get_messages()))


if __name__ == "__main__":
    main()
