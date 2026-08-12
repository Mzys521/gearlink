"""GearLink 自定义 ModelProvider 示例

演示实现 `ModelProvider` 抽象（四类扩展点之一）：只需覆写 `chat()`，
未覆写的 `chat_stream()` 会自动回退到非流式调用。

本示例使用无网络的 EchoProvider，无需任何 API key 即可运行：
    python examples/custom_provider_demo.py
"""

from typing import Any

from gearlink import ModelProvider, ModelResponse, ReactAgent


class EchoProvider(ModelProvider):
    """回声提供者：把最后一条用户消息原样返回，用于演示与测试。"""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return ModelResponse(content=f"[EchoProvider] 你说的是：{last_user}")


def main() -> None:
    agent = ReactAgent(provider=EchoProvider())

    # run：一次性拿到最终答案
    print(agent.run("你好，GearLink！"))

    # run_stream：EchoProvider 未覆写 chat_stream，自动回退非流式（一次性产出全文）
    for delta in agent.run_stream("流式接口也会自动回退"):
        print(delta, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
