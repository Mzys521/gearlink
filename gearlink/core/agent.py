import json

from gearlink.core.memory import Memory, MemoryManager, ShortTermMemory
from gearlink.core.tool import TOOL_SCHEMAS, call_tool
from gearlink.exceptions import GearLinkError
from gearlink.providers.base import ModelProvider, ModelResponse


SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "当需要实时信息（如当前时间）时，请调用可用的工具，而不是凭空回答。"
)
MAX_ITERATIONS = 10


class ReactAgent:
    """ReAct Agent：推理(Reason) -> 行动(Act) -> 观察(Observe) 循环"""

    def __init__(
        self, provider: ModelProvider, memory: Memory | MemoryManager | None = None
    ) -> None:
        """初始化 Agent。

        Args:
            provider: 模型提供者实例。
            memory: 记忆实现；None 时默认使用 ShortTermMemory(max_message=20)
                并附加内置 SYSTEM_PROMPT。传入 MemoryManager 时启用长期记忆检索注入。
        """
        self.provider = provider
        if memory is None:
            memory = ShortTermMemory(max_message=20)
            memory.add_message({"role": "system", "content": SYSTEM_PROMPT})
        self.memory = memory

    def _chat(self, query: str | None = None) -> ModelResponse:
        """基于记忆中的消息请求模型。

        Args:
            query: 语义检索查询；仅对 MemoryManager 生效，用于注入长期记忆相关历史。
        """
        if isinstance(self.memory, MemoryManager):
            messages = self.memory.build_context(query or "")
        else:
            messages = self.memory.get_messages()
        return self.provider.chat(messages=messages, tools=TOOL_SCHEMAS)

    def run(self, user_input: str) -> str:
        """执行一次用户请求，内部运行 ReAct 循环直到得到最终答案"""
        self.memory.add_message({"role": "user", "content": user_input})

        for iteration in range(MAX_ITERATIONS):
            # 仅首轮携带查询注入长期记忆，后续轮次上下文已在短期记忆中
            response = self._chat(query=user_input if iteration == 0 else None)

            # 无工具调用 -> 模型已给出最终答案，结束循环
            if not response.tool_calls:
                self.memory.add_message({"role": "assistant", "content": response.content})
                return response.content

            # 有工具调用 -> 记录助手的工具调用消息
            self.memory.add_message(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
            )

            # 逐个执行工具调用，并将结果写回记忆
            for tool_call in response.tool_calls:
                name = tool_call.name
                try:
                    arguments = json.loads(tool_call.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                try:
                    result = call_tool(name, arguments)
                    result_text = json.dumps(result, ensure_ascii=False)
                except GearLinkError as e:
                    # 工具失败属可恢复信号：写回消息交给模型处理，不中断循环
                    result_text = f"工具调用失败: {e}"

                print(f"[工具调用] {name}({arguments}) -> {result_text}")

                self.memory.add_message(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    }
                )
            # 携带工具结果进入下一轮推理

        return "已达到最大推理轮数，无法得出最终答案。"


if __name__ == "__main__":
    # 入口示例：core 不直接依赖 providers，仅在演示入口处导入具体实现
    from dotenv import load_dotenv

    from gearlink.providers.openai_provider import OpenAIProvider

    load_dotenv()  # 加载项目根目录 .env 中的配置（如 DEEPSEEK_API_KEY）

    agent = ReactAgent(provider=OpenAIProvider())
    while True:
        user_input = input("用户: ")
        if user_input == "exit":
            break
        print("助手:", agent.run(user_input))
