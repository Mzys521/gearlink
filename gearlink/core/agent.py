from openai_provider import OpenAIProvider
from base import ModelProvider, ModelResponse
import json
import os
import sys

from tool import TOOL_SCHEMAS, call_tool
from memory import ShortTermMemory

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "providers"))


SYSTEM_PROMPT = "You are a helpful assistant. 当需要实时信息（如当前时间）时，请调用可用的工具，而不是凭空回答。"
MAX_ITERATIONS = 10 


class ReactAgent:
    """ReAct Agent：推理(Reason) -> 行动(Act) -> 观察(Observe) 循环"""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider
        self.memory = ShortTermMemory(max_message=20)
        self.memory.add_message({"role": "system", "content": SYSTEM_PROMPT})

    def _chat(self) -> ModelResponse:
        """基于记忆中的消息请求模型"""
        return self.provider.chat(
            messages=self.memory.get_messages(),
            tools=TOOL_SCHEMAS,
        )

    def run(self, user_input: str) -> str:
        """执行一次用户请求，内部运行 ReAct 循环直到得到最终答案"""
        self.memory.add_message({"role": "user", "content": user_input})

        for _ in range(MAX_ITERATIONS):
            response = self._chat()

            # 无工具调用 -> 模型已给出最终答案，结束循环
            if not response.tool_calls:
                self.memory.add_message(
                    {"role": "assistant", "content": response.content}
                )
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
                except Exception as e:
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
    agent = ReactAgent(provider=OpenAIProvider())
    print(agent.run("现在几点了？"))
