import json
import logging

from gearlink.core.memory import Memory, MemoryManager, ShortTermMemory
from gearlink.core.tool import TOOL_SCHEMAS, call_tool
from gearlink.exceptions import GearLinkError
from gearlink.providers.base import ModelProvider, ModelResponse
from gearlink.utils.token_count import estimate_tokens


SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "当需要实时信息（如当前时间）时，请调用可用的工具，而不是凭空回答。"
)
MAX_ITERATIONS = 10

#: 工具结果写入记忆前的 token 估算上限，超出时截断防止撑爆上下文
MAX_TOOL_RESULT_TOKENS = 2000

#: 模块级日志器：记录 ReAct 循环中的工具调用等过程信息
logger = logging.getLogger(__name__)


class ReactAgent:
    """ReAct Agent：推理(Reason) -> 行动(Act) -> 观察(Observe) 循环"""

    def __init__(
        self,
        provider: ModelProvider,
        memory: Memory | MemoryManager | None = None,
        retrieve_every_iteration: bool = False,
    ) -> None:
        """初始化 Agent。

        Args:
            provider: 模型提供者实例。
            memory: 记忆实现；None 时默认使用 ShortTermMemory(max_message=20)
                并附加内置 SYSTEM_PROMPT。传入 MemoryManager 时启用长期记忆检索注入。
            retrieve_every_iteration: 是否每轮 ReAct 迭代都携带查询注入长期记忆；
                False 表示仅首轮注入（现状，后续轮次上下文已在短期记忆中）。
                开启时依赖 build_context 的短期窗口去重避免重复注入。
        """
        self.provider = provider
        self.retrieve_every_iteration = retrieve_every_iteration
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
            # 默认仅首轮携带查询注入长期记忆；开启 retrieve_every_iteration 时每轮都注入
            query = user_input if (iteration == 0 or self.retrieve_every_iteration) else None
            response = self._chat(query=query)

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
                    # 超出预算时按 4 字符/token 的启发式反推字符上限截断
                    if estimate_tokens(result_text) > MAX_TOOL_RESULT_TOKENS:
                        result_text = (
                            result_text[: MAX_TOOL_RESULT_TOKENS * 4]
                            + "\n...(工具结果过长，已截断)"
                        )
                except GearLinkError as e:
                    # 工具失败属可恢复信号：写回消息交给模型处理，不中断循环
                    result_text = f"工具调用失败: {e}"

                logger.info("[工具调用] %s(%s) -> %s", name, arguments, result_text)

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
    import logging

    import chromadb
    from dotenv import load_dotenv

    from gearlink.core.memory import LongTermMemory, MemoryManager, ShortTermMemory
    from gearlink.providers.openai_provider import OpenAIProvider

    logging.basicConfig(level=logging.INFO)  # 展示记忆沉淀/检索/去重/裁剪日志
    load_dotenv()  # 加载项目根目录 .env 中的配置（如 DEEPSEEK_API_KEY）

    # 长期记忆：chromadb 向量库持久化到 .chroma/（首次运行会下载嵌入模型）
    vector_db = chromadb.PersistentClient(path=".chroma")
    long_term = LongTermMemory(vector_db=vector_db, collection_name="chat_history")

    # 会话摘要生成器：应用层组装，调用同一模型做一次独立摘要请求（core 不依赖 providers）
    summarizer_provider = OpenAIProvider()

    def summarize(transcript: str) -> str:
        response = summarizer_provider.chat(
            messages=[
                {
                    "role": "system",
                    "content": "请用简体中文简要总结以下对话的要点，不超过 100 字。",
                },
                {"role": "user", "content": transcript},
            ]
        )
        return response.content or ""

    memory = MemoryManager(
        short_term=ShortTermMemory(max_message=20),
        long_term=long_term,
        max_context_tokens=4000,
        summarizer=summarize,
    )
    memory.add_message({"role": "system", "content": SYSTEM_PROMPT})

    agent = ReactAgent(provider=OpenAIProvider(), memory=memory)

    while True:
        user_input = input("用户: ")
        if user_input == "exit":
            memory.end_session()  # 结束会话：沉淀会话摘要并补沉淀剩余上下文
            break
        print("助手:", agent.run(user_input))
