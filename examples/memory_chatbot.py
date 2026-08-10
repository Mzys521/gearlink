"""GearLink 记忆对话示例

一个带「短期滑窗 + 长期向量检索 + 会话摘要沉淀」的记忆型对话助手，
通过 ReAct 循环调用内置工具（get_current_time）获取实时时间。

运行方式（项目根目录下，须先 `pip install -e .` 安装本包）：
    python examples/memory_chatbot.py

前置条件：
    - 根目录 .env 中配置 DEEPSEEK_API_KEY（或设置同名环境变量）
    - 依赖：openai、chromadb、python-dotenv（见 requirements.txt）

交互命令：
    exit    结束会话：沉淀会话摘要，并补沉淀剩余上下文到长期记忆
    clear   清空短期与长期记忆
"""

import logging
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from gearlink import (
    LongTermMemory,
    MemoryManager,
    OpenAIProvider,
    ReactAgent,
    ShortTermMemory,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # 加载根目录 .env 中的 DEEPSEEK_API_KEY

# 1) 长期记忆：chromadb 向量库，持久化到 examples/.chroma/（首次运行会加载嵌入模型）
vector_db = chromadb.PersistentClient(path=str(Path(__file__).resolve().parent / ".chroma"))
long_term = LongTermMemory(
    vector_db=vector_db,
    collection_name="demo_chat_history",
    recency_weight=0.5,  # 检索排序时新近记忆适度排前
    max_entries=200,  # 容量上限，超出时淘汰最旧条目
    dedupe=True,  # 同内容不重复写入
)

# 2) 会话摘要生成器：应用层装配，交给同一模型做一次独立摘要请求
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
    short_term=ShortTermMemory(max_message=20, max_tokens=3000),
    long_term=long_term,
    max_context_tokens=4000,
    system_budget_ratio=0.5,  # 非检索 system 消息占上下文预算的比例上限
    summarizer=summarize,
)

# 3) Agent：每轮迭代都携带查询注入长期检索，增强多轮一致性
agent = ReactAgent(
    provider=OpenAIProvider(),
    memory=memory,
    retrieve_every_iteration=True,
)


def main() -> None:
    print("GearLink 记忆对话示例（输入 exit 结束 / clear 清空记忆）")
    print("试试问：现在是几点？/ 我叫什么名字？/ 我上次聊了什么？")
    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input == "exit":
            memory.end_session()  # 沉淀会话摘要并清空短期记忆
            print("会话已结束，摘要已沉淀到长期记忆。")
            break
        if user_input == "clear":
            memory.clear()
            print("记忆已清空。")
            continue
        print("助手:", agent.run(user_input))


if __name__ == "__main__":
    main()
