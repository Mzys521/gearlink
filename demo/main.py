"""GearLink 综合 Agent 示例：工具 + 长期记忆 + 技能 + 流式输出

一个可交互的命令行助手，整合了框架的四个可插拔维度：

- 工具：自定义工具（add / multiply / get_weather）与内置工具（get_current_time），
  经 `register_tool` 显式登记后由 ReAct 循环按需调用；
- 记忆：短期滑窗 + chromadb 向量长期记忆 + 会话摘要沉淀（MemoryManager）；
- 技能：从 demo/skills/ 发现的技能包，模型经 `load_skill` 工具按需加载完整指令；
- 流式输出：`run_stream` 逐片段产出模型文本。

运行方式（项目根目录下，须先 `pip install -e .` 安装本包）：
    python demo/main.py

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

# 导入自定义工具模块，触发 add / multiply / get_weather 的注册
import tools  # noqa: F401
import gearlink.tools.load_skill  # noqa: F401  显式导入，触发 load_skill 工具注册
from gearlink import (
    LongTermMemory,
    MemoryManager,
    OpenAIProvider,
    ReactAgent,
    ShortTermMemory,
    SkillLoader,
    SkillRegistry,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

DEMO_DIR = Path(__file__).resolve().parent
ROOT = DEMO_DIR.parent
load_dotenv(ROOT / ".env")  # 加载根目录 .env 中的 DEEPSEEK_API_KEY


def build_skill_registry() -> SkillRegistry:
    """从 demo/skills/ 目录发现技能并登记到注册表。"""
    registry = SkillRegistry()
    for skill in SkillLoader.discover_from_directory(DEMO_DIR / "skills"):
        registry.register(skill)
    return registry


def build_system_prompt(registry: SkillRegistry) -> str:
    """组装系统提示：基础提示 + 可用技能列表。

    由于本示例自行注入 MemoryManager（而非默认短期记忆），ReactAgent 不会自动
    拼入技能列表，故在此手动组装并以 system 消息写入记忆。
    """
    base = (
        "你是一个有用的助手。当需要实时信息（如当前时间、天气）时，"
        "请调用可用的工具，而不是凭空回答。"
    )
    skills = registry.list_all()
    if not skills:
        return base
    skill_lines = "\n".join(f"- {s.name}: {s.description}" for s in skills)
    return (
        f"{base}\n"
        "当遇到需要特定专业知识的任务时，你可以调用 `load_skill` 工具加载相应的技能指令，"
        "然后严格遵循这些指令执行。\n"
        f"当前可用技能：\n{skill_lines}"
    )


def build_memory() -> MemoryManager:
    """组装「短期 + 长期 + 会话摘要」的记忆管理器。"""
    vector_db = chromadb.PersistentClient(path=str(DEMO_DIR / ".chroma"))
    long_term = LongTermMemory(
        vector_db=vector_db,
        collection_name="demo_agent_history",
        recency_weight=0.5,
        max_entries=200,
        dedupe=True,
    )

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

    return MemoryManager(
        short_term=ShortTermMemory(max_message=20, max_tokens=3000),
        long_term=long_term,
        max_context_tokens=4000,
        system_budget_ratio=0.5,
        summarizer=summarize,
    )


def main() -> None:
    skill_registry = build_skill_registry()
    memory = build_memory()

    # 自行注入 memory 时，须自行写入系统提示（含技能列表）
    memory.add_message({"role": "system", "content": build_system_prompt(skill_registry)})

    agent = ReactAgent(
        provider=OpenAIProvider(),
        memory=memory,
        retrieve_every_iteration=True,
        skill_registry=skill_registry,
    )

    skill_names = ", ".join(s.name for s in skill_registry.list_all()) or "（无）"
    print("GearLink 综合 Agent 示例（输入 exit 结束 / clear 清空记忆）")
    print(f"已加载技能：{skill_names}")
    print("试试问：现在几点了？/ 北京天气如何？/ 帮我生成一份每日简报 / 12.3 加 45.6 等于几？")
    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            user_input = "exit"
        if not user_input:
            continue
        if user_input == "exit":
            memory.end_session()  # 沉淀会话摘要并清空短期记忆
            print("会话已结束，摘要已沉淀到长期记忆。")
            break
        if user_input == "clear":
            memory.clear()
            # 清空后重新写入系统提示，保证下一轮仍有技能列表与角色设定
            memory.add_message({"role": "system", "content": build_system_prompt(skill_registry)})
            print("记忆已清空。")
            continue

        print("助手: ", end="", flush=True)
        for delta in agent.run_stream(user_input):
            print(delta, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
