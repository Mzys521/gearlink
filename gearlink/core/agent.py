# gearlink/core/agent.py
"""
ReAct Agent 实现，集成了 Skills 模块。
支持通过技能目录加载专业知识，并在系统提示中动态注入可用技能列表，
Agent 可以通过调用 load_skill 工具按需获取完整技能指令。
"""

import json
from pathlib import Path
from typing import Optional, Union, List

from gearlink.core.memory import ShortTermMemory
# 从工具模块导入：工具 Schema、调用函数、以及注入技能注册表的 setter
from gearlink.core.tool import TOOL_SCHEMAS, call_tool, set_skill_registry
from gearlink.exceptions import GearLinkError
from gearlink.providers.base import ModelProvider, ModelResponse
# 导入技能模块的核心类
from gearlink.skills import SkillRegistry, SkillLoader, Skill

# ------------------- 系统提示配置 -------------------
# 基础系统提示，引导 Agent 使用工具，并告知其可通过 load_skill 获取专业知识。
BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "当需要实时信息（如当前时间）时，请调用可用的工具，而不是凭空回答。\n"
    "当遇到需要特定专业知识（如代码审查、文档编写等）的任务时，你可以调用 `load_skill` "
    "工具加载相应的技能指令，然后严格遵循这些指令执行。"
)

# ReAct 循环最大迭代次数，防止死循环
MAX_ITERATIONS = 10


class ReactAgent:
    """
    ReAct Agent：推理(Reason) -> 行动(Act) -> 观察(Observe) 循环。
    集成了 Skills 模块，支持加载外部技能以扩展专业知识。
    """

    def __init__(
        self,
        provider: ModelProvider,
        skills: Optional[Union[SkillRegistry, Path, List[Path]]] = None,
    ) -> None:
        """
        初始化 Agent。

        Args:
            provider: 模型提供者实例，用于调用 LLM。
            skills: 可选，技能来源。可以是：
                - SkillRegistry 实例（直接使用已注册的技能）
                - Path 或 str（单个技能目录路径）
                - List[Path]（多个技能目录路径）
                如果为 None，则不加载任何技能，Agent 行为与原始版本完全一致。
        """
        self.provider = provider
        self.memory = ShortTermMemory(max_message=20)

        # ========== Skills 模块装配 ==========
        # 1. 创建技能注册表，用于存储所有已发现技能的元数据（L1 信息）
        self.skill_registry = SkillRegistry()

        # 2. 根据传入的 skills 参数加载技能元数据
        self._load_skills(skills)

        # 3. 将技能注册表注入到工具模块，使 load_skill 工具能访问
        #    注意：必须在工具调用前完成注入，否则 load_skill 会报错
        set_skill_registry(self.skill_registry)

        # ========== 动态构建系统提示 ==========
        # 基础提示 + 可用技能列表（如果有）
        system_prompt = BASE_SYSTEM_PROMPT
        skill_list = self.skill_registry.list_all()
        if skill_list:
            # 构建技能列表描述，供 Agent 了解有哪些技能可用
            prompt_append = "\n\n<available_skills>\n"
            for skill in skill_list:
                prompt_append += f"- {skill.name}: {skill.description}\n"
            # 引导 Agent 如何使用这些技能
            prompt_append += (
                "当需要使用某个技能时，调用 `load_skill` 工具，并传入 skill_name。\n"
                "加载后，请严格按照返回的指令执行任务。\n"
                "</available_skills>"
            )
            system_prompt += prompt_append

        # 将最终的系统提示存入记忆（作为第一条消息）
        self.memory.add_message({"role": "system", "content": system_prompt})

    def _load_skills(self, skills_input):
        """
        处理 skills 参数，将技能元数据加载到注册表中。
        支持多种输入格式，灵活适配不同使用场景。

        Args:
            skills_input: None, SkillRegistry, Path, str, 或 List[Path/str]。
        """
        # 未提供技能，直接返回
        if skills_input is None:
            return

        # 如果直接传入 SkillRegistry，则直接使用，不再扫描文件系统
        if isinstance(skills_input, SkillRegistry):
            self.skill_registry = skills_input
            return

        # 统一转换为 Path 列表
        if isinstance(skills_input, (Path, str)):
            paths = [Path(skills_input)]
        elif isinstance(skills_input, list):
            paths = [Path(p) for p in skills_input]
        else:
            raise ValueError("skills must be SkillRegistry, Path, or list of Paths")

        # 逐个路径扫描，发现并注册技能
        for base_path in paths:
            if not base_path.exists():
                raise FileNotFoundError(f"Skills path {base_path} not found.")
            # 使用 SkillLoader 扫描目录，返回 Skill 对象列表（仅含元数据，L1）
            discovered = SkillLoader.discover_from_directory(base_path)
            for skill in discovered:
                self.skill_registry.register(skill)

    def _chat(self) -> ModelResponse:
        """
        调用模型进行对话。
        使用当前记忆中的所有消息和已注册的工具 Schema（TOOL_SCHEMAS）。
        """
        return self.provider.chat(
            messages=self.memory.get_messages(),
            tools=TOOL_SCHEMAS,  # 已包含 load_skill 工具（若已注册）
        )

    def run(self, user_input: str) -> str:
        """
        执行一次用户请求，内部运行 ReAct 循环直到得到最终答案。

        Args:
            user_input: 用户的输入文本。

        Returns:
            模型给出的最终答案文本。
        """
        # 将用户输入添加到记忆
        self.memory.add_message({"role": "user", "content": user_input})

        # ReAct 循环
        for _ in range(MAX_ITERATIONS):
            # 1. 推理：让模型根据当前上下文决定下一步（生成回复或调用工具）
            response = self._chat()

            # 如果模型没有调用工具，说明已经生成了最终答案，直接返回
            if not response.tool_calls:
                self.memory.add_message({"role": "assistant", "content": response.content})
                return response.content

            # 2. 行动：模型决定调用工具，将工具调用记录到记忆
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

            # 3. 观察：执行每个工具调用，并将结果写回记忆
            for tool_call in response.tool_calls:
                name = tool_call.name
                # 解析参数（若解析失败则使用空字典）
                try:
                    arguments = json.loads(tool_call.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                # 执行工具（call_tool 会从 TOOL_REGISTRY 中查找并调用）
                try:
                    result = call_tool(name, arguments)
                    # 将结果转为 JSON 字符串，以便统一存储
                    result_text = json.dumps(result, ensure_ascii=False)
                except GearLinkError as e:
                    # 工具执行失败时，将错误信息作为结果返回，让模型自行处理
                    result_text = f"工具调用失败: {e}"

                # 打印日志便于调试（保留原有行为）
                print(f"[工具调用] {name}({arguments}) -> {result_text}")

                # 将工具执行结果作为 tool 角色消息存入记忆
                self.memory.add_message(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    }
                )
            # 循环继续，模型将基于工具结果进行下一轮推理

        # 达到最大迭代次数仍未得到答案
        return "已达到最大推理轮数，无法得出最终答案。"


# ------------------- 入口示例（方便直接运行测试） -------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    from gearlink.providers.openai_provider import OpenAIProvider

    load_dotenv()  # 加载 .env 配置（如 DEEPSEEK_API_KEY）

    # 实例化 Agent，装配示例技能（基于本文件定位，不依赖运行时工作目录）
    skill_dir = Path(__file__).resolve().parents[1] / "examples" / "skill_demo"
    agent = ReactAgent(
        provider=OpenAIProvider(),
        skills=skill_dir,  # 技能目录：gearlink/examples/skill_demo
    )

    # 交互式循环
    while True:
        user_input = input("用户: ")
        if user_input == "exit":
            break
        print("助手:", agent.run(user_input))