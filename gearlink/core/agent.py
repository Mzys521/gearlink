"""ReAct Agent 实现，支持按需注入技能注册表（Skills）。

装配期注入 `SkillRegistry` 后，Agent 会在默认系统提示中动态列出可用技能（L1 元数据），
模型可通过已注册的 `load_skill` 工具按名获取完整技能指令（L2）并严格遵循执行；
注册表经 `core.tool.set_skill_registry` 登记，供 `load_skill` 工具解析。
"""

import json
import logging
from collections.abc import Iterator

from gearlink.core.memory import Memory, MemoryManager, ShortTermMemory
from gearlink.core.tool import TOOL_SCHEMAS, call_tool, set_skill_registry
from gearlink.exceptions import GearLinkError
from gearlink.providers.base import ModelProvider, ModelResponse
from gearlink.skills import SkillRegistry
from gearlink.utils.token_count import estimate_tokens

# ------------------- 系统提示配置 -------------------

#: 内置系统提示：引导 Agent 优先使用工具回答实时问题
SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "当需要实时信息（如当前时间）时，请调用可用的工具，而不是凭空回答。"
)

#: 注入技能注册表后追加的提示：告知模型可用技能列表的用途与加载方式
_SKILL_HINT_PROMPT = (
    "当遇到需要特定专业知识（如代码审查、文档编写等）的任务时，你可以调用 `load_skill` "
    "工具加载相应的技能指令，然后严格遵循这些指令执行。\n"
    "当前可用技能："
)

# ReAct 循环最大迭代次数，防止死循环
MAX_ITERATIONS = 10

#: 工具结果写入记忆前的 token 估算上限，超出时截断防止撑爆上下文
MAX_TOOL_RESULT_TOKENS = 2000

#: 模块级日志器：记录 ReAct 循环中的工具调用等过程信息
logger = logging.getLogger(__name__)


def _build_system_prompt(skill_registry: SkillRegistry | None = None) -> str:
    """组装系统提示：基础提示 + 可用技能列表（仅当注入非空注册表时）。

    Args:
        skill_registry: 技能注册表；None 或不含技能时返回基础提示。

    Returns:
        组装后的系统提示文本。
    """
    skills = skill_registry.list_all() if skill_registry is not None else []
    if not skills:
        return SYSTEM_PROMPT
    skill_lines = "\n".join(f"- {skill.name}: {skill.description}" for skill in skills)
    return f"{SYSTEM_PROMPT}\n{_SKILL_HINT_PROMPT}\n{skill_lines}"


class ReactAgent:
    """ReAct Agent：推理(Reason) -> 行动(Act) -> 观察(Observe) 循环"""

    def __init__(
        self,
        provider: ModelProvider,
        memory: Memory | MemoryManager | None = None,
        retrieve_every_iteration: bool = False,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        """初始化 Agent。

        Args:
            provider: 模型提供者实例。
            memory: 记忆实现；None 时默认使用 ShortTermMemory(max_message=20)
                并附加内置 SYSTEM_PROMPT（注入技能时为含技能列表的组装提示）。
                传入 MemoryManager 时启用长期记忆检索注入。
            retrieve_every_iteration: 是否每轮 ReAct 迭代都携带查询注入长期记忆；
                False 表示仅首轮注入（现状，后续轮次上下文已在短期记忆中）。
                开启时依赖 build_context 的短期窗口去重避免重复注入。
            skill_registry: 技能注册表；非 None 时登记给 load_skill 工具使用，
                且仅在 memory 为 None 时把可用技能列表拼入默认系统提示。
                自行注入 memory 时须自行添加所需的系统提示消息。
        """
        self.provider = provider
        self.retrieve_every_iteration = retrieve_every_iteration
        if skill_registry is not None:
            # 供 load_skill 工具按名解析技能（core/tool.py 全局注入点）
            set_skill_registry(skill_registry)
        if memory is None:
            memory = ShortTermMemory(max_message=20)
            memory.add_message({"role": "system", "content": _build_system_prompt(skill_registry)})
        self.memory = memory

    def _chat(self, query: str | None = None) -> ModelResponse:
        """基于记忆中的消息请求模型。

        Args:
            query: 语义检索查询；仅对 MemoryManager 生效，用于注入长期记忆相关历史。

        Returns:
            ModelResponse: 统一响应结构，含文本内容与工具调用请求。

        Raises:
            ProviderError: 底层模型服务调用失败时由 provider 抛出，Agent 不捕获。
        """
        if isinstance(self.memory, MemoryManager):
            messages = self.memory.build_context(query or "")
        else:
            messages = self.memory.get_messages()
        return self.provider.chat(messages=messages, tools=TOOL_SCHEMAS)

    def run(self, user_input: str) -> str:
        """执行一次用户请求，内部运行 ReAct 循环直到得到最终答案。

        工具执行失败属可恢复信号：以文本结果写回记忆交给模型处理，不中断循环。

        Args:
            user_input: 用户的输入文本。

        Returns:
            模型给出的最终答案文本；达到 MAX_ITERATIONS 仍未收敛时返回兜底文案。

        Raises:
            ProviderError: 模型服务调用失败，由调用方决定重试策略。
        """
        # 将用户输入添加到记忆
        self.memory.add_message({"role": "user", "content": user_input})

        for iteration in range(MAX_ITERATIONS):
            # 推理：默认仅首轮携带查询注入长期记忆；开启 retrieve_every_iteration 时每轮都注入
            query = user_input if (iteration == 0 or self.retrieve_every_iteration) else None
            response = self._chat(query=query)

            # 如果模型没有调用工具，说明已经生成了最终答案，直接返回
            if not response.tool_calls:
                self.memory.add_message({"role": "assistant", "content": response.content})
                return response.content

            # 行动 + 观察：记录工具调用并执行，将结果写回记忆
            self._execute_tool_calls(response)
            # 循环继续，模型将基于工具结果进行下一轮推理

        # 达到最大迭代次数仍未得到答案
        return "已达到最大推理轮数，无法得出最终答案。"

    def run_stream(self, user_input: str) -> Iterator[str]:
        """执行一次用户请求，以流式逐步产出模型文本。

        与 `run` 共享 ReAct 循环与工具执行逻辑，区别在于模型输出经
        `provider.chat_stream` 逐片段产出；工具调用阶段自动执行并继续循环，
        该阶段若模型附带文本增量（如思考叙述）也会一并流出。

        Args:
            user_input: 用户的输入文本。

        Yields:
            模型输出的文本增量片段；全部片段拼接即为最终答案。

        Raises:
            ProviderError: 模型服务调用失败，由调用方决定重试策略。
            GearLinkError: 提供者的流式响应缺少携带完整响应的终止事件。
        """
        # 将用户输入添加到记忆
        self.memory.add_message({"role": "user", "content": user_input})

        for iteration in range(MAX_ITERATIONS):
            query = user_input if (iteration == 0 or self.retrieve_every_iteration) else None
            if isinstance(self.memory, MemoryManager):
                messages = self.memory.build_context(query or "")
            else:
                messages = self.memory.get_messages()

            # 流式推理：实时转出文本增量，终止事件携带累积后的完整响应
            response: ModelResponse | None = None
            for chunk in self.provider.chat_stream(messages=messages, tools=TOOL_SCHEMAS):
                if chunk.delta:
                    yield chunk.delta
                if chunk.response is not None:
                    response = chunk.response
            if response is None:
                raise GearLinkError("提供者的流式响应缺少终止事件（未携带完整 response）")

            # 没有工具调用即为最终答案：记入记忆并结束流
            if not response.tool_calls:
                self.memory.add_message({"role": "assistant", "content": response.content})
                return

            # 行动 + 观察：记录工具调用并执行，将结果写回记忆后继续循环
            self._execute_tool_calls(response)

        # 达到最大迭代次数仍未得到答案
        yield "已达到最大推理轮数，无法得出最终答案。"

    def _execute_tool_calls(self, response: ModelResponse) -> None:
        """记录助手的工具调用消息，执行各工具并将结果写回记忆。

        Args:
            response: 含工具调用请求的模型响应。
        """
        # 行动：模型决定调用工具，将工具调用记录到记忆
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

        # 观察：执行每个工具调用，并将结果写回记忆
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
                # 超出预算时按 4 字符/token 的启发式反推字符上限截断
                if estimate_tokens(result_text) > MAX_TOOL_RESULT_TOKENS:
                    result_text = (
                        result_text[: MAX_TOOL_RESULT_TOKENS * 4] + "\n...(工具结果过长，已截断)"
                    )
            except GearLinkError as e:
                # 工具执行失败是可恢复信号：将错误信息作为结果写回，让模型自行处理
                result_text = f"工具调用失败: {e}"

            logger.info("[工具调用] %s(%s) -> %s", name, arguments, result_text)

            # 将工具执行结果作为 tool 角色消息存入记忆
            self.memory.add_message(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                }
            )
