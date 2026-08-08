"""load_skill 工具：按名称加载技能的完整指令（L2 信息）。

依赖 Agent 装配时通过 set_skill_registry 注入的技能注册表；
未注入或技能不存在时抛出 ToolError，由调度器统一兜底。
"""

from gearlink.core import tool as _tool_module
from gearlink.core.tool import register_tool
from gearlink.exceptions import ToolError
from gearlink.skills import SkillLoader


def load_skill(skill_name: str) -> dict[str, str]:
    """加载指定技能的完整指令正文。

    Args:
        skill_name: 已注册的技能名称。

    Returns:
        包含 skill_name 与 instructions 的字典。

    Raises:
        ToolError: 技能注册表未注入，或技能不存在/加载失败时抛出。
    """
    registry = _tool_module._skill_registry
    if registry is None:
        raise ToolError("技能注册表未注入，无法加载技能")
    skill = registry.get(skill_name)  # 未找到时抛 SkillNotFoundError（GearLinkError 子类）
    instructions = SkillLoader.load_full_instructions(skill)
    return {"skill_name": skill_name, "instructions": instructions}


register_tool(
    "load_skill",
    load_skill,
    {
        "description": "按名称加载指定技能的完整指令，加载后须严格遵循返回的指令执行任务",
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "要加载的技能名称（取自可用技能列表）",
                },
            },
            "required": ["skill_name"],
        },
    },
)
