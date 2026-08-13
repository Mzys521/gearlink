"""load_skill 工具：按名称加载技能的完整指令（L2 信息）。

依赖 Agent 装配时通过 set_skill_registry 注入的技能注册表；
未注入或技能不存在时抛出 ToolError，由调度器统一兜底。

技能注册表解析顺序（开发方向 §6.4）：
1. 当前工具执行上下文中的 ToolRegistry（经 contextvars 设置，支持实例化注册表隔离）；
2. 回退到模块级默认注册表的 _skill_registry（向后兼容）。
"""

from gearlink.core import tool as _tool_module
from gearlink.core.tool import get_current_tool_registry, register_tool
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
    # 优先从当前工具执行上下文解析技能注册表（实例化注册表路径）
    current = get_current_tool_registry()
    if current is not None:
        registry = current.skill_registry
    else:
        # 回退到模块级默认注册表（向后兼容）
        registry = _tool_module._default_registry.skill_registry

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
