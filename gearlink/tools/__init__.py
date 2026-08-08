"""工具扩展：内置工具实现，注册进 core.tool 的 TOOL_REGISTRY。"""

from gearlink.tools.builtin import get_current_time
from gearlink.tools.load_skill import load_skill

__all__ = [
    "get_current_time",
    "load_skill"
]
