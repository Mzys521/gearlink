"""内置工具：工具实现与其 JSON Schema 同源定义，显式注册进 core.tool 的注册表。"""

from datetime import datetime

from gearlink.core.tool import register_tool


def get_current_time() -> str:
    """获取当前本地时间，格式为 YYYY-MM-DD HH:MM:SS。

    Returns:
        当前本地时间的格式化字符串。
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


register_tool(
    "get_current_time",
    get_current_time,
    {
        "description": "获取当前的本地时间，格式为 YYYY-MM-DD HH:MM:SS",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
)
