from typing import Any, Callable, Dict

from fastmcp import FastMCP
from datetime import datetime


server = FastMCP()


@server.tool()
def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---- OpenAI function calling 注册表 ----

# 工具名 -> 实现函数
TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "get_current_time": get_current_time,
}

# 工具名 -> OpenAI tools 参数所需的 JSON Schema 描述
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的本地时间，格式为 YYYY-MM-DD HH:MM:SS",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """根据工具名和参数执行工具，返回工具结果"""
    if name not in TOOL_REGISTRY:
        raise ValueError(f"未知工具: {name}")
    return TOOL_REGISTRY[name](**arguments)
