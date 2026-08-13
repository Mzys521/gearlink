"""demo 自定义工具：实现与 JSON Schema 同源定义，经 register_tool 显式登记。

导入本模块即触发工具注册（副作用注册，与 gearlink.tools.builtin 同风格）。
登记后，之后创建的 ReactAgent 每轮都会把这些工具的 schema 交给模型。
"""

from gearlink import register_tool


def add(a: float, b: float) -> float:
    """两数相加

    Args:
        a: 加数 a。
        b: 加数 b。

    Returns:
        两个数的和。
    """
    return a + b


def multiply(a: float, b: float) -> float:
    """两数相乘

    Args:
        a: 因数 a。
        b: 因数 b。

    Returns:
        两个数的积。
    """
    return a * b


def get_weather(city: str) -> str:
    """查询指定城市的天气（演示用，返回内置模拟数据）

    Args:
        city: 城市名称，如「北京」「上海」。

    Returns:
        该城市的模拟天气描述文本。
    """
    mock_weather = {
        "北京": "晴朗，气温 24℃，微风",
        "上海": "多云，气温 27℃，东南风 3 级",
        "广州": "小雨，气温 30℃，湿度较高",
    }
    return mock_weather.get(city, f"{city}：晴朗，气温 25℃，微风")


register_tool(
    "add",
    add,
    {
        "description": "计算两个数的和",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "加数 a"},
                "b": {"type": "number", "description": "加数 b"},
            },
            "required": ["a", "b"],
        },
    },
)

register_tool(
    "multiply",
    multiply,
    {
        "description": "计算两个数的积",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "因数 a"},
                "b": {"type": "number", "description": "因数 b"},
            },
            "required": ["a", "b"],
        },
    },
)

register_tool(
    "get_weather",
    get_weather,
    {
        "description": "查询指定城市的天气情况",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，如北京、上海"},
            },
            "required": ["city"],
        },
    },
)
