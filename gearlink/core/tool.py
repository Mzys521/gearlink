from collections.abc import Callable
from typing import Any

from gearlink.exceptions import ToolError, ToolNotFoundError

# ---- OpenAI function calling 注册表 ----
# 具体工具实现位于 gearlink/tools/，通过 register_tool 显式登记进本注册表。

# 工具名 -> 实现函数
TOOL_REGISTRY: dict[str, Callable[..., Any]] = {}

# 工具名 -> OpenAI tools 参数所需的 JSON Schema 描述
TOOL_SCHEMAS: list[dict[str, Any]] = []


def register_tool(name: str, func: Callable[..., Any], schema: dict[str, Any]) -> None:
    """显式登记一个工具实现及其 JSON Schema（禁止运行时隐式扫描）。

    Args:
        name: 全局唯一的工具名。
        func: 工具实现函数。
        schema: OpenAI tools 参数所需的 function 级 schema（不含外层 type 包装与 name），
            须包含 description 与 parameters 键，且 parameters 与 func 签名严格同源。

    Raises:
        ToolError: 工具名重复，或 schema 缺少必需键时抛出。
    """
    if name in TOOL_REGISTRY:
        raise ToolError(f"工具名重复: {name}")
    if "description" not in schema or "parameters" not in schema:
        raise ToolError(f"工具 {name} 的 schema 须包含 description 与 parameters 键")
    TOOL_REGISTRY[name] = func
    TOOL_SCHEMAS.append(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": schema["description"],
                "parameters": schema["parameters"],
            },
        }
    )


def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    """根据工具名和参数执行工具，返回工具结果。

    Args:
        name: 已注册的工具名称。
        arguments: 工具函数的关键字参数。

    Returns:
        工具函数的返回值。

    Raises:
        ToolNotFoundError: 工具名未注册时抛出。
        ToolError: 工具执行异常时抛出（保留原始异常为 __cause__）。
    """
    if name not in TOOL_REGISTRY:
        raise ToolNotFoundError(f"未知工具: {name}")
    try:
        return TOOL_REGISTRY[name](**arguments)
    except Exception as e:
        # 调度器统一兜底：包装为项目异常体系，保留原始异常链
        raise ToolError(f"工具 {name} 执行失败: {e}") from e
