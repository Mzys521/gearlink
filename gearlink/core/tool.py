import inspect
import types
from collections.abc import Callable
from typing import Any, Union, get_args, get_origin

from gearlink.exceptions import ToolError, ToolNotFoundError
from gearlink.skills import SkillRegistry  # 导入内存注册表

__all__ = [
    "TOOL_REGISTRY",
    "TOOL_SCHEMAS",
    "register_tool",
    "call_tool",
    "set_skill_registry",
    "build_tool_schema",
]

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


# ---- 全局技能注册表（由 Agent 初始化时注入） ----
_skill_registry: SkillRegistry | None = None


def set_skill_registry(registry: SkillRegistry) -> None:
    """由 Agent 调用，将当前使用的技能注册表注入工具模块。"""
    global _skill_registry
    _skill_registry = registry


# ---- schema 自动生成辅助（开发方向 §4.4，纯新增，不替代手写路径） ----

#: Python 类型标注到 JSON Schema 类型的映射
_JSON_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type_of(annotation: Any) -> tuple[str, bool]:
    """将类型标注映射为 JSON Schema 类型名。

    Args:
        annotation: 参数的类型标注（支持常见内置类型与 `X | None` 可选形式）。

    Returns:
        (类型名, 是否可为 None) 二元组；无法识别时回退 string。
    """
    if get_origin(annotation) in (Union, types.UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        nullable = len(args) < len(get_args(annotation))
        json_type = _JSON_TYPE_MAP.get(args[0], "string") if args else "string"
        return json_type, nullable
    if isinstance(annotation, type):
        return _JSON_TYPE_MAP.get(annotation, "string"), False
    return "string", False


def build_tool_schema(func: Callable[..., Any], description: str | None = None) -> dict[str, Any]:
    """从函数签名 + docstring 推导工具的 JSON Schema（降低接入成本）。

    纯新增辅助（开发方向 §4.4）：不替代手写 schema 路径，生成的 schema
    可直接传给 `register_tool`。参数描述无法从签名推导，仅给出类型；
    需要精细描述时仍建议手写。

    Args:
        func: 工具实现函数；docstring 首行作为工具描述（未提供时用 description）。
        description: 显式指定的工具描述；优先级高于 docstring。

    Returns:
        function 级 schema（含 description 与 parameters，不含外层 type 包装），
        与 `register_tool` 的 schema 参数格式一致。

    Raises:
        ToolError: 函数含无法映射的参数形态（*args / **kwargs / 无标注）时抛出。
    """
    signature = inspect.signature(func)
    hints = func.__annotations__
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ToolError(f"函数 {func.__name__} 含 *args/**kwargs，无法自动生成 schema")
        annotation = hints.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            raise ToolError(f"函数 {func.__name__} 的参数 {name} 缺少类型标注，无法自动生成 schema")
        json_type, _nullable = _json_type_of(annotation)
        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    doc_summary = (inspect.getdoc(func) or "").splitlines()
    return {
        "description": description or (doc_summary[0] if doc_summary else func.__name__),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
