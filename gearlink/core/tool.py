"""工具注册表 + JSON Schema + 调度器（三件套）。

工具注册表已从进程级全局单例升级为可实例化的 `ToolRegistry`（开发方向 §6.4），
支持通过 `ReactAgent(tool_registry=...)` 注入独立实例，使同一进程内多个 Agent
可持有隔离的工具集与技能集。模块级 `TOOL_REGISTRY` / `TOOL_SCHEMAS` /
`register_tool` / `call_tool` / `set_skill_registry` 保留为默认实例的兼容委托。

工具执行期间，当前 `ToolRegistry` 经 `contextvars` 暴露给工具函数（如 `load_skill`），
使其无需依赖全局状态即可解析所属 Agent 的技能注册表。
"""

from __future__ import annotations

import contextvars
import inspect
import types
from collections.abc import Callable
from typing import Any, Union, get_args, get_origin

from gearlink.exceptions import ToolError, ToolNotFoundError
from gearlink.skills import SkillRegistry  # 导入内存注册表（契约类型，非具体实现）

__all__ = [
    "ToolRegistry",
    "TOOL_REGISTRY",
    "TOOL_SCHEMAS",
    "register_tool",
    "call_tool",
    "set_skill_registry",
    "build_tool_schema",
    "get_current_tool_registry",
]

#: 工具执行期间的「当前注册表」上下文变量。
#: 由 `ToolRegistry.call_tool` 在调用工具函数前设置，供 `load_skill` 等工具
#: 解析所属 Agent 的技能注册表；未在工具执行上下文中时为 None（回退到全局）。
_current_tool_registry: contextvars.ContextVar[ToolRegistry | None] = contextvars.ContextVar(
    "gearlink_current_tool_registry", default=None
)


def get_current_tool_registry() -> ToolRegistry | None:
    """返回当前工具执行上下文中的 `ToolRegistry`，未在工具执行期间时返回 None。

    供 `load_skill` 等需要访问所属 Agent 注册表的工具函数使用。
    """
    return _current_tool_registry.get()


class ToolRegistry:
    """工具注册表：封装工具名→实现映射、JSON Schema 列表与技能注册表。

    每个实例独立维护自己的工具集与技能注册表，可经 `ReactAgent(tool_registry=...)`
    注入，使同一进程内多个 Agent 持有隔离的工具/技能配置。未注入时 Agent 使用
    模块级默认实例（`_default_registry`），行为与升级前完全一致。

    工具执行（`call_tool`）期间，本实例经 `contextvars` 注册为「当前注册表」，
    供 `load_skill` 等工具函数解析技能注册表，无需依赖全局状态。
    """

    def __init__(self) -> None:
        """初始化空注册表。"""
        self._registry: dict[str, Callable[..., Any]] = {}
        self._schemas: list[dict[str, Any]] = []
        self._skill_registry: SkillRegistry | None = None

    @property
    def registry(self) -> dict[str, Callable[..., Any]]:
        """工具名 → 实现函数的映射（只读视图）。"""
        return self._registry

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI tools 参数所需的 JSON Schema 列表（只读视图）。"""
        return self._schemas

    @property
    def skill_registry(self) -> SkillRegistry | None:
        """当前关联的技能注册表（由 Agent 注入）。"""
        return self._skill_registry

    def register_tool(self, name: str, func: Callable[..., Any], schema: dict[str, Any]) -> None:
        """显式登记一个工具实现及其 JSON Schema（禁止运行时隐式扫描）。

        Args:
            name: 本注册表内唯一的工具名。
            func: 工具实现函数。
            schema: OpenAI tools 参数所需的 function 级 schema（不含外层 type
                包装与 name），须包含 description 与 parameters 键，且 parameters
                与 func 签名严格同源。

        Raises:
            ToolError: 工具名重复，或 schema 缺少必需键时抛出。
        """
        if name in self._registry:
            raise ToolError(f"工具名重复: {name}")
        if "description" not in schema or "parameters" not in schema:
            raise ToolError(f"工具 {name} 的 schema 须包含 description 与 parameters 键")
        self._registry[name] = func
        self._schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": schema["description"],
                    "parameters": schema["parameters"],
                },
            }
        )

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """根据工具名和参数执行工具，返回工具结果。

        执行期间把本实例注册为「当前注册表」上下文，供工具函数解析技能注册表。

        Args:
            name: 已注册的工具名称。
            arguments: 工具函数的关键字参数。

        Returns:
            工具函数的返回值。

        Raises:
            ToolNotFoundError: 工具名未注册时抛出。
            ToolError: 工具执行异常时抛出（保留原始异常为 __cause__）。
        """
        if name not in self._registry:
            raise ToolNotFoundError(f"未知工具: {name}")
        token = _current_tool_registry.set(self)
        try:
            return self._registry[name](**arguments)
        except Exception as e:
            raise ToolError(f"工具 {name} 执行失败: {e}") from e
        finally:
            _current_tool_registry.reset(token)

    def set_skill_registry(self, registry: SkillRegistry) -> None:
        """将技能注册表注入本工具注册表，供 `load_skill` 工具解析。

        Args:
            registry: 技能注册表实例。
        """
        self._skill_registry = registry

    def get_schemas(self, whitelist: list[str] | None = None) -> list[dict[str, Any]]:
        """返回本轮可用的工具 schema（按白名单过滤）。

        Args:
            whitelist: 工具名白名单；None 表示返回全部。

        Returns:
            过滤后的 schema 列表。
        """
        if whitelist is None:
            return self._schemas
        allowed = set(whitelist)
        return [schema for schema in self._schemas if schema["function"]["name"] in allowed]

    def unregister_tool(self, name: str) -> None:
        """按名注销一个已注册的工具（幂等）。

        Args:
            name: 要注销的工具名。
        """
        self._registry.pop(name, None)
        index = next(
            (i for i, s in enumerate(self._schemas) if s["function"]["name"] == name), None
        )
        if index is not None:
            del self._schemas[index]


# ---- 模块级默认实例（向后兼容） ----

#: 默认注册表实例：模块级 `TOOL_REGISTRY` / `TOOL_SCHEMAS` / `register_tool` /
#: `call_tool` / `set_skill_registry` 均委托给此实例，行为与升级前完全一致。
_default_registry = ToolRegistry()

#: 工具名 → 实现函数（默认注册表视图，向后兼容）
TOOL_REGISTRY: dict[str, Callable[..., Any]] = _default_registry.registry

#: 工具名 → JSON Schema 列表（默认注册表视图，向后兼容）
TOOL_SCHEMAS: list[dict[str, Any]] = _default_registry.schemas


def register_tool(name: str, func: Callable[..., Any], schema: dict[str, Any]) -> None:
    """显式登记一个工具实现及其 JSON Schema（委托给默认注册表，向后兼容）。

    Args:
        name: 全局唯一的工具名。
        func: 工具实现函数。
        schema: OpenAI tools 参数所需的 function 级 schema，须包含 description
            与 parameters 键。

    Raises:
        ToolError: 工具名重复，或 schema 缺少必需键时抛出。
    """
    _default_registry.register_tool(name, func, schema)


def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    """根据工具名和参数执行工具（委托给默认注册表，向后兼容）。

    Args:
        name: 已注册的工具名称。
        arguments: 工具函数的关键字参数。

    Returns:
        工具函数的返回值。

    Raises:
        ToolNotFoundError: 工具名未注册时抛出。
        ToolError: 工具执行异常时抛出（保留原始异常为 __cause__）。
    """
    return _default_registry.call_tool(name, arguments)


def set_skill_registry(registry: SkillRegistry) -> None:
    """将技能注册表注入默认工具注册表（委托给默认注册表，向后兼容）。

    Args:
        registry: 技能注册表实例。
    """
    _default_registry.set_skill_registry(registry)


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
