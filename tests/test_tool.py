"""工具注册表与调度器测试：正常路径与异常路径。"""

import pytest

from gearlink.core.tool import (
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    ToolRegistry,
    build_tool_schema,
    call_tool,
    get_current_tool_registry,
    register_tool,
)
from gearlink.exceptions import ToolError, ToolNotFoundError


def test_call_tool_executes_registered_tool():
    result = call_tool("get_current_time", {})
    # 格式应为 YYYY-MM-DD HH:MM:SS
    assert len(result) == 19
    assert result[4] == "-" and result[10] == " "


def test_call_tool_raises_on_unknown_name():
    with pytest.raises(ToolNotFoundError):
        call_tool("no_such_tool", {})


def test_call_tool_wraps_execution_error():
    def boom() -> None:
        raise RuntimeError("boom")

    TOOL_REGISTRY["boom"] = boom
    try:
        with pytest.raises(ToolError) as exc_info:
            call_tool("boom", {})
        assert exc_info.value.__cause__ is not None
    finally:
        del TOOL_REGISTRY["boom"]


def test_schemas_match_registry():
    """schema 与实现同源：每个注册工具都有对应的 JSON Schema。"""
    schema_names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert schema_names == set(TOOL_REGISTRY)


def test_register_tool_adds_to_registry_and_schemas():
    def echo() -> str:
        return "ok"

    register_tool(
        "echo",
        echo,
        {
            "description": "回声工具",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    )
    try:
        assert TOOL_REGISTRY["echo"] is echo
        assert call_tool("echo", {}) == "ok"
        (schema,) = [s for s in TOOL_SCHEMAS if s["function"]["name"] == "echo"]
        assert schema["type"] == "function"
        assert schema["function"]["description"] == "回声工具"
    finally:
        del TOOL_REGISTRY["echo"]
        TOOL_SCHEMAS[:] = [s for s in TOOL_SCHEMAS if s["function"]["name"] != "echo"]


def test_register_tool_raises_on_duplicate_name():
    def dummy() -> None:
        pass

    schema = {
        "description": "重复名工具",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    with pytest.raises(ToolError):
        register_tool("get_current_time", dummy, schema)


def test_register_tool_raises_on_incomplete_schema():
    def dummy() -> None:
        pass

    with pytest.raises(ToolError):
        register_tool("incomplete", dummy, {"description": "缺少 parameters"})
    # 注册失败不得留下残留登记
    assert "incomplete" not in TOOL_REGISTRY
    assert all(s["function"]["name"] != "incomplete" for s in TOOL_SCHEMAS)


# ------------------- schema 自动生成（build_tool_schema，开发方向 §4.4） -------------------


def test_build_tool_schema_from_signature_and_docstring():
    def multiply(a: float, b: float) -> float:
        """计算两个数的乘积"""
        return a * b

    schema = build_tool_schema(multiply)
    assert schema["description"] == "计算两个数的乘积"
    assert schema["parameters"] == {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }


def test_build_tool_schema_optional_params_not_required():
    def greet(name: str, times: int = 1, note: str | None = None) -> str:
        return name

    schema = build_tool_schema(greet)
    assert schema["parameters"]["required"] == ["name"]
    assert schema["parameters"]["properties"]["times"] == {"type": "integer"}
    assert schema["parameters"]["properties"]["note"] == {"type": "string"}


def test_build_tool_schema_explicit_description_wins():
    def echo(text: str) -> str:
        """docstring 描述"""
        return text

    assert build_tool_schema(echo, description="显式描述")["description"] == "显式描述"


def test_build_tool_schema_generated_works_with_register_tool():
    def square(x: int) -> int:
        """计算平方"""
        return x * x

    register_tool("auto_schema_square", square, build_tool_schema(square))
    assert call_tool("auto_schema_square", {"x": 4}) == 16


def test_build_tool_schema_raises_on_missing_annotation():
    def bad(x):  # noqa: ANN001
        return x

    with pytest.raises(ToolError, match="缺少类型标注"):
        build_tool_schema(bad)


def test_build_tool_schema_raises_on_varargs():
    def bad(*args: str) -> str:
        return ""

    with pytest.raises(ToolError, match="无法自动生成"):
        build_tool_schema(bad)


# ------------------- ToolRegistry 实例化与隔离（开发方向 §6.4） -------------------


def test_tool_registry_instance_isolation():
    """两个 ToolRegistry 实例的注册表相互隔离：在一个实例注册的工具不出现在另一个。"""
    registry_a = ToolRegistry()
    registry_b = ToolRegistry()

    def echo() -> str:
        return "ok"

    registry_a.register_tool(
        "iso_echo",
        echo,
        {
            "description": "隔离测试工具",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    )

    assert "iso_echo" in registry_a.registry
    assert "iso_echo" not in registry_b.registry
    assert any(s["function"]["name"] == "iso_echo" for s in registry_a.schemas)
    assert not any(s["function"]["name"] == "iso_echo" for s in registry_b.schemas)


def test_tool_registry_register_and_call():
    """在实例上注册工具并调用，验证返回结果。"""
    registry = ToolRegistry()

    def add(a: int, b: int) -> int:
        return a + b

    registry.register_tool(
        "iso_add",
        add,
        {
            "description": "加法",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        },
    )

    assert registry.call_tool("iso_add", {"a": 2, "b": 3}) == 5


def test_tool_registry_call_tool_sets_context():
    """call_tool 执行期间 get_current_tool_registry() 返回当前注册表实例。"""
    registry = ToolRegistry()
    captured = []

    def capture_registry() -> str:
        captured.append(get_current_tool_registry())
        return "ok"

    registry.register_tool(
        "iso_capture",
        capture_registry,
        {
            "description": "捕获当前注册表",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    )

    registry.call_tool("iso_capture", {})

    assert captured == [registry]


def test_tool_registry_get_schemas_with_whitelist():
    """get_schemas 按白名单过滤：注册 2 个工具，白名单仅保留 1 个。"""
    registry = ToolRegistry()
    schema = {
        "description": "测试",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    registry.register_tool("iso_alpha", lambda: "a", schema)
    registry.register_tool("iso_beta", lambda: "b", schema)

    filtered = registry.get_schemas(["iso_alpha"])
    assert len(filtered) == 1
    assert filtered[0]["function"]["name"] == "iso_alpha"


def test_tool_registry_unregister_tool():
    """unregister_tool 从注册表和 schema 列表中同时移除工具。"""
    registry = ToolRegistry()

    def dummy() -> str:
        return "ok"

    registry.register_tool(
        "iso_dummy",
        dummy,
        {
            "description": "临时工具",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    )

    assert "iso_dummy" in registry.registry
    assert any(s["function"]["name"] == "iso_dummy" for s in registry.schemas)

    registry.unregister_tool("iso_dummy")

    assert "iso_dummy" not in registry.registry
    assert not any(s["function"]["name"] == "iso_dummy" for s in registry.schemas)


def test_tool_registry_duplicate_name_raises():
    """同一实例上重复注册同名工具时抛出 ToolError。"""
    registry = ToolRegistry()

    def dummy() -> str:
        return "ok"

    schema = {
        "description": "重复名",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    registry.register_tool("iso_dup", dummy, schema)

    with pytest.raises(ToolError):
        registry.register_tool("iso_dup", dummy, schema)


def test_tool_registry_skill_registry():
    """set_skill_registry 注入的技能注册表可通过 skill_registry 属性读取。"""
    from gearlink.skills import SkillRegistry

    registry = ToolRegistry()
    assert registry.skill_registry is None

    skills = SkillRegistry()
    registry.set_skill_registry(skills)

    assert registry.skill_registry is skills
