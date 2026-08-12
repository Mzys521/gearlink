"""工具注册表与调度器测试：正常路径与异常路径。"""

import pytest

from gearlink.core.tool import (
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    build_tool_schema,
    call_tool,
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
