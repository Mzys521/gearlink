"""工具注册表与调度器测试：正常路径与异常路径。"""

import pytest

from gearlink.core.tool import TOOL_REGISTRY, TOOL_SCHEMAS, call_tool, register_tool
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
