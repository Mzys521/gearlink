"""McpClient 测试：mock 远端 MCP 会话，不进行真实连接。"""

from types import SimpleNamespace

import pytest

from gearlink.core.tool import TOOL_REGISTRY, TOOL_SCHEMAS, call_tool
from gearlink.exceptions import ToolError
from gearlink.mcp import McpClient


@pytest.fixture(autouse=True)
def clean_registry():
    """隔离全局注册表：测试前后恢复原状，避免 MCP 工具残留影响其他用例。"""
    registry_backup = dict(TOOL_REGISTRY)
    schemas_backup = list(TOOL_SCHEMAS)
    yield
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(registry_backup)
    TOOL_SCHEMAS.clear()
    TOOL_SCHEMAS.extend(schemas_backup)


class FakeSession:
    """模拟 MCP ClientSession 的两个协程方法。"""

    def __init__(self, tools, results=None, list_error=None) -> None:
        self.tools = tools
        self.results = results or {}
        self.list_error = list_error
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self):
        if self.list_error:
            raise self.list_error
        return SimpleNamespace(tools=self.tools)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.results[name]


def make_result(text: str, is_error: bool = False):
    return SimpleNamespace(content=[SimpleNamespace(text=text)], isError=is_error)


def make_tools():
    return [
        SimpleNamespace(
            name="echo",
            description="回声工具",
            inputSchema={"type": "object", "properties": {"text": {"type": "string"}}},
        ),
        SimpleNamespace(name="bare", description="", inputSchema=None),
    ]


def test_register_tools_maps_remote_tools_with_namespace_prefix():
    session = FakeSession(make_tools())
    client = McpClient("demo", session)

    names = client.register_tools()

    assert names == ["mcp_demo_echo", "mcp_demo_bare"]
    assert "mcp_demo_echo" in TOOL_REGISTRY
    schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "mcp_demo_echo")
    assert schema["function"]["description"] == "回声工具"
    assert schema["function"]["parameters"]["properties"] == {"text": {"type": "string"}}
    # 缺失 inputSchema / description 时给出兜底值
    bare = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "mcp_demo_bare")
    assert bare["function"]["parameters"] == {"type": "object", "properties": {}}
    assert bare["function"]["description"]


def test_call_forwards_arguments_and_normalizes_result():
    session = FakeSession(make_tools(), results={"echo": make_result("回声内容")})
    client = McpClient("demo", session)
    client.register_tools()

    result = call_tool("mcp_demo_echo", {"text": "回声内容"})

    assert result == "回声内容"
    # 参数被原样转发给远端工具
    assert session.calls == [("echo", {"text": "回声内容"})]


def test_remote_error_is_normalized_to_text():
    session = FakeSession(make_tools(), results={"echo": make_result("参数不合法", is_error=True)})
    client = McpClient("demo", session)
    client.register_tools()

    result = call_tool("mcp_demo_echo", {"text": "x"})

    assert "返回错误" in result and "参数不合法" in result


def test_unregister_tools_removes_registry_and_schemas():
    client = McpClient("demo", FakeSession(make_tools()))
    client.register_tools()

    client.unregister_tools()
    client.unregister_tools()  # 幂等

    assert "mcp_demo_echo" not in TOOL_REGISTRY
    assert not any(s["function"]["name"].startswith("mcp_demo_") for s in TOOL_SCHEMAS)


def test_duplicate_registration_raises_tool_error():
    client = McpClient("demo", FakeSession(make_tools()))
    client.register_tools()

    another = McpClient("demo", FakeSession(make_tools()))
    with pytest.raises(ToolError, match="重复"):
        another.register_tools()


def test_list_tools_failure_raises_tool_error():
    session = FakeSession([], list_error=ConnectionError("连不上"))
    client = McpClient("demo", session)

    with pytest.raises(ToolError, match="工具清单失败"):
        client.register_tools()
