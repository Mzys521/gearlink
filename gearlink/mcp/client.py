"""MCP（Model Context Protocol）工具客户端：把外部 MCP 服务器的工具
映射进 GearLink 工具注册表（开发方向 §4.6，纯新增目录）。

设计要点：

- 远端工具以 ``mcp_<server>_<tool>`` 命名登记进 ``TOOL_REGISTRY``，
  ``core/`` 无感知，复用现有调度器与 ReAct 循环；
- 调用时转发参数、归一化结果为纯文本；远端报错也归一化为文本交还模型；
- 会话对象（``mcp.ClientSession``）由应用层注入，本模块不在导入期依赖
  ``mcp`` SDK（optional dependency：``pip install gearlink[mcp]``）。
"""

import asyncio
import concurrent.futures
from collections.abc import Awaitable
from typing import Any

from gearlink.core.tool import TOOL_REGISTRY, TOOL_SCHEMAS, register_tool
from gearlink.exceptions import ToolError

#: 远端工具登记进本地注册表时的命名空间前缀格式
_TOOL_NAME_TEMPLATE = "mcp_{server}_{tool}"


def _run_async(coro: Awaitable[Any]) -> Any:
    """同步执行一个协程；已处于事件循环中时改在新线程运行，避免死锁。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _normalize_result(result: Any) -> str:
    """把 MCP call_tool 的结果内容块归一化为纯文本。"""
    parts = [block.text for block in getattr(result, "content", []) if getattr(block, "text", "")]
    return "\n".join(parts)


class McpClient:
    """MCP 工具客户端：登记/注销一台远端 MCP 服务器的全部工具。

    典型用法（stdio 传输，需 ``pip install gearlink[mcp]``）::

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command="npx", args=["-y", "@modelcontextprotocol/server-everything"]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                client = McpClient("everything", session)
                client.register_tools()   # 之后 ReactAgent 即可调用远端工具
                ...
                client.unregister_tools()
    """

    def __init__(self, server_name: str, session: Any) -> None:
        """初始化 MCP 工具客户端。

        Args:
            server_name: 服务器命名空间标识（仅含字母/数字/下划线为佳），
                用于拼接本地工具名 ``mcp_<server>_<tool>``。
            session: 已建立并完成 initialize 的 MCP ClientSession（异步对象），
                须提供 ``list_tools()`` 与 ``call_tool(name, arguments)`` 协程方法。
        """
        self.server_name = server_name
        self.session = session
        self._registered: list[str] = []

    def register_tools(self) -> list[str]:
        """拉取远端工具清单并登记进本地注册表。

        Returns:
            本次登记的本地工具名列表（含命名空间前缀）。

        Raises:
            ToolError: 远端清单获取失败，或工具名与已注册工具冲突时抛出
                （冲突时已成功登记的部分保持登记，可调用 unregister_tools 回滚）。
        """
        try:
            listing = _run_async(self.session.list_tools())
        except Exception as e:
            raise ToolError(f"获取 MCP 服务器 {self.server_name} 的工具清单失败: {e}") from e

        names: list[str] = []
        for tool in listing.tools:
            local_name = _TOOL_NAME_TEMPLATE.format(server=self.server_name, tool=tool.name)
            register_tool(
                local_name,
                self._make_invoker(tool.name),
                {
                    "description": getattr(tool, "description", "") or f"MCP 工具 {tool.name}",
                    "parameters": getattr(tool, "inputSchema", None)
                    or {"type": "object", "properties": {}},
                },
            )
            names.append(local_name)
        self._registered.extend(names)
        return names

    def unregister_tools(self) -> None:
        """注销本客户端登记的全部工具（幂等）。"""
        for name in self._registered:
            TOOL_REGISTRY.pop(name, None)
            index = next(
                (i for i, s in enumerate(TOOL_SCHEMAS) if s["function"]["name"] == name), None
            )
            if index is not None:
                del TOOL_SCHEMAS[index]
        self._registered = []

    def _make_invoker(self, remote_name: str):
        """生成远端工具的本地调用函数：转发参数并归一化结果为文本。"""

        def invoke(**arguments: Any) -> str:
            result = _run_async(self.session.call_tool(remote_name, arguments))
            text = _normalize_result(result)
            if getattr(result, "isError", False):
                # 远端业务错误不归一化为异常：作为工具结果交还模型自行处置
                return (
                    f"MCP 工具 {remote_name} 返回错误: {text}"
                    if text
                    else (f"MCP 工具 {remote_name} 返回错误")
                )
            return text

        return invoke
