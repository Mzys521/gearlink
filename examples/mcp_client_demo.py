"""GearLink MCP 客户端示例（消费外部 MCP 服务器的工具）

演示 `McpClient`（开发方向 §4.6）：把远端 MCP 服务器的工具以
``mcp_<server>_<tool>`` 命名登记进本地注册表，ReAct Agent 无感知复用。

前置条件：
    pip install gearlink[mcp]
    根目录 .env 中配置 DEEPSEEK_API_KEY（驱动 Agent 的模型）
    设置环境变量指定一台 MCP 服务器（stdio 传输），例如：
        MCP_SERVER_COMMAND=npx
        MCP_SERVER_ARGS=-y @modelcontextprotocol/server-everything

运行方式：
    python examples/mcp_client_demo.py

缺少 mcp 依赖或未配置服务器命令时，本示例会给出友好提示。
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from gearlink import McpClient, OpenAIProvider, ReactAgent

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("本示例需要 mcp 依赖：pip install gearlink[mcp]")
    sys.exit(0)


async def main() -> None:
    command = os.environ.get("MCP_SERVER_COMMAND")
    if not command:
        print("请先设置 MCP_SERVER_COMMAND / MCP_SERVER_ARGS 环境变量指定 MCP 服务器，例如：")
        print("  MCP_SERVER_COMMAND=npx")
        print("  MCP_SERVER_ARGS=-y @modelcontextprotocol/server-everything")
        return

    args = os.environ.get("MCP_SERVER_ARGS", "").split()
    params = StdioServerParameters(command=command, args=args)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1) 登记远端工具：core/ 无感知，复用现有调度器与 ReAct 循环
            client = McpClient("external", session)
            names = client.register_tools()
            print("已登记远端工具：", names)

            # 2) 照常运行 Agent：模型可直接调用 mcp_external_* 工具
            agent = ReactAgent(provider=OpenAIProvider())
            answer = agent.run("请从已注册的工具中挑选一个演示调用，并总结结果")
            print("助手：", answer)

            # 3) 会话结束注销工具（幂等）
            client.unregister_tools()


if __name__ == "__main__":
    asyncio.run(main())
