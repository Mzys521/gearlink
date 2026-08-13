"""GearLink 自定义工具示例

演示工具扩展点（四类扩展点之一）：`register_tool` 显式登记工具与 JSON Schema，
`call_tool` 统一调度（集中错误兜底）。

无需 API key 即可运行前两部分（工具注册与直接调度）：
    python examples/custom_tool_demo.py

配置了 DEEPSEEK_API_KEY 时还会演示 Agent 在 ReAct 循环中自动调用自定义工具。
"""

import os

from dotenv import load_dotenv

from gearlink import TOOL_REGISTRY, ToolError, ToolNotFoundError, call_tool, register_tool

load_dotenv()


def multiply(a: float, b: float) -> float:
    """两数相乘"""
    return a * b


def main() -> None:
    # 1. 注册自定义工具：schema 与函数签名同源定义（开发规范 §6）
    register_tool(
        "multiply",
        multiply,
        {
            "description": "计算两个数的乘积",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "乘数 a"},
                    "b": {"type": "number", "description": "乘数 b"},
                },
                "required": ["a", "b"],
            },
        },
    )
    print("已注册工具:", list(TOOL_REGISTRY))

    # 2. 直接调度：call_tool 是集中错误兜底的调度器
    print("multiply(6, 7) =", call_tool("multiply", {"a": 6, "b": 7}))

    # 3. 错误兜底演示：未知工具与执行异常都被包装为项目异常体系
    try:
        call_tool("not_registered", {})
    except ToolNotFoundError as e:
        print("捕获 ToolNotFoundError:", e)

    try:
        call_tool("multiply", {"a": "非数字", "b": "也不是数字"})
    except ToolError as e:
        print("捕获 ToolError:", e)

    # 4. （可选）Agent 在 ReAct 循环中自动选用自定义工具
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("\n未配置 DEEPSEEK_API_KEY，跳过 Agent 联动演示。")
        return

    from gearlink import OpenAIProvider, ReactAgent

    agent = ReactAgent(provider=OpenAIProvider())
    print("\nAgent 答案:", agent.run("帮我算一下 12.5 乘以 8 等于多少"))


if __name__ == "__main__":
    main()
