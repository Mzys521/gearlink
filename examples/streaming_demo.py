"""GearLink 流式输出示例

演示 `ReactAgent.run_stream` 逐片段产出模型文本；工具调用阶段自动执行并继续循环。

运行方式（项目根目录下，须先 `pip install -e .` 安装本包）：
    python examples/streaming_demo.py

前置条件：
    - 根目录 .env 中配置 DEEPSEEK_API_KEY（或设置同名环境变量）
"""

from dotenv import load_dotenv

from gearlink import OpenAIProvider, ReactAgent

load_dotenv()


def main() -> None:
    agent = ReactAgent(provider=OpenAIProvider())

    print("助手: ", end="", flush=True)
    for delta in agent.run_stream("用一段话介绍 GearLink 是什么"):
        print(delta, end="", flush=True)
    print()

    # 工具调用阶段自动执行（get_current_time），最终答案同样流式输出
    print("助手: ", end="", flush=True)
    for delta in agent.run_stream("现在几点了？"):
        print(delta, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
