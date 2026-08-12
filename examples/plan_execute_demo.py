"""GearLink 规划-执行（PlanExecuteAgent）示例

演示 `PlanExecuteAgent` 先规划后执行：规划器把任务分解为步骤清单，每步经内部
`ReactAgent` 子循环执行（支持工具），最后整合为最终答案；同时演示用 `hooks`
回调观察规划与逐步执行事件。

运行方式（项目根目录下，须先 `pip install -e .` 安装本包）：
    python examples/plan_execute_demo.py

前置条件：
    - 根目录 .env 中配置 DEEPSEEK_API_KEY（或设置同名环境变量）
"""

from dotenv import load_dotenv

from gearlink import (
    OpenAIProvider,
    PlanExecuteAgent,
    PlanGeneratedEvent,
    PlanStepEndEvent,
    PlanStepStartEvent,
)

load_dotenv()


def on_step(event) -> None:
    """观察规划-执行事件流：打印规划结果与每步开始/结束。"""
    if isinstance(event, PlanGeneratedEvent):
        print(f"[规划] 共 {len(event.steps)} 步")
    elif isinstance(event, PlanStepStartEvent):
        print(f"[步骤 {event.index + 1}] {event.step}")
    elif isinstance(event, PlanStepEndEvent):
        print(f"[步骤 {event.index + 1} 完成] {event.result[:60]}...")


def main() -> None:
    agent = PlanExecuteAgent(
        provider=OpenAIProvider(),
        max_steps=5,
        hooks=[on_step],
    )

    print("助手: ", end="", flush=True)
    answer = agent.run("对比 Python 与 Go 的适用场景，并给出选型建议")
    print(answer)


if __name__ == "__main__":
    main()
