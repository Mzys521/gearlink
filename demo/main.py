"""GearLink CLI 智能体：基于当前 GearLink 框架的可交互命令行助手。

整合框架的可插拔维度与能力：

- 模型：``--provider`` 选择 deepseek / ollama / anthropic（OpenAI 兼容接口）；
- 工具：自定义工具（add / multiply / get_weather）+ 内置 get_current_time，
  经 ``register_tool`` 显式登记后由 ReAct 循环按需调用；
- 记忆：短期滑窗 + chromadb 向量长期记忆 + 会话摘要沉淀（MemoryManager）；
- 技能：从 demo/skills/ 发现的技能包，模型经 ``load_skill`` 工具按需加载完整指令；
- 流式输出：``run_stream`` 逐片段产出模型文本；
- 事件钩子：``on_step`` 实时打印工具调用状态（ToolCallStartEvent /
  ToolCallEndEvent，观察 ReAct 循环）；
- 日志开关：``--debug`` 一键开启框架内部日志。

运行方式（项目根目录下，须先 ``pip install -e .`` 安装本包）：

    python demo/main.py
    python demo/main.py --provider ollama --model qwen2.5:7b
    python demo/main.py --debug
    python demo/main.py --no-ltm          # 禁用长期记忆，仅短期滑窗

前置条件：
    - deepseek：根目录 .env 配置 DEEPSEEK_API_KEY（或设同名环境变量）；
    - ollama：本地启动 Ollama 服务并拉取模型（``ollama serve`` / ``ollama pull``）；
    - anthropic：安装 ``gearlink[anthropic]`` 并配置 ANTHROPIC_API_KEY。

交互命令（``/`` 前缀，避免与自然语言冲突；裸 ``exit``/``quit`` 亦可退出）：
    /exit, /quit  结束会话：沉淀会话摘要并补沉淀剩余上下文到长期记忆
    /clear        清空短期与长期记忆
    /reset        仅清空短期对话，保留长期记忆
    /history      查看近期对话消息
    /help         显示命令帮助
"""

import argparse
import logging
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv

# 确保 demo 目录在 sys.path 上，使 ``import tools`` 在
# ``python demo/main.py`` 与 ``python -m demo.main`` 两种运行方式下都可用
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 导入自定义工具模块，触发 add / multiply / get_weather 的注册
import tools  # noqa: F401
# 显式导入，触发 load_skill 工具注册（注册表禁止运行时隐式扫描）
import gearlink.tools.load_skill  # noqa: F401
from gearlink import (
    AnthropicProvider,
    GearLinkError,
    LongTermMemory,
    MemoryManager,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    ReactAgent,
    ShortTermMemory,
    SkillLoader,
    SkillRegistry,
    ToolCallEndEvent,
    ToolCallStartEvent,
    disable_logging,
    enable_logging,
)

DEMO_DIR = Path(__file__).resolve().parent
ROOT = DEMO_DIR.parent
load_dotenv(ROOT / ".env")  # 加载根目录 .env 中的 API 密钥

#: 长期记忆向量库的持久化路径
_CHROMA_PATH = DEMO_DIR / ".chroma"

#: 各角色消息在 /history 中的显示名
_ROLE_LABELS = {"system": "系统", "user": "用户", "assistant": "助手", "tool": "工具"}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="GearLink CLI 智能体：可交互的命令行助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        choices=["deepseek", "ollama", "anthropic"],
        default="deepseek",
        help="模型提供者（默认 deepseek）",
    )
    parser.add_argument("--model", help="模型名称（覆盖提供者默认值）")
    parser.add_argument("--base-url", help="服务地址（覆盖提供者默认值）")
    parser.add_argument("--debug", action="store_true", help="开启框架内部日志输出")
    parser.add_argument(
        "--no-ltm", action="store_true", help="禁用长期记忆，仅使用短期滑窗"
    )
    parser.add_argument("--no-skill", action="store_true", help="不加载技能包")
    return parser.parse_args()


def build_provider(args: argparse.Namespace):
    """按 --provider 构造模型提供者实例。"""
    if args.provider == "ollama":
        return OllamaProvider(model=args.model, base_url=args.base_url)
    if args.provider == "anthropic":
        return AnthropicProvider(model=args.model, base_url=args.base_url)
    return OpenAIProvider(model=args.model, base_url=args.base_url)


def build_skill_registry() -> SkillRegistry:
    """从 demo/skills/ 目录发现技能并登记到注册表。"""
    registry = SkillRegistry()
    for skill in SkillLoader.discover_from_directory(DEMO_DIR / "skills"):
        registry.register(skill)
    return registry


def build_system_prompt(skill_registry: SkillRegistry | None) -> str:
    """组装系统提示：基础提示 + 可用技能列表（仅当注入非空注册表时）。

    自行注入 MemoryManager 时，ReactAgent 不会自动拼入技能列表，
    故在此手动组装并以 system 消息写入记忆。
    """
    base = (
        "你是一个有用的助手。当需要实时信息（如当前时间、天气）时，"
        "请调用可用的工具，而不是凭空回答。"
    )
    skills = skill_registry.list_all() if skill_registry is not None else []
    if not skills:
        return base
    skill_lines = "\n".join(f"- {s.name}: {s.description}" for s in skills)
    return (
        f"{base}\n"
        "当遇到需要特定专业知识的任务时，你可以调用 `load_skill` 工具加载相应的技能指令，"
        "然后严格遵循这些指令执行。\n"
        f"当前可用技能：\n{skill_lines}"
    )


def build_memory(provider, *, enable_long_term: bool) -> MemoryManager:
    """组装「短期 + 长期 + 会话摘要」的记忆管理器。

    Args:
        provider: 模型提供者实例；长期记忆开启时复用其作为会话摘要生成器。
        enable_long_term: True 启用 chromadb 长期记忆；False 仅短期滑窗。
    """
    short_term = ShortTermMemory(max_message=20, max_tokens=3000)

    if not enable_long_term:
        return MemoryManager(short_term=short_term)

    vector_db = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    long_term = LongTermMemory(
        vector_db=vector_db,
        collection_name="demo_agent_history",
        recency_weight=0.5,
        max_entries=200,
        dedupe=True,
    )

    def summarize(transcript: str) -> str:
        response = provider.chat(
            messages=[
                {
                    "role": "system",
                    "content": "请用简体中文简要总结以下对话的要点，不超过 100 字。",
                },
                {"role": "user", "content": transcript},
            ]
        )
        return response.content or ""

    return MemoryManager(
        short_term=short_term,
        long_term=long_term,
        max_context_tokens=4000,
        system_budget_ratio=0.5,
        summarizer=summarize,
    )


def on_step(event):
    """on_step 事件钩子：实时展示 ReAct 循环中的工具调用（事件流回调示例）。

    Returns:
        None 表示不修改事件（纯观察）。
    """
    if isinstance(event, ToolCallStartEvent):
        print(f"\n  [工具] {event.name}({event.arguments})", flush=True)
    elif isinstance(event, ToolCallEndEvent):
        if event.error is None:
            preview = event.result if len(event.result) <= 60 else event.result[:60] + "…"
            print(f"  [工具] {event.name} 完成: {preview}", flush=True)
        else:
            print(f"  [工具] {event.name} 失败: {event.error}", flush=True)
    return None


def print_banner(skill_registry: SkillRegistry | None, args: argparse.Namespace) -> None:
    """打印启动横幅：当前配置与示例提问。"""
    skills = skill_registry.list_all() if skill_registry is not None else []
    skill_names = ", ".join(s.name for s in skills) or "（无）"
    model_desc = args.model or "（提供者默认）"
    print("=" * 60)
    print("GearLink CLI 智能体")
    print(f"提供者: {args.provider}  |  模型: {model_desc}")
    print(f"已加载技能: {skill_names}")
    print(f"长期记忆: {'关闭' if args.no_ltm else '开启'}")
    print("-" * 60)
    print("输入 /help 查看命令；直接输入文本即可对话。")
    print("试试问：现在几点了？/ 北京天气如何？/ 12.3 加 45.6 等于几？/ 生成每日简报")
    print("=" * 60)


def print_help() -> None:
    """打印可用命令。"""
    print("可用命令（/ 前缀）：")
    print("  /exit, /quit  结束会话（沉淀摘要到长期记忆）")
    print("  /clear        清空短期与长期记忆")
    print("  /reset        仅清空短期对话，保留长期记忆")
    print("  /history      查看近期对话消息")
    print("  /help         显示本帮助")
    print("裸 exit / quit 亦可退出；直接输入文本即与助手对话。")


def print_history(memory: MemoryManager) -> None:
    """打印短期记忆中的近期对话消息。"""
    messages = memory.short_term.get_messages()
    if not messages:
        print("（暂无对话记录）")
        return
    for message in messages:
        role = _ROLE_LABELS.get(message.get("role"), message.get("role"))
        content = message.get("content") or "(空)"
        if len(content) > 80:
            content = content[:80] + "…"
        print(f"  [{role}] {content}")


def run_chat(agent: ReactAgent, user_input: str) -> None:
    """执行一轮对话：流式输出助手回复，并对模型/框架错误做友好兜底。"""
    print("助手: ", end="", flush=True)
    try:
        for delta in agent.run_stream(user_input):
            print(delta, end="", flush=True)
        print()
    except ProviderError as e:
        print(f"\n[模型调用失败] {e}（可重试: {e.retryable}）")
    except GearLinkError as e:
        print(f"\n[框架错误] {e}")


def main() -> None:
    args = parse_args()

    # 日志开关：--debug 开启框架内部日志，否则保持静默
    if args.debug:
        enable_logging(logging.DEBUG)
    else:
        disable_logging()

    provider = build_provider(args)
    skill_registry = None if args.no_skill else build_skill_registry()
    memory = build_memory(provider, enable_long_term=not args.no_ltm)

    # 自行注入 memory 时，须自行写入系统提示（含技能列表）
    memory.add_message({"role": "system", "content": build_system_prompt(skill_registry)})

    agent = ReactAgent(
        provider=provider,
        memory=memory,
        retrieve_every_iteration=True,
        skill_registry=skill_registry,
        hooks=[on_step],
        max_retries=2,
    )

    print_banner(skill_registry, args)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            user_input = "/exit"

        if not user_input:
            continue

        low = user_input.lower()
        # 兼容裸 exit / quit
        if low in ("exit", "quit"):
            low = "/exit"

        if not low.startswith("/"):
            run_chat(agent, user_input)
            continue

        # 命令处理
        if low in ("/exit", "/quit"):
            memory.end_session()  # 沉淀会话摘要并清空短期记忆
            if args.no_ltm:
                print("会话已结束。")
            else:
                print("会话已结束，摘要已沉淀到长期记忆。")
            break
        if low == "/clear":
            memory.clear()  # 清空短期与长期记忆
            memory.add_message({"role": "system", "content": build_system_prompt(skill_registry)})
            print("短期与长期记忆已清空。")
            continue
        if low == "/reset":
            memory.short_term.clear()  # 仅清空短期对话，保留长期记忆
            memory.add_message({"role": "system", "content": build_system_prompt(skill_registry)})
            print("短期对话已重置（长期记忆保留）。")
            continue
        if low == "/history":
            print_history(memory)
            continue
        if low == "/help":
            print_help()
            continue
        print(f"未知命令: {user_input}（输入 /help 查看可用命令）")


if __name__ == "__main__":
    main()
