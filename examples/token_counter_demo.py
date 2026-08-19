"""可注入 TokenCounter：默认启发式与可选 tiktoken 精确计数示例。"""

from gearlink import (
    HeuristicTokenCounter,
    ShortTermMemory,
    TiktokenTokenCounter,
    truncate_text,
)


def main() -> None:
    heuristic = HeuristicTokenCounter()
    text = "中文、code and emoji 🚀" * 20
    print("启发式估算:", heuristic.count_text(text))
    truncated = truncate_text(text, 20, token_counter=heuristic, suffix="...[截断]")
    # unicode_escape 便于在 Windows GBK 终端也安全展示 emoji。
    print("预算内文本:", truncated.encode("unicode_escape").decode("ascii"))

    memory = ShortTermMemory(max_tokens=12, max_message=None, token_counter=heuristic)
    memory.add_message({"role": "user", "content": "第一条消息"})
    memory.add_message({"role": "assistant", "content": "第二条较长消息"})
    print("预算裁剪后的记忆:", memory.get_messages())

    try:
        exact = TiktokenTokenCounter(encoding_name="cl100k_base")
    except ImportError:
        print("精确计数示例需要: pip install 'gearlink[tokenizers]'")
    except Exception as exc:  # 首次加载 encoding 可能需要下载词表，离线时跳过
        print(f"当前环境无法加载 tiktoken encoding: {type(exc).__name__}")
    else:
        print("tiktoken 精确计数:", exact.count_text(text))


if __name__ == "__main__":
    main()
