"""token_count 测试：启发式计数与消息级计数的边界场景。"""

from gearlink.utils.token_count import count_message_tokens, estimate_tokens


def test_estimate_tokens_returns_zero_for_empty_text():
    assert estimate_tokens("") == 0


def test_estimate_tokens_counts_cjk_as_one_token_per_char():
    # 4 个汉字 + 4 个全角字符
    assert estimate_tokens("你好世界") == 4
    assert estimate_tokens("，。！？") == 4


def test_estimate_tokens_counts_ascii_as_four_chars_per_token():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_estimate_tokens_mixes_cjk_and_ascii():
    # 2 个汉字 + 8 个 ASCII 字符 = 2 + 2
    assert estimate_tokens("你好" + "x" * 8) == 4


def test_count_message_tokens_counts_content():
    assert count_message_tokens({"role": "user", "content": "你好"}) == 2


def test_count_message_tokens_counts_tool_calls():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_time", "arguments": "{}"},
            }
        ],
    }
    # tool_calls 参数部分计入 token
    assert count_message_tokens(message) > 0


def test_count_message_tokens_handles_missing_fields():
    assert count_message_tokens({"role": "tool"}) == 0
