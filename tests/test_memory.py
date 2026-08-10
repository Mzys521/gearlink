"""ShortTermMemory 测试：正常路径与边界场景。"""

from gearlink.core.memory import ShortTermMemory


def test_short_term_memory_add_and_get_messages():
    memory = ShortTermMemory(max_message=20)
    memory.add_message({"role": "user", "content": "你好"})
    memory.add_message({"role": "assistant", "content": "你好！"})

    messages = memory.get_messages()
    assert len(messages) == 2
    assert messages[0]["content"] == "你好"


def test_short_term_memory_truncates_over_limit():
    memory = ShortTermMemory(max_message=3)
    for i in range(5):
        memory.add_message({"role": "user", "content": str(i)})

    messages = memory.get_messages()
    assert len(messages) == 3
    assert [m["content"] for m in messages] == ["2", "3", "4"]


def test_short_term_memory_get_messages_with_limit():
    memory = ShortTermMemory(max_message=20)
    for i in range(5):
        memory.add_message({"role": "user", "content": str(i)})

    assert [m["content"] for m in memory.get_messages(limit=2)] == ["3", "4"]


def test_short_term_memory_clear():
    memory = ShortTermMemory(max_message=20)
    memory.add_message({"role": "user", "content": "你好"})
    memory.clear()

    assert memory.get_messages() == []


def test_short_term_memory_truncates_by_max_tokens():
    memory = ShortTermMemory(max_tokens=6, max_message=None)
    # 每条 2 token（2 个汉字），6 token 预算最多容纳 3 条
    for i in range(5):
        memory.add_message({"role": "user", "content": "你好"})

    assert len(memory.get_messages()) == 3


def test_short_term_memory_keeps_latest_when_single_message_exceeds_budget():
    memory = ShortTermMemory(max_tokens=4, max_message=None)
    memory.add_message({"role": "user", "content": "你好"})
    # 单条 13 token 超出预算：软约束，保留最新一条不清空
    memory.add_message({"role": "user", "content": "十个汉字的测试内容超过预算"})

    messages = memory.get_messages()
    assert len(messages) == 1
    assert messages[0]["content"] == "十个汉字的测试内容超过预算"


def test_short_term_memory_removes_oldest_first_by_tokens():
    memory = ShortTermMemory(max_tokens=6, max_message=None)
    memory.add_message({"role": "user", "content": "a" * 20})  # 5 token
    memory.add_message({"role": "user", "content": "你好"})  # 2 token

    # 首条被移除后仅剩 2 token，符合预算
    messages = memory.get_messages()
    assert len(messages) == 1
    assert messages[0]["content"] == "你好"
