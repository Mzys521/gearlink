"""MemoryManager 测试：短期/长期记忆组合行为。"""

import pytest

from gearlink.core.memory import LongTermMemory, MemoryManager, ShortTermMemory
from gearlink.exceptions import MemoryError


class FakeCollection:
    """模拟 chromadb Collection 的最小子集"""

    def __init__(self) -> None:
        self.records: dict[str, tuple[str, dict]] = {}  # id -> (document, metadata)

    def add(self, ids, documents, metadatas) -> None:
        for record_id, doc, meta in zip(ids, documents, metadatas):
            self.records[record_id] = (doc, meta)

    def get(self, include=None):
        ids = list(self.records)
        return {
            "ids": ids,
            "documents": [self.records[i][0] for i in ids],
            "metadatas": [self.records[i][1] for i in ids],
        }

    def query(self, query_texts, n_results):
        items = list(self.records.items())[:n_results]
        return {
            "ids": [[record_id for record_id, _ in items]],
            "documents": [[doc for _, (doc, _) in items]],
            "metadatas": [[meta for _, (_, meta) in items]],
            "distances": [[float(i) for i in range(len(items))]],
        }

    def delete(self, ids) -> None:
        for record_id in ids:
            self.records.pop(record_id, None)


class FakeClient:
    def get_or_create_collection(self, name):
        return FakeCollection()


@pytest.fixture
def manager():
    return MemoryManager(
        short_term=ShortTermMemory(max_message=20),
        long_term=LongTermMemory(vector_db=FakeClient(), collection_name="test"),
    )


def test_add_message_writes_to_short_term(manager):
    manager.add_message({"role": "user", "content": "你好"})

    assert [m["content"] for m in manager.short_term.get_messages()] == ["你好"]


def test_add_message_sediments_user_and_assistant_to_long_term(manager):
    manager.add_message({"role": "user", "content": "我喜欢喝红茶"})
    manager.add_message({"role": "assistant", "content": "好的，记住了"})

    contents = [doc for doc, _ in manager.long_term.collection.records.values()]
    assert contents == ["我喜欢喝红茶", "好的，记住了"]


def test_add_message_skips_system_and_tool_messages(manager):
    manager.add_message({"role": "system", "content": "系统提示"})
    manager.add_message({"role": "tool", "tool_call_id": "t1", "content": "工具结果"})

    assert manager.long_term.get_messages() == []


def test_add_message_skips_empty_content(manager):
    manager.add_message({"role": "assistant", "content": None, "tool_calls": []})

    assert manager.long_term.get_messages() == []


def test_add_message_without_long_term_only_writes_short_term():
    manager = MemoryManager(short_term=ShortTermMemory(max_message=20))
    manager.add_message({"role": "user", "content": "你好"})

    assert manager.long_term is None
    assert [m["content"] for m in manager.short_term.get_messages()] == ["你好"]


def test_build_context_injects_relevant_memory_as_system(manager):
    manager.add_message({"role": "user", "content": "我喜欢喝红茶"})
    manager.short_term.clear()  # 清空短期，只验证注入部分
    manager.short_term.add_message({"role": "user", "content": "我喜欢喝什么？"})

    messages = manager.build_context("我喜欢喝什么？")

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "我喜欢喝红茶" in messages[0]["content"]
    assert messages[1]["content"] == "我喜欢喝什么？"


def test_build_context_without_long_term_returns_short_term_only():
    manager = MemoryManager(short_term=ShortTermMemory(max_message=20))
    manager.add_message({"role": "user", "content": "你好"})

    assert manager.build_context("你好") == [{"role": "user", "content": "你好"}]


def test_build_context_empty_long_term_has_no_system_prefix(manager):
    manager.add_message({"role": "system", "content": "系统提示"})

    messages = manager.build_context("任意查询")

    # 长期记忆无内容时不注入 system 检索消息
    assert messages == [{"role": "system", "content": "系统提示"}]


def test_build_context_respects_relevant_limit(manager):
    for i in range(5):
        manager.add_message({"role": "user", "content": f"记忆{i}"})
    manager.short_term.clear()

    messages = manager.build_context("查询", relevant_limit=2)

    injected_lines = [line for line in messages[0]["content"].splitlines() if line.startswith("- ")]
    assert len(injected_lines) == 2


def test_end_session_sediments_remaining_and_clears_short_term(manager):
    manager.add_message({"role": "system", "content": "系统提示"})
    manager.add_message({"role": "user", "content": "你好"})

    manager.end_session()

    assert manager.short_term.get_messages() == []
    contents = [doc for doc, _ in manager.long_term.collection.records.values()]
    assert "系统提示" in contents  # end_session 补齐了未即时沉淀的消息
    assert "你好" in contents


def test_end_session_without_long_term_only_clears_short_term():
    manager = MemoryManager(short_term=ShortTermMemory(max_message=20))
    manager.add_message({"role": "user", "content": "你好"})

    manager.end_session()

    assert manager.short_term.get_messages() == []


def test_end_session_sediments_summary_when_summarizer_set():
    """配置 summarizer 时，end_session 应把会话文本拼成 transcript 并沉淀摘要到长期库"""
    captured: dict[str, str] = {}

    def fake_summarizer(transcript: str) -> str:
        captured["transcript"] = transcript
        return "会话要点总结"

    manager = MemoryManager(
        short_term=ShortTermMemory(max_message=20),
        long_term=LongTermMemory(vector_db=FakeClient(), collection_name="test"),
        summarizer=fake_summarizer,
    )
    manager.add_message({"role": "user", "content": "你好"})
    manager.add_message({"role": "assistant", "content": "你好，有什么可以帮你"})

    manager.end_session()

    contents = [doc for doc, _ in manager.long_term.collection.records.values()]
    assert "[会话摘要] 会话要点总结" in contents
    assert captured["transcript"] == "用户: 你好\n助手: 你好，有什么可以帮你"


def test_end_session_without_summarizer_keeps_legacy_behavior(manager):
    """未配置 summarizer 时，end_session 行为与现状一致：不产生摘要沉淀"""
    manager.add_message({"role": "user", "content": "你好"})

    manager.end_session()

    contents = [doc for doc, _ in manager.long_term.collection.records.values()]
    assert "你好" in contents
    assert not any("[会话摘要]" in doc for doc in contents)


def test_end_session_ignores_summarizer_failure(manager):
    """summarizer 抛异常时不应中断会话结束：跳过摘要但保留补沉淀逻辑"""

    def broken_summarizer(transcript: str) -> str:
        raise RuntimeError("模型不可用")

    manager = MemoryManager(
        short_term=ShortTermMemory(max_message=20),
        long_term=LongTermMemory(vector_db=FakeClient(), collection_name="test"),
        summarizer=broken_summarizer,
    )
    manager.add_message({"role": "user", "content": "你好"})

    manager.end_session()  # 不应抛出异常

    assert manager.short_term.get_messages() == []
    contents = [doc for doc, _ in manager.long_term.collection.records.values()]
    assert "你好" in contents  # 原有补沉淀逻辑仍执行
    assert not any("[会话摘要]" in doc for doc in contents)


def test_clear_removes_both_memories(manager):
    manager.add_message({"role": "user", "content": "你好"})

    manager.clear()

    assert manager.short_term.get_messages() == []
    assert manager.long_term.get_messages() == []


def test_add_message_propagates_long_term_write_error(manager):
    def broken_add(**kwargs):
        raise RuntimeError("disk full")

    manager.long_term.collection.add = broken_add

    with pytest.raises(MemoryError) as exc_info:
        manager.add_message({"role": "user", "content": "写入失败"})
    assert exc_info.value.__cause__ is not None
    # 短期记忆已写入成功，不受长期记忆失败影响
    assert [m["content"] for m in manager.short_term.get_messages()] == ["写入失败"]


def test_build_context_trims_short_term_to_budget():
    manager = MemoryManager(short_term=ShortTermMemory(max_message=20), max_context_tokens=8)
    for _ in range(5):
        manager.add_message({"role": "user", "content": "记忆"})

    messages = manager.build_context("查询")

    # 每条 2 token（2 个汉字），8 token 预算保留最新的 4 条
    assert [m["content"] for m in messages] == ["记忆"] * 4


def test_build_context_always_keeps_system_messages():
    manager = MemoryManager(short_term=ShortTermMemory(max_message=20), max_context_tokens=4)
    manager.add_message({"role": "system", "content": "系统提示"})  # 4 token
    manager.add_message({"role": "user", "content": "你好"})  # 2 token

    messages = manager.build_context("查询")

    # 系统消息始终保留，用户消息超出剩余预算被裁剪
    assert messages == [{"role": "system", "content": "系统提示"}]


def test_build_context_caps_system_prompts_when_ratio_set():
    """system_budget_ratio < 1.0 时，超出配额的最旧非检索 system 被裁剪，检索注入消息保留"""
    manager = MemoryManager(
        short_term=ShortTermMemory(max_message=20),
        long_term=LongTermMemory(vector_db=FakeClient(), collection_name="test"),
        max_context_tokens=16,
        system_budget_ratio=0.25,
    )
    manager.long_term.add_message({"role": "user", "content": "检索记忆"})
    for i in range(4):
        manager.add_message({"role": "system", "content": f"系统{i}"})  # 每条 3 token

    messages = manager.build_context("查询")

    # 非检索 system 配额 = 16 * 25% = 4 token，4 条共 12 token，仅保留最新 1 条
    assert len(messages) == 2
    assert "检索记忆" in messages[0]["content"]  # 受保护的检索注入消息未被裁剪
    assert messages[1]["content"] == "系统3"
    assert not any(
        "系统0" in m["content"] or "系统1" in m["content"] or "系统2" in m["content"]
        for m in messages
    )


def test_build_context_trims_retrieval_entries_to_ratio():
    manager = MemoryManager(
        short_term=ShortTermMemory(max_message=20),
        long_term=LongTermMemory(vector_db=FakeClient(), collection_name="test"),
        max_context_tokens=16,
    )
    for _ in range(5):
        manager.add_message({"role": "user", "content": "长期记忆"})
    manager.short_term.clear()

    messages = manager.build_context("查询")

    # 检索预算 = 16 * 25% = 4 token，仅保留第一条（4 token）
    injected_lines = [line for line in messages[0]["content"].splitlines() if line.startswith("- ")]
    assert len(injected_lines) == 1


def test_build_context_injects_entries_with_role_labels(manager):
    manager.add_message({"role": "user", "content": "我喜欢喝红茶"})
    manager.add_message({"role": "assistant", "content": "好的，记住了"})
    manager.short_term.clear()
    manager.short_term.add_message({"role": "user", "content": "我喝什么？"})

    messages = manager.build_context("我喝什么？")

    content = messages[0]["content"]
    assert "[用户]" in content and "我喜欢喝红茶" in content
    assert "[助手]" in content and "好的，记住了" in content


def test_build_context_labels_unknown_role_with_raw_name(manager):
    manager.long_term.add_message({"role": "tool", "content": "工具时间戳结果"})
    manager.short_term.clear()
    manager.short_term.add_message({"role": "user", "content": "查询"})

    messages = manager.build_context("查询")

    # 未知角色回退为原始 role 值作为说话人标签
    assert "[tool]" in messages[0]["content"]


def test_build_context_dedupes_content_already_in_short_term(manager):
    manager.add_message({"role": "user", "content": "我喜欢喝红茶"})

    messages = manager.build_context("我喜欢喝什么？")

    # 检索结果与短期窗口内容重复，不再重复注入
    assert messages == [{"role": "user", "content": "我喜欢喝红茶"}]


def test_build_context_keeps_non_duplicate_entries(manager):
    manager.add_message({"role": "user", "content": "我喜欢喝红茶"})
    manager.add_message({"role": "user", "content": "我喜欢喝绿茶"})
    manager.short_term.clear()
    manager.short_term.add_message({"role": "user", "content": "我喜欢喝红茶"})

    messages = manager.build_context("我喜欢喝什么？")

    # 仅过滤与短期窗口重复的红茶条目，保留绿茶
    content = messages[0]["content"]
    assert "绿茶" in content
    assert "红茶" not in content
