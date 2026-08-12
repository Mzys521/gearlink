"""MemoryManager 测试：短期/长期记忆组合行为。"""

import json

import pytest

from gearlink.core.memory import LongTermMemory, MemoryManager, Session, ShortTermMemory
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


# ==================== 会话持久化（snapshot / restore） ====================


def test_snapshot_captures_short_term_messages_and_metadata(manager):
    manager.add_message({"role": "user", "content": "你好"})
    manager.add_message({"role": "assistant", "content": "你好，有什么可以帮你？"})

    snapshot = manager.snapshot(model="gpt-4o", metadata={"app": "demo"})

    assert snapshot["id"]
    assert snapshot["created_at"] > 0
    assert snapshot["model"] == "gpt-4o"
    assert snapshot["metadata"] == {"app": "demo"}
    assert [m["content"] for m in snapshot["messages"]] == ["你好", "你好，有什么可以帮你？"]


def test_snapshot_is_json_serializable(manager):
    manager.add_message({"role": "user", "content": "落盘测试"})

    snapshot = manager.snapshot()
    restored_dict = json.loads(json.dumps(snapshot))

    assert restored_dict == snapshot


def test_snapshot_restore_roundtrip_keeps_build_context_identical():
    # 验收标准（开发方向 §4.5）：snapshot → restore 往返后 build_context 输出一致
    manager_a = MemoryManager(short_term=ShortTermMemory(max_message=20))
    manager_a.add_message({"role": "system", "content": "你是一个助手"})
    manager_a.add_message({"role": "user", "content": "我喜欢喝红茶"})
    manager_a.add_message({"role": "assistant", "content": "好的，记住了"})
    before = manager_a.build_context("我喜欢喝什么？")

    snapshot = manager_a.snapshot()

    # 模拟进程重启：全新 MemoryManager 从快照恢复
    manager_b = MemoryManager(short_term=ShortTermMemory(max_message=20))
    session = manager_b.restore(snapshot)

    assert manager_b.build_context("我喜欢喝什么？") == before
    assert session.id == snapshot["id"]


def test_restore_replaces_existing_short_term_content():
    manager_a = MemoryManager(short_term=ShortTermMemory(max_message=20))
    manager_a.add_message({"role": "user", "content": "旧会话内容"})
    snapshot = manager_a.snapshot()

    manager_b = MemoryManager(short_term=ShortTermMemory(max_message=20))
    manager_b.add_message({"role": "user", "content": "无关的新消息"})
    manager_b.restore(snapshot)

    assert [m["content"] for m in manager_b.short_term.get_messages()] == ["旧会话内容"]


def test_restore_returns_session_with_metadata():
    manager = MemoryManager(short_term=ShortTermMemory(max_message=20))
    snapshot = manager.snapshot(model="gpt-4o", metadata={"user_id": "u-1"})

    session = manager.restore(snapshot)

    assert isinstance(session, Session)
    assert session.model == "gpt-4o"
    assert session.metadata == {"user_id": "u-1"}


def test_restore_invalid_snapshot_raises_memory_error(manager):
    with pytest.raises(MemoryError):
        manager.restore({"messages": []})  # 缺少 id / created_at 必要字段


def test_session_dict_roundtrip():
    session = Session(
        id="s-1",
        created_at=1000.0,
        model="gpt-4o",
        messages=({"role": "user", "content": "你好"},),
        metadata={"k": "v"},
    )

    restored = Session.from_dict(session.to_dict())

    assert restored == session


def test_session_from_dict_tolerates_missing_optional_fields():
    session = Session.from_dict({"id": "s-2", "created_at": 2000.0})

    assert session.model is None
    assert session.messages == ()
    assert session.metadata is None


# ==================== 上下文摘要压缩（开发方向 §5.2） ====================


def make_long_manager(summarizer, compress_context=True):
    """构造小预算的记忆管理器，便于触发压缩阈值。"""
    return MemoryManager(
        short_term=ShortTermMemory(max_message=50),
        max_context_tokens=120,
        summarizer=summarizer,
        compress_context=compress_context,
    )


def test_compress_context_replaces_oldest_segment_with_summary():
    manager = make_long_manager(summarizer=lambda text: "前文摘要")
    for i in range(6):
        manager.add_message({"role": "user", "content": f"第 {i} 轮较长的对话内容" * 5})

    messages = manager.build_context("继续")

    # 短期记忆被重写：首位是 [上下文摘要] system，其后仅保留最新 2 条
    short_term = manager.short_term.get_messages()
    assert short_term[0]["role"] == "system"
    assert short_term[0]["content"] == "[上下文摘要] 前文摘要"
    assert len(short_term) == 3
    assert short_term[-1]["content"].startswith("第 5 轮")
    assert messages[0]["content"] == "[上下文摘要] 前文摘要"


def test_compress_context_disabled_by_default():
    # 验收：默认行为等价现状——不开启时仅静态裁剪，消息不丢
    manager = MemoryManager(
        short_term=ShortTermMemory(max_message=50),
        max_context_tokens=120,
        summarizer=lambda text: "摘要",
    )
    for i in range(6):
        manager.add_message({"role": "user", "content": f"第 {i} 轮较长的对话内容" * 5})

    manager.build_context("继续")

    assert len(manager.short_term.get_messages()) == 6


def test_compress_context_keeps_messages_when_summarizer_fails():
    def broken_summarizer(text):
        raise RuntimeError("摘要服务不可用")

    manager = make_long_manager(summarizer=broken_summarizer)
    for i in range(6):
        manager.add_message({"role": "user", "content": f"第 {i} 轮较长的对话内容" * 5})

    # 压缩失败不中断主流程，消息原样保留
    messages = manager.build_context("继续")

    assert len(manager.short_term.get_messages()) == 6
    assert messages


# ==================== 用户画像 profile 钩子（开发方向 §5.2） ====================


def test_profile_hook_updates_profile_on_end_session():
    hook_calls = []

    def hook(new_messages, current_profile):
        hook_calls.append((len(new_messages), current_profile))
        return {"偏好": "红茶"}

    manager = MemoryManager(short_term=ShortTermMemory(max_message=20), profile_hook=hook)
    manager.add_message({"role": "user", "content": "我喜欢喝红茶"})
    manager.end_session()

    assert manager.profile == {"偏好": "红茶"}
    assert hook_calls == [(1, None)]


def test_profile_injected_first_in_build_context():
    manager = MemoryManager(
        short_term=ShortTermMemory(max_message=20),
        long_term=LongTermMemory(vector_db=FakeClient(), collection_name="test"),
    )
    manager.profile = {"偏好": "红茶"}
    manager.long_term.add_message({"role": "user", "content": "无关的历史记忆"})
    manager.short_term.add_message({"role": "user", "content": "推荐饮品"})

    messages = manager.build_context("推荐饮品")

    # 画像优先携带：置于检索注入之前的首位 system
    assert messages[0]["content"].startswith("[用户画像]")
    assert "偏好: 红茶" in messages[0]["content"]


def test_profile_hook_none_or_error_keeps_old_profile():
    manager = MemoryManager(
        short_term=ShortTermMemory(max_message=20),
        profile_hook=lambda msgs, profile: None,  # 返回 None 不更新
    )
    manager.profile = {"偏好": "红茶"}
    manager.add_message({"role": "user", "content": "新消息"})
    manager.end_session()
    assert manager.profile == {"偏好": "红茶"}

    broken = MemoryManager(
        short_term=ShortTermMemory(max_message=20),
        profile_hook=lambda msgs, profile: (_ for _ in ()).throw(RuntimeError("钩子异常")),
    )
    broken.profile = {"偏好": "绿茶"}
    broken.add_message({"role": "user", "content": "新消息"})
    # 钩子异常不中断会话结束，画像保持不变
    broken.end_session()
    assert broken.profile == {"偏好": "绿茶"}
