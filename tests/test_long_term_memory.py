"""LongTermMemory 测试：使用 Fake 集合替代真实 chromadb。"""

import time

import pytest

from gearlink import ChromaVectorStore, VectorStore
from gearlink.core.memory import LongTermMemory, MemoryEntry
from gearlink.exceptions import MemoryError


class FakeCollection:
    """模拟 chromadb Collection 的最小子集"""

    def __init__(self) -> None:
        self.records: dict[str, tuple[str, dict]] = {}  # id -> (document, metadata)

    def add(self, ids, documents, metadatas) -> None:
        for record_id, doc, meta in zip(ids, documents, metadatas):
            self.records[record_id] = (doc, meta)

    def get(self, include=None, where=None):
        records = self.records
        if where:
            records = {
                rid: (doc, meta)
                for rid, (doc, meta) in records.items()
                if all(meta.get(key) == value for key, value in where.items())
            }
        ids = list(records)
        return {
            "ids": ids,
            "documents": [records[i][0] for i in ids],
            "metadatas": [records[i][1] for i in ids],
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
    def __init__(self) -> None:
        self.collection = FakeCollection()
        self.last_embedding_function = None

    def get_or_create_collection(self, name, embedding_function=None):
        self.last_embedding_function = embedding_function
        return self.collection


@pytest.fixture
def memory():
    return LongTermMemory(vector_db=FakeClient(), collection_name="test")


def test_add_message_writes_to_collection(memory):
    memory.add_message({"role": "user", "content": "我喜欢喝红茶"})

    assert len(memory.collection.records) == 1
    doc, meta = next(iter(memory.collection.records.values()))
    assert doc == "我喜欢喝红茶"
    assert meta["role"] == "user"
    assert meta["timestamp"] > 0


def test_add_message_requires_role_and_content(memory):
    with pytest.raises(ValueError):
        memory.add_message({"role": "user"})
    with pytest.raises(ValueError):
        memory.add_message({"content": "没有角色"})


def test_get_messages_sorted_by_timestamp(memory):
    memory.add_message({"role": "user", "content": "第一条"})
    memory.add_message({"role": "assistant", "content": "第二条"})

    messages = memory.get_messages()
    assert len(messages) == 2
    assert [m["content"] for m in messages] == ["第一条", "第二条"]
    assert messages[0]["timestamp"] <= messages[1]["timestamp"]


def test_get_messages_with_limit(memory):
    for i in range(5):
        memory.add_message({"role": "user", "content": str(i)})

    assert [m["content"] for m in memory.get_messages(limit=2)] == ["3", "4"]


def test_get_relevant_messages_returns_entries(memory):
    memory.add_message({"role": "user", "content": "北京今天天气很好"})
    memory.add_message({"role": "user", "content": "我喜欢喝红茶"})

    results = memory.get_relevant_messages("天气怎么样", limit=2)

    assert len(results) == 2
    assert all(isinstance(r, MemoryEntry) for r in results)
    assert results[0].content == "北京今天天气很好"
    assert results[0].distance < results[1].distance


def test_get_relevant_messages_respects_recency_weight():
    """开启时间衰减后，新近记忆应在综合得分上排前"""
    memory = LongTermMemory(vector_db=FakeClient(), collection_name="test", recency_weight=2.0)
    memory.add_message({"role": "user", "content": "旧消息"})
    memory.add_message({"role": "user", "content": "新消息"})
    # 手动调整时间戳：旧消息 40 天前（age_ratio 封顶 1.0），新消息 5 天前
    records = list(memory.collection.records.values())
    records[0][1]["timestamp"] = time.time() - 40 * 86400
    records[1][1]["timestamp"] = time.time() - 5 * 86400

    results = memory.get_relevant_messages("查询", limit=2)

    # distance 上旧消息更小，但时间衰减后新消息综合得分更低，排前
    assert [r.content for r in results] == ["新消息", "旧消息"]


def test_get_relevant_messages_default_order_unchanged(memory):
    """默认 recency_weight=0 时仍按相关度（distance 升序）排列"""
    memory.add_message({"role": "user", "content": "旧消息"})
    memory.add_message({"role": "user", "content": "新消息"})
    records = list(memory.collection.records.values())
    records[0][1]["timestamp"] = time.time() - 40 * 86400
    records[1][1]["timestamp"] = time.time() - 5 * 86400

    results = memory.get_relevant_messages("查询", limit=2)

    assert [r.content for r in results] == ["旧消息", "新消息"]


def test_add_message_evicts_oldest_over_max_entries():
    """配置 max_entries 后，写入超出上限应淘汰时间戳最旧的条目"""
    memory = LongTermMemory(vector_db=FakeClient(), collection_name="test", max_entries=3)
    for i in range(5):
        memory.add_message({"role": "user", "content": f"消息{i}"})

    records = list(memory.collection.records.values())
    assert len(records) == 3
    assert [doc for doc, _ in records] == ["消息2", "消息3", "消息4"]


def test_get_relevant_messages_dedupes_identical_content(memory):
    """检索结果含重复 content 时只保留相关度首个"""
    memory.add_message({"role": "user", "content": "内容甲"})
    memory.add_message({"role": "user", "content": "内容乙"})
    memory.add_message({"role": "user", "content": "内容甲"})

    results = memory.get_relevant_messages("查询", limit=3)

    assert [r.content for r in results] == ["内容甲", "内容乙"]


def test_add_message_with_dedupe_skips_duplicate_content():
    """dedupe=True 时，相同内容重复写入应被跳过"""
    memory = LongTermMemory(vector_db=FakeClient(), collection_name="test", dedupe=True)
    memory.add_message({"role": "user", "content": "相同内容"})
    memory.add_message({"role": "user", "content": "相同内容"})

    assert len(memory.collection.records) == 1


def test_add_message_without_dedupe_keeps_duplicates(memory):
    """默认 dedupe=False 时相同内容原样写入"""
    memory.add_message({"role": "user", "content": "相同内容"})
    memory.add_message({"role": "user", "content": "相同内容"})

    assert len(memory.collection.records) == 2


def test_clear_removes_all_records(memory):
    memory.add_message({"role": "user", "content": "待清除"})
    memory.clear()

    assert memory.get_messages() == []


def test_clear_on_empty_memory_does_not_fail(memory):
    memory.clear()  # 空集合不应抛出异常
    assert memory.get_messages() == []


def test_add_message_wraps_storage_error(memory):
    def broken_add(**kwargs):
        raise RuntimeError("disk full")

    memory.collection.add = broken_add
    with pytest.raises(MemoryError) as exc_info:
        memory.add_message({"role": "user", "content": "写入失败"})
    assert exc_info.value.__cause__ is not None


def test_memory_entry_dict_roundtrip():
    entry = MemoryEntry(id="e1", role="user", content="你好", timestamp=1723000000.0, distance=0.5)
    assert MemoryEntry.from_dict(entry.to_dict()) == entry

    # 兼容缺少 distance 的旧数据
    legacy = {"id": "e2", "role": "user", "content": "旧数据", "timestamp": 1.0}
    assert MemoryEntry.from_dict(legacy).distance is None


def test_default_uses_chromadb_default_embedding():
    client = FakeClient()
    LongTermMemory(vector_db=client, collection_name="test")
    assert client.last_embedding_function is None


def test_injected_embedding_function_passed_to_collection():
    calls: list[list[str]] = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[float(len(text))] for text in texts]

    client = FakeClient()
    memory = LongTermMemory(vector_db=client, collection_name="test", embedding_function=fake_embed)
    assert memory.embedding_function is fake_embed
    assert client.last_embedding_function is not None

    # 适配器转发到注入的函数（写入与检索统一走注入 embedding）
    vectors = client.last_embedding_function(["你好", " GearLink"])
    assert vectors == [[2.0], [9.0]]
    assert calls == [["你好", " GearLink"]]

    # 注入后读写路径正常
    memory.add_message({"role": "user", "content": "我喜欢喝茶"})
    hits = memory.get_relevant_messages("茶", limit=1)
    assert len(hits) == 1


def test_relevance_threshold_filters_low_relevance():
    """配置 relevance_threshold 后，distance 超阈值的条目被过滤"""
    memory = LongTermMemory(vector_db=FakeClient(), collection_name="test", relevance_threshold=1.5)
    for content in ["相关甲", "相关乙", "偏远丙"]:
        memory.add_message({"role": "user", "content": content})

    # FakeCollection 按插入序返回 distance 0.0 / 1.0 / 2.0
    results = memory.get_relevant_messages("查询", limit=3)

    assert [r.content for r in results] == ["相关甲", "相关乙"]


def test_relevance_threshold_default_keeps_all(memory):
    """默认未配置阈值时不过滤（行为与现状一致）"""
    for content in ["甲", "乙", "丙"]:
        memory.add_message({"role": "user", "content": content})

    results = memory.get_relevant_messages("查询", limit=3)

    assert len(results) == 3


def test_mmr_rerank_prefers_diverse_content():
    """配置 mmr_lambda 后，MMR 优先选取与已选集差异大的条目"""
    memory = LongTermMemory(vector_db=FakeClient(), collection_name="test", mmr_lambda=0.5)
    memory.add_message({"role": "user", "content": "我喜欢喝红茶奶茶"})
    memory.add_message({"role": "user", "content": "我喜欢喝红茶奶茶加糖"})
    memory.add_message({"role": "user", "content": "北京今天天气很好"})

    results = memory.get_relevant_messages("查询", limit=2)

    # 最相关的首条不变；第二条选多样性更高的天气记忆而非近重复句
    assert [r.content for r in results] == ["我喜欢喝红茶奶茶", "北京今天天气很好"]


def test_mmr_lambda_one_keeps_relevance_order():
    """mmr_lambda=1.0 退化为纯相关度排序（等价不去冗余）"""
    memory = LongTermMemory(vector_db=FakeClient(), collection_name="test", mmr_lambda=1.0)
    memory.add_message({"role": "user", "content": "我喜欢喝红茶奶茶"})
    memory.add_message({"role": "user", "content": "我喜欢喝红茶奶茶加糖"})
    memory.add_message({"role": "user", "content": "北京今天天气很好"})

    results = memory.get_relevant_messages("查询", limit=2)

    assert [r.content for r in results] == ["我喜欢喝红茶奶茶", "我喜欢喝红茶奶茶加糖"]


class InMemoryStore:
    """VectorStore 协议的自定义后端示例（内存实现）"""

    def __init__(self) -> None:
        self.records: dict[str, tuple[str, dict]] = {}

    def add(self, ids, documents, metadatas) -> None:
        for record_id, doc, meta in zip(ids, documents, metadatas):
            self.records[record_id] = (doc, meta)

    def get(self, include=None, where=None):
        records = self.records
        if where:
            records = {
                rid: value
                for rid, value in records.items()
                if all(value[1].get(key) == val for key, val in where.items())
            }
        ids = list(records)
        return {
            "ids": ids,
            "documents": [records[i][0] for i in ids],
            "metadatas": [records[i][1] for i in ids],
        }

    def query(self, query_text, n_results):
        items = list(self.records.items())[:n_results]
        return {
            "ids": [record_id for record_id, _ in items],
            "documents": [doc for _, (doc, _) in items],
            "metadatas": [meta for _, (_, meta) in items],
            "distances": [float(i) for i in range(len(items))],
        }

    def delete(self, ids) -> None:
        for record_id in ids:
            self.records.pop(record_id, None)


def test_custom_vector_store_injection():
    """直接注入 VectorStore 后端时读写检索路径全部可用"""
    store = InMemoryStore()
    memory = LongTermMemory(store=store)
    assert isinstance(store, VectorStore)
    assert memory.collection is None

    memory.add_message({"role": "user", "content": "自定义后端记忆"})
    assert len(store.records) == 1

    results = memory.get_relevant_messages("查询", limit=1)
    assert [r.content for r in results] == ["自定义后端记忆"]

    memory.clear()
    assert memory.get_messages() == []


def test_custom_store_takes_precedence_over_vector_db():
    """store 与 vector_db 同时提供时 store 优先"""
    store = InMemoryStore()
    client = FakeClient()
    memory = LongTermMemory(vector_db=client, collection_name="test", store=store)

    memory.add_message({"role": "user", "content": "优先级验证"})
    assert len(store.records) == 1
    assert len(client.collection.records) == 0


def test_missing_vector_db_and_store_raises():
    """vector_db 与 store 均未提供时报 ValueError"""
    with pytest.raises(ValueError):
        LongTermMemory()


def test_chroma_vector_store_flattens_query_result():
    """ChromaVectorStore 把 chromadb 嵌套列表结果扁平化为单列表"""
    collection = FakeCollection()
    collection.add(ids=["a"], documents=["文档"], metadatas=[{"role": "user", "timestamp": 1.0}])
    store = ChromaVectorStore(collection)

    result = store.query(query_text="查询", n_results=1)

    assert result["ids"] == ["a"]
    assert result["documents"] == ["文档"]
    assert result["distances"] == [0.0]
