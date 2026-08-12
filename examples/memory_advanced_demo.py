"""GearLink 记忆深化示例（开发方向 §5.2）

演示四组进阶能力（均默认关闭，本示例显式开启）：

- 上下文摘要压缩：`compress_context=True` + `summarizer` + `max_context_tokens`，
  短期记忆超出预算阈值时把最旧一段压缩为 `[上下文摘要]`；
- 用户画像：`profile_hook` 在 `end_session` 时沉淀用户背景，
  之后 `build_context` 优先以 `[用户画像]` system 消息注入；
- 检索质量：`relevance_threshold` 阈值过滤 + `mmr_lambda` MMR 去冗余；
- 存储后端抽象：实现 `VectorStore` 协议注入自定义后端（此处为内存实现，
  无需 chromadb 与任何 API key）。

本示例全部使用本地内存实现，无需任何 API key 即可运行：
    python examples/memory_advanced_demo.py
"""

from typing import Any

from gearlink import LongTermMemory, MemoryManager, ShortTermMemory


class InMemoryVectorStore:
    """VectorStore 协议的内存实现（演示用；真实场景可用 chromadb 等后端）。"""

    def __init__(self) -> None:
        self.records: dict[str, tuple[str, dict[str, Any]]] = {}
        self._counter = 0

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        for record_id, doc, meta in zip(ids, documents, metadatas):
            self.records[record_id] = (doc, meta)

    def get(
        self, include: list[str] | None = None, where: dict[str, Any] | None = None
    ) -> dict[str, Any]:
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

    def query(self, query_text: str, n_results: int) -> dict[str, list[Any]]:
        # 演示用：按与查询的字面重合度粗排，distance 为不重合字符数
        query_chars = set(query_text)

        def distance(doc: str) -> float:
            return float(len(set(doc) - query_chars))

        items = sorted(self.records.items(), key=lambda kv: distance(kv[1][0]))[:n_results]
        return {
            "ids": [rid for rid, _ in items],
            "documents": [doc for _, (doc, _) in items],
            "metadatas": [meta for _, (_, meta) in items],
            "distances": [distance(doc) for _, (doc, _) in items],
        }

    def delete(self, ids: list[str]) -> None:
        for record_id in ids:
            self.records.pop(record_id, None)


def fake_summarizer(transcript: str) -> str:
    """演示用摘要器：真实场景可注入一次模型调用生成摘要。"""
    return f"此前对话共 {len(transcript.splitlines())} 轮，要点已由摘要器提炼。"


def fake_profile_hook(messages: list[dict[str, Any]], profile: dict[str, Any] | None):
    """演示用画像钩子：从本轮消息提取用户偏好，返回新画像。"""
    merged = dict(profile or {})
    for message in messages:
        if message.get("role") == "user" and "喜欢" in str(message.get("content", "")):
            merged["偏好"] = message["content"]
    return merged or None


def main() -> None:
    # ---- 1) 上下文摘要压缩 + 用户画像 ----
    manager = MemoryManager(
        short_term=ShortTermMemory(max_message=50),
        max_context_tokens=60,  # 刻意调小以便触发压缩
        summarizer=fake_summarizer,
        compress_context=True,
        profile_hook=fake_profile_hook,
    )
    for i in range(6):
        manager.add_message({"role": "user", "content": f"第 {i} 轮：我喜欢喝红茶，请展开讲讲。"})
        manager.add_message({"role": "assistant", "content": f"第 {i} 轮：好的，红茶的要点是……"})

    context = manager.build_context("红茶")
    print("压缩前上下文条数:", len(manager.short_term.get_messages()))
    print("首条消息（摘要已写回短期记忆头部）:")
    print(" ", manager.short_term.get_messages()[0]["content"])

    manager.end_session()  # 触发 profile_hook 沉淀画像
    print("沉淀的用户画像:", manager.profile)

    manager.add_message({"role": "user", "content": "再来一杯"})
    context = manager.build_context("红茶")
    print("画像注入后的首条上下文:", context[0]["content"])

    # ---- 2) 检索阈值过滤 + MMR 去冗余 + 自定义 VectorStore ----
    memory = LongTermMemory(
        store=InMemoryVectorStore(),
        relevance_threshold=3.0,  # distance 超过 3.0 的低相关条目被过滤
        mmr_lambda=0.5,  # 平衡相关度与多样性，避免 top-k 全是重复语义
    )
    memory.add_message({"role": "user", "content": "我喜欢喝红茶"})
    memory.add_message({"role": "user", "content": "我喜欢喝红茶加奶"})
    memory.add_message({"role": "assistant", "content": "北京今天天气很好"})
    memory.add_message({"role": "assistant", "content": "量子物理与茶毫无关系"})

    hits = memory.get_relevant_messages("我喜欢喝红茶", limit=2)
    print("\n阈值过滤 + MMR 去冗余后的检索结果：")
    for hit in hits:
        print(f"  [{hit.role}] {hit.content}（distance={hit.distance}）")


if __name__ == "__main__":
    main()
