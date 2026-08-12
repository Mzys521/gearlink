"""GearLink 会话持久化与恢复示例（断线恢复）

演示 `MemoryManager.snapshot()` / `restore()`（开发方向 §4.5）：
把进行中的会话导出为 JSON 快照落盘，进程退出/断线后从快照恢复，
无缝继续对话。

本示例无需 API key，直接运行：
    python examples/session_restore_demo.py

真实应用中：
- 每轮对话结束后调用 `snapshot()` 落盘（或在 `Ctrl+C` 信号处理中保存）；
- 下次启动时读取快照文件调用 `restore()`，再把 memory 注入
  `ReactAgent(memory=...)` 即可继续上次的会话。
"""

import json
import tempfile
from pathlib import Path

from gearlink import MemoryManager, ShortTermMemory

SNAPSHOT_FILE = Path(tempfile.gettempdir()) / "gearlink_session_demo.json"


def phase_one_chat() -> None:
    """第一阶段：进行中的对话，随后导出快照落盘（模拟断线前）。"""
    print("=== 第一阶段：正常对话 ===")
    manager = MemoryManager(short_term=ShortTermMemory(max_message=20))
    manager.add_message({"role": "system", "content": "你是一个旅行规划助手"})
    manager.add_message({"role": "user", "content": "帮我规划一次三天的杭州旅行"})
    manager.add_message({"role": "assistant", "content": "好的，第一天建议游览西湖与灵隐寺"})
    manager.add_message({"role": "user", "content": "第二天我想去爬山"})

    # 导出快照并落盘：model / metadata 由应用层按需记录
    snapshot = manager.snapshot(model="gpt-4o", metadata={"user_id": "u-42"})
    SNAPSHOT_FILE.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"会话快照已保存到 {SNAPSHOT_FILE}（id={snapshot['id'][:8]}...）")
    print("（此时模拟进程退出 / 断线）\n")


def phase_two_restore() -> None:
    """第二阶段：全新进程从快照恢复会话，继续对话。"""
    print("=== 第二阶段：重启后恢复 ===")
    manager = MemoryManager(short_term=ShortTermMemory(max_message=20))

    snapshot = json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
    session = manager.restore(snapshot)
    print(f"已恢复会话 {session.id[:8]}...（模型: {session.model}，元数据: {session.metadata}）")

    # 恢复后无缝继续对话：新消息直接追加，历史上下文完整保留
    manager.add_message({"role": "assistant", "content": "第二天推荐前往北高峰徒步"})
    manager.add_message({"role": "user", "content": "第三天呢？"})

    context = manager.build_context("第三天呢？")
    print(f"\n本轮请求的完整上下文（共 {len(context)} 条消息）：")
    for message in context:
        preview = message["content"][:40]
        print(f"  [{message['role']}] {preview}")


if __name__ == "__main__":
    phase_one_chat()
    phase_two_restore()
    SNAPSHOT_FILE.unlink(missing_ok=True)
    print("\n演示完成，临时快照文件已清理。")
