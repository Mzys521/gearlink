"""skills 模块测试：Skill 数据契约、SkillRegistry 与 SkillLoader（不涉及外部服务）。"""

from pathlib import Path

import pytest

from gearlink.exceptions import SkillLoadError, SkillNotFoundError, SkillValidationError
from gearlink.skills import Skill, SkillLoader, SkillRegistry

SKILL_MD = """---
name: demo-skill
description: 演示技能
author: gearlink
---

# 示例指令

按步骤执行。
"""


def make_skill_dir(base_dir: Path, dirname: str = "demo-skill") -> Path:
    """在 base_dir 下创建一个合法的测试技能目录"""
    skill_dir = base_dir / dirname
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    return skill_dir


def test_skill_rejects_invalid_name():
    with pytest.raises(SkillValidationError):
        Skill(name="非法名称!", description="描述", path=Path("."))


def test_skill_rejects_empty_description():
    with pytest.raises(SkillValidationError):
        Skill(name="ok-name", description="", path=Path("."))


def test_registry_register_and_get():
    registry = SkillRegistry()
    skill = Skill(name="demo-skill", description="演示", path=Path("."))

    registry.register(skill)

    assert registry.get("demo-skill") is skill
    assert registry.contains("demo-skill")
    assert registry.list_all() == [skill]


def test_registry_get_missing_raises_not_found():
    with pytest.raises(SkillNotFoundError):
        SkillRegistry().get("no-such-skill")


def test_loader_discovers_skills(tmp_path):
    make_skill_dir(tmp_path)
    # 非技能文件应被忽略
    (tmp_path / "not-a-skill.txt").write_text("忽略我", encoding="utf-8")

    skills = SkillLoader.discover_from_directory(tmp_path)

    assert [s.name for s in skills] == ["demo-skill"]
    assert skills[0].description == "演示技能"
    assert skills[0].metadata == {"author": "gearlink"}
    assert skills[0].instructions is None  # 仅 L1，L2 按需加载


def test_loader_discover_missing_directory_returns_empty(tmp_path):
    assert SkillLoader.discover_from_directory(tmp_path / "不存在") == []


def test_loader_discover_skips_invalid_skill(tmp_path):
    # 非法技能（无 frontmatter）仅跳过，不中断整体发现
    bad_dir = tmp_path / "bad-skill"
    bad_dir.mkdir()
    (bad_dir / "SKILL.md").write_text("没有 frontmatter", encoding="utf-8")
    make_skill_dir(tmp_path)

    skills = SkillLoader.discover_from_directory(tmp_path)

    assert [s.name for s in skills] == ["demo-skill"]


def test_loader_loads_full_instructions(tmp_path):
    make_skill_dir(tmp_path)
    skill = SkillLoader.discover_from_directory(tmp_path)[0]

    instructions = SkillLoader.load_full_instructions(skill)

    assert instructions.startswith("# 示例指令")


def test_loader_raises_when_skill_md_missing(tmp_path):
    skill = Skill(name="demo-skill", description="演示", path=tmp_path)

    with pytest.raises(SkillLoadError):
        SkillLoader.load_full_instructions(skill)
