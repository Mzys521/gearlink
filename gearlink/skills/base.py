"""技能扩展契约与基础设施：Skill 数据模型、注册表与文件系统加载器。

采用渐进式披露设计：注册表只保存技能的 L1 元数据（name / description），
完整指令正文（L2）按需经 `gearlink.tools.load_skill` 工具加载。
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gearlink.exceptions import SkillLoadError, SkillNotFoundError, SkillValidationError

#: 模块级日志器：记录技能发现与加载过程的问题
logger = logging.getLogger(__name__)


# ---------- Skill 数据契约 ----------
@dataclass
class Skill:
    """技能核心数据模型（渐进式披露设计）。

    Attributes:
        name: 技能唯一名称，仅允许字母、数字与连字符。
        description: 技能用途描述（L1 元数据，注入系统提示供模型选择）。
        path: 技能目录路径，须包含 SKILL.md。
        instructions: L2 完整指令正文，未加载时为 None。
        metadata: frontmatter 中除 name / description 外的附加元数据。

    Raises:
        SkillValidationError: name 或 description 不合法时由 __post_init__ 抛出。
    """

    name: str
    description: str
    path: Path
    instructions: str | None = None  # L2 完整指令，初次为 None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验元数据合法性：name 仅允许字母数字与连字符，description 非空。"""
        if not self.name or not self.name.replace("-", "").isalnum():
            raise SkillValidationError(f"技能名仅允许字母、数字与连字符，实际为 '{self.name}'")
        if not self.description:
            raise SkillValidationError("技能描述不能为空")


# ---------- SkillRegistry 注册表 ----------
class SkillRegistry:
    """技能内存注册表：存储已发现技能的 L1 元数据，显式注册、禁止运行时隐式扫描。"""

    def __init__(self) -> None:
        """初始化空注册表。"""
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """登记一个技能；同名技能重复登记时发出警告并覆盖。

        Args:
            skill: 待登记的技能对象。
        """
        if skill.name in self._skills:
            logger.warning("技能 '%s' 已注册，将被覆盖", skill.name)
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        """按名称获取技能。

        Args:
            name: 已注册的技能名称。

        Returns:
            对应的技能对象。

        Raises:
            SkillNotFoundError: 技能名未注册时抛出。
        """
        skill = self._skills.get(name)
        if skill is None:
            raise SkillNotFoundError(f"技能 '{name}' 未在注册表中找到")
        return skill

    def list_all(self) -> list[Skill]:
        """返回全部已注册技能的列表。

        Returns:
            技能对象列表（注册顺序）。
        """
        return list(self._skills.values())

    def contains(self, name: str) -> bool:
        """判断技能名是否已注册。

        Args:
            name: 技能名称。

        Returns:
            已注册返回 True，否则 False。
        """
        return name in self._skills


# ---------- SkillLoader 加载器 ----------
class SkillLoader:
    """技能文件系统加载器：从目录发现技能（仅 L1）并按需加载完整指令（L2）。"""

    @staticmethod
    def discover_from_directory(base_dir: Path) -> list[Skill]:
        """扫描目录发现技能，返回所有有效技能的元数据列表（仅 L1）。

        约定：base_dir 下每个子目录视为一个技能，须包含带 YAML frontmatter
        （name / description 必填）的 SKILL.md；单个技能解析失败仅记录警告并跳过，
        不中断整体发现流程。

        Args:
            base_dir: 技能根目录；不存在时返回空列表。

        Returns:
            发现的技能列表（仅含 L1 元数据，instructions 为 None）。
        """
        skills: list[Skill] = []
        if not base_dir.exists():
            return skills

        for skill_dir in base_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            try:
                content = skill_md.read_text(encoding="utf-8")
                # 简单解析 YAML frontmatter（--- 包裹的头部）
                parts = content.split("---", 2)
                if len(parts) < 3:
                    raise SkillValidationError(f"SKILL.md 格式无效: {skill_md}")

                frontmatter = yaml.safe_load(parts[1])
                if not frontmatter or "name" not in frontmatter or "description" not in frontmatter:
                    raise SkillValidationError("frontmatter 缺少 'name' 或 'description' 字段")

                skills.append(
                    Skill(
                        name=frontmatter["name"],
                        description=frontmatter["description"],
                        path=skill_dir,
                        metadata={
                            k: v for k, v in frontmatter.items() if k not in ("name", "description")
                        },
                    )
                )
            except Exception as e:
                logger.warning("技能加载失败，已跳过 %s: %s", skill_dir, e)

        return skills

    @staticmethod
    def load_full_instructions(skill: Skill) -> str:
        """加载技能的完整指令正文（L2 信息）。

        Args:
            skill: 已发现的技能对象。

        Returns:
            SKILL.md 的正文部分（去除 frontmatter）。

        Raises:
            SkillLoadError: SKILL.md 不存在或读取失败时抛出（保留原始异常为 __cause__）。
        """
        md_path = skill.path / "SKILL.md"
        if not md_path.exists():
            raise SkillLoadError(f"SKILL.md 不存在: {md_path}")
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError as e:
            # 包装第三方/系统异常为项目异常体系，保留异常链
            raise SkillLoadError(f"读取 SKILL.md 失败 {md_path}: {e}") from e
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()  # 只返回正文部分
        return content
