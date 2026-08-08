from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional,Dict,Any,List
import yaml
import warnings
import logging

#【关键】从上级包 gearlink 导入异常（使用相对导入）
from ..exceptions import (
    SkillValidationError,
    SkillNotFoundError,
    SkillLoadError,
)

# ---------- Skill 数据契约 ----------
@dataclass
class Skill:
    """技能核心数据模型(符号渐进式披露设计)"""
    name: str
    description: str
    path: Path
    instructions: Optional[str] = None          # L2 完整指令，初次为 None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.name or not self.name.replace('-','').isalnum():
            raise SkillValidationError(
                f"Skill name must be alphanumeric with hyphens, got '{self.name}'"
            )
        if not self.description:
            raise SkillValidationError("Skill description cannot be empty.")
# ---------- SkillRegistry 注册表 ----------
class SkillRegistry:
    """内存注册表，存储已发现技能的 L1 元数据。"""
    
    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if skill.name in self._skills:
            warnings.warn(f"Skill '{skill.name}' already registered, overwriting.")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        skill = self._skills.get(name)
        if skill is None:
            raise SkillNotFoundError(f"Skill '{name}' not found in registry.")
        return skill

    def list_all(self) -> List[Skill]:
        return list(self._skills.values())

    def contains(self, name: str) -> bool:
        return name in self._skills

# ---------- SkillLoader 加载器 ----------
class SkillLoader:
    """从文件系统发现并加载技能。"""

    @staticmethod
    def discover_from_directory(base_dir: Path) -> List[Skill]:
        """扫描目录，返回所有有效技能的元数据列表（仅 L1）。"""
        skills = []
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
                # 简单解析 YAML frontmatter
                parts = content.split("---", 2)
                if len(parts) < 3:
                    raise SkillValidationError(f"Invalid SKILL.md format in {skill_md}")

                frontmatter = yaml.safe_load(parts[1])
                if not frontmatter or "name" not in frontmatter or "description" not in frontmatter:
                    raise SkillValidationError("Missing 'name' or 'description' in frontmatter")

                skill = Skill(
                    name=frontmatter["name"],
                    description=frontmatter["description"],
                    path=skill_dir,
                    metadata={k: v for k, v in frontmatter.items() if k not in ("name", "description")}
                )
                skills.append(skill)
            except Exception as e:
                logging.warning(f"Failed to load skill from {skill_dir}: {e}")

        return skills

    @staticmethod
    def load_full_instructions(skill: Skill) -> str:
        """加载技能的完整指令正文（L2 信息）。"""
        md_path = skill.path / "SKILL.md"
        if not md_path.exists():
            raise SkillLoadError(f"SKILL.md not found at {md_path}")
        try:
            content = md_path.read_text(encoding="utf-8")
            parts = content.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()   # 只返回正文部分
            return content
        except Exception as e:
            raise SkillLoadError(f"Failed to read {md_path}: {e}")