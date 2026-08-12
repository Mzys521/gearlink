"""GearLink 统一异常层次，遵循 docs/接口设计规范.md 第 5 节。

公共 API 只抛出本模块定义的异常；捕获第三方异常后须转换包装并保留 __cause__。
"""


class GearLinkError(Exception):
    """所有 GearLink 项目异常的基类"""


class ProviderError(GearLinkError):
    """模型服务调用失败（网络、鉴权、限流等）。

    Attributes:
        retryable: 是否为可重试错误（网络故障、限流等瞬时异常）；
            鉴权失败等确定性错误为 False。重试策略由调用方决定
            （如 `ReactAgent(max_retries=...)`，开发方向 §4.3）。
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        """初始化异常。

        Args:
            message: 面向开发者的错误描述（含模型名等定位信息，不泄露密钥）。
            retryable: 是否可重试；默认 False（等价于无此字段的历史行为）。
        """
        super().__init__(message)
        self.retryable = retryable


class ToolError(GearLinkError):
    """工具注册/执行失败"""


class ToolNotFoundError(ToolError):
    """请求的工具名未注册"""


class MemoryError(GearLinkError):
    """记忆读写失败"""


# ========== Skill 相关异常 ==========
class SkillError(GearLinkError):
    """所有 Skill 相关异常的基类"""


class SkillNotFoundError(SkillError):
    """按名称查找技能时未找到"""


class SkillLoadError(SkillError):
    """加载技能内容（如 SKILL.md 解析失败）时抛出"""


class SkillValidationError(SkillError):
    """技能元数据（YAML frontmatter）验证失败"""


class SkillExecutionError(SkillError):
    """执行技能辅助脚本时出错"""
