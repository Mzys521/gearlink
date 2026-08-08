"""GearLink 统一异常层次，遵循 docs/接口设计规范.md 第 5 节。

公共 API 只抛出本模块定义的异常；捕获第三方异常后须转换包装并保留 __cause__。
"""


class GearLinkError(Exception):
    """所有 GearLink 项目异常的基类"""


class ProviderError(GearLinkError):
    """模型服务调用失败（网络、鉴权、限流等）"""


class ToolError(GearLinkError):
    """工具注册/执行失败"""


class ToolNotFoundError(ToolError):
    """请求的工具名未注册"""


class MemoryError(GearLinkError):
    """记忆读写失败"""
