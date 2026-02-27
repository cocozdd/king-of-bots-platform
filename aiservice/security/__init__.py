"""
Security 模块 - 安全防护

包含：
- PromptSecurityService: Prompt 安全服务（注入检测、输入校验）

注意：主要的安全功能在 guardrails.py 中实现
此模块提供更高级的封装和额外功能
"""
from typing import Tuple

# 从 guardrails 导入核心功能
from guardrails import (
    InputGuard,
    OutputGuard,
    RateLimiter,
    ValidationResult,
    get_input_guard,
    get_output_guard,
    get_rate_limiter,
    validate_input,
    filter_output,
)


class PromptSecurityService:
    """
    Prompt 安全服务 - 对齐 Java PromptSecurityService
    
    提供统一的安全检查接口
    """
    
    def __init__(self):
        self.input_guard = get_input_guard()
        self.output_guard = get_output_guard()
    
    def validate_input(self, text: str, max_length: int = 10000) -> Tuple[bool, str]:
        """
        输入校验
        
        Returns:
            (is_safe, error_message)
        """
        result = self.input_guard.validate(text)
        return result.is_safe, result.reason
    
    def detect_injection(self, text: str) -> bool:
        """Prompt Injection 检测"""
        result = self.input_guard._detect_injection(text)
        return not result.is_safe
    
    def sanitize_output(self, text: str) -> str:
        """输出脱敏"""
        return self.output_guard.filter(text)
    
    def get_refusal_response(self, reason: str = "安全策略") -> str:
        """标准化拒答响应"""
        return f"抱歉，由于{reason}，我无法处理这个请求。请换一种方式提问。"


# 全局实例
_security_service = None


def get_security_service() -> PromptSecurityService:
    """获取全局安全服务实例"""
    global _security_service
    if _security_service is None:
        _security_service = PromptSecurityService()
    return _security_service


__all__ = [
    "PromptSecurityService",
    "get_security_service",
    "InputGuard",
    "OutputGuard",
    "RateLimiter",
    "ValidationResult",
    "validate_input",
    "filter_output",
]
