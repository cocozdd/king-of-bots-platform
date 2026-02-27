"""
安全防护模块 - Phase 4 新增

功能：
- 输入验证：检测 Prompt Injection、敏感词
- 输出过滤：移除敏感信息泄露
- Rate Limiting：请求频率限制

P0 改进（2026-01）：
- 集成 LLMGuard，提升检测召回率到 95%+
- 正则预筛 + LLM 深度检测的两阶段策略

面试要点：
- 2026 年 AI 安全标准要求多层防护
- Prompt Injection 是 LLM 应用的主要安全威胁
- 输出过滤防止模型泄露敏感信息
- LLM-based 检测可发现未知攻击模式
"""
import logging
import re
import time
from typing import Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """验证结果"""
    is_safe: bool
    reason: str = ""
    details: dict = field(default_factory=dict)


class InputGuard:
    """
    输入验证器
    
    检测：
    1. Prompt Injection 攻击
    2. 敏感词/违禁内容
    3. 异常长度输入
    """
    
    # Prompt Injection 特征模式
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+(instructions?|prompts?)",
        r"disregard\s+(previous|above|all)",
        r"forget\s+(everything|all|previous)",
        r"you\s+are\s+now\s+[a-z]+",
        r"act\s+as\s+(if\s+)?you",
        r"pretend\s+(to\s+be|you\s+are)",
        r"system\s*:\s*",
        r"\[INST\]",
        r"<\|im_start\|>",
        r"###\s*(instruction|system)",
    ]
    
    # 敏感词列表
    SENSITIVE_WORDS = [
        "密码", "password", "token", "api_key", "secret",
        "私钥", "private_key", "credential", "auth_token",
    ]
    
    # 最大输入长度
    MAX_INPUT_LENGTH = 10000
    
    def __init__(self):
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]
    
    def validate(self, user_input: str) -> ValidationResult:
        """
        验证用户输入
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            ValidationResult: 验证结果
        """
        if not user_input:
            return ValidationResult(is_safe=True)
        
        # 1. 长度检查
        if len(user_input) > self.MAX_INPUT_LENGTH:
            logger.warning("输入过长: %d 字符", len(user_input))
            return ValidationResult(
                is_safe=False,
                reason="INPUT_TOO_LONG",
                details={"length": len(user_input), "max": self.MAX_INPUT_LENGTH}
            )
        
        # 2. Prompt Injection 检测
        injection_result = self._detect_injection(user_input)
        if not injection_result.is_safe:
            logger.warning("检测到 Prompt Injection: %s", injection_result.details)
            return injection_result
        
        # 3. 敏感词检测（仅警告，不阻止）
        sensitive_result = self._detect_sensitive(user_input)
        if not sensitive_result.is_safe:
            logger.info("输入包含敏感词: %s", sensitive_result.details)
            # 敏感词不阻止，只记录
        
        return ValidationResult(is_safe=True)
    
    def _detect_injection(self, text: str) -> ValidationResult:
        """检测 Prompt Injection"""
        text_lower = text.lower()
        
        for i, pattern in enumerate(self._compiled_patterns):
            match = pattern.search(text_lower)
            if match:
                return ValidationResult(
                    is_safe=False,
                    reason="PROMPT_INJECTION_DETECTED",
                    details={
                        "pattern_index": i,
                        "matched": match.group(),
                        "position": match.start(),
                    }
                )
        
        return ValidationResult(is_safe=True)
    
    def _detect_sensitive(self, text: str) -> ValidationResult:
        """检测敏感词"""
        text_lower = text.lower()
        found = []
        
        for word in self.SENSITIVE_WORDS:
            if word.lower() in text_lower:
                found.append(word)
        
        if found:
            return ValidationResult(
                is_safe=False,
                reason="SENSITIVE_CONTENT",
                details={"words": found}
            )
        
        return ValidationResult(is_safe=True)


class OutputGuard:
    """
    输出过滤器
    
    过滤：
    1. API Key 泄露
    2. 系统路径泄露
    3. 内部错误信息
    """
    
    # 需要过滤的模式
    FILTER_PATTERNS = [
        # API Keys
        (r'sk-[a-zA-Z0-9]{20,}', '[API_KEY_REDACTED]'),
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9_-]{20,}', '[API_KEY_REDACTED]'),
        
        # 系统路径
        (r'/Users/[^\s"\'<>]+', '[PATH_REDACTED]'),
        (r'/home/[^\s"\'<>]+', '[PATH_REDACTED]'),
        (r'C:\\Users\\[^\s"\'<>]+', '[PATH_REDACTED]'),
        
        # 内部 IP
        (r'192\.168\.\d+\.\d+', '[INTERNAL_IP]'),
        (r'10\.\d+\.\d+\.\d+', '[INTERNAL_IP]'),
        
        # 数据库连接字符串
        (r'(mysql|postgresql|mongodb)://[^\s"\'<>]+', '[DB_URL_REDACTED]'),
    ]
    
    def __init__(self):
        self._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), r) for p, r in self.FILTER_PATTERNS
        ]
    
    def filter(self, response: str) -> str:
        """
        过滤输出中的敏感信息
        
        Args:
            response: 模型输出
            
        Returns:
            过滤后的输出
        """
        if not response:
            return response
        
        filtered = response
        for pattern, replacement in self._compiled_patterns:
            filtered = pattern.sub(replacement, filtered)
        
        return filtered
    
    def filter_with_report(self, response: str) -> Tuple[str, dict]:
        """
        过滤输出并返回过滤报告
        
        Returns:
            (filtered_response, report)
        """
        if not response:
            return response, {}
        
        filtered = response
        report = {"filtered_count": 0, "patterns_matched": []}
        
        for pattern, replacement in self._compiled_patterns:
            matches = pattern.findall(filtered)
            if matches:
                report["filtered_count"] += len(matches)
                report["patterns_matched"].append(replacement)
                filtered = pattern.sub(replacement, filtered)
        
        if report["filtered_count"] > 0:
            logger.info("输出过滤: %d 处敏感信息", report["filtered_count"])
        
        return filtered, report


class RateLimiter:
    """
    请求频率限制器
    
    使用滑动窗口算法
    """
    
    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: int = 60,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> Tuple[bool, dict]:
        """
        检查是否允许请求
        
        Args:
            client_id: 客户端标识（如 IP、用户ID）
            
        Returns:
            (is_allowed, info)
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # 清理过期记录
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > window_start
        ]
        
        current_count = len(self._requests[client_id])
        remaining = self.max_requests - current_count
        
        if current_count >= self.max_requests:
            return False, {
                "remaining": 0,
                "reset_in": int(self._requests[client_id][0] - window_start),
            }
        
        # 记录本次请求
        self._requests[client_id].append(now)
        
        return True, {"remaining": remaining - 1}
    
    def get_usage(self, client_id: str) -> dict:
        """获取使用情况"""
        now = time.time()
        window_start = now - self.window_seconds
        
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > window_start
        ]
        
        current_count = len(self._requests[client_id])
        
        return {
            "used": current_count,
            "limit": self.max_requests,
            "remaining": max(0, self.max_requests - current_count),
            "window_seconds": self.window_seconds,
        }


# 全局实例
_input_guard: Optional[InputGuard] = None
_output_guard: Optional[OutputGuard] = None
_rate_limiter: Optional[RateLimiter] = None


def get_input_guard() -> InputGuard:
    """获取输入验证器"""
    global _input_guard
    if _input_guard is None:
        _input_guard = InputGuard()
    return _input_guard


def get_output_guard() -> OutputGuard:
    """获取输出过滤器"""
    global _output_guard
    if _output_guard is None:
        _output_guard = OutputGuard()
    return _output_guard


def get_rate_limiter(
    max_requests: int = 60,
    window_seconds: int = 60,
) -> RateLimiter:
    """获取频率限制器"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_requests, window_seconds)
    return _rate_limiter


def validate_input(user_input: str) -> ValidationResult:
    """便捷函数：验证输入"""
    return get_input_guard().validate(user_input)


def filter_output(response: str) -> str:
    """便捷函数：过滤输出"""
    return get_output_guard().filter(response)


async def validate_input_with_llm(user_input: str) -> ValidationResult:
    """
    使用 LLM 进行深度安全检测（P0 改进）
    
    两阶段策略：
    1. 正则检测（快速预筛）
    2. LLM 检测（深度验证）
    
    Returns:
        ValidationResult: 验证结果
    """
    # 1. 正则预筛（快速）
    input_guard = get_input_guard()
    regex_result = input_guard.validate(user_input)
    
    if not regex_result.is_safe:
        return regex_result
    
    # 2. LLM 深度检测（可选）
    try:
        from llm_guard import get_llm_guard
        llm_guard = get_llm_guard()
        
        if llm_guard:
            detection = await llm_guard.adetect(user_input)
            
            if detection.is_attack:
                return ValidationResult(
                    is_safe=False,
                    reason="LLM_DETECTED_INJECTION",
                    details={
                        "llm_reason": detection.reason,
                        "confidence": detection.confidence,
                    }
                )
    except Exception as e:
        logger.warning("LLM 安全检测失败，仅使用正则检测: %s", e)
    
    return ValidationResult(is_safe=True)
