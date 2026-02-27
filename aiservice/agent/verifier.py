"""
Verifier 模块 - Bot 代码校验

第一级：静态结构校验（非空、class 声明、入口方法、安全黑名单、括号平衡）
第二级：编译验证钩子（可选，调用后端 /ai/bot/compile-check）
"""
import re
import logging
from dataclasses import dataclass
from typing import Optional

import requests as http_requests

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """校验结果"""
    passed: bool
    reason: Optional[str] = None
    suggestions: Optional[str] = None


# 安全黑名单模式
_SECURITY_BLACKLIST = [
    (r"\bSystem\s*\.\s*exit\b", "禁止调用 System.exit"),
    (r"\bRuntime\s*\.\s*getRuntime\s*\(\s*\)\s*\.\s*exec\b", "禁止调用 Runtime.exec"),
    (r"\bProcessBuilder\b", "禁止使用 ProcessBuilder"),
    (r"\bjava\s*\.\s*net\s*\.", "禁止直接网络访问 (java.net)"),
    (r"\bjava\s*\.\s*io\s*\.\s*File\b", "禁止直接文件系统访问 (java.io.File)"),
    (r"\bjava\s*\.\s*nio\s*\.\s*file\b", "禁止直接文件系统访问 (java.nio.file)"),
]


def verify_bot_code(code: str) -> VerificationResult:
    """
    第一级：静态结构校验
    
    1. 非空检查
    2. 有 class 声明
    3. 有 nextMove 方法或 main 入口
    4. 安全黑名单（System.exit, Runtime.exec, ProcessBuilder, 网络访问, 文件系统）
    5. 读取 INPUT 环境变量（main 入口时必需）
    6. 括号/花括号平衡检查（检测 LLM 截断输出）
    """
    # 1. 非空检查
    if not code or not code.strip():
        return VerificationResult(
            passed=False,
            reason="代码为空",
            suggestions="请提供有效的 Bot 代码。",
        )

    stripped = code.strip()

    # 2. 有 class 声明
    if not re.search(r"\bclass\s+\w+", stripped):
        return VerificationResult(
            passed=False,
            reason="缺少 class 声明",
            suggestions="Bot 代码必须包含至少一个 class 声明。",
        )

    # 3. 有 nextMove 方法或 main 入口
    has_next_move = bool(re.search(r"\bInteger\s+nextMove\b|\bint\s+nextMove\b", stripped))
    has_main = bool(re.search(r"\bpublic\s+static\s+void\s+main\b", stripped))
    if not has_next_move and not has_main:
        return VerificationResult(
            passed=False,
            reason="缺少 nextMove 方法或 main 入口",
            suggestions="Bot 代码必须包含 nextMove 方法（实现 BotInterface）或 public static void main 入口。",
        )

    # 4. 安全黑名单
    for pattern, message in _SECURITY_BLACKLIST:
        if re.search(pattern, stripped):
            return VerificationResult(
                passed=False,
                reason=f"安全违规: {message}",
                suggestions=f"请移除 {message.split('禁止')[1] if '禁止' in message else message} 相关代码。",
            )

    # 5. main 入口时检查 INPUT 环境变量读取
    if has_main and not has_next_move:
        if "System.getenv" not in stripped and "INPUT" not in stripped:
            return VerificationResult(
                passed=False,
                reason="main 入口缺少 INPUT 环境变量读取",
                suggestions="使用 main 入口时，必须通过 System.getenv(\"INPUT\") 读取输入。",
            )

    # 6. 括号/花括号平衡检查
    brace_count = stripped.count("{") - stripped.count("}")
    paren_count = stripped.count("(") - stripped.count(")")
    if brace_count != 0:
        return VerificationResult(
            passed=False,
            reason=f"花括号不平衡（差值: {brace_count}），可能是 LLM 截断输出",
            suggestions="请检查代码的花括号是否完整闭合。",
        )
    if paren_count != 0:
        return VerificationResult(
            passed=False,
            reason=f"圆括号不平衡（差值: {paren_count}），可能是 LLM 截断输出",
            suggestions="请检查代码的圆括号是否完整闭合。",
        )

    return VerificationResult(passed=True)


def try_compile_check(code: str) -> VerificationResult:
    """
    第二级：编译验证钩子（可选）。
    
    调用 /ai/bot/compile-check 端点（不存在则跳过，视为通过）。
    生产可接入 BotRunningSystem 编译沙箱。
    """
    try:
        resp = http_requests.post(
            "http://127.0.0.1:3000/ai/bot/compile-check",
            json={"code": code},
            timeout=5,
        )
        if resp.status_code == 404:
            # 端点不存在，跳过编译检查
            return VerificationResult(passed=True)
        data = resp.json()
        if data.get("success"):
            return VerificationResult(passed=True)
        return VerificationResult(
            passed=False,
            reason=f"编译失败: {data.get('error', '未知错误')}",
            suggestions=data.get("suggestions"),
        )
    except http_requests.ConnectionError:
        # 后端不可达，跳过编译检查
        logger.debug("compile-check 端点不可达，跳过编译验证")
        return VerificationResult(passed=True)
    except Exception as e:
        logger.warning("编译检查异常: %s", e)
        return VerificationResult(passed=True)
