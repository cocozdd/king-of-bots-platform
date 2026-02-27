"""
LLM 客户端模块 - Phase 4 升级

功能：
- 同步/异步 LLM 客户端构建
- 支持 Structured Output
- 环境变量配置

支持模型：
- DeepSeek V3（默认，性价比高）
- GLM-4（智谱 AI，国内可用）
- OpenAI GPT-4o

面试要点：
- 2026 年标准：异步优先，提升并发性能
- Structured Output 保证输出格式可控
"""
import logging
import os
import threading
from typing import Optional, Type, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


def _normalize_provider(value: Optional[str]) -> str:
    return (value or "auto").strip().lower()


def _is_deepseek_base(base_url: Optional[str]) -> bool:
    if not base_url:
        return False
    return "deepseek" in base_url.lower()


def get_openai_compatible_credentials() -> dict:
    """
    返回 OpenAI 兼容的 api_key/base_url。
    优先级: AI_PROVIDER -> 自动推断。
    
    支持: deepseek | openai | glm | auto
    """
    provider = _normalize_provider(os.getenv("AI_PROVIDER"))
    openai_key = os.getenv("OPENAI_API_KEY") or ""
    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or ""
    glm_key = os.getenv("GLM_API_KEY") or os.getenv("ZHIPU_API_KEY") or ""
    openai_base = os.getenv("OPENAI_API_BASE")
    deepseek_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    glm_base = os.getenv("GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4")

    if provider == "openai":
        if not openai_key:
            logger.warning("AI_PROVIDER=openai 但 OPENAI_API_KEY 未设置")
        return {"api_key": openai_key or None, "api_base": openai_base}
    
    if provider == "deepseek":
        if not deepseek_key:
            logger.warning("AI_PROVIDER=deepseek 但 DEEPSEEK_API_KEY 未设置")
        return {"api_key": deepseek_key or None, "api_base": deepseek_base}
    
    if provider == "glm" or provider == "zhipu":
        if not glm_key:
            logger.warning("AI_PROVIDER=glm 但 GLM_API_KEY 未设置")
        return {"api_key": glm_key or None, "api_base": glm_base, "model": "glm-4"}

    # auto 模式：优先 DeepSeek > GLM > OpenAI
    if deepseek_key and (_is_deepseek_base(openai_base) or not openai_key):
        return {"api_key": deepseek_key, "api_base": openai_base or deepseek_base}
    
    if glm_key:
        return {"api_key": glm_key, "api_base": glm_base, "model": "glm-4"}

    return {
        "api_key": openai_key or deepseek_key or None,
        "api_base": openai_base or (deepseek_base if deepseek_key else None),
    }


def should_use_llm() -> bool:
    """判断是否使用真实 LLM"""
    mode = os.getenv("AI_SERVICE_MODE", "auto").lower()
    if mode == "mock":
        return False
    if mode == "llm":
        return True
    creds = get_openai_compatible_credentials()
    return bool(creds.get("api_key"))


def _get_llm_config() -> dict:
    """获取 LLM 配置"""
    creds = get_openai_compatible_credentials()
    api_key = creds.get("api_key")
    if not api_key:
        return {}

    api_base = creds.get("api_base")

    model = os.getenv("AI_CHAT_MODEL", "deepseek-chat")
    
    config = {
        "model": model,
        "temperature": float(os.getenv("AI_TEMPERATURE", "0.2")),
        "api_key": api_key,  # 新版 langchain-openai 使用 api_key
    }
    if api_base:
        config["base_url"] = api_base  # 新版使用 base_url
    
    return config


def build_llm(streaming: bool = False) -> Optional[ChatOpenAI]:
    """
    构建同步 LLM 客户端
    
    Args:
        streaming: 是否启用流式输出
        
    Returns:
        ChatOpenAI 实例，配置无效时返回 None
    """
    config = _get_llm_config()
    if not config:
        return None
    
    config["streaming"] = streaming
    return ChatOpenAI(**config)


def build_llm_async() -> Optional[ChatOpenAI]:
    """
    构建异步 LLM 客户端（推荐用于 FastAPI）
    
    使用方式：
        llm = build_llm_async()
        response = await llm.ainvoke(messages)
    
    Returns:
        ChatOpenAI 实例（支持 ainvoke）
    """
    config = _get_llm_config()
    if not config:
        return None
    
    config["streaming"] = False
    return ChatOpenAI(**config)


def build_structured_llm(
    output_schema: Type[T],
    streaming: bool = False,
) -> Optional[ChatOpenAI]:
    """
    构建支持 Structured Output 的 LLM
    
    Args:
        output_schema: Pydantic 模型，定义输出结构
        streaming: 是否流式
        
    Returns:
        绑定了结构化输出的 LLM
        
    使用方式：
        class Answer(BaseModel):
            answer: str
            confidence: float
        
        llm = build_structured_llm(Answer)
        result = await llm.ainvoke(messages)  # result 是 Answer 类型
    """
    llm = build_llm(streaming)
    if llm is None:
        return None
    
    return llm.with_structured_output(output_schema)


# 全局 LLM 实例缓存 - 线程安全
_llm_instance: Optional[ChatOpenAI] = None
_llm_async_instance: Optional[ChatOpenAI] = None
_llm_lock = threading.Lock()
_llm_async_lock = threading.Lock()


def get_llm() -> Optional[ChatOpenAI]:
    """获取全局同步 LLM 实例（线程安全）"""
    global _llm_instance
    if _llm_instance is None:
        with _llm_lock:
            # Double-check locking pattern
            if _llm_instance is None:
                _llm_instance = build_llm(streaming=False)
    return _llm_instance


def get_llm_async() -> Optional[ChatOpenAI]:
    """获取全局异步 LLM 实例（线程安全）"""
    global _llm_async_instance
    if _llm_async_instance is None:
        with _llm_async_lock:
            if _llm_async_instance is None:
                _llm_async_instance = build_llm_async()
    return _llm_async_instance
