"""
可观测性模块 - Phase 4

功能：
- LangSmith 集成（推荐用于开发调试）
- LangFuse 集成（推荐用于生产监控）
- 自定义追踪回调

面试要点：
- 2026 标准：AI 应用必须具备完整可观测性
- LangSmith 提供 Trace 可视化和调试
- LangFuse 提供生产级监控和分析
"""
import os
import logging
import functools
from typing import Optional, Any, Dict, List, Callable, TypeVar
from contextlib import contextmanager

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


def is_langsmith_enabled() -> bool:
    """检查 LangSmith 是否启用"""
    return bool(os.getenv("LANGCHAIN_API_KEY")) and os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"


def is_langfuse_enabled() -> bool:
    """检查 LangFuse 是否启用"""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))


def get_langsmith_callback():
    """获取 LangSmith 回调处理器"""
    if not is_langsmith_enabled():
        return None
    
    try:
        # LangSmith 通过环境变量自动启用，无需显式回调
        # 设置以下环境变量即可：
        # LANGCHAIN_TRACING_V2=true (or LANGSMITH_TRACING=true)
        # LANGCHAIN_API_KEY=your-api-key (or LANGSMITH_API_KEY)
        # LANGCHAIN_PROJECT=kob-ai-service
        logger.info("LangSmith tracing 已启用")
        return None  # LangChain 自动处理
    except Exception as e:
        logger.warning("LangSmith 初始化失败: %s", e)
        return None


def traceable(name: str = None, run_type: str = "chain") -> Callable[[F], F]:
    """
    LangSmith traceable 装饰器 - 2026 标准
    
    使用方式：
        @traceable(name="my_function", run_type="tool")
        def my_function(x):
            return x * 2
    
    Args:
        name: 追踪名称，默认使用函数名
        run_type: 运行类型，可选 chain/tool/llm/retriever
    """
    def decorator(func: F) -> F:
        # 尝试使用 LangSmith 的 traceable
        if is_langsmith_enabled():
            try:
                from langsmith import traceable as ls_traceable
                return ls_traceable(name=name or func.__name__, run_type=run_type)(func)
            except ImportError:
                pass
        
        # 回退：简单日志装饰器
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name or func.__name__
            logger.debug("Trace start: %s", func_name)
            try:
                result = func(*args, **kwargs)
                logger.debug("Trace end: %s", func_name)
                return result
            except Exception as e:
                logger.warning("Trace error: %s - %s", func_name, e)
                raise
        
        return wrapper  # type: ignore
    
    return decorator


def async_traceable(name: str = None, run_type: str = "chain") -> Callable[[F], F]:
    """
    异步版 traceable 装饰器
    """
    def decorator(func: F) -> F:
        if is_langsmith_enabled():
            try:
                from langsmith import traceable as ls_traceable
                return ls_traceable(name=name or func.__name__, run_type=run_type)(func)
            except ImportError:
                pass
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            func_name = name or func.__name__
            logger.debug("Async trace start: %s", func_name)
            try:
                result = await func(*args, **kwargs)
                logger.debug("Async trace end: %s", func_name)
                return result
            except Exception as e:
                logger.warning("Async trace error: %s - %s", func_name, e)
                raise
        
        return wrapper  # type: ignore
    
    return decorator


def get_langfuse_callback():
    """获取 LangFuse 回调处理器"""
    if not is_langfuse_enabled():
        return None
    
    try:
        from langfuse.callback import CallbackHandler
        
        handler = CallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        logger.info("LangFuse tracing 已启用")
        return handler
    except ImportError:
        logger.warning("langfuse 未安装，跳过 LangFuse 集成")
        return None
    except Exception as e:
        logger.warning("LangFuse 初始化失败: %s", e)
        return None


def get_tracing_callbacks() -> List[Any]:
    """获取所有可用的追踪回调"""
    callbacks = []
    
    langfuse = get_langfuse_callback()
    if langfuse:
        callbacks.append(langfuse)
    
    return callbacks


class TraceContext:
    """追踪上下文管理器"""
    
    def __init__(
        self,
        trace_id: str,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.trace_id = trace_id
        self.operation = operation
        self.metadata = metadata or {}
        self._callbacks = []
    
    def __enter__(self):
        self._callbacks = get_tracing_callbacks()
        
        # 记录开始
        logger.info(
            "Trace start: trace_id=%s operation=%s",
            self.trace_id,
            self.operation,
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.warning(
                "Trace error: trace_id=%s operation=%s error=%s",
                self.trace_id,
                self.operation,
                exc_val,
            )
        else:
            logger.info(
                "Trace end: trace_id=%s operation=%s",
                self.trace_id,
                self.operation,
            )
        
        # 刷新 LangFuse
        for cb in self._callbacks:
            if hasattr(cb, "flush"):
                try:
                    cb.flush()
                except Exception:
                    pass
        
        return False  # 不抑制异常
    
    @property
    def callbacks(self) -> List[Any]:
        """获取回调列表，用于传递给 LLM"""
        return self._callbacks


@contextmanager
def trace(
    trace_id: str,
    operation: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    追踪上下文管理器
    
    使用方式：
        with trace(trace_id, "chat") as ctx:
            llm.invoke(messages, config={"callbacks": ctx.callbacks})
    """
    ctx = TraceContext(trace_id, operation, metadata)
    with ctx:
        yield ctx


def log_llm_usage(
    trace_id: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
):
    """记录 LLM 使用情况"""
    logger.info(
        "LLM usage: trace_id=%s model=%s input_tokens=%d output_tokens=%d latency_ms=%d",
        trace_id,
        model,
        input_tokens,
        output_tokens,
        latency_ms,
    )


def get_observability_status() -> Dict[str, Any]:
    """获取可观测性状态"""
    return {
        "langsmith_enabled": is_langsmith_enabled(),
        "langfuse_enabled": is_langfuse_enabled(),
        "langsmith_project": os.getenv("LANGCHAIN_PROJECT", "default"),
        "langfuse_host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    }
