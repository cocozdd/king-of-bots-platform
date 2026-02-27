"""
Memory 模块 - 会话管理和对话记忆 (2026 升级版)

包含：
- SessionManager: 会话生命周期管理
- ConversationMemoryService: 对话记忆（短期+摘要压缩）
- SessionStore: 会话持久化存储（支持内存/Redis）
"""
from .session_manager import SessionManager, get_session_manager
from .conversation_memory import (
    ConversationMemoryService,
    get_memory_service,
    MemoryEntry,
)
from .session_store import (
    SessionStore,
    InMemorySessionStore,
    RedisSessionStore,
    get_session_store,
    set_session_store,
)

__all__ = [
    "SessionManager",
    "get_session_manager",
    "ConversationMemoryService", 
    "get_memory_service",
    "MemoryEntry",
    "SessionStore",
    "InMemorySessionStore",
    "RedisSessionStore",
    "get_session_store",
    "set_session_store",
]
