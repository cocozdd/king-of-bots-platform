"""
会话管理器 - 对齐 Java SessionController

功能：
- 会话创建/获取/删除
- 会话元数据管理（标题、创建时间）
- 支持内存存储（可扩展为 Redis）

2026 年标准：
- 使用 LangGraph checkpointer 概念
- 支持 thread_id 作为会话标识
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """会话数据结构"""
    id: str
    user_id: int
    title: str
    created_at: str
    updated_at: str
    metadata: Dict = field(default_factory=dict)


class SessionManager:
    """
    会话管理器 - 对齐 Java SessionController
    
    特性：
    - 内存存储（生产环境可替换为 Redis）
    - 支持会话 CRUD
    - 支持元数据管理
    """
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        logger.info("SessionManager 初始化完成")
    
    def create_session(
        self,
        user_id: int,
        title: str = "新对话",
        metadata: Dict = None,
    ) -> Session:
        """
        创建新会话
        
        Args:
            user_id: 用户 ID
            title: 会话标题
            metadata: 额外元数据
            
        Returns:
            新创建的会话
        """
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        session = Session(
            id=session_id,
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        
        self._sessions[session_id] = session
        logger.info(f"创建会话: {session_id} for user {user_id}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        return self._sessions.get(session_id)
    
    def list_sessions(
        self,
        user_id: int,
        limit: int = 50,
    ) -> List[Session]:
        """
        列出用户的所有会话
        
        Args:
            user_id: 用户 ID
            limit: 最大返回数量
            
        Returns:
            会话列表（按更新时间倒序）
        """
        user_sessions = [
            s for s in self._sessions.values()
            if s.user_id == user_id
        ]
        
        # 按更新时间倒序
        user_sessions.sort(key=lambda x: x.updated_at, reverse=True)
        
        return user_sessions[:limit]
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除会话
        
        Returns:
            是否删除成功
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"删除会话: {session_id}")
            return True
        return False
    
    def update_title(self, session_id: str, title: str) -> bool:
        """更新会话标题"""
        session = self._sessions.get(session_id)
        if session:
            session.title = title
            session.updated_at = datetime.now().isoformat()
            return True
        return False
    
    def update_metadata(self, session_id: str, metadata: Dict) -> bool:
        """更新会话元数据"""
        session = self._sessions.get(session_id)
        if session:
            session.metadata.update(metadata)
            session.updated_at = datetime.now().isoformat()
            return True
        return False
    
    def touch(self, session_id: str):
        """更新会话的 updated_at 时间"""
        session = self._sessions.get(session_id)
        if session:
            session.updated_at = datetime.now().isoformat()
    
    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        return session_id in self._sessions
    
    def get_session_count(self, user_id: int) -> int:
        """获取用户的会话数量"""
        return sum(1 for s in self._sessions.values() if s.user_id == user_id)


# 全局实例
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """获取全局会话管理器实例"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
