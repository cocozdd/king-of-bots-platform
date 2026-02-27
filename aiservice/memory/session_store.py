"""
会话存储抽象层 - 2026 最佳实践

支持多种后端：
- 内存存储（开发环境）
- Redis 存储（生产环境推荐）
- PostgreSQL 存储（需要持久化时）

面试要点：
- 抽象存储层便于切换后端
- Redis 适合高并发场景
- PostgreSQL 适合需要事务和持久化的场景
"""
import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class SessionStore(ABC):
    """会话存储抽象基类"""
    
    @abstractmethod
    def create_session(self, session_id: str, user_id: int, title: str = "新对话") -> Dict:
        """创建会话"""
        pass
    
    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Dict]:
        """获取会话"""
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        pass
    
    @abstractmethod
    def list_sessions(self, user_id: int) -> List[Dict]:
        """列出用户会话"""
        pass
    
    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """添加消息"""
        pass
    
    @abstractmethod
    def get_messages(self, session_id: str, limit: int = 50) -> List[Dict]:
        """获取消息"""
        pass
    
    @abstractmethod
    def update_session_title(self, session_id: str, title: str) -> bool:
        """更新会话标题"""
        pass


class InMemorySessionStore(SessionStore):
    """内存存储实现 - 开发环境使用"""
    
    def __init__(self):
        self._sessions: Dict[str, Dict] = {}
        self._messages: Dict[str, List[Dict]] = {}
        logger.info("使用内存会话存储（仅限开发环境）")
    
    def create_session(self, session_id: str, user_id: int, title: str = "新对话") -> Dict:
        now = datetime.now().isoformat()
        session = {
            "id": session_id,
            "user_id": user_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }
        self._sessions[session_id] = session
        self._messages[session_id] = []
        return session
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        return self._sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        self._sessions.pop(session_id, None)
        self._messages.pop(session_id, None)
        return True
    
    def list_sessions(self, user_id: int) -> List[Dict]:
        sessions = [
            {**s, "message_count": len(self._messages.get(s["id"], []))}
            for s in self._sessions.values()
            if s["user_id"] == user_id
        ]
        sessions.sort(key=lambda x: x["updated_at"], reverse=True)
        return sessions
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._messages:
            self._messages[session_id] = []
        
        self._messages[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        
        if session_id in self._sessions:
            self._sessions[session_id]["updated_at"] = datetime.now().isoformat()
    
    def get_messages(self, session_id: str, limit: int = 50) -> List[Dict]:
        messages = self._messages.get(session_id, [])
        return messages[-limit:] if len(messages) > limit else messages
    
    def update_session_title(self, session_id: str, title: str) -> bool:
        if session_id in self._sessions:
            self._sessions[session_id]["title"] = title
            self._sessions[session_id]["updated_at"] = datetime.now().isoformat()
            return True
        return False
    
    def clear_all(self) -> None:
        """清除所有数据"""
        self._sessions.clear()
        self._messages.clear()


class RedisSessionStore(SessionStore):
    """Redis 存储实现 - 生产环境推荐"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client = None
        self._prefix = "kob:session:"
        logger.info("使用 Redis 会话存储: %s", self.redis_url.split("@")[-1])  # 隐藏密码
    
    def _get_client(self):
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(self.redis_url, decode_responses=True)
            except ImportError:
                logger.error("redis 包未安装，请运行: pip install redis")
                raise
        return self._client
    
    def create_session(self, session_id: str, user_id: int, title: str = "新对话") -> Dict:
        client = self._get_client()
        now = datetime.now().isoformat()
        session = {
            "id": session_id,
            "user_id": user_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }
        
        # 存储会话
        client.hset(f"{self._prefix}{session_id}", mapping={
            "data": json.dumps(session),
            "user_id": str(user_id),
        })
        
        # 添加到用户会话列表
        client.sadd(f"{self._prefix}user:{user_id}", session_id)
        
        # 设置过期时间（7天）
        client.expire(f"{self._prefix}{session_id}", 7 * 24 * 3600)
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        client = self._get_client()
        data = client.hget(f"{self._prefix}{session_id}", "data")
        return json.loads(data) if data else None
    
    def delete_session(self, session_id: str) -> bool:
        client = self._get_client()
        session = self.get_session(session_id)
        if session:
            user_id = session.get("user_id")
            client.srem(f"{self._prefix}user:{user_id}", session_id)
        
        client.delete(f"{self._prefix}{session_id}")
        client.delete(f"{self._prefix}{session_id}:messages")
        return True
    
    def list_sessions(self, user_id: int) -> List[Dict]:
        client = self._get_client()
        session_ids = client.smembers(f"{self._prefix}user:{user_id}")
        
        sessions = []
        for sid in session_ids:
            session = self.get_session(sid)
            if session:
                session["message_count"] = client.llen(f"{self._prefix}{sid}:messages")
                sessions.append(session)
        
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        client = self._get_client()
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        client.rpush(f"{self._prefix}{session_id}:messages", json.dumps(message))
        
        # 更新会话时间
        session = self.get_session(session_id)
        if session:
            session["updated_at"] = datetime.now().isoformat()
            client.hset(f"{self._prefix}{session_id}", "data", json.dumps(session))
    
    def get_messages(self, session_id: str, limit: int = 50) -> List[Dict]:
        client = self._get_client()
        messages_raw = client.lrange(f"{self._prefix}{session_id}:messages", -limit, -1)
        return [json.loads(m) for m in messages_raw]
    
    def update_session_title(self, session_id: str, title: str) -> bool:
        session = self.get_session(session_id)
        if session:
            session["title"] = title
            session["updated_at"] = datetime.now().isoformat()
            client = self._get_client()
            client.hset(f"{self._prefix}{session_id}", "data", json.dumps(session))
            return True
        return False


# 全局存储实例
_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """获取会话存储实例"""
    global _store
    if _store is None:
        store_type = os.getenv("SESSION_STORE", "memory").lower()
        
        if store_type == "redis":
            _store = RedisSessionStore()
        else:
            _store = InMemorySessionStore()
    
    return _store


def set_session_store(store: SessionStore) -> None:
    """设置会话存储实例（用于测试）"""
    global _store
    _store = store
