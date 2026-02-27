"""
会话持久化存储 - P0 改进

功能：
- Redis 持久化会话数据
- 支持多实例部署
- 自动过期清理（TTL）
- 降级为内存存储

面试要点：
- 为什么需要 Redis：服务重启不丢失、多实例共享、可水平扩展
- TTL 过期策略：自动清理过期会话，节省内存
- 降级策略：Redis 不可用时降级为内存，保证服务可用性
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# 默认 TTL（24小时）
DEFAULT_SESSION_TTL = 24 * 60 * 60


class SessionStoreBase(ABC):
    """会话存储基类"""
    
    @abstractmethod
    def save_session(self, session_id: str, data: dict) -> bool:
        """保存会话元数据"""
        pass
    
    @abstractmethod
    def get_session(self, session_id: str) -> Optional[dict]:
        """获取会话元数据"""
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        pass
    
    @abstractmethod
    def list_sessions(self, user_id: int) -> List[dict]:
        """列出用户的所有会话"""
        pass
    
    @abstractmethod
    def save_messages(self, session_id: str, messages: List[dict]) -> bool:
        """保存会话消息历史"""
        pass
    
    @abstractmethod
    def get_messages(self, session_id: str) -> List[dict]:
        """获取会话消息历史"""
        pass
    
    @abstractmethod
    def add_message(self, session_id: str, message: dict) -> bool:
        """添加单条消息"""
        pass


class MemorySessionStore(SessionStoreBase):
    """
    内存会话存储（降级方案）
    
    用于 Redis 不可用时的降级
    注意：服务重启会丢失数据
    """
    
    def __init__(self):
        self._sessions: Dict[str, dict] = {}
        self._messages: Dict[str, List[dict]] = {}
        logger.warning("使用内存会话存储（数据不持久化）")
    
    def save_session(self, session_id: str, data: dict) -> bool:
        self._sessions[session_id] = data
        return True
    
    def get_session(self, session_id: str) -> Optional[dict]:
        return self._sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        self._sessions.pop(session_id, None)
        self._messages.pop(session_id, None)
        return True
    
    def list_sessions(self, user_id: int) -> List[dict]:
        return [
            {**s, "message_count": len(self._messages.get(s["id"], []))}
            for s in self._sessions.values()
            if s.get("user_id") == user_id
        ]
    
    def save_messages(self, session_id: str, messages: List[dict]) -> bool:
        self._messages[session_id] = messages
        return True
    
    def get_messages(self, session_id: str) -> List[dict]:
        return self._messages.get(session_id, [])
    
    def add_message(self, session_id: str, message: dict) -> bool:
        if session_id not in self._messages:
            self._messages[session_id] = []
        self._messages[session_id].append(message)
        return True
    
    def clear_all(self):
        """清空所有数据"""
        self._sessions.clear()
        self._messages.clear()


class RedisSessionStore(SessionStoreBase):
    """
    Redis 会话存储
    
    特性：
    - 持久化存储，服务重启不丢失
    - 支持多实例部署（会话共享）
    - TTL 自动过期
    - 原子操作保证数据一致性
    """
    
    def __init__(
        self,
        redis_url: str = None,
        ttl: int = DEFAULT_SESSION_TTL,
        key_prefix: str = "kob:session:",
    ):
        self.ttl = ttl
        self.key_prefix = key_prefix
        self._redis = None
        
        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        try:
            import redis
            self._redis = redis.from_url(redis_url, decode_responses=True)
            # 测试连接
            self._redis.ping()
            logger.info("✓ Redis 会话存储初始化成功: %s", redis_url.split("@")[-1])
        except ImportError:
            logger.error("redis 库未安装，请运行: pip install redis")
            raise
        except Exception as e:
            logger.error("Redis 连接失败: %s", e)
            raise
    
    def _session_key(self, session_id: str) -> str:
        """生成会话元数据 key"""
        return f"{self.key_prefix}{session_id}"
    
    def _messages_key(self, session_id: str) -> str:
        """生成消息历史 key"""
        return f"{self.key_prefix}{session_id}:messages"
    
    def _user_sessions_key(self, user_id: int) -> str:
        """生成用户会话索引 key"""
        return f"{self.key_prefix}user:{user_id}:sessions"
    
    def save_session(self, session_id: str, data: dict) -> bool:
        """保存会话元数据"""
        try:
            key = self._session_key(session_id)
            self._redis.setex(key, self.ttl, json.dumps(data, ensure_ascii=False))
            
            # 添加到用户会话索引
            user_id = data.get("user_id")
            if user_id:
                user_key = self._user_sessions_key(user_id)
                self._redis.sadd(user_key, session_id)
                self._redis.expire(user_key, self.ttl * 7)  # 索引保留更久
            
            return True
        except Exception as e:
            logger.error("保存会话失败: %s", e)
            return False
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """获取会话元数据"""
        try:
            key = self._session_key(session_id)
            data = self._redis.get(key)
            if data:
                # 刷新 TTL
                self._redis.expire(key, self.ttl)
                return json.loads(data)
            return None
        except Exception as e:
            logger.error("获取会话失败: %s", e)
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        try:
            # 获取会话以找到 user_id
            session = self.get_session(session_id)
            
            # 删除会话数据
            self._redis.delete(
                self._session_key(session_id),
                self._messages_key(session_id),
            )
            
            # 从用户索引中移除
            if session and session.get("user_id"):
                user_key = self._user_sessions_key(session["user_id"])
                self._redis.srem(user_key, session_id)
            
            return True
        except Exception as e:
            logger.error("删除会话失败: %s", e)
            return False
    
    def list_sessions(self, user_id: int) -> List[dict]:
        """列出用户的所有会话"""
        try:
            user_key = self._user_sessions_key(user_id)
            session_ids = self._redis.smembers(user_key)
            
            sessions = []
            for session_id in session_ids:
                session = self.get_session(session_id)
                if session:
                    # 获取消息数量
                    msg_key = self._messages_key(session_id)
                    msg_count = self._redis.llen(msg_key)
                    session["message_count"] = msg_count
                    sessions.append(session)
                else:
                    # 清理无效索引
                    self._redis.srem(user_key, session_id)
            
            # 按更新时间排序
            sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            return sessions
        except Exception as e:
            logger.error("列出会话失败: %s", e)
            return []
    
    def save_messages(self, session_id: str, messages: List[dict]) -> bool:
        """保存会话消息历史（覆盖）"""
        try:
            key = self._messages_key(session_id)
            
            # 使用 pipeline 批量操作
            pipe = self._redis.pipeline()
            pipe.delete(key)
            
            for msg in messages:
                pipe.rpush(key, json.dumps(msg, ensure_ascii=False))
            
            pipe.expire(key, self.ttl)
            pipe.execute()
            
            return True
        except Exception as e:
            logger.error("保存消息历史失败: %s", e)
            return False
    
    def get_messages(self, session_id: str) -> List[dict]:
        """获取会话消息历史"""
        try:
            key = self._messages_key(session_id)
            messages_raw = self._redis.lrange(key, 0, -1)
            
            # 刷新 TTL
            if messages_raw:
                self._redis.expire(key, self.ttl)
            
            return [json.loads(m) for m in messages_raw]
        except Exception as e:
            logger.error("获取消息历史失败: %s", e)
            return []
    
    def add_message(self, session_id: str, message: dict) -> bool:
        """添加单条消息"""
        try:
            key = self._messages_key(session_id)
            self._redis.rpush(key, json.dumps(message, ensure_ascii=False))
            self._redis.expire(key, self.ttl)
            return True
        except Exception as e:
            logger.error("添加消息失败: %s", e)
            return False
    
    def get_stats(self) -> dict:
        """获取存储统计信息"""
        try:
            info = self._redis.info("memory")
            keys = self._redis.keys(f"{self.key_prefix}*")
            
            return {
                "type": "redis",
                "connected": True,
                "memory_used": info.get("used_memory_human", "unknown"),
                "total_keys": len(keys),
            }
        except Exception as e:
            return {"type": "redis", "connected": False, "error": str(e)}


# ============ 全局实例管理 ============

_session_store: Optional[SessionStoreBase] = None


def get_session_store() -> SessionStoreBase:
    """
    获取会话存储实例
    
    优先使用 Redis，Redis 不可用时降级为内存存储
    """
    global _session_store
    
    if _session_store is not None:
        return _session_store
    
    # 检查是否启用 Redis
    use_redis = os.getenv("SESSION_STORE", "redis").lower() == "redis"
    redis_url = os.getenv("REDIS_URL")
    
    if use_redis and redis_url:
        try:
            _session_store = RedisSessionStore(redis_url=redis_url)
            return _session_store
        except Exception as e:
            logger.warning("Redis 初始化失败，降级为内存存储: %s", e)
    
    # 降级为内存存储
    _session_store = MemorySessionStore()
    return _session_store


def reset_session_store():
    """重置会话存储（用于测试）"""
    global _session_store
    _session_store = None
