"""
Embedding 缓存 - P0 改进

功能：
- L1 缓存：LRU 内存缓存（快速访问）
- L2 缓存：Redis 缓存（持久化）
- 缓存命中率监控

面试要点：
- 为什么需要缓存：减少 API 调用，降低成本，提升响应速度
- 两级缓存策略：内存热数据 + Redis 持久化
- 缓存 key 设计：MD5(query) 保证唯一性
- TTL 策略：Embedding 相对稳定，可设置较长 TTL（7天）
"""
import hashlib
import json
import logging
import os
from functools import lru_cache
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_L1_CACHE_SIZE = 1000  # 内存缓存条目数
DEFAULT_L2_TTL = 7 * 24 * 60 * 60  # Redis 缓存 TTL（7天）


@dataclass
class CacheStats:
    """缓存统计"""
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    api_calls: int = 0
    
    @property
    def total_requests(self) -> int:
        return self.l1_hits + self.l1_misses
    
    @property
    def l1_hit_rate(self) -> float:
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0
    
    @property
    def overall_hit_rate(self) -> float:
        total = self.total_requests
        hits = self.l1_hits + self.l2_hits
        return hits / total if total > 0 else 0
    
    @property
    def cost_savings_percent(self) -> float:
        """计算成本节省百分比"""
        total = self.total_requests
        if total == 0:
            return 0
        return (1 - self.api_calls / total) * 100
    
    def to_dict(self) -> dict:
        return {
            "l1_hits": self.l1_hits,
            "l1_misses": self.l1_misses,
            "l2_hits": self.l2_hits,
            "l2_misses": self.l2_misses,
            "api_calls": self.api_calls,
            "total_requests": self.total_requests,
            "l1_hit_rate": f"{self.l1_hit_rate:.2%}",
            "overall_hit_rate": f"{self.overall_hit_rate:.2%}",
            "cost_savings": f"{self.cost_savings_percent:.1f}%",
        }


class CachedEmbedder:
    """
    带缓存的 Embedding 生成器
    
    两级缓存架构：
    - L1：LRU 内存缓存，容量有限但极快
    - L2：Redis 缓存，容量大且持久化
    
    查询流程：L1 -> L2 -> API
    """
    
    def __init__(
        self,
        embedder,
        redis_client=None,
        l1_size: int = DEFAULT_L1_CACHE_SIZE,
        l2_ttl: int = DEFAULT_L2_TTL,
        key_prefix: str = "kob:embedding:",
    ):
        self.embedder = embedder
        self.redis = redis_client
        self.l1_size = l1_size
        self.l2_ttl = l2_ttl
        self.key_prefix = key_prefix
        self.stats = CacheStats()
        
        # L1 缓存（使用 dict 手动实现 LRU）
        self._l1_cache: Dict[str, List[float]] = {}
        self._l1_order: List[str] = []  # 访问顺序，最新在后
        
        logger.info(
            "CachedEmbedder 初始化: L1=%d条, L2=%s, TTL=%d秒",
            l1_size,
            "Redis" if redis_client else "disabled",
            l2_ttl,
        )
    
    def _cache_key(self, query: str) -> str:
        """生成缓存 key"""
        # 使用 MD5 保证 key 长度固定且唯一
        hash_val = hashlib.md5(query.encode("utf-8")).hexdigest()
        return f"{self.key_prefix}{hash_val}"
    
    def _l1_get(self, key: str) -> Optional[List[float]]:
        """L1 缓存读取"""
        if key in self._l1_cache:
            # 更新访问顺序（LRU）
            self._l1_order.remove(key)
            self._l1_order.append(key)
            self.stats.l1_hits += 1
            return self._l1_cache[key]
        
        self.stats.l1_misses += 1
        return None
    
    def _l1_set(self, key: str, value: List[float]):
        """L1 缓存写入"""
        if key in self._l1_cache:
            # 已存在，更新顺序
            self._l1_order.remove(key)
        elif len(self._l1_cache) >= self.l1_size:
            # 缓存满，淘汰最旧的
            oldest_key = self._l1_order.pop(0)
            del self._l1_cache[oldest_key]
        
        self._l1_cache[key] = value
        self._l1_order.append(key)
    
    def _l2_get(self, key: str) -> Optional[List[float]]:
        """L2 缓存（Redis）读取"""
        if self.redis is None:
            return None
        
        try:
            data = self.redis.get(key)
            if data:
                self.stats.l2_hits += 1
                return json.loads(data)
        except Exception as e:
            logger.warning("L2 缓存读取失败: %s", e)
        
        self.stats.l2_misses += 1
        return None
    
    def _l2_set(self, key: str, value: List[float]):
        """L2 缓存（Redis）写入"""
        if self.redis is None:
            return
        
        try:
            self.redis.setex(key, self.l2_ttl, json.dumps(value))
        except Exception as e:
            logger.warning("L2 缓存写入失败: %s", e)
    
    def embed_query(self, query: str) -> Optional[List[float]]:
        """
        生成查询向量（带缓存）
        
        查询流程：L1 -> L2 -> API
        """
        key = self._cache_key(query)
        
        # 1. L1 缓存查询
        embedding = self._l1_get(key)
        if embedding is not None:
            logger.debug("Embedding L1 命中: %s", query[:30])
            return embedding
        
        # 2. L2 缓存查询
        embedding = self._l2_get(key)
        if embedding is not None:
            logger.debug("Embedding L2 命中: %s", query[:30])
            # 回填 L1
            self._l1_set(key, embedding)
            return embedding
        
        # 3. 调用 API
        try:
            self.stats.api_calls += 1
            embedding = self.embedder.embed_query(query)
            
            if embedding:
                # 写入缓存
                self._l1_set(key, embedding)
                self._l2_set(key, embedding)
                logger.debug("Embedding API 调用: %s", query[:30])
            
            return embedding
        except Exception as e:
            logger.error("Embedding API 调用失败: %s", e)
            return None
    
    async def aembed_query(self, query: str) -> Optional[List[float]]:
        """
        异步生成查询向量（带缓存）
        """
        key = self._cache_key(query)
        
        # 1. L1 缓存查询（同步，因为是内存操作）
        embedding = self._l1_get(key)
        if embedding is not None:
            return embedding
        
        # 2. L2 缓存查询
        embedding = self._l2_get(key)
        if embedding is not None:
            self._l1_set(key, embedding)
            return embedding
        
        # 3. 调用 API
        try:
            self.stats.api_calls += 1
            
            # 尝试异步调用
            if hasattr(self.embedder, "aembed_query"):
                embedding = await self.embedder.aembed_query(query)
            else:
                import asyncio
                embedding = await asyncio.to_thread(
                    self.embedder.embed_query, query
                )
            
            if embedding:
                self._l1_set(key, embedding)
                self._l2_set(key, embedding)
            
            return embedding
        except Exception as e:
            logger.error("异步 Embedding 调用失败: %s", e)
            return None
    
    def get_stats(self) -> dict:
        """获取缓存统计"""
        return self.stats.to_dict()
    
    def clear_cache(self):
        """清空缓存"""
        self._l1_cache.clear()
        self._l1_order.clear()
        self.stats = CacheStats()
        logger.info("Embedding 缓存已清空")
    
    def warmup(self, queries: List[str]):
        """
        缓存预热
        
        批量加载常用查询到缓存
        """
        logger.info("开始缓存预热: %d 个查询", len(queries))
        
        for query in queries:
            self.embed_query(query)
        
        logger.info(
            "缓存预热完成: API调用=%d, 缓存命中=%d",
            self.stats.api_calls,
            self.stats.l2_hits,
        )


# ============ 全局实例管理 ============

_cached_embedder: Optional[CachedEmbedder] = None


def get_cached_embedder(embedder=None) -> Optional[CachedEmbedder]:
    """
    获取带缓存的 Embedder
    
    如果 embedder 为 None，返回已存在的实例或 None
    """
    global _cached_embedder
    
    if _cached_embedder is not None:
        return _cached_embedder
    
    if embedder is None:
        return None
    
    # 尝试初始化 Redis
    redis_client = None
    redis_url = os.getenv("REDIS_URL")
    
    if redis_url:
        try:
            import redis
            redis_client = redis.from_url(redis_url, decode_responses=True)
            redis_client.ping()
            logger.info("✓ Embedding 缓存 Redis 连接成功")
        except Exception as e:
            logger.warning("Embedding 缓存 Redis 连接失败，仅使用 L1 缓存: %s", e)
            redis_client = None
    
    _cached_embedder = CachedEmbedder(
        embedder=embedder,
        redis_client=redis_client,
    )
    
    return _cached_embedder


def get_embedding_cache_stats() -> dict:
    """获取缓存统计（便捷函数）"""
    if _cached_embedder:
        return _cached_embedder.get_stats()
    return {"status": "not_initialized"}
