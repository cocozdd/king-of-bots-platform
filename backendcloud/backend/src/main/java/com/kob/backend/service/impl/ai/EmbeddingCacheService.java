package com.kob.backend.service.impl.ai;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.stats.CacheStats;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Embedding 向量缓存服务
 * 
 * 功能:
 * 1. 缓存 Embedding 向量，避免重复调用 API
 * 2. 统计缓存命中率
 * 3. 估算节省的成本
 * 
 * 配置:
 * - 最大缓存 1000 条
 * - 写入后 1 小时过期
 * - 基于大小的驱逐策略
 * 
 * @author KOB Team
 */
@Service
public class EmbeddingCacheService {
    
    private static final Logger log = LoggerFactory.getLogger(EmbeddingCacheService.class);
    
    // 缓存配置
    private static final int MAX_CACHE_SIZE = 1000;
    private static final int EXPIRE_HOURS = 1;
    
    // Caffeine 缓存实例
    private final Cache<String, double[]> embeddingCache;
    
    // 统计数据
    private final AtomicLong totalRequests = new AtomicLong(0);
    private final AtomicLong cacheHits = new AtomicLong(0);
    private final AtomicLong cacheMisses = new AtomicLong(0);
    private final AtomicLong savedTokens = new AtomicLong(0);
    private final Instant startTime = Instant.now();
    
    // Token 估算：平均每个问题约 50 tokens
    private static final int AVG_TOKENS_PER_QUERY = 50;
    // text-embedding-v2 价格: ¥0.0007/1k tokens
    private static final double COST_PER_1K_TOKENS = 0.0007;
    
    public EmbeddingCacheService() {
        this.embeddingCache = Caffeine.newBuilder()
                .maximumSize(MAX_CACHE_SIZE)
                .expireAfterWrite(EXPIRE_HOURS, TimeUnit.HOURS)
                .recordStats()
                .build();
        
        log.info("EmbeddingCacheService 初始化完成 - maxSize={}, expireHours={}", 
                MAX_CACHE_SIZE, EXPIRE_HOURS);
    }
    
    /**
     * 从缓存获取 Embedding
     * 
     * @param text 输入文本
     * @return 缓存的向量，如果不存在则返回 null
     */
    public double[] get(String text) {
        totalRequests.incrementAndGet();
        String key = generateKey(text);
        double[] cached = embeddingCache.getIfPresent(key);
        
        if (cached != null) {
            cacheHits.incrementAndGet();
            savedTokens.addAndGet(AVG_TOKENS_PER_QUERY);
            log.debug("缓存命中: key={}", key.substring(0, Math.min(20, key.length())));
            return cached;
        }
        
        cacheMisses.incrementAndGet();
        log.debug("缓存未命中: key={}", key.substring(0, Math.min(20, key.length())));
        return null;
    }
    
    /**
     * 将 Embedding 存入缓存
     * 
     * @param text 输入文本
     * @param embedding Embedding 向量
     */
    public void put(String text, double[] embedding) {
        String key = generateKey(text);
        embeddingCache.put(key, embedding);
        log.debug("缓存写入: key={}, dim={}", 
                key.substring(0, Math.min(20, key.length())), embedding.length);
    }
    
    /**
     * 生成缓存 Key
     * 使用文本内容的规范化版本作为 key
     */
    private String generateKey(String text) {
        // 规范化：去除多余空白，转小写
        return text.trim().toLowerCase().replaceAll("\\s+", " ");
    }
    
    /**
     * 获取缓存统计信息
     */
    public CacheSummary getSummary() {
        CacheStats stats = embeddingCache.stats();
        long total = totalRequests.get();
        long hits = cacheHits.get();
        long misses = cacheMisses.get();
        long saved = savedTokens.get();
        
        double hitRate = total > 0 ? (double) hits / total * 100 : 0;
        double savedCost = saved * COST_PER_1K_TOKENS / 1000;
        
        return new CacheSummary(
                total,
                hits,
                misses,
                hitRate,
                embeddingCache.estimatedSize(),
                MAX_CACHE_SIZE,
                saved,
                savedCost,
                Instant.now().toString()
        );
    }
    
    /**
     * 清空缓存
     */
    public void clear() {
        embeddingCache.invalidateAll();
        totalRequests.set(0);
        cacheHits.set(0);
        cacheMisses.set(0);
        savedTokens.set(0);
        log.info("缓存已清空");
    }
    
    /**
     * 缓存统计摘要
     */
    public static class CacheSummary {
        private final long totalRequests;
        private final long cacheHits;
        private final long cacheMisses;
        private final double hitRatePercent;
        private final long currentSize;
        private final long maxSize;
        private final long savedTokens;
        private final double savedCostYuan;
        private final String timestamp;
        
        public CacheSummary(long totalRequests, long cacheHits, long cacheMisses,
                          double hitRatePercent, long currentSize, long maxSize,
                          long savedTokens, double savedCostYuan, String timestamp) {
            this.totalRequests = totalRequests;
            this.cacheHits = cacheHits;
            this.cacheMisses = cacheMisses;
            this.hitRatePercent = hitRatePercent;
            this.currentSize = currentSize;
            this.maxSize = maxSize;
            this.savedTokens = savedTokens;
            this.savedCostYuan = savedCostYuan;
            this.timestamp = timestamp;
        }
        
        // Getters
        public long getTotalRequests() { return totalRequests; }
        public long getCacheHits() { return cacheHits; }
        public long getCacheMisses() { return cacheMisses; }
        public double getHitRatePercent() { return hitRatePercent; }
        public long getCurrentSize() { return currentSize; }
        public long getMaxSize() { return maxSize; }
        public long getSavedTokens() { return savedTokens; }
        public double getSavedCostYuan() { return savedCostYuan; }
        public String getTimestamp() { return timestamp; }
        
        @Override
        public String toString() {
            return String.format(
                "CacheSummary{total=%d, hits=%d, misses=%d, hitRate=%.1f%%, " +
                "size=%d/%d, savedTokens=%d, savedCost=¥%.6f}",
                totalRequests, cacheHits, cacheMisses, hitRatePercent,
                currentSize, maxSize, savedTokens, savedCostYuan
            );
        }
    }
}
