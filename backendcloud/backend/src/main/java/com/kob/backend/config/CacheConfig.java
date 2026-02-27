package com.kob.backend.config;

import com.github.benmanes.caffeine.cache.Caffeine;
import org.springframework.cache.CacheManager;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.cache.caffeine.CaffeineCacheManager;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.concurrent.TimeUnit;

/**
 * Caffeine 缓存配置
 * 
 * 用于缓存 Embedding 向量，避免重复调用 API
 * 
 * 配置策略:
 * - 最大缓存 1000 条
 * - 写入后 1 小时过期
 * - 基于大小的驱逐策略
 * 
 * @author KOB Team
 */
@Configuration
@EnableCaching
public class CacheConfig {
    
    public static final String EMBEDDING_CACHE = "embeddingCache";
    
    @Bean
    public CacheManager cacheManager() {
        CaffeineCacheManager cacheManager = new CaffeineCacheManager(EMBEDDING_CACHE);
        cacheManager.setCaffeine(Caffeine.newBuilder()
                // 最大缓存 1000 条
                .maximumSize(1000)
                // 写入后 1 小时过期
                .expireAfterWrite(1, TimeUnit.HOURS)
                // 开启统计
                .recordStats()
        );
        return cacheManager;
    }
}
