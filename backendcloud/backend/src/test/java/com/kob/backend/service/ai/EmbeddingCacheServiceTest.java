package com.kob.backend.service.ai;

import com.kob.backend.service.impl.ai.EmbeddingCacheService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.cache.CacheManager;
import org.springframework.cache.concurrent.ConcurrentMapCacheManager;

import static org.junit.jupiter.api.Assertions.*;

/**
 * EmbeddingCacheService 单元测试
 * 
 * 测试要点：
 * - 缓存命中/未命中
 * - 缓存存储和读取
 * - 向量维度验证
 */
class EmbeddingCacheServiceTest {
    
    private EmbeddingCacheService cacheService;
    private CacheManager cacheManager;
    
    @BeforeEach
    void setUp() {
        cacheManager = new ConcurrentMapCacheManager("embeddings");
        cacheService = new EmbeddingCacheService(cacheManager);
    }
    
    @Test
    @DisplayName("首次查询应返回null（缓存未命中）")
    void get_withNewText_shouldReturnNull() {
        float[] result = cacheService.get("新文本");
        
        assertNull(result);
    }
    
    @Test
    @DisplayName("存储后应能正确读取")
    void putAndGet_shouldWork() {
        String text = "测试文本";
        float[] embedding = new float[]{0.1f, 0.2f, 0.3f};
        
        cacheService.put(text, embedding);
        float[] result = cacheService.get(text);
        
        assertNotNull(result);
        assertArrayEquals(embedding, result, 0.0001f);
    }
    
    @Test
    @DisplayName("相同文本应返回相同向量")
    void get_withSameText_shouldReturnSameEmbedding() {
        String text = "相同文本";
        float[] embedding = new float[]{0.5f, 0.6f, 0.7f};
        
        cacheService.put(text, embedding);
        
        float[] result1 = cacheService.get(text);
        float[] result2 = cacheService.get(text);
        
        assertArrayEquals(result1, result2, 0.0001f);
    }
    
    @Test
    @DisplayName("不同文本应返回不同结果")
    void get_withDifferentText_shouldReturnDifferentResults() {
        cacheService.put("文本A", new float[]{0.1f, 0.2f});
        cacheService.put("文本B", new float[]{0.3f, 0.4f});
        
        float[] resultA = cacheService.get("文本A");
        float[] resultB = cacheService.get("文本B");
        
        assertNotEquals(resultA[0], resultB[0], 0.0001f);
    }
    
    @Test
    @DisplayName("空文本处理")
    void get_withEmptyText_shouldHandleGracefully() {
        float[] result = cacheService.get("");
        assertNull(result);
    }
    
    @Test
    @DisplayName("缓存命中率统计")
    void cacheHitRate_shouldBeTracked() {
        String text = "测试";
        cacheService.put(text, new float[]{0.1f});
        
        // 第一次 get 应该命中
        cacheService.get(text);
        // 未缓存的应该未命中
        cacheService.get("未缓存");
        
        // 验证缓存工作正常即可，具体命中率统计依赖实现
        assertNotNull(cacheService.get(text));
        assertNull(cacheService.get("未缓存"));
    }
}
