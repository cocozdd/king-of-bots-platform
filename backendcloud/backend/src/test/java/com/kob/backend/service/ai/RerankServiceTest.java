package com.kob.backend.service.ai;

import com.kob.backend.controller.ai.dto.AiDoc;
import com.kob.backend.service.impl.ai.AiMetricsService;
import com.kob.backend.service.impl.ai.RerankService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.verify;

/**
 * RerankService 单元测试
 * 
 * 测试要点：
 * - 空输入处理
 * - 相关性排序准确性
 * - TopK 截断
 * - 边界条件
 */
@ExtendWith(MockitoExtension.class)
class RerankServiceTest {
    
    @Mock
    private AiMetricsService metricsService;
    
    @InjectMocks
    private RerankService rerankService;
    
    private List<AiDoc> testDocs;
    
    @BeforeEach
    void setUp() {
        testDocs = Arrays.asList(
            createDoc("1", "Bot移动策略", "如何让Bot更智能地移动，避开障碍物", "strategy"),
            createDoc("2", "算法优化", "使用BFS算法寻找最短路径", "algorithm"),
            createDoc("3", "代码示例", "Java代码实现贪吃蛇Bot", "code"),
            createDoc("4", "游戏规则", "贪吃蛇对战平台的基本规则说明", "rule"),
            createDoc("5", "高级技巧", "Bot策略进阶，如何预判对手移动", "strategy")
        );
    }
    
    @Test
    @DisplayName("空列表输入应返回空列表")
    void rerank_withEmptyList_shouldReturnEmptyList() {
        List<AiDoc> result = rerankService.rerank("测试查询", new ArrayList<>(), 5);
        
        assertTrue(result.isEmpty());
    }
    
    @Test
    @DisplayName("null输入应返回空列表")
    void rerank_withNull_shouldReturnEmptyList() {
        List<AiDoc> result = rerankService.rerank("测试查询", null, 5);
        
        assertTrue(result.isEmpty());
    }
    
    @Test
    @DisplayName("查询'Bot策略'应将策略相关文档排在前面")
    void rerank_withStrategyQuery_shouldPrioritizeStrategyDocs() {
        List<AiDoc> result = rerankService.rerank("Bot策略", testDocs, 3);
        
        assertEquals(3, result.size());
        // 策略相关的文档应该排在前面
        assertTrue(result.stream()
                .anyMatch(doc -> doc.getTitle().contains("策略") || doc.getTitle().contains("技巧")));
    }
    
    @Test
    @DisplayName("查询'算法'应将算法相关文档排在前面")
    void rerank_withAlgorithmQuery_shouldPrioritizeAlgorithmDocs() {
        List<AiDoc> result = rerankService.rerank("算法优化", testDocs, 3);
        
        assertEquals(3, result.size());
        // 第一个应该是算法相关
        assertEquals("算法优化", result.get(0).getTitle());
    }
    
    @Test
    @DisplayName("TopK应正确截断结果")
    void rerank_withTopK_shouldTruncateResults() {
        List<AiDoc> result = rerankService.rerank("Bot", testDocs, 2);
        
        assertEquals(2, result.size());
    }
    
    @Test
    @DisplayName("TopK大于文档数时应返回所有文档")
    void rerank_withLargeTopK_shouldReturnAllDocs() {
        List<AiDoc> result = rerankService.rerank("Bot", testDocs, 100);
        
        assertEquals(testDocs.size(), result.size());
    }
    
    @Test
    @DisplayName("应记录指标")
    void rerank_shouldRecordMetrics() {
        rerankService.rerank("测试", testDocs, 3);
        
        verify(metricsService).recordRerankCall(anyInt(), anyInt(), anyLong());
    }
    
    private AiDoc createDoc(String id, String title, String content, String category) {
        AiDoc doc = new AiDoc();
        doc.setDocId(id);
        doc.setTitle(title);
        doc.setContent(content);
        doc.setCategory(category);
        return doc;
    }
}
