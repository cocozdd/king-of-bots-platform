package com.kob.backend.service.impl.ai;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.concurrent.atomic.AtomicLong;

/**
 * AI 指标监控服务
 * 
 * 功能：
 * - Token 使用统计
 * - 成本计算和累积
 * - API 调用性能监控
 * 
 * 用于：
 * - 成本控制和预算管理
 * - 性能分析和优化
 * - 审计和报表
 */
@Service
public class AiMetricsService {
    
    private static final Logger log = LoggerFactory.getLogger(AiMetricsService.class);
    
    // Token 统计
    private final AtomicLong totalTokens = new AtomicLong(0);
    private final AtomicLong totalEmbeddingTokens = new AtomicLong(0);
    private final AtomicLong totalChatTokens = new AtomicLong(0);
    
    // 成本统计（单位：分）
    private final AtomicLong totalCost = new AtomicLong(0);
    
    // 调用次数统计
    private final AtomicLong totalApiCalls = new AtomicLong(0);
    private final AtomicLong embeddingCalls = new AtomicLong(0);
    private final AtomicLong chatCalls = new AtomicLong(0);
    
    // 延迟统计（毫秒）
    private final AtomicLong totalLatency = new AtomicLong(0);
    
    /**
     * 估算文本的 Token 数量
     * 
     * 粗略估算规则：
     * - 中文：约 1.5 字符/token
     * - 英文：约 4 字符/token
     * - 混合文本：取平均 2 字符/token
     * 
     * @param text 输入文本
     * @return 估算的 Token 数量
     */
    public int estimateTokens(String text) {
        if (text == null || text.isEmpty()) {
            return 0;
        }
        
        // 简单估算：平均 2 字符 = 1 token
        return (int) Math.ceil(text.length() / 2.0);
    }
    
    /**
     * 记录 Embedding API 调用
     * 
     * @param model 模型名称
     * @param inputTokens 输入 Token 数
     * @param outputTokens 输出 Token 数（对于 Embedding 通常是固定的维度）
     * @param latencyMs 延迟（毫秒）
     */
    public void recordEmbeddingCall(String model, int inputTokens, int outputTokens, long latencyMs) {
        int total = inputTokens + outputTokens;
        
        // 更新统计
        totalTokens.addAndGet(total);
        totalEmbeddingTokens.addAndGet(total);
        embeddingCalls.incrementAndGet();
        totalApiCalls.incrementAndGet();
        totalLatency.addAndGet(latencyMs);
        
        // 计算成本
        long cost = calculateEmbeddingCost(model, inputTokens);
        totalCost.addAndGet(cost);
        
        log.info("Embedding调用 - model={}, inputTokens={}, outputDim={}, latency={}ms, cost={}分", 
            model, inputTokens, outputTokens, latencyMs, cost / 100.0);
    }
    
    /**
     * 记录 Chat API 调用
     * 
     * @param model 模型名称
     * @param inputTokens 输入 Token 数
     * @param outputTokens 输出 Token 数
     * @param latencyMs 延迟（毫秒）
     */
    public void recordChatCall(String model, int inputTokens, int outputTokens, long latencyMs) {
        int total = inputTokens + outputTokens;
        
        // 更新统计
        totalTokens.addAndGet(total);
        totalChatTokens.addAndGet(total);
        chatCalls.incrementAndGet();
        totalApiCalls.incrementAndGet();
        totalLatency.addAndGet(latencyMs);
        
        // 计算成本
        long cost = calculateChatCost(model, inputTokens, outputTokens);
        totalCost.addAndGet(cost);
        
        log.info("Chat调用 - model={}, input={}, output={}, total={}, latency={}ms, cost={}分", 
            model, inputTokens, outputTokens, total, latencyMs, cost / 100.0);
    }
    
    /**
     * 计算 Embedding 成本
     * 
     * DashScope 定价（2025）：
     * - text-embedding-v2: ¥0.0007/1k tokens
     * - text-embedding-v3: ¥0.0007/1k tokens
     * 
     * @param model 模型名称
     * @param tokens Token 数量
     * @return 成本（分）
     */
    private long calculateEmbeddingCost(String model, int tokens) {
        // text-embedding-v2/v3: ¥0.0007/1k = 0.07分/1000 tokens
        double pricePerThousand = 0.07; // 分/1000 tokens
        return Math.round(tokens * pricePerThousand / 1000.0);
    }
    
    /**
     * 计算 Chat 成本
     * 
     * DeepSeek 定价（2025）：
     * - deepseek-chat: 输入 ¥0.001/1k, 输出 ¥0.002/1k
     * 
     * @param model 模型名称
     * @param inputTokens 输入 Token 数
     * @param outputTokens 输出 Token 数
     * @return 成本（分）
     */
    private long calculateChatCost(String model, int inputTokens, int outputTokens) {
        if (model.contains("deepseek")) {
            // deepseek-chat: 输入 ¥0.001/1k = 0.1分/1000, 输出 ¥0.002/1k = 0.2分/1000
            double inputPrice = 0.1;  // 分/1000 tokens
            double outputPrice = 0.2; // 分/1000 tokens
            
            long inputCost = Math.round(inputTokens * inputPrice / 1000.0);
            long outputCost = Math.round(outputTokens * outputPrice / 1000.0);
            
            return inputCost + outputCost;
        }
        
        // 默认定价
        return Math.round((inputTokens + outputTokens) * 0.1 / 1000.0);
    }
    
    /**
     * 获取统计摘要
     * 
     * @return 当前的统计数据
     */
    public MetricsSummary getSummary() {
        long calls = totalApiCalls.get();
        long avgLatency = calls > 0 ? totalLatency.get() / calls : 0;
        
        return new MetricsSummary(
            totalTokens.get(),
            totalEmbeddingTokens.get(),
            totalChatTokens.get(),
            totalCost.get() / 100.0,  // 转换为元
            totalApiCalls.get(),
            embeddingCalls.get(),
            chatCalls.get(),
            avgLatency,
            Instant.now()
        );
    }
    
    // 新增统计
    private final AtomicLong rerankCalls = new AtomicLong(0);
    private final AtomicLong hybridSearchCalls = new AtomicLong(0);
    private final AtomicLong queryRewriteCalls = new AtomicLong(0);
    private final AtomicLong codeGenCalls = new AtomicLong(0);
    
    /**
     * 记录 Rerank 调用
     */
    public void recordRerankCall(int inputDocs, int outputDocs, long latencyMs) {
        rerankCalls.incrementAndGet();
        totalLatency.addAndGet(latencyMs);
        log.info("Rerank调用 - 输入{}篇, 输出{}篇, 耗时{}ms", inputDocs, outputDocs, latencyMs);
    }
    
    /**
     * 记录混合检索调用
     */
    public void recordHybridSearchCall(int vectorResults, int bm25Results, long latencyMs) {
        hybridSearchCalls.incrementAndGet();
        totalLatency.addAndGet(latencyMs);
        log.info("混合检索调用 - 向量{}篇, BM25 {}篇, 耗时{}ms", vectorResults, bm25Results, latencyMs);
    }
    
    /**
     * 记录查询改写调用
     */
    public void recordQueryRewriteCall(String strategy, long latencyMs) {
        queryRewriteCalls.incrementAndGet();
        totalLatency.addAndGet(latencyMs);
        log.info("查询改写调用 - 策略={}, 耗时{}ms", strategy, latencyMs);
    }
    
    /**
     * 记录代码生成调用
     */
    public void recordCodeGenCall(String type, int inputTokens, int outputTokens, long latencyMs) {
        codeGenCalls.incrementAndGet();
        totalApiCalls.incrementAndGet();
        totalTokens.addAndGet(inputTokens + outputTokens);
        totalChatTokens.addAndGet(inputTokens + outputTokens);
        totalLatency.addAndGet(latencyMs);
        
        long cost = calculateChatCost("deepseek-coder", inputTokens, outputTokens);
        totalCost.addAndGet(cost);
        
        log.info("代码生成调用 - type={}, input={}, output={}, latency={}ms, cost={}分",
                type, inputTokens, outputTokens, latencyMs, cost / 100.0);
    }
    
    /**
     * 重置统计数据（仅用于测试）
     */
    public void reset() {
        totalTokens.set(0);
        totalEmbeddingTokens.set(0);
        totalChatTokens.set(0);
        totalCost.set(0);
        totalApiCalls.set(0);
        embeddingCalls.set(0);
        chatCalls.set(0);
        totalLatency.set(0);
        log.info("指标已重置");
    }
    
    /**
     * 统计摘要数据类
     */
    public record MetricsSummary(
        long totalTokens,           // 总 Token 数
        long embeddingTokens,       // Embedding Token 数
        long chatTokens,            // Chat Token 数
        double totalCostYuan,       // 总成本（元）
        long totalCalls,            // 总调用次数
        long embeddingCalls,        // Embedding 调用次数
        long chatCalls,             // Chat 调用次数
        long avgLatencyMs,          // 平均延迟（毫秒）
        Instant timestamp           // 统计时间
    ) {
        /**
         * 格式化输出
         */
        @Override
        public String toString() {
            return String.format(
                "AI指标摘要: 总调用=%d次(Embed=%d, Chat=%d), " +
                "总Token=%d(Embed=%d, Chat=%d), " +
                "总成本=¥%.4f, 平均延迟=%dms, 时间=%s",
                totalCalls, embeddingCalls, chatCalls,
                totalTokens, embeddingTokens, chatTokens,
                totalCostYuan, avgLatencyMs, timestamp
            );
        }
    }
}
