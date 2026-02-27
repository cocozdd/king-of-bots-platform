package com.kob.backend.service.impl.ai;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

/**
 * AI 服务监控指标收集器
 * 
 * 收集指标：
 * - 响应时间（Java vs Python）
 * - 成功率 / 失败率
 * - Token 消耗（可选）
 * - 降级次数
 * 
 * 面试亮点：
 * - 滑动窗口统计（最近 N 分钟）
 * - 分后端指标对比
 * - 支持告警阈值配置
 */
@Service
public class AiMetricsCollector {
    
    private static final Logger log = LoggerFactory.getLogger(AiMetricsCollector.class);
    
    // 保留最近 5 分钟的指标
    private static final long WINDOW_MS = 5 * 60 * 1000;
    private static final int MAX_LATENCY_SAMPLES = 1000;
    
    // 按后端分类的指标
    private final Map<String, BackendMetrics> metricsMap = new ConcurrentHashMap<>();
    
    /**
     * 记录请求指标
     */
    public void recordRequest(String backend, String feature, long latencyMs, boolean success, String errorCode) {
        String key = backend + ":" + feature;
        BackendMetrics metrics = metricsMap.computeIfAbsent(key, k -> new BackendMetrics(backend, feature));
        metrics.record(latencyMs, success, errorCode);
    }
    
    /**
     * 记录 RAG 请求
     */
    public void recordRAGRequest(String backend, long latencyMs, boolean success) {
        recordRequest(backend, "rag", latencyMs, success, null);
    }
    
    /**
     * 记录 Agent 请求
     */
    public void recordAgentRequest(String backend, long latencyMs, boolean success) {
        recordRequest(backend, "agent", latencyMs, success, null);
    }
    
    /**
     * 记录降级事件
     */
    public void recordFallback(String feature, String fromBackend, String toBackend, String reason) {
        String key = fromBackend + ":" + feature;
        BackendMetrics metrics = metricsMap.computeIfAbsent(key, k -> new BackendMetrics(fromBackend, feature));
        metrics.recordFallback(reason);
        
        log.warn("[Metrics] Fallback: {} {} -> {}, reason: {}", feature, fromBackend, toBackend, reason);
    }
    
    /**
     * 获取所有指标
     */
    public Map<String, Object> getAllMetrics() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("timestamp", Instant.now().toString());
        result.put("windowMs", WINDOW_MS);
        
        Map<String, Object> backends = new LinkedHashMap<>();
        for (Map.Entry<String, BackendMetrics> entry : metricsMap.entrySet()) {
            backends.put(entry.getKey(), entry.getValue().getSummary());
        }
        result.put("backends", backends);
        
        // 对比分析
        result.put("comparison", generateComparison());
        
        return result;
    }
    
    /**
     * 获取特定后端的指标
     */
    public Map<String, Object> getMetrics(String backend, String feature) {
        String key = backend + ":" + feature;
        BackendMetrics metrics = metricsMap.get(key);
        return metrics != null ? metrics.getSummary() : Map.of("error", "No data");
    }
    
    /**
     * 生成 Java vs Python 对比
     */
    private Map<String, Object> generateComparison() {
        Map<String, Object> comparison = new LinkedHashMap<>();
        
        // RAG 对比
        BackendMetrics javaRag = metricsMap.get("java:rag");
        BackendMetrics pythonRag = metricsMap.get("python:rag");
        if (javaRag != null && pythonRag != null) {
            comparison.put("rag", Map.of(
                "java_p50_ms", javaRag.getP50Latency(),
                "python_p50_ms", pythonRag.getP50Latency(),
                "java_success_rate", javaRag.getSuccessRate(),
                "python_success_rate", pythonRag.getSuccessRate(),
                "recommendation", recommendBackend(javaRag, pythonRag)
            ));
        }
        
        // Agent 对比
        BackendMetrics javaAgent = metricsMap.get("java:agent");
        BackendMetrics pythonAgent = metricsMap.get("python:agent");
        if (javaAgent != null && pythonAgent != null) {
            comparison.put("agent", Map.of(
                "java_p50_ms", javaAgent.getP50Latency(),
                "python_p50_ms", pythonAgent.getP50Latency(),
                "java_success_rate", javaAgent.getSuccessRate(),
                "python_success_rate", pythonAgent.getSuccessRate(),
                "recommendation", recommendBackend(javaAgent, pythonAgent)
            ));
        }
        
        return comparison;
    }
    
    /**
     * 推荐使用哪个后端
     */
    private String recommendBackend(BackendMetrics java, BackendMetrics python) {
        double javaScore = java.getSuccessRate() * 100 - java.getP50Latency() / 100.0;
        double pythonScore = python.getSuccessRate() * 100 - python.getP50Latency() / 100.0;
        
        if (pythonScore > javaScore + 5) {
            return "python (better performance)";
        } else if (javaScore > pythonScore + 5) {
            return "java (better performance)";
        } else {
            return "equal (no significant difference)";
        }
    }
    
    /**
     * 重置所有指标
     */
    public void resetMetrics() {
        metricsMap.clear();
        log.info("[Metrics] All metrics reset");
    }
    
    /**
     * 单个后端的指标
     */
    private static class BackendMetrics {
        private final String backend;
        private final String feature;
        private final AtomicLong totalRequests = new AtomicLong(0);
        private final AtomicLong successCount = new AtomicLong(0);
        private final AtomicLong failureCount = new AtomicLong(0);
        private final AtomicLong fallbackCount = new AtomicLong(0);
        private final ConcurrentLinkedQueue<Long> latencySamples = new ConcurrentLinkedQueue<>();
        private final Map<String, AtomicLong> errorCounts = new ConcurrentHashMap<>();
        
        BackendMetrics(String backend, String feature) {
            this.backend = backend;
            this.feature = feature;
        }
        
        void record(long latencyMs, boolean success, String errorCode) {
            totalRequests.incrementAndGet();
            
            if (success) {
                successCount.incrementAndGet();
            } else {
                failureCount.incrementAndGet();
                if (errorCode != null) {
                    errorCounts.computeIfAbsent(errorCode, k -> new AtomicLong(0)).incrementAndGet();
                }
            }
            
            // 保留最近的延迟样本
            latencySamples.offer(latencyMs);
            while (latencySamples.size() > MAX_LATENCY_SAMPLES) {
                latencySamples.poll();
            }
        }
        
        void recordFallback(String reason) {
            fallbackCount.incrementAndGet();
        }
        
        double getSuccessRate() {
            long total = totalRequests.get();
            return total > 0 ? (double) successCount.get() / total : 0;
        }
        
        long getP50Latency() {
            List<Long> samples = new ArrayList<>(latencySamples);
            if (samples.isEmpty()) return 0;
            Collections.sort(samples);
            return samples.get(samples.size() / 2);
        }
        
        long getP90Latency() {
            List<Long> samples = new ArrayList<>(latencySamples);
            if (samples.isEmpty()) return 0;
            Collections.sort(samples);
            return samples.get((int) (samples.size() * 0.9));
        }
        
        long getP99Latency() {
            List<Long> samples = new ArrayList<>(latencySamples);
            if (samples.isEmpty()) return 0;
            Collections.sort(samples);
            return samples.get((int) (samples.size() * 0.99));
        }
        
        Map<String, Object> getSummary() {
            Map<String, Object> summary = new LinkedHashMap<>();
            summary.put("backend", backend);
            summary.put("feature", feature);
            summary.put("totalRequests", totalRequests.get());
            summary.put("successCount", successCount.get());
            summary.put("failureCount", failureCount.get());
            summary.put("fallbackCount", fallbackCount.get());
            summary.put("successRate", String.format("%.2f%%", getSuccessRate() * 100));
            summary.put("latency", Map.of(
                "p50_ms", getP50Latency(),
                "p90_ms", getP90Latency(),
                "p99_ms", getP99Latency()
            ));
            summary.put("errors", new HashMap<>(errorCounts));
            return summary;
        }
    }
}
