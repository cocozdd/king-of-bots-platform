package com.kob.backend.service.impl.ai;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A/B 测试路由器 - 根据配置决定请求路由到 Java 或 Python 后端
 * 
 * 支持两种路由策略：
 * 1. 基于用户 ID 哈希（确保同一用户始终使用同一后端）
 * 2. 基于随机数（纯流量百分比控制）
 * 
 * 配置项：
 * - ai.abtest.enabled=true/false - 是否启用 A/B 测试
 * - ai.abtest.python-traffic-percentage=50 - Python 流量占比 (0-100)
 * - ai.abtest.strategy=hash/random - 路由策略
 * 
 * 面试亮点：
 * - 支持灰度发布，逐步提升 Python 流量占比
 * - 用户粘性：基于哈希确保体验一致性
 * - 可观测性：记录路由决策便于分析
 */
@Service
public class ABTestRouter {
    
    private static final Logger log = LoggerFactory.getLogger(ABTestRouter.class);
    
    @Value("${ai.abtest.enabled:false}")
    private boolean abTestEnabled;
    
    @Value("${ai.abtest.python-traffic-percentage:0}")
    private int pythonTrafficPercentage;
    
    @Value("${ai.abtest.strategy:hash}")
    private String routingStrategy;
    
    @Value("${ai.rag.backend:java}")
    private String ragBackend;
    
    @Value("${ai.agent.backend:java}")
    private String agentBackend;
    
    // 统计数据
    private final AtomicLong totalRequests = new AtomicLong(0);
    private final AtomicLong pythonRequests = new AtomicLong(0);
    private final AtomicLong javaRequests = new AtomicLong(0);
    private final AtomicLong fallbackCount = new AtomicLong(0);
    
    // 用户路由缓存（确保同一用户始终路由到同一后端）
    private final Map<Integer, Boolean> userRouteCache = new ConcurrentHashMap<>();
    
    /**
     * 路由决策结果
     */
    public enum RouteDecision {
        PYTHON("python"),
        JAVA("java");
        
        private final String value;
        
        RouteDecision(String value) {
            this.value = value;
        }
        
        public String getValue() {
            return value;
        }
    }
    
    /**
     * RAG 路由决策
     */
    public RouteDecision routeRAG(Integer userId) {
        return route("rag", ragBackend, userId);
    }
    
    /**
     * Agent 路由决策
     */
    public RouteDecision routeAgent(Integer userId) {
        return route("agent", agentBackend, userId);
    }
    
    /**
     * 通用路由逻辑
     * 
     * @param feature 功能名称（rag/agent）
     * @param configBackend 配置的默认后端
     * @param userId 用户 ID（可为 null）
     * @return 路由决策
     */
    private RouteDecision route(String feature, String configBackend, Integer userId) {
        totalRequests.incrementAndGet();
        
        // 如果未启用 A/B 测试，直接使用配置的后端
        if (!abTestEnabled) {
            RouteDecision decision = "python".equalsIgnoreCase(configBackend) 
                    ? RouteDecision.PYTHON 
                    : RouteDecision.JAVA;
            recordDecision(decision);
            log.debug("[{}] A/B 未启用, 使用配置后端: {}", feature, decision);
            return decision;
        }
        
        // A/B 测试路由逻辑
        boolean usePython;
        
        if ("hash".equalsIgnoreCase(routingStrategy) && userId != null) {
            // 基于用户 ID 哈希的路由（确保用户粘性）
            usePython = userRouteCache.computeIfAbsent(userId, id -> {
                int hash = Math.abs(id.hashCode() % 100);
                return hash < pythonTrafficPercentage;
            });
        } else {
            // 基于随机数的路由
            usePython = Math.random() * 100 < pythonTrafficPercentage;
        }
        
        RouteDecision decision = usePython ? RouteDecision.PYTHON : RouteDecision.JAVA;
        recordDecision(decision);
        
        log.info("[A/B Test] feature={}, userId={}, decision={}, strategy={}, pythonPct={}%",
                feature, userId, decision, routingStrategy, pythonTrafficPercentage);
        
        return decision;
    }
    
    /**
     * 记录路由决策统计
     */
    private void recordDecision(RouteDecision decision) {
        if (decision == RouteDecision.PYTHON) {
            pythonRequests.incrementAndGet();
        } else {
            javaRequests.incrementAndGet();
        }
    }
    
    /**
     * 记录降级事件
     */
    public void recordFallback(String feature, String reason) {
        fallbackCount.incrementAndGet();
        log.warn("[Fallback] feature={}, reason={}", feature, reason);
    }
    
    /**
     * 获取 A/B 测试统计数据
     */
    public Map<String, Object> getStats() {
        long total = totalRequests.get();
        long python = pythonRequests.get();
        long java = javaRequests.get();
        long fallback = fallbackCount.get();
        
        return Map.of(
            "enabled", abTestEnabled,
            "pythonTrafficPercentage", pythonTrafficPercentage,
            "strategy", routingStrategy,
            "totalRequests", total,
            "pythonRequests", python,
            "javaRequests", java,
            "fallbackCount", fallback,
            "actualPythonPercentage", total > 0 ? (python * 100.0 / total) : 0,
            "actualJavaPercentage", total > 0 ? (java * 100.0 / total) : 0
        );
    }
    
    /**
     * 重置统计数据
     */
    public void resetStats() {
        totalRequests.set(0);
        pythonRequests.set(0);
        javaRequests.set(0);
        fallbackCount.set(0);
        userRouteCache.clear();
        log.info("[A/B Test] 统计数据已重置");
    }
    
    /**
     * 动态更新 Python 流量占比（用于灰度发布）
     */
    public void updatePythonTrafficPercentage(int percentage) {
        if (percentage < 0 || percentage > 100) {
            throw new IllegalArgumentException("流量占比必须在 0-100 之间");
        }
        int oldPercentage = this.pythonTrafficPercentage;
        this.pythonTrafficPercentage = percentage;
        userRouteCache.clear(); // 清除缓存，让新配置生效
        log.info("[A/B Test] Python 流量占比更新: {}% -> {}%", oldPercentage, percentage);
    }
    
    /**
     * 动态启用/禁用 A/B 测试
     */
    public void setAbTestEnabled(boolean enabled) {
        this.abTestEnabled = enabled;
        log.info("[A/B Test] A/B 测试状态: {}", enabled ? "启用" : "禁用");
    }
    
    // Getters
    public boolean isAbTestEnabled() { return abTestEnabled; }
    public int getPythonTrafficPercentage() { return pythonTrafficPercentage; }
    public String getRoutingStrategy() { return routingStrategy; }
}
