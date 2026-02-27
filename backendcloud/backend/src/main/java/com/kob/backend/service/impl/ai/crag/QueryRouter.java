package com.kob.backend.service.impl.ai.crag;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.regex.Pattern;

/**
 * 查询路由器 - 智能决策检索策略
 * 
 * 面试要点:
 * - 不是所有问题都需要检索 (Adaptive RAG)
 * - 路由决策影响延迟和质量
 * - 分类: 简单问答 / 知识检索 / 实时信息 / 多步推理
 * 
 * 路由策略:
 * 1. NO_RETRIEVAL - 直接回答 (简单问候、常识问题)
 * 2. VECTOR_SEARCH - 向量检索 (语义匹配)
 * 3. KEYWORD_SEARCH - 关键词检索 (精确匹配)
 * 4. HYBRID_SEARCH - 混合检索 (复杂问题)
 * 5. GRAPH_SEARCH - 图谱检索 (关系推理)
 * 6. WEB_SEARCH - 网络搜索 (实时信息)
 */
@Service
public class QueryRouter {
    
    private static final Logger log = LoggerFactory.getLogger(QueryRouter.class);
    
    // 简单问候模式
    private static final Pattern GREETING_PATTERN = Pattern.compile(
            "^(你好|hi|hello|嗨|早上好|晚上好|谢谢|感谢|再见|拜拜).*", 
            Pattern.CASE_INSENSITIVE);
    
    // 代码相关关键词
    private static final Set<String> CODE_KEYWORDS = Set.of(
            "代码", "code", "bug", "错误", "报错", "实现", "编写", 
            "函数", "方法", "类", "接口", "java", "python");
    
    // 策略相关关键词
    private static final Set<String> STRATEGY_KEYWORDS = Set.of(
            "策略", "算法", "优化", "bot", "贪吃蛇", "移动", 
            "避障", "寻路", "bfs", "a*", "dfs");
    
    // 实时信息关键词
    private static final Set<String> REALTIME_KEYWORDS = Set.of(
            "最新", "今天", "现在", "当前", "实时", "新闻");
    
    // 关系推理关键词
    private static final Set<String> RELATION_KEYWORDS = Set.of(
            "关系", "联系", "区别", "对比", "为什么", "如何影响", 
            "依赖", "调用", "继承");
    
    /**
     * 路由决策
     */
    public RouteDecision route(String query) {
        log.info("[Router] 分析查询: {}", truncate(query, 50));
        
        RouteDecision decision = new RouteDecision();
        decision.query = query;
        decision.features = analyzeFeatures(query);
        
        // 1. 检查是否简单问候
        if (isGreeting(query)) {
            decision.route = RouteType.NO_RETRIEVAL;
            decision.reason = "简单问候，无需检索";
            decision.confidence = 0.95;
            return decision;
        }
        
        // 2. 检查是否需要实时信息
        if (containsKeywords(query, REALTIME_KEYWORDS)) {
            decision.route = RouteType.WEB_SEARCH;
            decision.reason = "需要实时信息";
            decision.confidence = 0.8;
            return decision;
        }
        
        // 3. 检查是否涉及关系推理
        if (containsKeywords(query, RELATION_KEYWORDS) && query.length() > 20) {
            decision.route = RouteType.GRAPH_SEARCH;
            decision.reason = "涉及关系推理，使用图谱检索";
            decision.confidence = 0.75;
            return decision;
        }
        
        // 4. 检查是否代码相关
        if (containsKeywords(query, CODE_KEYWORDS)) {
            // 代码问题通常需要精确匹配
            decision.route = RouteType.HYBRID_SEARCH;
            decision.reason = "代码相关问题，使用混合检索";
            decision.confidence = 0.85;
            return decision;
        }
        
        // 5. 检查是否策略相关
        if (containsKeywords(query, STRATEGY_KEYWORDS)) {
            decision.route = RouteType.VECTOR_SEARCH;
            decision.reason = "策略相关问题，使用语义检索";
            decision.confidence = 0.8;
            return decision;
        }
        
        // 6. 根据查询长度和复杂度决定
        if (query.length() < 10) {
            decision.route = RouteType.KEYWORD_SEARCH;
            decision.reason = "短查询，使用关键词检索";
            decision.confidence = 0.7;
        } else if (query.length() > 50) {
            decision.route = RouteType.HYBRID_SEARCH;
            decision.reason = "复杂查询，使用混合检索";
            decision.confidence = 0.75;
        } else {
            decision.route = RouteType.VECTOR_SEARCH;
            decision.reason = "常规查询，使用向量检索";
            decision.confidence = 0.7;
        }
        
        log.info("[Router] 决策: {} (置信度: {})", decision.route, decision.confidence);
        return decision;
    }
    
    /**
     * 分析查询特征
     */
    private Map<String, Object> analyzeFeatures(String query) {
        Map<String, Object> features = new HashMap<>();
        features.put("length", query.length());
        features.put("wordCount", query.split("\\s+").length);
        features.put("hasQuestion", query.contains("?") || query.contains("？") || 
                query.contains("如何") || query.contains("怎么") || query.contains("什么"));
        features.put("hasCode", containsKeywords(query, CODE_KEYWORDS));
        features.put("hasStrategy", containsKeywords(query, STRATEGY_KEYWORDS));
        features.put("complexity", calculateComplexity(query));
        return features;
    }
    
    /**
     * 计算查询复杂度 (0-1)
     */
    private double calculateComplexity(String query) {
        double score = 0;
        
        // 长度因子
        score += Math.min(query.length() / 100.0, 0.3);
        
        // 问题词因子
        if (query.contains("为什么") || query.contains("如何")) score += 0.2;
        if (query.contains("对比") || query.contains("区别")) score += 0.2;
        
        // 技术词因子
        if (containsKeywords(query, CODE_KEYWORDS)) score += 0.15;
        if (containsKeywords(query, STRATEGY_KEYWORDS)) score += 0.15;
        
        return Math.min(score, 1.0);
    }
    
    private boolean isGreeting(String query) {
        return GREETING_PATTERN.matcher(query.trim()).matches();
    }
    
    private boolean containsKeywords(String query, Set<String> keywords) {
        String lower = query.toLowerCase();
        return keywords.stream().anyMatch(lower::contains);
    }
    
    private String truncate(String text, int maxLen) {
        if (text == null) return "";
        return text.length() > maxLen ? text.substring(0, maxLen) + "..." : text;
    }
    
    // ===== 路由类型 =====
    
    public enum RouteType {
        NO_RETRIEVAL,    // 直接回答
        VECTOR_SEARCH,   // 向量检索
        KEYWORD_SEARCH,  // 关键词检索
        HYBRID_SEARCH,   // 混合检索
        GRAPH_SEARCH,    // 图谱检索
        WEB_SEARCH       // 网络搜索
    }
    
    public static class RouteDecision {
        public String query;
        public RouteType route;
        public String reason;
        public double confidence;
        public Map<String, Object> features;
    }
}
