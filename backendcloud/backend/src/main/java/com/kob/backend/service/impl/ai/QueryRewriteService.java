package com.kob.backend.service.impl.ai;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * 查询改写服务
 * 
 * 功能：
 * - HyDE (Hypothetical Document Embeddings): 先生成假设答案再检索
 * - Multi-Query: 生成多个查询变体，扩大检索召回
 * - Query Expansion: 查询扩展，添加同义词
 * 
 * 优势：
 * - 解决用户查询与文档表述不匹配的问题
 * - 提升检索召回率
 * - 改善短查询的检索效果
 */
@Service
public class QueryRewriteService {
    
    private static final Logger log = LoggerFactory.getLogger(QueryRewriteService.class);
    
    @Autowired
    private AiMetricsService metricsService;
    
    /**
     * HyDE 查询改写
     * 
     * 原理：先让 LLM 生成一个假设的答案，然后用这个答案去检索
     * 优势：假设答案与真实文档的表述更接近，提升检索准确率
     * 
     * @param query 原始查询
     * @param deepseekClient DeepSeek 客户端
     * @return 生成的假设文档
     */
    public String hydeRewrite(String query, DeepseekClient deepseekClient) {
        if (deepseekClient == null || !deepseekClient.enabled()) {
            log.warn("DeepSeek 不可用，跳过 HyDE 改写");
            return query;
        }
        
        long startTime = System.currentTimeMillis();
        
        String systemPrompt = """
            你是一个 Bot 对战游戏的技术文档专家。
            用户会提出关于 Bot 开发的问题，请直接给出一段简洁的技术说明作为答案。
            不要使用"我认为"、"可能"等不确定词汇。
            直接描述技术要点，100-200字即可。
            """;
        
        String userPrompt = "问题：" + query + "\n\n请给出简洁的技术说明：";
        
        try {
            String hypotheticalDoc = deepseekClient.chat(systemPrompt, userPrompt, List.of());
            
            long latency = System.currentTimeMillis() - startTime;
            log.info("HyDE 改写完成: 原查询长度={}, 假设文档长度={}, 耗时{}ms",
                    query.length(), hypotheticalDoc.length(), latency);
            
            if (metricsService != null) {
                metricsService.recordQueryRewriteCall("hyde", latency);
            }
            
            return hypotheticalDoc;
        } catch (Exception e) {
            log.warn("HyDE 改写失败，使用原查询: {}", e.getMessage());
            return query;
        }
    }
    
    /**
     * Multi-Query 多查询生成
     * 
     * 原理：将用户查询改写为多个不同表述的查询
     * 优势：扩大检索召回，覆盖不同的表述方式
     * 
     * @param query 原始查询
     * @param deepseekClient DeepSeek 客户端
     * @return 多个查询变体
     */
    public List<String> multiQueryRewrite(String query, DeepseekClient deepseekClient) {
        List<String> queries = new ArrayList<>();
        queries.add(query); // 始终包含原始查询
        
        if (deepseekClient == null || !deepseekClient.enabled()) {
            log.warn("DeepSeek 不可用，仅使用原始查询");
            return queries;
        }
        
        long startTime = System.currentTimeMillis();
        
        String systemPrompt = """
            你是一个搜索查询专家。
            请将用户的问题改写成 3 个不同的表述方式，每行一个。
            保持原意，但使用不同的词汇和句式。
            只输出改写后的查询，不要编号，不要解释。
            """;
        
        String userPrompt = "原始问题：" + query;
        
        try {
            String response = deepseekClient.chat(systemPrompt, userPrompt, List.of());
            
            // 解析多个查询
            String[] lines = response.split("\n");
            for (String line : lines) {
                line = line.trim();
                // 移除可能的编号前缀
                line = line.replaceAll("^\\d+[.、)\\s]+", "");
                if (!line.isEmpty() && !line.equals(query) && line.length() > 5) {
                    queries.add(line);
                }
            }
            
            long latency = System.currentTimeMillis() - startTime;
            log.info("Multi-Query 改写完成: 生成{}个查询变体, 耗时{}ms", queries.size(), latency);
            
            if (metricsService != null) {
                metricsService.recordQueryRewriteCall("multi-query", latency);
            }
            
        } catch (Exception e) {
            log.warn("Multi-Query 改写失败: {}", e.getMessage());
        }
        
        return queries;
    }
    
    /**
     * 查询扩展 - 添加同义词和相关词
     * 
     * @param query 原始查询
     * @return 扩展后的查询
     */
    public String expandQuery(String query) {
        // 领域特定的同义词映射
        String expanded = query;
        
        // Bot 开发相关同义词
        expanded = expandWithSynonyms(expanded, "移动", "走位", "位移", "行动");
        expanded = expandWithSynonyms(expanded, "策略", "战术", "方法", "算法");
        expanded = expandWithSynonyms(expanded, "攻击", "进攻", "出击");
        expanded = expandWithSynonyms(expanded, "防御", "防守", "躲避");
        expanded = expandWithSynonyms(expanded, "代码", "程序", "实现");
        expanded = expandWithSynonyms(expanded, "优化", "改进", "提升", "增强");
        expanded = expandWithSynonyms(expanded, "bug", "错误", "问题", "异常");
        expanded = expandWithSynonyms(expanded, "蛇", "snake", "玩家");
        expanded = expandWithSynonyms(expanded, "地图", "map", "棋盘", "网格");
        expanded = expandWithSynonyms(expanded, "障碍", "墙", "阻挡");
        
        if (!expanded.equals(query)) {
            log.info("查询扩展: {} -> {}", query, expanded);
        }
        
        return expanded;
    }
    
    private String expandWithSynonyms(String query, String... synonyms) {
        for (String synonym : synonyms) {
            if (query.toLowerCase().contains(synonym.toLowerCase())) {
                // 添加其他同义词
                StringBuilder expanded = new StringBuilder(query);
                for (String s : synonyms) {
                    if (!query.toLowerCase().contains(s.toLowerCase())) {
                        expanded.append(" ").append(s);
                        break; // 只添加一个同义词，避免查询过长
                    }
                }
                return expanded.toString();
            }
        }
        return query;
    }
    
    /**
     * 智能查询改写 - 综合多种策略
     * 
     * @param query 原始查询
     * @param deepseekClient DeepSeek 客户端
     * @param strategy 改写策略: "hyde", "multi", "expand", "auto"
     * @return 改写结果
     */
    public QueryRewriteResult smartRewrite(String query, DeepseekClient deepseekClient, String strategy) {
        QueryRewriteResult result = new QueryRewriteResult();
        result.originalQuery = query;
        result.queries = new ArrayList<>();
        result.queries.add(query);
        
        switch (strategy.toLowerCase()) {
            case "hyde":
                result.hydeQuery = hydeRewrite(query, deepseekClient);
                result.queries.add(result.hydeQuery);
                break;
                
            case "multi":
                result.queries = multiQueryRewrite(query, deepseekClient);
                break;
                
            case "expand":
                String expanded = expandQuery(query);
                if (!expanded.equals(query)) {
                    result.queries.add(expanded);
                }
                break;
                
            case "auto":
            default:
                // 自动选择策略
                if (query.length() < 10) {
                    // 短查询：使用扩展
                    String exp = expandQuery(query);
                    if (!exp.equals(query)) {
                        result.queries.add(exp);
                    }
                } else if (query.contains("?") || query.contains("？") || 
                           query.contains("怎么") || query.contains("如何")) {
                    // 问句：使用 HyDE
                    result.hydeQuery = hydeRewrite(query, deepseekClient);
                    result.queries.add(result.hydeQuery);
                } else {
                    // 其他：使用 Multi-Query
                    result.queries = multiQueryRewrite(query, deepseekClient);
                }
                break;
        }
        
        result.strategy = strategy;
        return result;
    }
    
    /**
     * 查询改写结果
     */
    public static class QueryRewriteResult {
        public String originalQuery;
        public String hydeQuery;
        public List<String> queries;
        public String strategy;
        
        public String getPrimaryQuery() {
            if (hydeQuery != null && !hydeQuery.isEmpty()) {
                return hydeQuery;
            }
            return queries != null && !queries.isEmpty() ? queries.get(0) : originalQuery;
        }
    }
}
