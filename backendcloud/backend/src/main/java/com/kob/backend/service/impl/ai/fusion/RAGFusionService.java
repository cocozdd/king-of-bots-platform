package com.kob.backend.service.impl.ai.fusion;

import com.kob.backend.controller.ai.dto.AiDoc;
import com.kob.backend.service.impl.ai.AiMetricsService;
import com.kob.backend.service.impl.ai.DeepseekClient;
import com.kob.backend.service.impl.ai.DashscopeEmbeddingClient;
import com.kob.backend.service.impl.ai.PromptSecurityService;
import com.kob.backend.service.impl.ai.search.HybridSearchService;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * RAG Fusion 服务
 * 
 * 核心思想：
 * - 生成多个查询变体（Query Expansion）
 * - 对每个查询分别检索
 * - 使用 RRF 融合所有检索结果
 * 
 * 论文参考: RAG-Fusion: a New Take on Retrieval Augmented Generation
 * 
 * 面试要点：
 * - 单一查询可能遗漏相关文档
 * - 多查询覆盖不同语义角度
 * - RRF 融合多源结果，提升召回率
 * - 与 HyDE 的区别：HyDE 生成假设文档，RAG Fusion 生成多查询
 */
@Service
public class RAGFusionService {
    
    private static final Logger log = LoggerFactory.getLogger(RAGFusionService.class);
    
    // 默认生成查询数量
    private static final int DEFAULT_QUERY_COUNT = 4;
    // RRF 常数
    private static final int RRF_K = 60;
    
    @Autowired
    private HybridSearchService hybridSearchService;

    @Autowired(required = false)
    private DashscopeEmbeddingClient embeddingClient;

    @Autowired(required = false)
    private AiMetricsService metricsService;

    @Autowired
    private PromptSecurityService securityService;

    private DeepseekClient deepseekClient;
    private ExecutorService executor;
    
    @PostConstruct
    public void init() {
        this.deepseekClient = new DeepseekClient(metricsService);
        this.executor = Executors.newFixedThreadPool(4);
        log.info("RAG Fusion Service 初始化完成");
    }
    
    /**
     * RAG Fusion 主流程
     * 
     * @param originalQuery 原始查询
     * @param topK 返回文档数量
     * @return 融合结果
     */
    public RAGFusionResult search(String originalQuery, int topK) {
        long startTime = System.currentTimeMillis();
        RAGFusionResult result = new RAGFusionResult();
        result.originalQuery = originalQuery;

        // ===== 鲁棒性检查：提前拦截问题查询 =====
        PromptSecurityService.SecurityCheckResult securityCheck = securityService.performFullSecurityCheck(originalQuery);
        if (!securityCheck.isPassed()) {
            log.warn("[RAG Fusion] 安全检查拦截: type={}, query={}", securityCheck.getRejectType(), truncate(originalQuery, 50));
            result.generatedQueries = Collections.emptyList();
            result.fusedDocs = Collections.emptyList();
            result.documents = Collections.emptyList();
            result.latencyMs = System.currentTimeMillis() - startTime;
            return result;
        }

        log.info("[RAG Fusion] 开始处理: {}", truncate(originalQuery, 50));

        // Step 1: 生成多个查询变体
        List<String> queries = generateQueryVariants(originalQuery, DEFAULT_QUERY_COUNT);
        queries.add(0, originalQuery); // 原始查询放在第一位
        result.generatedQueries = queries;
        
        log.info("[RAG Fusion] 生成 {} 个查询变体", queries.size());
        
        // Step 2: 并行执行检索
        Map<String, List<RankedDoc>> queryResults = parallelSearch(queries, topK * 2);
        result.queryResults = queryResults;
        
        // Step 3: RRF 融合
        List<RankedDoc> fusedDocs = reciprocalRankFusion(queryResults, topK);
        result.fusedDocs = fusedDocs;
        
        // Step 4: 转换为 AiDoc
        result.documents = fusedDocs.stream()
                .map(RankedDoc::toAiDoc)
                .collect(Collectors.toList());
        
        result.latencyMs = System.currentTimeMillis() - startTime;
        
        log.info("[RAG Fusion] 完成: {} 个查询, {} 篇文档, {}ms",
                queries.size(), fusedDocs.size(), result.latencyMs);
        
        return result;
    }
    
    /**
     * 生成查询变体
     * 
     * 使用 LLM 从不同角度重新表述查询
     */
    private List<String> generateQueryVariants(String query, int count) {
        if (!deepseekClient.enabled()) {
            // 降级：使用规则生成
            return generateRuleBasedVariants(query, count);
        }
        
        String systemPrompt = """
            你是一个查询扩展专家。给定用户查询，生成多个不同角度的查询变体。
            
            要求：
            1. 每个变体从不同角度表达相同的信息需求
            2. 包括：同义词替换、问法变化、更具体/更抽象的表达
            3. 保持查询简洁，每个不超过 30 字
            4. 返回 JSON 数组格式
            
            示例：
            输入: "如何用 BFS 实现蛇的移动"
            输出: ["BFS 寻路算法实现", "广度优先搜索 蛇移动策略", "蛇如何使用 BFS 找最短路径", "贪吃蛇 BFS 代码实现"]
            """;
        
        String userMessage = String.format("为以下查询生成 %d 个变体（返回 JSON 数组）：\n%s", count, query);
        
        try {
            String response = deepseekClient.chat(systemPrompt, userMessage, Collections.emptyList());
            return parseQueryVariants(response, count);
        } catch (Exception e) {
            log.warn("[RAG Fusion] LLM 生成查询变体失败: {}", e.getMessage());
            return generateRuleBasedVariants(query, count);
        }
    }
    
    /**
     * 解析 LLM 返回的查询变体
     */
    private List<String> parseQueryVariants(String response, int maxCount) {
        List<String> variants = new ArrayList<>();
        
        try {
            // 提取 JSON 数组
            int start = response.indexOf("[");
            int end = response.lastIndexOf("]");
            if (start >= 0 && end > start) {
                String jsonStr = response.substring(start, end + 1);
                JSONArray arr = JSON.parseArray(jsonStr);
                for (int i = 0; i < Math.min(arr.size(), maxCount); i++) {
                    String variant = arr.getString(i);
                    if (variant != null && !variant.isEmpty()) {
                        variants.add(variant.trim());
                    }
                }
            }
        } catch (Exception e) {
            log.warn("[RAG Fusion] JSON 解析失败: {}", e.getMessage());
        }
        
        return variants;
    }
    
    /**
     * 规则生成查询变体（降级方案）
     */
    private List<String> generateRuleBasedVariants(String query, int count) {
        List<String> variants = new ArrayList<>();
        
        // 变体 1: 添加"如何"前缀
        if (!query.startsWith("如何")) {
            variants.add("如何 " + query);
        }
        
        // 变体 2: 添加"什么是"
        String[] terms = query.split("\\s+");
        if (terms.length > 0) {
            variants.add("什么是 " + terms[0]);
        }
        
        // 变体 3: 添加"教程"
        variants.add(query + " 教程");
        
        // 变体 4: 添加"代码"
        variants.add(query + " 代码实现");
        
        // 变体 5: 添加"示例"
        variants.add(query + " 示例");
        
        return variants.stream().limit(count).collect(Collectors.toList());
    }
    
    /**
     * 并行检索
     */
    private Map<String, List<RankedDoc>> parallelSearch(List<String> queries, int topK) {
        Map<String, List<RankedDoc>> results = new ConcurrentHashMap<>();
        List<CompletableFuture<Void>> futures = new ArrayList<>();
        
        for (String query : queries) {
            CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                try {
                    double[] embedding = getQueryEmbedding(query);
                    List<AiDoc> docs = hybridSearchService.hybridSearch(query, embedding, topK);
                    
                    List<RankedDoc> rankedDocs = new ArrayList<>();
                    for (int i = 0; i < docs.size(); i++) {
                        AiDoc doc = docs.get(i);
                        RankedDoc ranked = new RankedDoc();
                        ranked.id = doc.getId();
                        ranked.title = doc.getTitle();
                        ranked.content = doc.getContent();
                        ranked.category = doc.getCategory();
                        ranked.rank = i + 1;
                        ranked.sourceQuery = query;
                        rankedDocs.add(ranked);
                    }
                    
                    results.put(query, rankedDocs);
                } catch (Exception e) {
                    log.warn("[RAG Fusion] 检索失败 query='{}': {}", 
                            truncate(query, 30), e.getMessage());
                    results.put(query, Collections.emptyList());
                }
            }, executor);
            
            futures.add(future);
        }
        
        // 等待所有检索完成
        try {
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                    .get(30, TimeUnit.SECONDS);
        } catch (Exception e) {
            log.error("[RAG Fusion] 并行检索超时: {}", e.getMessage());
        }
        
        return results;
    }
    
    /**
     * RRF (Reciprocal Rank Fusion) 融合算法
     * 
     * score(d) = Σ 1 / (k + rank_i(d))
     * 
     * 对于每个文档 d，累加它在每个查询结果中的 RRF 分数
     */
    private List<RankedDoc> reciprocalRankFusion(
            Map<String, List<RankedDoc>> queryResults, int topK) {
        
        Map<String, RankedDoc> fusedMap = new HashMap<>();
        Map<String, Double> scoreMap = new HashMap<>();
        Map<String, Set<String>> sourceQueries = new HashMap<>();
        
        for (Map.Entry<String, List<RankedDoc>> entry : queryResults.entrySet()) {
            String query = entry.getKey();
            List<RankedDoc> docs = entry.getValue();
            
            for (RankedDoc doc : docs) {
                String docId = doc.id;
                
                // 累加 RRF 分数
                double rrfScore = 1.0 / (RRF_K + doc.rank);
                scoreMap.merge(docId, rrfScore, Double::sum);
                
                // 记录来源查询
                sourceQueries.computeIfAbsent(docId, k -> new HashSet<>()).add(query);
                
                // 保存文档信息
                fusedMap.putIfAbsent(docId, doc);
            }
        }
        
        // 按融合分数排序
        return scoreMap.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(topK)
                .map(entry -> {
                    RankedDoc doc = fusedMap.get(entry.getKey());
                    doc.fusedScore = entry.getValue();
                    doc.matchedQueries = sourceQueries.get(entry.getKey());
                    return doc;
                })
                .collect(Collectors.toList());
    }
    
    /**
     * 获取查询向量
     */
    private double[] getQueryEmbedding(String query) {
        if (embeddingClient != null && embeddingClient.enabled()) {
            return embeddingClient.embed(query);
        }
        // 返回零向量作为降级
        return new double[1536];
    }
    
    private String truncate(String text, int maxLen) {
        if (text == null) return "";
        return text.length() > maxLen ? text.substring(0, maxLen) + "..." : text;
    }
    
    // ===== 结果类 =====
    
    public static class RankedDoc {
        public String id;
        public String title;
        public String content;
        public String category;
        public int rank;           // 原始排名
        public String sourceQuery; // 来源查询
        public double fusedScore;  // RRF 融合分数
        public Set<String> matchedQueries; // 匹配的查询列表
        
        public AiDoc toAiDoc() {
            return new AiDoc(id, title, content, category);
        }
    }
    
    public static class RAGFusionResult {
        public String originalQuery;
        public List<String> generatedQueries;
        public Map<String, List<RankedDoc>> queryResults;
        public List<RankedDoc> fusedDocs;
        public List<AiDoc> documents;
        public long latencyMs;
        
        public Map<String, Object> toMap() {
            Map<String, Object> map = new HashMap<>();
            map.put("originalQuery", originalQuery);
            map.put("generatedQueries", generatedQueries);
            map.put("documentCount", documents != null ? documents.size() : 0);
            map.put("latencyMs", latencyMs);
            
            if (fusedDocs != null) {
                List<Map<String, Object>> docMaps = fusedDocs.stream()
                        .limit(5)
                        .map(d -> {
                            Map<String, Object> dm = new HashMap<>();
                            dm.put("id", d.id);
                            dm.put("title", d.title);
                            dm.put("fusedScore", String.format("%.4f", d.fusedScore));
                            dm.put("matchedQueryCount", d.matchedQueries != null ? d.matchedQueries.size() : 0);
                            return dm;
                        })
                        .collect(Collectors.toList());
                map.put("topDocuments", docMaps);
            }
            
            return map;
        }
    }
}
