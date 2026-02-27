package com.kob.backend.service.impl.ai;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.kob.backend.config.AiServiceProperties;
import com.kob.backend.controller.ai.dto.AiDoc;
import com.kob.backend.service.impl.ai.search.HybridSearchService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.*;

/**
 * RAG 路由器 - 根据配置决定调用 Java 或 Python 实现
 * 
 * 支持：
 * - 混合检索（hybrid-search）
 * - 重排序（rerank）
 * - 查询改写（query-rewrite）
 * 
 * 配置：
 * - ai.rag.backend=python 使用 Python 实现
 * - ai.rag.backend=java 使用 Java 实现（默认）
 * 
 * Phase 3 增强：
 * - 支持 A/B 测试路由
 * - 集成监控指标收集
 * - 自动降级机制
 */
@Service
public class RAGRouter {
    
    private static final Logger log = LoggerFactory.getLogger(RAGRouter.class);
    
    private final RestTemplate restTemplate;
    private final AiServiceProperties properties;
    private final HybridSearchService javaHybridSearch;
    private final ABTestRouter abTestRouter;
    private final AiMetricsCollector metricsCollector;
    private final ObjectMapper mapper = new ObjectMapper();
    
    @Value("${ai.rag.backend:java}")
    private String ragBackend;
    
    public RAGRouter(
            RestTemplate restTemplate,
            AiServiceProperties properties,
            HybridSearchService javaHybridSearch,
            ABTestRouter abTestRouter,
            AiMetricsCollector metricsCollector) {
        this.restTemplate = restTemplate;
        this.properties = properties;
        this.javaHybridSearch = javaHybridSearch;
        this.abTestRouter = abTestRouter;
        this.metricsCollector = metricsCollector;
    }
    
    /**
     * 判断是否使用 Python 后端
     */
    public boolean usePython() {
        return "python".equalsIgnoreCase(ragBackend) && properties.isEnabled();
    }
    
    /**
     * 混合检索（支持 A/B 测试）
     * 
     * @param query 查询文本
     * @param queryEmbedding 查询向量（可选）
     * @param topK 返回数量
     * @param userId 用户 ID（用于 A/B 测试路由）
     * @return 检索结果列表
     */
    public List<AiDoc> hybridSearch(String query, double[] queryEmbedding, int topK, Integer userId) {
        long startTime = System.currentTimeMillis();
        boolean success = false;
        String backend = "java";
        
        try {
            // A/B 测试路由决策
            ABTestRouter.RouteDecision decision = abTestRouter.routeRAG(userId);
            backend = decision.getValue();
            
            List<AiDoc> results;
            if (decision == ABTestRouter.RouteDecision.PYTHON && properties.isEnabled()) {
                results = pythonHybridSearchWithFallback(query, queryEmbedding, topK);
            } else {
                results = javaHybridSearch.hybridSearch(query, queryEmbedding, topK);
            }
            
            success = true;
            return results;
        } finally {
            long latency = System.currentTimeMillis() - startTime;
            metricsCollector.recordRAGRequest(backend, latency, success);
        }
    }
    
    /**
     * 混合检索（兼容旧接口）
     */
    public List<AiDoc> hybridSearch(String query, double[] queryEmbedding, int topK) {
        return hybridSearch(query, queryEmbedding, topK, null);
    }
    
    /**
     * Python 检索带自动降级
     */
    private List<AiDoc> pythonHybridSearchWithFallback(String query, double[] queryEmbedding, int topK) {
        try {
            return pythonHybridSearch(query, queryEmbedding, topK);
        } catch (Exception e) {
            log.warn("Python RAG 调用失败，降级到 Java: {}", e.getMessage());
            metricsCollector.recordFallback("rag", "python", "java", e.getMessage());
            abTestRouter.recordFallback("rag", e.getMessage());
            return javaHybridSearch.hybridSearch(query, queryEmbedding, topK);
        }
    }
    
    /**
     * 调用 Python 混合检索
     */
    private List<AiDoc> pythonHybridSearch(String query, double[] queryEmbedding, int topK) {
        try {
            String url = properties.getBaseUrl() + "/api/rag/hybrid-search";
            
            Map<String, Object> request = new HashMap<>();
            request.put("query", query);
            request.put("top_k", topK);
            request.put("trace_id", UUID.randomUUID().toString());
            if (queryEmbedding != null) {
                request.put("queryEmbedding", queryEmbedding);
            }
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<Map<String, Object>> entity = new HttpEntity<>(request, headers);
            
            ResponseEntity<String> response = restTemplate.exchange(
                url, HttpMethod.POST, entity, String.class);
            
            JsonNode root = mapper.readTree(response.getBody());
            if (!root.path("success").asBoolean(false)) {
                log.warn("Python RAG 失败，降级到 Java: {}", root.path("error"));
                return javaHybridSearch.hybridSearch(query, queryEmbedding, topK);
            }
            
            List<AiDoc> results = new ArrayList<>();
            JsonNode resultsNode = root.path("results");
            if (resultsNode.isArray()) {
                for (JsonNode node : resultsNode) {
                    results.add(new AiDoc(
                        node.path("id").asText(),
                        node.path("title").asText(),
                        node.path("content").asText(),
                        node.path("category").asText()
                    ));
                }
            }
            
            log.debug("Python RAG 返回 {} 结果", results.size());
            return results;
        } catch (Exception e) {
            log.warn("Python RAG 调用失败，降级到 Java: {}", e.getMessage());
            return javaHybridSearch.hybridSearch(query, queryEmbedding, topK);
        }
    }
    
    /**
     * 重排序
     * 
     * @param query 查询文本
     * @param documents 文档列表
     * @param topK 返回数量
     * @return 重排序后的文档列表
     */
    public List<AiDoc> rerank(String query, List<AiDoc> documents, Integer topK) {
        if (!usePython()) {
            // Java 端暂无重排序实现，直接返回原列表
            return documents;
        }
        
        try {
            String url = properties.getBaseUrl() + "/api/rag/rerank";
            
            List<Map<String, Object>> docMaps = new ArrayList<>();
            for (AiDoc doc : documents) {
                Map<String, Object> docMap = new HashMap<>();
                docMap.put("id", doc.getId());
                docMap.put("title", doc.getTitle());
                docMap.put("content", doc.getContent());
                docMap.put("category", doc.getCategory());
                docMaps.add(docMap);
            }
            
            Map<String, Object> request = new HashMap<>();
            request.put("query", query);
            request.put("documents", docMaps);
            request.put("trace_id", UUID.randomUUID().toString());
            if (topK != null) {
                request.put("top_k", topK);
            }
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<Map<String, Object>> entity = new HttpEntity<>(request, headers);
            
            ResponseEntity<String> response = restTemplate.exchange(
                url, HttpMethod.POST, entity, String.class);
            
            JsonNode root = mapper.readTree(response.getBody());
            if (!root.path("success").asBoolean(false)) {
                log.warn("Python Rerank 失败: {}", root.path("error"));
                return documents;
            }
            
            List<AiDoc> results = new ArrayList<>();
            JsonNode resultsNode = root.path("results");
            if (resultsNode.isArray()) {
                for (JsonNode node : resultsNode) {
                    results.add(new AiDoc(
                        node.path("id").asText(),
                        node.path("title").asText(),
                        node.path("content").asText(),
                        node.path("category").asText()
                    ));
                }
            }
            
            return results;
        } catch (Exception e) {
            log.warn("Python Rerank 调用失败: {}", e.getMessage());
            return documents;
        }
    }
    
    /**
     * 查询改写
     * 
     * @param query 原始查询
     * @return 改写结果（包含 rewrittenQuery 和 expandedQueries）
     */
    public QueryRewriteResult rewriteQuery(String query) {
        if (!usePython()) {
            // Java 端暂无查询改写实现
            return new QueryRewriteResult(query, query, Collections.emptyList(), "search");
        }
        
        try {
            String url = properties.getBaseUrl() + "/api/rag/query-rewrite";
            
            Map<String, Object> request = new HashMap<>();
            request.put("query", query);
            request.put("trace_id", UUID.randomUUID().toString());
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<Map<String, Object>> entity = new HttpEntity<>(request, headers);
            
            ResponseEntity<String> response = restTemplate.exchange(
                url, HttpMethod.POST, entity, String.class);
            
            JsonNode root = mapper.readTree(response.getBody());
            if (!root.path("success").asBoolean(false)) {
                log.warn("Python Query Rewrite 失败: {}", root.path("error"));
                return new QueryRewriteResult(query, query, Collections.emptyList(), "search");
            }
            
            String rewritten = root.path("rewritten_query").asText(query);
            String intent = root.path("intent").asText("search");
            
            List<String> expanded = new ArrayList<>();
            JsonNode expandedNode = root.path("expanded_queries");
            if (expandedNode.isArray()) {
                for (JsonNode node : expandedNode) {
                    expanded.add(node.asText());
                }
            }
            
            return new QueryRewriteResult(query, rewritten, expanded, intent);
        } catch (Exception e) {
            log.warn("Python Query Rewrite 调用失败: {}", e.getMessage());
            return new QueryRewriteResult(query, query, Collections.emptyList(), "search");
        }
    }
    
    /**
     * 查询改写结果
     */
    public static class QueryRewriteResult {
        private final String originalQuery;
        private final String rewrittenQuery;
        private final List<String> expandedQueries;
        private final String intent;
        
        public QueryRewriteResult(String originalQuery, String rewrittenQuery, 
                                   List<String> expandedQueries, String intent) {
            this.originalQuery = originalQuery;
            this.rewrittenQuery = rewrittenQuery;
            this.expandedQueries = expandedQueries;
            this.intent = intent;
        }
        
        public String getOriginalQuery() { return originalQuery; }
        public String getRewrittenQuery() { return rewrittenQuery; }
        public List<String> getExpandedQueries() { return expandedQueries; }
        public String getIntent() { return intent; }
    }
}
