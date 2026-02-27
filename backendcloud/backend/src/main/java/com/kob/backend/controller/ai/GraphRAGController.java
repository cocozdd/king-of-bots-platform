package com.kob.backend.controller.ai;

import com.kob.backend.controller.ai.dto.AiDoc;
import com.kob.backend.controller.ai.dto.AiHintRequest;
import com.kob.backend.service.impl.ai.*;
import com.kob.backend.service.impl.ai.search.HybridSearchService;
import com.kob.backend.service.impl.ai.graphrag.GraphRAGService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.annotation.PostConstruct;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

/**
 * GraphRAG 控制器
 * 
 * 提供 GraphRAG 相关的 API：
 * - GraphRAG 智能问答（Local + Global Search）
 * - 传统 RAG vs GraphRAG 对比测试
 * - 知识图谱状态查询
 */
@RestController
@RequestMapping("/ai/graphrag")
public class GraphRAGController {
    
    private static final Logger log = LoggerFactory.getLogger(GraphRAGController.class);
    
    @Autowired
    private GraphRAGService graphRAGService;
    
    @Autowired
    private HybridSearchService hybridSearchService;
    
    @Autowired
    private RerankService rerankService;
    
    @Autowired
    private QueryRewriteService queryRewriteService;
    
    @Autowired
    private AiMetricsService metricsService;
    
    @Autowired
    private PromptSecurityService securityService;
    
    @Autowired
    private EmbeddingCacheService cacheService;
    
    @Value("${dashscope.api.key:}")
    private String dashscopeApiKey;
    
    private DashscopeEmbeddingClient dashscopeClient;
    private DeepseekClient deepseekClient;
    
    private final ExecutorService executor = Executors.newFixedThreadPool(4);
    
    @PostConstruct
    public void init() {
        String key = dashscopeApiKey;
        if (key == null || key.isEmpty()) {
            key = System.getenv("DASHSCOPE_API_KEY");
        }
        dashscopeClient = new DashscopeEmbeddingClient(key, metricsService, cacheService);
        deepseekClient = new DeepseekClient(metricsService);
        log.info("GraphRAG Controller 初始化完成");
    }
    
    /**
     * GraphRAG 智能问答
     * 
     * 自动选择 Local 或 Global Search
     */
    @PostMapping("/ask")
    public CompletableFuture<ResponseEntity<Map<String, Object>>> ask(@RequestBody AiHintRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            Map<String, Object> result = new HashMap<>();
            long startTime = System.currentTimeMillis();
            
            try {
                // 1. 安全验证
                PromptSecurityService.ValidationResult validation = 
                        securityService.validateQuestion(request.getQuestion());
                if (!validation.isSafe()) {
                    result.put("success", false);
                    result.put("error", "输入不合法: " + validation.getReason());
                    return ResponseEntity.ok(result);
                }
                
                String query = validation.getSanitizedInput();
                
                // 2. 生成 Embedding
                if (!dashscopeClient.enabled()) {
                    result.put("success", false);
                    result.put("error", "Embedding 服务未配置");
                    return ResponseEntity.ok(result);
                }
                double[] embedding = dashscopeClient.embed(query);
                
                // 3. GraphRAG 检索
                GraphRAGService.GraphRAGResult graphResult = 
                        graphRAGService.search(query, embedding, 5);
                
                // 4. 生成回答
                if (!deepseekClient.enabled()) {
                    result.put("success", false);
                    result.put("error", "Chat 服务未配置");
                    return ResponseEntity.ok(result);
                }
                
                // 构建增强的上下文（文档内容 + 图谱上下文）
                List<String> contexts = graphResult.getDocuments().stream()
                        .map(AiDoc::getContent)
                        .collect(Collectors.toList());
                
                // 添加图谱上下文
                if (graphResult.getGraphContext() != null) {
                    contexts.add(0, graphResult.getGraphContext());
                }
                
                String systemPrompt = """
                    你是 Bot 开发助手，基于提供的参考文档和知识图谱信息回答问题。
                    
                    格式要求：
                    - 使用 Markdown 格式输出
                    - 段落之间用空行分隔
                    - 使用 ## 作为二级标题，### 作为三级标题
                    - 代码必须用 ```语言名 和 ``` 包裹，每行代码单独一行
                    - 列表项之间适当换行
                    
                    内容要求：
                    - 回答要简洁准确
                    - 充分利用图谱中的概念关系
                    - 引用相关文档
                    - 不知道就说不知道
                    """;
                
                String answer = deepseekClient.chat(systemPrompt, query, contexts);
                
                // 5. 构建响应
                List<Map<String, String>> sources = graphResult.getDocuments().stream()
                        .map(doc -> {
                            Map<String, String> source = new HashMap<>();
                            source.put("id", doc.getId());
                            source.put("title", doc.getTitle());
                            source.put("category", doc.getCategory());
                            return source;
                        })
                        .collect(Collectors.toList());
                
                long latency = System.currentTimeMillis() - startTime;
                
                result.put("success", true);
                result.put("answer", answer);
                result.put("sources", sources);
                result.put("searchType", graphResult.getSearchType());
                result.put("extractedEntities", graphResult.getExtractedEntities());
                result.put("relevantEntities", graphResult.getRelevantEntities());
                result.put("graphContext", graphResult.getGraphContext());
                result.put("latencyMs", latency);
                result.put("graphLatencyMs", graphResult.getLatencyMs());
                
                log.info("GraphRAG 问答完成: type={}, 检索{}篇, 实体{}个, 耗时{}ms", 
                        graphResult.getSearchType(), 
                        graphResult.getDocuments().size(),
                        graphResult.getRelevantEntities().size(),
                        latency);
                
            } catch (Exception e) {
                log.error("GraphRAG 问答失败: {}", e.getMessage(), e);
                result.put("success", false);
                result.put("error", "服务暂时不可用: " + e.getMessage());
            }
            
            return ResponseEntity.ok(result);
        }, executor);
    }
    
    /**
     * 对比测试：传统 RAG vs GraphRAG
     */
    @PostMapping("/compare")
    public CompletableFuture<ResponseEntity<Map<String, Object>>> compare(@RequestBody AiHintRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            Map<String, Object> result = new HashMap<>();
            
            try {
                // 安全验证
                PromptSecurityService.ValidationResult validation = 
                        securityService.validateQuestion(request.getQuestion());
                if (!validation.isSafe()) {
                    result.put("success", false);
                    result.put("error", "输入不合法");
                    return ResponseEntity.ok(result);
                }
                
                String query = validation.getSanitizedInput();
                
                if (!dashscopeClient.enabled() || !deepseekClient.enabled()) {
                    result.put("success", false);
                    result.put("error", "AI 服务未配置");
                    return ResponseEntity.ok(result);
                }
                
                double[] embedding = dashscopeClient.embed(query);
                
                // ==================== 传统 RAG ====================
                long ragStart = System.currentTimeMillis();
                
                List<AiDoc> ragCandidates = hybridSearchService.hybridSearch(query, embedding, 20);
                List<AiDoc> ragDocs = rerankService.rerank(query, ragCandidates, 5);
                
                List<String> ragContexts = ragDocs.stream()
                        .map(AiDoc::getContent)
                        .collect(Collectors.toList());
                
                String ragAnswer = deepseekClient.chat(
                        "你是 Bot 开发助手，简洁回答问题。",
                        query,
                        ragContexts
                );
                
                long ragLatency = System.currentTimeMillis() - ragStart;
                
                // ==================== GraphRAG ====================
                long graphStart = System.currentTimeMillis();
                
                GraphRAGService.GraphRAGResult graphResult = 
                        graphRAGService.search(query, embedding, 5);
                
                List<String> graphContexts = graphResult.getDocuments().stream()
                        .map(AiDoc::getContent)
                        .collect(Collectors.toList());
                
                if (graphResult.getGraphContext() != null) {
                    graphContexts.add(0, graphResult.getGraphContext());
                }
                
                String graphAnswer = deepseekClient.chat(
                        "你是 Bot 开发助手，基于文档和知识图谱回答问题。",
                        query,
                        graphContexts
                );
                
                long graphLatency = System.currentTimeMillis() - graphStart;
                
                // ==================== 构建对比结果 ====================
                Map<String, Object> ragResult = new HashMap<>();
                ragResult.put("answer", ragAnswer);
                ragResult.put("latencyMs", ragLatency);
                ragResult.put("docsCount", ragDocs.size());
                ragResult.put("sources", ragDocs.stream()
                        .map(d -> Map.of("id", d.getId(), "title", d.getTitle()))
                        .collect(Collectors.toList()));
                
                Map<String, Object> graphRAGResult = new HashMap<>();
                graphRAGResult.put("answer", graphAnswer);
                graphRAGResult.put("latencyMs", graphLatency);
                graphRAGResult.put("searchType", graphResult.getSearchType());
                graphRAGResult.put("docsCount", graphResult.getDocuments().size());
                graphRAGResult.put("entitiesCount", graphResult.getRelevantEntities().size());
                graphRAGResult.put("extractedEntities", graphResult.getExtractedEntities());
                graphRAGResult.put("graphContext", graphResult.getGraphContext());
                graphRAGResult.put("sources", graphResult.getDocuments().stream()
                        .map(d -> Map.of("id", d.getId(), "title", d.getTitle()))
                        .collect(Collectors.toList()));
                
                result.put("success", true);
                result.put("query", query);
                result.put("traditionalRAG", ragResult);
                result.put("graphRAG", graphRAGResult);
                result.put("comparison", Map.of(
                        "latencyDiffMs", graphLatency - ragLatency,
                        "graphRAGHasMoreContext", graphResult.getGraphContext() != null,
                        "entitiesDiscovered", graphResult.getRelevantEntities().size()
                ));
                
                log.info("RAG 对比完成: 传统RAG {}ms, GraphRAG {}ms", ragLatency, graphLatency);
                
            } catch (Exception e) {
                log.error("对比测试失败: {}", e.getMessage(), e);
                result.put("success", false);
                result.put("error", "服务暂时不可用: " + e.getMessage());
            }
            
            return ResponseEntity.ok(result);
        }, executor);
    }
    
    /**
     * 构建/重建知识图谱
     */
    @PostMapping("/build")
    public ResponseEntity<Map<String, Object>> buildGraph() {
        Map<String, Object> result = new HashMap<>();
        try {
            long startTime = System.currentTimeMillis();
            graphRAGService.buildGraph();
            long elapsed = System.currentTimeMillis() - startTime;
            
            result.put("success", true);
            result.put("message", "知识图谱构建完成");
            result.put("latencyMs", elapsed);
            result.put("stats", graphRAGService.getStats());
        } catch (Exception e) {
            log.error("构建知识图谱失败: {}", e.getMessage(), e);
            result.put("success", false);
            result.put("error", e.getMessage());
        }
        return ResponseEntity.ok(result);
    }
    
    /**
     * 获取知识图谱状态
     */
    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> getStatus() {
        Map<String, Object> status = new HashMap<>();
        status.put("embeddingEnabled", dashscopeClient != null && dashscopeClient.enabled());
        status.put("chatEnabled", deepseekClient != null && deepseekClient.enabled());
        status.put("graphStats", graphRAGService.getStats());
        return ResponseEntity.ok(status);
    }
}
