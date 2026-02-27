package com.kob.backend.controller.ai;

import com.kob.backend.controller.ai.dto.AiDoc;
import com.kob.backend.service.impl.ai.*;
import com.kob.backend.service.impl.ai.crag.*;
import com.kob.backend.service.impl.ai.search.HybridSearchService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import com.kob.backend.service.impl.ai.PromptSecurityService;

import javax.annotation.PostConstruct;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

/**
 * CRAG (Corrective RAG) 控制器
 * 
 * 提供:
 * - CRAG vs 传统 RAG 对比
 * - 查询路由分析
 * - RAG 质量评估
 * 
 * 面试演示要点:
 * - CRAG 如何修正低质量检索
 * - 查询路由如何优化延迟
 * - 评估指标如何量化 RAG 质量
 */
@RestController
@RequestMapping("/ai/crag")
public class CRAGController {
    
    private static final Logger log = LoggerFactory.getLogger(CRAGController.class);
    
    @Autowired
    private CRAGService cragService;
    
    @Autowired
    private QueryRouter queryRouter;
    
    @Autowired
    private RAGEvaluator ragEvaluator;
    
    @Autowired
    private HybridSearchService hybridSearchService;
    
    @Autowired(required = false)
    private AiMetricsService metricsService;
    
    @Autowired(required = false)
    private EmbeddingCacheService cacheService;

    @Autowired
    private PromptSecurityService securityService;

    @Value("${dashscope.api.key:}")
    private String dashscopeApiKey;
    
    private DashscopeEmbeddingClient dashscopeClient;
    private DeepseekClient deepseekClient;
    private final ExecutorService executor = Executors.newFixedThreadPool(2);
    
    @PostConstruct
    public void init() {
        String key = dashscopeApiKey;
        if (key == null || key.isEmpty()) {
            key = System.getenv("DASHSCOPE_API_KEY");
        }
        dashscopeClient = new DashscopeEmbeddingClient(key, metricsService, cacheService);
        deepseekClient = new DeepseekClient(metricsService);
        log.info("CRAG Controller 初始化完成");
    }
    
    /**
     * CRAG vs 传统 RAG 对比
     * 
     * 演示: 相同问题，两种方法的效果差异
     */
    @PostMapping("/compare")
    public CompletableFuture<ResponseEntity<Map<String, Object>>> compare(
            @RequestBody Map<String, String> request) {

        return CompletableFuture.supplyAsync(() -> {
            Map<String, Object> result = new HashMap<>();
            String question = request.get("question");

            if (question == null || question.trim().isEmpty()) {
                result.put("success", false);
                result.put("error", "请提供问题");
                return ResponseEntity.ok(result);
            }

            // 安全验证
            PromptSecurityService.SecurityCheckResult securityCheck = securityService.performFullSecurityCheck(question);
            if (!securityCheck.isPassed()) {
                result.put("success", false);
                result.put("error", securityCheck.getRejectReason());
                result.put("rejectType", securityCheck.getRejectType());
                return ResponseEntity.ok(result);
            }

            try {
                // 获取 embedding
                double[] embedding = dashscopeClient.enabled() ? 
                        dashscopeClient.embed(question) : new double[0];
                
                // 1. 传统 RAG
                long tradStart = System.currentTimeMillis();
                List<AiDoc> tradDocs = hybridSearchService.hybridSearch(question, embedding, 5);
                List<String> tradContexts = tradDocs.stream()
                        .map(AiDoc::getContent)
                        .collect(Collectors.toList());
                String tradSystemPrompt = """
                        你是 Bot 开发助手，基于提供的参考文档回答问题。
                        
                        格式要求：
                        - 使用 Markdown 格式输出
                        - 段落之间用空行分隔
                        - 使用 ## 作为二级标题，### 作为三级标题
                        - 代码必须用 ```语言名 和 ``` 包裹，每行代码单独一行
                        - 列表项之间适当换行
                        
                        内容要求：
                        - 回答要简洁准确
                        - 引用相关文档
                        - 不知道就说不知道
                        """;
                String tradAnswer = deepseekClient.chat(tradSystemPrompt, question, tradContexts);
                long tradLatency = System.currentTimeMillis() - tradStart;
                
                // 评估传统 RAG
                RAGEvaluator.EvaluationResult tradEval = 
                        ragEvaluator.evaluate(question, tradAnswer, tradDocs);
                
                Map<String, Object> tradResult = new HashMap<>();
                tradResult.put("answer", tradAnswer);
                tradResult.put("docsCount", tradDocs.size());
                tradResult.put("latencyMs", tradLatency);
                tradResult.put("evaluation", tradEval.toScoreMap());
                tradResult.put("evalSummary", tradEval.summary);
                
                // 2. CRAG
                long cragStart = System.currentTimeMillis();
                CRAGService.CRAGResult cragResult = cragService.process(question, embedding);
                long cragLatency = System.currentTimeMillis() - cragStart;
                
                // 评估 CRAG
                RAGEvaluator.EvaluationResult cragEval = 
                        ragEvaluator.evaluate(question, cragResult.answer, cragResult.finalDocs);
                
                Map<String, Object> cragResultMap = new HashMap<>();
                cragResultMap.put("answer", cragResult.answer);
                cragResultMap.put("action", cragResult.action.name());
                cragResultMap.put("steps", cragResult.steps);
                cragResultMap.put("docsCount", cragResult.finalDocs.size());
                cragResultMap.put("avgRelevance", cragResult.avgRelevance);
                cragResultMap.put("latencyMs", cragLatency);
                cragResultMap.put("evaluation", cragEval.toScoreMap());
                cragResultMap.put("evalSummary", cragEval.summary);
                
                // 3. 对比分析
                Map<String, Object> comparison = new HashMap<>();
                comparison.put("qualityImprovement", 
                        String.format("%.1f%%", (cragEval.overallScore - tradEval.overallScore) * 100));
                comparison.put("latencyDiff", cragLatency - tradLatency);
                comparison.put("cragAction", cragResult.action.name());
                comparison.put("recommendation", generateRecommendation(tradEval, cragEval, cragResult));
                
                result.put("success", true);
                result.put("query", question);
                result.put("traditionalRAG", tradResult);
                result.put("crag", cragResultMap);
                result.put("comparison", comparison);
                
            } catch (Exception e) {
                log.error("CRAG 对比失败", e);
                result.put("success", false);
                result.put("error", e.getMessage());
            }
            
            return ResponseEntity.ok(result);
        }, executor);
    }
    
    /**
     * 查询路由分析
     */
    @PostMapping("/route")
    public ResponseEntity<Map<String, Object>> analyzeRoute(
            @RequestBody Map<String, String> request) {

        Map<String, Object> result = new HashMap<>();
        String question = request.get("question");

        if (question == null || question.trim().isEmpty()) {
            result.put("success", false);
            result.put("error", "请提供问题");
            return ResponseEntity.ok(result);
        }

        // 安全验证
        PromptSecurityService.SecurityCheckResult securityCheck = securityService.performFullSecurityCheck(question);
        if (!securityCheck.isPassed()) {
            result.put("success", false);
            result.put("error", securityCheck.getRejectReason());
            result.put("rejectType", securityCheck.getRejectType());
            return ResponseEntity.ok(result);
        }

        QueryRouter.RouteDecision decision = queryRouter.route(question);
        
        result.put("success", true);
        result.put("query", question);
        result.put("route", decision.route.name());
        result.put("reason", decision.reason);
        result.put("confidence", decision.confidence);
        result.put("features", decision.features);
        
        // 添加路由建议
        result.put("suggestion", getRouteSuggestion(decision.route));
        
        return ResponseEntity.ok(result);
    }
    
    /**
     * RAG 质量评估
     */
    @PostMapping("/evaluate")
    public CompletableFuture<ResponseEntity<Map<String, Object>>> evaluate(
            @RequestBody Map<String, String> request) {

        return CompletableFuture.supplyAsync(() -> {
            Map<String, Object> result = new HashMap<>();

            String question = request.get("question");
            String answer = request.get("answer");

            if (question == null || answer == null) {
                result.put("success", false);
                result.put("error", "请提供 question 和 answer");
                return ResponseEntity.ok(result);
            }

            // 安全验证
            PromptSecurityService.ValidationResult questionValidation = securityService.validateQuestion(question);
            if (!questionValidation.isSafe()) {
                result.put("success", false);
                result.put("error", questionValidation.getReason());
                return ResponseEntity.ok(result);
            }

            try {
                // 获取相关文档用于评估
                double[] embedding = dashscopeClient.enabled() ? 
                        dashscopeClient.embed(question) : new double[0];
                List<AiDoc> docs = hybridSearchService.hybridSearch(question, embedding, 5);
                
                RAGEvaluator.EvaluationResult eval = ragEvaluator.evaluate(question, answer, docs);
                
                result.put("success", true);
                result.put("query", question);
                result.put("scores", eval.toScoreMap());
                result.put("summary", eval.summary);
                result.put("details", Map.of(
                        "contextRelevance", describeScore("检索相关性", eval.contextRelevance),
                        "answerRelevance", describeScore("答案相关性", eval.answerRelevance),
                        "faithfulness", describeScore("忠实度(非幻觉)", eval.faithfulness),
                        "contextUtilization", describeScore("上下文利用率", eval.contextUtilization)
                ));
                
            } catch (Exception e) {
                log.error("RAG 评估失败", e);
                result.put("success", false);
                result.put("error", e.getMessage());
            }
            
            return ResponseEntity.ok(result);
        }, executor);
    }
    
    /**
     * 获取 CRAG 系统状态
     */
    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> getStatus() {
        Map<String, Object> status = new HashMap<>();
        
        status.put("cragEnabled", true);
        status.put("queryRouterEnabled", true);
        status.put("evaluatorEnabled", true);
        status.put("embeddingEnabled", dashscopeClient.enabled());
        status.put("llmEnabled", deepseekClient.enabled());
        
        status.put("thresholds", Map.of(
                "correctThreshold", 0.7,
                "incorrectThreshold", 0.3
        ));
        
        status.put("routeTypes", Arrays.stream(QueryRouter.RouteType.values())
                .map(Enum::name)
                .collect(Collectors.toList()));
        
        return ResponseEntity.ok(status);
    }
    
    // ===== 辅助方法 =====
    
    private String generateRecommendation(RAGEvaluator.EvaluationResult trad,
                                          RAGEvaluator.EvaluationResult crag,
                                          CRAGService.CRAGResult cragResult) {
        double improvement = crag.overallScore - trad.overallScore;
        
        if (improvement > 0.1) {
            return String.format("CRAG 显著提升质量 (+%.0f%%)，建议使用 CRAG", improvement * 100);
        } else if (improvement > 0) {
            return "CRAG 略有提升，对质量敏感场景建议使用";
        } else if (cragResult.action == CRAGService.CRAGAction.CORRECT) {
            return "检索质量高，传统 RAG 已足够";
        } else {
            return "两种方法效果相近，可根据延迟需求选择";
        }
    }
    
    private String getRouteSuggestion(QueryRouter.RouteType route) {
        return switch (route) {
            case NO_RETRIEVAL -> "简单问题，直接使用 LLM 回答，无需检索";
            case VECTOR_SEARCH -> "语义检索适合模糊匹配和概念理解";
            case KEYWORD_SEARCH -> "关键词检索适合精确匹配";
            case HYBRID_SEARCH -> "混合检索兼顾语义和精确匹配";
            case GRAPH_SEARCH -> "图谱检索适合关系推理和多跳问答";
            case WEB_SEARCH -> "需要实时信息，建议接入搜索引擎";
        };
    }
    
    private String describeScore(String name, double score) {
        String level = score >= 0.8 ? "优秀" : score >= 0.6 ? "良好" : score >= 0.4 ? "一般" : "较差";
        return String.format("%s: %.2f (%s)", name, score, level);
    }
}
