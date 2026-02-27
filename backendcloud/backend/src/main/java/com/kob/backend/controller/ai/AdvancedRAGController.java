package com.kob.backend.controller.ai;

import com.kob.backend.controller.ai.dto.AiDoc;
import com.kob.backend.service.impl.ai.DashscopeEmbeddingClient;
import com.kob.backend.service.impl.ai.AiMetricsService;
import com.kob.backend.service.impl.ai.EmbeddingCacheService;
import com.kob.backend.service.impl.ai.PromptSecurityService;
import org.springframework.beans.factory.annotation.Value;
import com.kob.backend.service.impl.ai.agentic.AgenticRAGService;
import com.kob.backend.service.impl.ai.speculative.SpeculativeRAGService;
import com.kob.backend.service.impl.ai.evaluation.RAGEvaluationPipeline;
import com.kob.backend.service.impl.ai.longcontext.LongContextRAGService;
import com.kob.backend.service.impl.ai.search.HybridSearchService;
import com.kob.backend.service.impl.ai.fusion.RAGFusionService;
import com.kob.backend.service.impl.ai.ragas.RAGASEvaluationService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 高级 RAG API 控制器
 *
 * 提供 2024-2026 年前沿 RAG 技术的 API 接口:
 * - Agentic RAG: Agent 自主决策检索策略
 * - Speculative RAG: 多路并行生成 + 选优
 * - RAGAS 评估: 标准化 RAG 质量评估
 * - Long-Context + RAG: 自适应策略选择
 */
@RestController
@RequestMapping("/ai/advanced")
public class AdvancedRAGController {

    private static final Logger log = LoggerFactory.getLogger(AdvancedRAGController.class);

    @Autowired
    private AgenticRAGService agenticRAGService;

    @Autowired
    private SpeculativeRAGService speculativeRAGService;

    @Autowired
    private RAGEvaluationPipeline evaluationPipeline;

    @Autowired
    private LongContextRAGService longContextRAGService;

    @Autowired
    private HybridSearchService hybridSearchService;

    @Autowired(required = false)
    private RAGFusionService ragFusionService;

    @Autowired(required = false)
    private RAGASEvaluationService ragasEvaluationService;

    @Autowired(required = false)
    private AiMetricsService metricsService;

    @Autowired(required = false)
    private EmbeddingCacheService embeddingCacheService;

    @Autowired
    private PromptSecurityService securityService;

    @Value("${dashscope.api.key:}")
    private String dashscopeApiKey;

    private DashscopeEmbeddingClient embeddingClient;

    @Autowired
    public void setEmbeddingClient() {
        this.embeddingClient = new DashscopeEmbeddingClient(dashscopeApiKey, metricsService, embeddingCacheService);
    }

    /**
     * Agentic RAG - Agent 自主决策检索
     *
     * Agent 会自动分析问题，决定:
     * - 是否需要检索
     * - 检索什么内容
     * - 何时停止检索并生成答案
     */
    @PostMapping("/agentic")
    public ResponseEntity<?> agenticRAG(@RequestBody Map<String, String> request) {
        String query = request.get("query");
        if (query == null || query.trim().isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "query 不能为空"));
        }

        log.info("[API] Agentic RAG 请求: {}", truncate(query, 50));

        try {
            // 获取查询向量
            double[] embedding = embeddingClient.embed(query);

            // 执行 Agentic RAG
            AgenticRAGService.AgenticRAGResult result = agenticRAGService.process(query, embedding);

            return ResponseEntity.ok(result.toMap());
        } catch (Exception e) {
            log.error("[API] Agentic RAG 错误: {}", e.getMessage());
            return ResponseEntity.internalServerError()
                .body(Map.of("error", e.getMessage()));
        }
    }

    /**
     * Speculative RAG - 多路并行生成 + 选优
     *
     * 并行使用多种策略生成答案，然后选择最优结果
     */
    @PostMapping("/speculative")
    public ResponseEntity<?> speculativeRAG(@RequestBody Map<String, String> request) {
        String query = request.get("query");
        if (query == null || query.trim().isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "query 不能为空"));
        }

        // 安全验证
        PromptSecurityService.SecurityCheckResult securityCheck = securityService.performFullSecurityCheck(query);
        if (!securityCheck.isPassed()) {
            return ResponseEntity.ok(Map.of(
                "error", securityCheck.getRejectReason(),
                "rejectType", securityCheck.getRejectType()
            ));
        }

        log.info("[API] Speculative RAG 请求: {}", truncate(query, 50));

        try {
            double[] embedding = embeddingClient.embed(query);

            SpeculativeRAGService.SpeculativeRAGResult result =
                speculativeRAGService.process(query, embedding);

            return ResponseEntity.ok(result.toMap());
        } catch (Exception e) {
            log.error("[API] Speculative RAG 错误: {}", e.getMessage());
            return ResponseEntity.internalServerError()
                .body(Map.of("error", e.getMessage()));
        }
    }

    /**
     * Long-Context + RAG 自适应策略
     *
     * 根据知识库大小和查询特征，自动选择最优策略:
     * - 全上下文
     * - 分块处理
     * - 混合模式
     * - 纯 RAG
     */
    @PostMapping("/longcontext")
    public ResponseEntity<?> longContextRAG(@RequestBody Map<String, Object> request) {
        String query = (String) request.get("query");
        if (query == null || query.trim().isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "query 不能为空"));
        }

        // 安全验证
        PromptSecurityService.SecurityCheckResult securityCheck = securityService.performFullSecurityCheck(query);
        if (!securityCheck.isPassed()) {
            return ResponseEntity.ok(Map.of(
                "error", securityCheck.getRejectReason(),
                "rejectType", securityCheck.getRejectType()
            ));
        }

        log.info("[API] Long-Context RAG 请求: {}", truncate(query, 50));

        try {
            double[] embedding = embeddingClient.embed(query);

            // 获取知识库（这里使用检索获取，实际可以从数据库加载全部）
            List<AiDoc> knowledgeBase = hybridSearchService.hybridSearch(query, embedding, 50);

            LongContextRAGService.LongContextRAGResult result =
                longContextRAGService.process(query, embedding, knowledgeBase);

            return ResponseEntity.ok(result.toMap());
        } catch (Exception e) {
            log.error("[API] Long-Context RAG 错误: {}", e.getMessage());
            return ResponseEntity.internalServerError()
                .body(Map.of("error", e.getMessage()));
        }
    }

    /**
     * RAGAS 评估 - 单个样本
     */
    @PostMapping("/evaluate")
    public ResponseEntity<?> evaluateRAGAS(@RequestBody Map<String, Object> request) {
        String query = (String) request.get("query");
        String answer = (String) request.get("answer");
        @SuppressWarnings("unchecked")
        List<String> contexts = (List<String>) request.get("contexts");

        if (query == null || answer == null || contexts == null) {
            return ResponseEntity.badRequest()
                .body(Map.of("error", "需要 query, answer, contexts 参数"));
        }

        log.info("[API] RAGAS 评估请求");

        try {
            RAGEvaluationPipeline.EvaluationInput input = new RAGEvaluationPipeline.EvaluationInput();
            input.query = query;
            input.answer = answer;
            input.contexts = contexts;
            input.groundTruthAnswer = (String) request.get("groundTruthAnswer");
            input.groundTruthContext = (String) request.get("groundTruthContext");

            RAGEvaluationPipeline.RAGASResult result = evaluationPipeline.evaluateRAGAS(input);

            return ResponseEntity.ok(result.toMap());
        } catch (Exception e) {
            log.error("[API] RAGAS 评估错误: {}", e.getMessage());
            return ResponseEntity.internalServerError()
                .body(Map.of("error", e.getMessage()));
        }
    }

    /**
     * RAGAS 批量评估
     */
    @PostMapping("/evaluate/batch")
    public ResponseEntity<?> evaluateBatch(@RequestBody Map<String, Object> request) {
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> samples = (List<Map<String, Object>>) request.get("samples");

        if (samples == null || samples.isEmpty()) {
            return ResponseEntity.badRequest()
                .body(Map.of("error", "需要 samples 参数"));
        }

        log.info("[API] RAGAS 批量评估请求: {} 样本", samples.size());

        try {
            List<RAGEvaluationPipeline.EvaluationInput> inputs = samples.stream()
                .map(this::mapToEvaluationInput)
                .collect(Collectors.toList());

            RAGEvaluationPipeline.BatchEvaluationResult result =
                evaluationPipeline.evaluateBatch(inputs);

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("sampleCount", result.sampleCount);
            response.put("aggregateMetrics", result.aggregateMetrics);
            response.put("latencyMs", result.latencyMs);

            return ResponseEntity.ok(response);
        } catch (Exception e) {
            log.error("[API] RAGAS 批量评估错误: {}", e.getMessage());
            return ResponseEntity.internalServerError()
                .body(Map.of("error", e.getMessage()));
        }
    }

    /**
     * RAG Fusion - 多查询生成 + RRF 融合
     *
     * 2024-2025 年热门技术：
     * - 生成多个查询变体覆盖不同语义角度
     * - 并行检索后使用 RRF 融合
     * - 提升召回率 10-20%
     */
    @PostMapping("/fusion")
    public ResponseEntity<?> ragFusion(@RequestBody Map<String, Object> request) {
        String query = (String) request.get("query");
        Integer topK = (Integer) request.getOrDefault("topK", 5);

        if (query == null || query.trim().isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "query 不能为空"));
        }

        // 安全验证
        PromptSecurityService.SecurityCheckResult securityCheck = securityService.performFullSecurityCheck(query);
        if (!securityCheck.isPassed()) {
            return ResponseEntity.ok(Map.of(
                "error", securityCheck.getRejectReason(),
                "rejectType", securityCheck.getRejectType()
            ));
        }

        log.info("[API] RAG Fusion 请求: {}", truncate(query, 50));

        try {
            if (ragFusionService == null) {
                return ResponseEntity.status(503)
                    .body(Map.of("error", "RAG Fusion 服务未初始化"));
            }

            RAGFusionService.RAGFusionResult result = ragFusionService.search(query, topK);

            return ResponseEntity.ok(result.toMap());
        } catch (Exception e) {
            log.error("[API] RAG Fusion 错误: {}", e.getMessage());
            return ResponseEntity.internalServerError()
                .body(Map.of("error", e.getMessage()));
        }
    }

    /**
     * RAGAS 评估 - 使用标准 RAGAS 指标
     *
     * 指标：
     * - Faithfulness: 答案是否基于上下文
     * - Answer Relevancy: 答案是否回答问题
     * - Context Precision: 检索是否精确
     * - Context Recall: 检索是否完整
     */
    @PostMapping("/ragas")
    public ResponseEntity<?> ragasEvaluate(@RequestBody Map<String, Object> request) {
        String question = (String) request.get("question");
        String answer = (String) request.get("answer");
        @SuppressWarnings("unchecked")
        List<String> contexts = (List<String>) request.get("contexts");
        String groundTruth = (String) request.get("groundTruth");

        if (question == null || answer == null) {
            return ResponseEntity.badRequest()
                .body(Map.of("error", "需要 question 和 answer 参数"));
        }

        // 安全验证
        PromptSecurityService.ValidationResult validation = securityService.validateQuestion(question);
        if (!validation.isSafe()) {
            return ResponseEntity.ok(Map.of("error", validation.getReason()));
        }

        log.info("[API] RAGAS 评估请求: {}", truncate(question, 50));

        try {
            if (ragasEvaluationService == null) {
                return ResponseEntity.status(503)
                    .body(Map.of("error", "RAGAS 评估服务未初始化"));
            }

            List<String> ctxList = contexts != null ? contexts : Collections.emptyList();
            RAGASEvaluationService.RAGASResult result = 
                ragasEvaluationService.evaluate(question, answer, ctxList, groundTruth);

            return ResponseEntity.ok(result.toMap());
        } catch (Exception e) {
            log.error("[API] RAGAS 评估错误: {}", e.getMessage());
            return ResponseEntity.internalServerError()
                .body(Map.of("error", e.getMessage()));
        }
    }

    /**
     * 服务状态检查
     */
    @GetMapping("/status")
    public ResponseEntity<?> status() {
        Map<String, Object> status = new LinkedHashMap<>();
        status.put("service", "Advanced RAG API");
        status.put("version", "1.0.0");

        Map<String, Boolean> features = new LinkedHashMap<>();
        features.put("agenticRAG", agenticRAGService != null);
        features.put("speculativeRAG", speculativeRAGService != null);
        features.put("ragasEvaluation", evaluationPipeline != null);
        features.put("longContextRAG", longContextRAGService != null);
        features.put("embeddingEnabled", embeddingClient != null && embeddingClient.enabled());
        features.put("ragFusion", ragFusionService != null);
        features.put("ragasEvaluation", ragasEvaluationService != null);
        status.put("features", features);

        status.put("timestamp", System.currentTimeMillis());

        return ResponseEntity.ok(status);
    }

    /**
     * 技术说明文档
     */
    @GetMapping("/docs")
    public ResponseEntity<?> docs() {
        Map<String, Object> docs = new LinkedHashMap<>();

        docs.put("title", "高级 RAG API 文档");
        docs.put("version", "1.0.0 (2026)");

        List<Map<String, Object>> endpoints = new ArrayList<>();

        // Agentic RAG
        Map<String, Object> agentic = new LinkedHashMap<>();
        agentic.put("path", "/ai/advanced/agentic");
        agentic.put("method", "POST");
        agentic.put("description", "Agentic RAG - Agent 自主决策检索策略");
        agentic.put("request", Map.of("query", "string"));
        agentic.put("features", List.of(
            "Agent 自动分析问题复杂度",
            "动态决定是否需要检索",
            "支持多轮检索迭代",
            "支持图谱检索和知识精炼"
        ));
        endpoints.add(agentic);

        // Speculative RAG
        Map<String, Object> speculative = new LinkedHashMap<>();
        speculative.put("path", "/ai/advanced/speculative");
        speculative.put("method", "POST");
        speculative.put("description", "Speculative RAG - 多路并行生成 + 选优");
        speculative.put("request", Map.of("query", "string"));
        speculative.put("features", List.of(
            "并行使用多种策略生成答案",
            "自动评分和选优",
            "提高答案质量",
            "降低幻觉风险"
        ));
        endpoints.add(speculative);

        // Long-Context RAG
        Map<String, Object> longContext = new LinkedHashMap<>();
        longContext.put("path", "/ai/advanced/longcontext");
        longContext.put("method", "POST");
        longContext.put("description", "Long-Context + RAG 自适应策略");
        longContext.put("request", Map.of("query", "string"));
        longContext.put("features", List.of(
            "根据知识库大小自动选择策略",
            "支持全上下文、分块、混合模式",
            "成本-效果权衡",
            "适应 128K 长窗口模型"
        ));
        endpoints.add(longContext);

        // RAGAS Evaluation
        Map<String, Object> evaluate = new LinkedHashMap<>();
        evaluate.put("path", "/ai/advanced/evaluate");
        evaluate.put("method", "POST");
        evaluate.put("description", "RAGAS 标准化评估");
        evaluate.put("request", Map.of(
            "query", "string",
            "answer", "string",
            "contexts", "list<string>",
            "groundTruthAnswer", "string (optional)",
            "groundTruthContext", "string (optional)"
        ));
        evaluate.put("metrics", List.of(
            "contextPrecision - 检索精度",
            "contextRecall - 检索召回",
            "faithfulness - 忠实度（非幻觉）",
            "answerRelevancy - 答案相关性"
        ));
        endpoints.add(evaluate);

        docs.put("endpoints", endpoints);

        // 技术参考
        docs.put("references", List.of(
            "Agentic RAG - RAG 与 Agent 融合的 2024-2026 主流趋势",
            "Speculative RAG - 借鉴 Speculative Decoding 的多路生成",
            "RAGAS - Retrieval-Augmented Generation Assessment (2023)",
            "Long-Context + RAG - 适应 128K 窗口的新策略"
        ));

        return ResponseEntity.ok(docs);
    }

    // ==================== 辅助方法 ====================

    private RAGEvaluationPipeline.EvaluationInput mapToEvaluationInput(Map<String, Object> map) {
        RAGEvaluationPipeline.EvaluationInput input = new RAGEvaluationPipeline.EvaluationInput();
        input.query = (String) map.get("query");
        input.answer = (String) map.get("answer");
        @SuppressWarnings("unchecked")
        List<String> contexts = (List<String>) map.get("contexts");
        input.contexts = contexts;
        input.groundTruthAnswer = (String) map.get("groundTruthAnswer");
        input.groundTruthContext = (String) map.get("groundTruthContext");
        return input;
    }

    private String truncate(String text, int maxLen) {
        if (text == null) return "";
        return text.length() > maxLen ? text.substring(0, maxLen) + "..." : text;
    }
}
