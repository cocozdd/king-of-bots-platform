package com.kob.backend.service.impl.ai.longcontext;

import com.kob.backend.controller.ai.dto.AiDoc;
import com.kob.backend.service.impl.ai.AiMetricsService;
import com.kob.backend.service.impl.ai.DeepseekClient;
import com.kob.backend.service.impl.ai.PromptSecurityService;
import com.kob.backend.service.impl.ai.search.HybridSearchService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Long-Context + RAG 自适应策略服务
 *
 * 背景:
 * - 2024-2026 年 LLM 上下文窗口大幅扩展 (128K-1M tokens)
 * - 传统 RAG 基于"上下文有限"的假设需要重新审视
 * - 需要智能决策：何时使用 RAG，何时直接使用长上下文
 *
 * 核心策略:
 * 1. 小规模知识库 (< 50K tokens) → 直接放入上下文
 * 2. 中等规模 + 简单查询 → 直接上下文
 * 3. 大规模/复杂查询 → 传统 RAG 检索
 * 4. 实时性需求 → 必须 RAG（检索最新数据）
 *
 * 决策因素:
 * - 知识库大小
 * - 查询复杂度
 * - 实时性需求
 * - 成本敏感度（长上下文更贵）
 * - 精度需求（RAG 更精准但可能丢失全局视角）
 *
 * 面试亮点:
 * - 展示对 LLM 发展趋势的理解
 * - 成本-效果权衡的工程思维
 * - 不是"RAG 或 LC"，而是"RAG + LC"混合策略
 */
@Service
public class LongContextRAGService {

    private static final Logger log = LoggerFactory.getLogger(LongContextRAGService.class);

    // 阈值配置
    private static final int SMALL_KB_THRESHOLD = 50000;      // 50K tokens = 小知识库
    private static final int MEDIUM_KB_THRESHOLD = 200000;    // 200K tokens = 中等知识库
    private static final int MAX_CONTEXT_WINDOW = 128000;     // 模型最大上下文窗口

    // 成本系数（长上下文输入成本更高）
    private static final double LONG_CONTEXT_COST_MULTIPLIER = 1.5;

    @Autowired
    private HybridSearchService hybridSearchService;

    @Autowired(required = false)
    private AiMetricsService metricsService;

    @Autowired
    private PromptSecurityService securityService;

    private DeepseekClient deepseekClient;

    @PostConstruct
    public void init() {
        this.deepseekClient = new DeepseekClient(metricsService);
        log.info("Long-Context RAG Service 初始化完成");
    }

    /**
     * 自适应处理入口
     *
     * @param query 用户查询
     * @param queryEmbedding 查询向量
     * @param knowledgeBase 完整知识库（用于 LC 模式）
     * @return 处理结果
     */
    public LongContextRAGResult process(String query, double[] queryEmbedding,
                                        List<AiDoc> knowledgeBase) {
        long startTime = System.currentTimeMillis();
        LongContextRAGResult result = new LongContextRAGResult();
        result.query = query;

        // ===== 鲁棒性检查：提前拦截问题查询 =====
        PromptSecurityService.SecurityCheckResult securityCheck = securityService.performFullSecurityCheck(query);
        if (!securityCheck.isPassed()) {
            log.warn("[LC-RAG] 安全检查拦截: type={}, query={}", securityCheck.getRejectType(), truncate(query, 50));
            result.answer = securityCheck.getRejectReason();
            result.strategy = new Strategy();
            result.strategy.type = StrategyType.PURE_RAG;
            result.strategy.reason = "安全拦截: " + securityCheck.getRejectType();
            result.queryAnalysis = new QueryAnalysis();
            result.latencyMs = System.currentTimeMillis() - startTime;
            return result;
        }

        // Step 1: 分析查询特征
        QueryAnalysis analysis = analyzeQuery(query);
        result.queryAnalysis = analysis;

        // Step 2: 评估知识库规模
        int kbTokens = estimateTokens(knowledgeBase);
        result.knowledgeBaseTokens = kbTokens;

        // Step 3: 决策：使用哪种策略
        Strategy strategy = decideStrategy(analysis, kbTokens, knowledgeBase);
        result.strategy = strategy;

        log.info("[LC-RAG] 策略决策: {} (KB tokens={}, query complexity={})",
            strategy.type, kbTokens, analysis.complexity);

        // Step 4: 执行策略
        switch (strategy.type) {
            case FULL_CONTEXT:
                // 直接使用全部知识库作为上下文
                result = executeFullContext(query, knowledgeBase, result);
                break;

            case CHUNKED_CONTEXT:
                // 分块处理：先用 RAG 粗筛，再用长上下文精炼
                result = executeChunkedContext(query, queryEmbedding, knowledgeBase, result);
                break;

            case HYBRID:
                // 混合模式：RAG 检索 + 长上下文补充
                result = executeHybrid(query, queryEmbedding, knowledgeBase, result);
                break;

            case PURE_RAG:
            default:
                // 纯 RAG 模式
                result = executePureRAG(query, queryEmbedding, result);
                break;
        }

        result.latencyMs = System.currentTimeMillis() - startTime;
        return result;
    }

    /**
     * 分析查询特征
     */
    private QueryAnalysis analyzeQuery(String query) {
        QueryAnalysis analysis = new QueryAnalysis();

        // 1. 查询长度
        analysis.length = query.length();

        // 2. 复杂度评估
        analysis.complexity = estimateQueryComplexity(query);

        // 3. 是否需要全局视角
        analysis.needsGlobalView = needsGlobalPerspective(query);

        // 4. 是否需要实时信息
        analysis.needsRealtimeInfo = needsRealtimeInformation(query);

        // 5. 问题类型
        analysis.questionType = classifyQuestionType(query);

        return analysis;
    }

    /**
     * 估计查询复杂度 (0-1)
     */
    private double estimateQueryComplexity(String query) {
        double complexity = 0.3; // 基础复杂度

        // 多跳推理指标
        if (query.contains("和") || query.contains("与") ||
            query.contains("比较") || query.contains("区别")) {
            complexity += 0.2;
        }

        // 条件查询
        if (query.contains("如果") || query.contains("当") ||
            query.contains("假设") || query.contains("情况下")) {
            complexity += 0.2;
        }

        // 聚合查询
        if (query.contains("所有") || query.contains("有哪些") ||
            query.contains("总结") || query.contains("列举")) {
            complexity += 0.2;
        }

        // 查询长度
        if (query.length() > 50) {
            complexity += 0.1;
        }

        return Math.min(1.0, complexity);
    }

    /**
     * 判断是否需要全局视角
     */
    private boolean needsGlobalPerspective(String query) {
        String[] globalIndicators = {"全部", "所有", "总结", "概述", "整体",
            "归纳", "有哪些", "列举", "分类", "体系"};

        for (String indicator : globalIndicators) {
            if (query.contains(indicator)) return true;
        }
        return false;
    }

    /**
     * 判断是否需要实时信息
     */
    private boolean needsRealtimeInformation(String query) {
        String[] realtimeIndicators = {"最新", "今天", "昨天", "最近",
            "现在", "当前", "实时", "2024", "2025", "2026"};

        for (String indicator : realtimeIndicators) {
            if (query.contains(indicator)) return true;
        }
        return false;
    }

    /**
     * 分类问题类型
     */
    private String classifyQuestionType(String query) {
        if (query.contains("如何") || query.contains("怎么") || query.contains("怎样")) {
            return "HOW";
        }
        if (query.contains("为什么") || query.contains("原因")) {
            return "WHY";
        }
        if (query.contains("什么") || query.contains("是什么")) {
            return "WHAT";
        }
        if (query.contains("哪些") || query.contains("列举")) {
            return "LIST";
        }
        if (query.contains("比较") || query.contains("区别") || query.contains("异同")) {
            return "COMPARE";
        }
        return "GENERAL";
    }

    /**
     * 决策：使用哪种策略
     */
    private Strategy decideStrategy(QueryAnalysis analysis, int kbTokens,
                                    List<AiDoc> knowledgeBase) {
        Strategy strategy = new Strategy();

        // 规则 1: 小知识库 + 需要全局视角 → 全上下文
        if (kbTokens < SMALL_KB_THRESHOLD && analysis.needsGlobalView) {
            strategy.type = StrategyType.FULL_CONTEXT;
            strategy.reason = "小知识库 + 需要全局视角，直接使用全部上下文";
            return strategy;
        }

        // 规则 2: 小知识库 + 低复杂度 → 全上下文
        if (kbTokens < SMALL_KB_THRESHOLD && analysis.complexity < 0.5) {
            strategy.type = StrategyType.FULL_CONTEXT;
            strategy.reason = "小知识库 + 简单查询，直接使用全部上下文";
            return strategy;
        }

        // 规则 3: 需要实时信息 → 纯 RAG（确保检索最新）
        if (analysis.needsRealtimeInfo) {
            strategy.type = StrategyType.PURE_RAG;
            strategy.reason = "需要实时信息，使用 RAG 确保检索最新数据";
            return strategy;
        }

        // 规则 4: 中等知识库 + 复杂/全局查询 → 混合模式
        if (kbTokens < MEDIUM_KB_THRESHOLD &&
            (analysis.complexity > 0.6 || analysis.needsGlobalView)) {
            strategy.type = StrategyType.HYBRID;
            strategy.reason = "中等知识库 + 复杂查询，RAG 检索 + 长上下文补充";
            strategy.ragTopK = 10;
            strategy.supplementTopK = 20;
            return strategy;
        }

        // 规则 5: 大知识库 + 列举/比较类 → 分块处理
        if (kbTokens >= MEDIUM_KB_THRESHOLD &&
            (analysis.questionType.equals("LIST") || analysis.questionType.equals("COMPARE"))) {
            strategy.type = StrategyType.CHUNKED_CONTEXT;
            strategy.reason = "大知识库 + 聚合查询，分块 + Map-Reduce";
            return strategy;
        }

        // 规则 6: 中等知识库 + 简单查询 → 分块上下文
        if (kbTokens < MEDIUM_KB_THRESHOLD && analysis.complexity < 0.5) {
            strategy.type = StrategyType.CHUNKED_CONTEXT;
            strategy.reason = "中等知识库 + 简单查询，分块上下文";
            return strategy;
        }

        // 默认: 纯 RAG
        strategy.type = StrategyType.PURE_RAG;
        strategy.reason = "大知识库/默认策略，使用传统 RAG";
        return strategy;
    }

    /**
     * 执行策略: 全上下文
     */
    private LongContextRAGResult executeFullContext(String query,
                                                    List<AiDoc> knowledgeBase,
                                                    LongContextRAGResult result) {
        log.info("[LC-RAG] 执行全上下文策略，文档数: {}", knowledgeBase.size());

        // 将全部知识库作为上下文
        List<String> contexts = knowledgeBase.stream()
            .map(d -> "## " + d.getTitle() + "\n" + d.getContent())
            .collect(Collectors.toList());

        String systemPrompt = """
            你是 Bot 开发助手。以下是完整的知识库内容。
            请基于这些知识全面、准确地回答问题。

            格式要求:
            - 使用 Markdown 格式
            - 段落之间用空行分隔
            - 代码用 ```语言名 ``` 包裹
            """;

        result.answer = deepseekClient.chat(systemPrompt, query, contexts);
        result.usedDocs = knowledgeBase;
        result.tokensUsed = estimateTokens(knowledgeBase);

        return result;
    }

    /**
     * 执行策略: 分块上下文 (Map-Reduce 风格)
     */
    private LongContextRAGResult executeChunkedContext(String query,
                                                       double[] queryEmbedding,
                                                       List<AiDoc> knowledgeBase,
                                                       LongContextRAGResult result) {
        log.info("[LC-RAG] 执行分块上下文策略");

        // Step 1: 分块
        int chunkSize = MAX_CONTEXT_WINDOW / 4; // 每块约 32K tokens
        List<List<AiDoc>> chunks = chunkDocuments(knowledgeBase, chunkSize);

        // Step 2: Map - 对每个块进行处理
        List<String> chunkAnswers = new ArrayList<>();
        for (int i = 0; i < chunks.size(); i++) {
            List<AiDoc> chunk = chunks.get(i);
            List<String> contexts = chunk.stream()
                .map(d -> d.getTitle() + ": " + d.getContent())
                .collect(Collectors.toList());

            String mapPrompt = """
                你是 Bot 开发助手。这是知识库的第 %d/%d 部分。
                请提取与问题相关的关键信息（如果有的话）。
                如果这部分没有相关信息，回答"无相关信息"。
                """.formatted(i + 1, chunks.size());

            String chunkAnswer = deepseekClient.chat(mapPrompt, query, contexts);
            if (!chunkAnswer.contains("无相关信息")) {
                chunkAnswers.add("【部分 " + (i + 1) + "】\n" + chunkAnswer);
            }
        }

        // Step 3: Reduce - 汇总所有块的结果
        String reducePrompt = """
            你是 Bot 开发助手。以下是从知识库各部分提取的相关信息。
            请综合这些信息，给出完整、连贯的答案。

            格式要求:
            - 使用 Markdown 格式
            - 去除重复信息
            - 确保答案完整
            """;

        result.answer = deepseekClient.chat(reducePrompt, query, chunkAnswers);
        result.usedDocs = knowledgeBase;
        result.chunksProcessed = chunks.size();

        return result;
    }

    /**
     * 执行策略: 混合模式 (RAG + 长上下文)
     */
    private LongContextRAGResult executeHybrid(String query,
                                               double[] queryEmbedding,
                                               List<AiDoc> knowledgeBase,
                                               LongContextRAGResult result) {
        log.info("[LC-RAG] 执行混合策略");

        // Step 1: RAG 检索最相关的文档
        List<AiDoc> ragDocs = hybridSearchService.hybridSearch(
            query, queryEmbedding, result.strategy.ragTopK);

        // Step 2: 补充上下文（随机采样或按主题采样）
        Set<String> ragIds = ragDocs.stream()
            .map(AiDoc::getId).collect(Collectors.toSet());

        List<AiDoc> supplementDocs = knowledgeBase.stream()
            .filter(d -> !ragIds.contains(d.getId()))
            .limit(result.strategy.supplementTopK)
            .collect(Collectors.toList());

        // Step 3: 合并上下文
        List<AiDoc> allDocs = new ArrayList<>(ragDocs);
        allDocs.addAll(supplementDocs);

        List<String> contexts = allDocs.stream()
            .map(d -> "## " + d.getTitle() + "\n" + d.getContent())
            .collect(Collectors.toList());

        // Step 4: 生成答案
        String systemPrompt = """
            你是 Bot 开发助手。以下是检索到的最相关文档（标记为核心）和补充上下文。
            请优先参考核心文档，补充文档用于提供背景知识。

            核心文档数量: %d
            补充文档数量: %d

            格式要求:
            - 使用 Markdown 格式
            - 段落之间用空行分隔
            - 代码用 ```语言名 ``` 包裹
            """.formatted(ragDocs.size(), supplementDocs.size());

        result.answer = deepseekClient.chat(systemPrompt, query, contexts);
        result.usedDocs = allDocs;
        result.ragDocsCount = ragDocs.size();
        result.supplementDocsCount = supplementDocs.size();

        return result;
    }

    /**
     * 执行策略: 纯 RAG
     */
    private LongContextRAGResult executePureRAG(String query,
                                                double[] queryEmbedding,
                                                LongContextRAGResult result) {
        log.info("[LC-RAG] 执行纯 RAG 策略");

        List<AiDoc> docs = hybridSearchService.hybridSearch(query, queryEmbedding, 5);

        List<String> contexts = docs.stream()
            .map(d -> d.getTitle() + ": " + d.getContent())
            .collect(Collectors.toList());

        String systemPrompt = """
            你是 Bot 开发助手。基于检索到的参考文档回答问题。

            格式要求:
            - 使用 Markdown 格式
            - 段落之间用空行分隔
            - 代码用 ```语言名 ``` 包裹
            - 回答要准确、简洁
            """;

        result.answer = deepseekClient.chat(systemPrompt, query, contexts);
        result.usedDocs = docs;
        result.ragDocsCount = docs.size();

        return result;
    }

    // ==================== 辅助方法 ====================

    private String truncate(String text, int maxLen) {
        if (text == null) return "";
        return text.length() > maxLen ? text.substring(0, maxLen) + "..." : text;
    }

    /**
     * 估算知识库的 token 数
     */
    private int estimateTokens(List<AiDoc> docs) {
        if (docs == null || docs.isEmpty()) return 0;

        int totalChars = docs.stream()
            .mapToInt(d -> {
                int titleLen = d.getTitle() != null ? d.getTitle().length() : 0;
                int contentLen = d.getContent() != null ? d.getContent().length() : 0;
                return titleLen + contentLen;
            })
            .sum();

        // 粗略估算：中文约 1.5 字符/token，英文约 4 字符/token
        // 取平均 2 字符/token
        return totalChars / 2;
    }

    /**
     * 将文档分块
     */
    private List<List<AiDoc>> chunkDocuments(List<AiDoc> docs, int maxTokensPerChunk) {
        List<List<AiDoc>> chunks = new ArrayList<>();
        List<AiDoc> currentChunk = new ArrayList<>();
        int currentTokens = 0;

        for (AiDoc doc : docs) {
            int docTokens = estimateTokens(List.of(doc));

            if (currentTokens + docTokens > maxTokensPerChunk && !currentChunk.isEmpty()) {
                chunks.add(new ArrayList<>(currentChunk));
                currentChunk.clear();
                currentTokens = 0;
            }

            currentChunk.add(doc);
            currentTokens += docTokens;
        }

        if (!currentChunk.isEmpty()) {
            chunks.add(currentChunk);
        }

        return chunks;
    }

    // ==================== 数据类 ====================

    public enum StrategyType {
        FULL_CONTEXT,      // 全上下文
        CHUNKED_CONTEXT,   // 分块上下文 (Map-Reduce)
        HYBRID,            // 混合 (RAG + 长上下文)
        PURE_RAG           // 纯 RAG
    }

    public static class QueryAnalysis {
        public int length;
        public double complexity;
        public boolean needsGlobalView;
        public boolean needsRealtimeInfo;
        public String questionType;
    }

    public static class Strategy {
        public StrategyType type;
        public String reason;
        public int ragTopK = 5;
        public int supplementTopK = 10;
    }

    public static class LongContextRAGResult {
        public String query;
        public String answer;

        public QueryAnalysis queryAnalysis;
        public Strategy strategy;

        public int knowledgeBaseTokens;
        public int tokensUsed;
        public int chunksProcessed;
        public int ragDocsCount;
        public int supplementDocsCount;

        public List<AiDoc> usedDocs;
        public long latencyMs;

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("query", query);
            map.put("answer", answer);
            map.put("strategy", strategy.type.name());
            map.put("strategyReason", strategy.reason);
            map.put("knowledgeBaseTokens", knowledgeBaseTokens);
            map.put("latencyMs", latencyMs);

            Map<String, Object> analysisMap = new LinkedHashMap<>();
            analysisMap.put("complexity", queryAnalysis.complexity);
            analysisMap.put("needsGlobalView", queryAnalysis.needsGlobalView);
            analysisMap.put("needsRealtimeInfo", queryAnalysis.needsRealtimeInfo);
            analysisMap.put("questionType", queryAnalysis.questionType);
            map.put("queryAnalysis", analysisMap);

            return map;
        }
    }
}
