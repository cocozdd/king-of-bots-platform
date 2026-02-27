package com.kob.backend.service.impl.ai.speculative;

import com.kob.backend.controller.ai.dto.AiDoc;
import com.kob.backend.service.impl.ai.AiMetricsService;
import com.kob.backend.service.impl.ai.DeepseekClient;
import com.kob.backend.service.impl.ai.PromptSecurityService;
import com.kob.backend.service.impl.ai.search.HybridSearchService;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * Speculative RAG 服务
 *
 * 论文: Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting (2024)
 *
 * 核心思想:
 * 1. 多路并行生成：使用不同的检索策略/Prompt 并行生成多个候选答案
 * 2. 验证与选择：使用验证器评估每个候选答案的质量
 * 3. 选优输出：选择质量最高的答案作为最终输出
 *
 * 优势:
 * - 提高答案质量：多样化策略增加找到最优答案的概率
 * - 降低幻觉风险：多路验证减少单一路径的错误
 * - 利用并行加速：并行生成不增加太多延迟
 *
 * 面试亮点:
 * - 借鉴 Speculative Decoding 思想应用于 RAG
 * - 并行计算 + 质量验证的工程实践
 * - 2024 年前沿技术，展示技术敏锐度
 */
@Service
public class SpeculativeRAGService {

    private static final Logger log = LoggerFactory.getLogger(SpeculativeRAGService.class);

    // 并行路径数量（Draft 数量）
    private static final int NUM_DRAFTS = 3;

    // 线程池
    private ExecutorService executorService;

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
        this.executorService = Executors.newFixedThreadPool(NUM_DRAFTS);
        log.info("Speculative RAG Service 初始化完成，并行路径数: {}", NUM_DRAFTS);
    }

    /**
     * Speculative RAG 主流程
     *
     * @param query 用户查询
     * @param queryEmbedding 查询向量
     * @return Speculative RAG 结果
     */
    public SpeculativeRAGResult process(String query, double[] queryEmbedding) {
        long startTime = System.currentTimeMillis();
        SpeculativeRAGResult result = new SpeculativeRAGResult();
        result.query = query;
        result.drafts = new ArrayList<>();

        // ===== 鲁棒性检查：提前拦截问题查询 =====
        PromptSecurityService.SecurityCheckResult securityCheck = securityService.performFullSecurityCheck(query);
        if (!securityCheck.isPassed()) {
            log.warn("[Speculative RAG] 安全检查拦截: type={}, query={}", securityCheck.getRejectType(), truncate(query, 50));
            result.answer = securityCheck.getRejectReason();
            result.selectedStrategy = "SECURITY_REJECT";
            result.latencyMs = System.currentTimeMillis() - startTime;
            return result;
        }

        log.info("[Speculative RAG] 开始处理: {}", truncate(query, 50));

        // Step 1: 并行生成多个草稿（Draft）
        List<DraftStrategy> strategies = createDraftStrategies(query);
        List<Future<Draft>> futures = new ArrayList<>();

        for (DraftStrategy strategy : strategies) {
            futures.add(executorService.submit(() ->
                generateDraft(query, queryEmbedding, strategy)));
        }

        // 收集所有草稿
        List<Draft> drafts = new ArrayList<>();
        for (Future<Draft> future : futures) {
            try {
                Draft draft = future.get(30, TimeUnit.SECONDS);
                if (draft != null) {
                    drafts.add(draft);
                }
            } catch (TimeoutException e) {
                log.warn("[Speculative RAG] 草稿生成超时");
            } catch (Exception e) {
                log.error("[Speculative RAG] 草稿生成失败: {}", e.getMessage());
            }
        }

        if (drafts.isEmpty()) {
            // 所有草稿都失败，降级处理
            log.warn("[Speculative RAG] 所有草稿生成失败，降级为普通 RAG");
            result.answer = fallbackGenerate(query, queryEmbedding);
            result.selectedStrategy = "FALLBACK";
            result.latencyMs = System.currentTimeMillis() - startTime;
            return result;
        }

        result.drafts = drafts;
        log.info("[Speculative RAG] 生成 {} 个草稿", drafts.size());

        // Step 2: 验证和评分每个草稿
        List<ScoredDraft> scoredDrafts = verifyDrafts(query, drafts);

        // Step 3: 选择最优草稿
        ScoredDraft best = scoredDrafts.stream()
            .max(Comparator.comparingDouble(d -> d.score))
            .orElse(scoredDrafts.get(0));

        result.answer = best.draft.answer;
        result.selectedStrategy = best.draft.strategy.name;
        result.selectedScore = best.score;
        result.scoredDrafts = scoredDrafts;
        result.finalDocs = best.draft.docs;

        log.info("[Speculative RAG] 选择策略: {}, 得分: {:.2f}",
            best.draft.strategy.name, best.score);

        result.latencyMs = System.currentTimeMillis() - startTime;
        return result;
    }

    /**
     * 创建多样化的草稿策略
     */
    private List<DraftStrategy> createDraftStrategies(String query) {
        List<DraftStrategy> strategies = new ArrayList<>();

        // 策略 1: 标准 RAG - 直接检索
        strategies.add(new DraftStrategy(
            "STANDARD",
            query,
            "基于检索到的参考文档，准确回答问题。",
            5,
            0.3
        ));

        // 策略 2: 深度 RAG - 更多文档，详细回答
        strategies.add(new DraftStrategy(
            "DETAILED",
            query + " 详细解释 原理",
            "基于参考文档，给出详细、深入的回答，包括原理和示例。",
            8,
            0.5
        ));

        // 策略 3: 简洁 RAG - 少量文档，简洁回答
        strategies.add(new DraftStrategy(
            "CONCISE",
            extractKeywords(query),
            "给出简洁、直接的回答，只包含最关键的信息。",
            3,
            0.2
        ));

        return strategies;
    }

    /**
     * 生成单个草稿
     */
    private Draft generateDraft(String query, double[] queryEmbedding, DraftStrategy strategy) {
        Draft draft = new Draft();
        draft.strategy = strategy;

        try {
            // 检索文档
            List<AiDoc> docs = hybridSearchService.hybridSearch(
                strategy.searchQuery,
                queryEmbedding,
                strategy.topK
            );
            draft.docs = docs;

            if (docs.isEmpty()) {
                draft.answer = generateWithoutContext(query, strategy);
            } else {
                draft.answer = generateWithContext(query, docs, strategy);
            }

            log.debug("[Speculative RAG] 策略 {} 生成完成", strategy.name);
            return draft;

        } catch (Exception e) {
            log.error("[Speculative RAG] 策略 {} 生成失败: {}", strategy.name, e.getMessage());
            return null;
        }
    }

    /**
     * 验证和评分草稿
     */
    private List<ScoredDraft> verifyDrafts(String query, List<Draft> drafts) {
        List<ScoredDraft> scored = new ArrayList<>();

        for (Draft draft : drafts) {
            double score = scoreDraft(query, draft);
            scored.add(new ScoredDraft(draft, score));
        }

        return scored;
    }

    /**
     * 评分单个草稿
     *
     * 评分维度:
     * 1. 相关性: 答案与问题的关联度
     * 2. 完整性: 答案是否完整回答了问题
     * 3. 一致性: 答案与检索文档的一致性（忠实度）
     * 4. 质量: 语言质量、格式等
     */
    private double scoreDraft(String query, Draft draft) {
        double relevanceScore = calculateRelevance(query, draft.answer);
        double completenessScore = calculateCompleteness(query, draft.answer);
        double faithfulnessScore = calculateFaithfulness(draft.answer, draft.docs);
        double qualityScore = calculateQuality(draft.answer);

        // 加权平均
        double score =
            relevanceScore * 0.30 +
            completenessScore * 0.25 +
            faithfulnessScore * 0.30 +
            qualityScore * 0.15;

        // 应用策略权重（一些策略可能更适合某些查询）
        score *= (1.0 + draft.strategy.weight);

        log.debug("[Speculative RAG] 策略 {} 得分: rel={:.2f}, comp={:.2f}, faith={:.2f}, qual={:.2f}, total={:.2f}",
            draft.strategy.name, relevanceScore, completenessScore, faithfulnessScore, qualityScore, score);

        return score;
    }

    /**
     * 相关性评分: 答案是否与问题相关
     */
    private double calculateRelevance(String query, String answer) {
        if (answer == null || answer.isEmpty()) return 0;

        String queryLower = query.toLowerCase();
        String answerLower = answer.toLowerCase();

        // 关键词覆盖率
        String[] queryTerms = queryLower.split("\\s+");
        long matchCount = Arrays.stream(queryTerms)
            .filter(term -> term.length() > 1 && answerLower.contains(term))
            .count();

        double keywordCoverage = queryTerms.length > 0 ?
            (double) matchCount / queryTerms.length : 0;

        // 长度合理性（太短可能不完整，太长可能冗余）
        int answerLength = answer.length();
        double lengthScore;
        if (answerLength < 50) {
            lengthScore = 0.3;
        } else if (answerLength < 200) {
            lengthScore = 0.7;
        } else if (answerLength < 1000) {
            lengthScore = 1.0;
        } else {
            lengthScore = 0.8;
        }

        return keywordCoverage * 0.6 + lengthScore * 0.4;
    }

    /**
     * 完整性评分: 答案是否完整回答了问题
     */
    private double calculateCompleteness(String query, String answer) {
        if (answer == null || answer.isEmpty()) return 0;

        // 检测问题类型并验证答案是否包含相应内容
        double score = 0.5; // 基础分

        // 如果问题包含"如何"/"怎么"，答案应该包含步骤或方法
        if (query.contains("如何") || query.contains("怎么") || query.contains("how")) {
            if (answer.contains("1.") || answer.contains("首先") ||
                answer.contains("步骤") || answer.contains("方法")) {
                score += 0.3;
            }
        }

        // 如果问题包含"什么"/"是什么"，答案应该包含定义或解释
        if (query.contains("什么") || query.contains("是什么") || query.contains("what")) {
            if (answer.length() > 100) {
                score += 0.2;
            }
        }

        // 如果问题包含"为什么"，答案应该包含原因
        if (query.contains("为什么") || query.contains("why")) {
            if (answer.contains("因为") || answer.contains("原因") ||
                answer.contains("由于") || answer.contains("是因为")) {
                score += 0.3;
            }
        }

        // 答案包含代码（如果问题可能需要代码）
        if (query.contains("代码") || query.contains("实现") ||
            query.contains("code") || query.contains("示例")) {
            if (answer.contains("```")) {
                score += 0.2;
            }
        }

        return Math.min(1.0, score);
    }

    /**
     * 忠实度评分: 答案是否与检索文档一致（避免幻觉）
     */
    private double calculateFaithfulness(String answer, List<AiDoc> docs) {
        if (answer == null || answer.isEmpty() || docs == null || docs.isEmpty()) {
            return 0.5; // 无法验证
        }

        // 合并所有文档内容
        String allContent = docs.stream()
            .map(d -> (d.getTitle() + " " + d.getContent()).toLowerCase())
            .collect(Collectors.joining(" "));

        // 提取答案中的关键句子
        String[] sentences = answer.split("[。.!?！？]");
        int supportedCount = 0;
        int totalSentences = 0;

        for (String sentence : sentences) {
            sentence = sentence.trim();
            if (sentence.length() < 10) continue; // 跳过太短的句子

            totalSentences++;
            String sentenceLower = sentence.toLowerCase();

            // 检查句子中的关键词是否出现在文档中
            String[] words = sentenceLower.split("\\s+");
            long matchedWords = Arrays.stream(words)
                .filter(w -> w.length() > 2 && allContent.contains(w))
                .count();

            if (words.length > 0 && (double) matchedWords / words.length > 0.3) {
                supportedCount++;
            }
        }

        return totalSentences > 0 ? (double) supportedCount / totalSentences : 0.5;
    }

    /**
     * 质量评分: 语言质量、格式等
     */
    private double calculateQuality(String answer) {
        if (answer == null || answer.isEmpty()) return 0;

        double score = 0.5;

        // Markdown 格式使用
        if (answer.contains("##") || answer.contains("**")) {
            score += 0.1;
        }

        // 代码块格式
        if (answer.contains("```")) {
            score += 0.1;
        }

        // 列表使用
        if (answer.contains("1.") || answer.contains("- ")) {
            score += 0.1;
        }

        // 段落分隔
        if (answer.contains("\n\n")) {
            score += 0.1;
        }

        // 无明显错误标记
        if (!answer.contains("错误") && !answer.contains("失败") &&
            !answer.contains("无法")) {
            score += 0.1;
        }

        return Math.min(1.0, score);
    }

    /**
     * 基于上下文生成答案
     */
    private String generateWithContext(String query, List<AiDoc> docs, DraftStrategy strategy) {
        if (!deepseekClient.enabled()) {
            return "AI 服务未配置";
        }

        String systemPrompt = """
            你是 Bot 开发助手。""" + strategy.promptStyle + """

            格式要求:
            - 使用 Markdown 格式
            - 段落之间用空行分隔
            - 代码用 ```语言名 ``` 包裹
            """;

        List<String> contexts = docs.stream()
            .map(d -> d.getTitle() + ":\n" + d.getContent())
            .collect(Collectors.toList());

        return deepseekClient.chat(systemPrompt, query, contexts);
    }

    /**
     * 无上下文生成
     */
    private String generateWithoutContext(String query, DraftStrategy strategy) {
        if (!deepseekClient.enabled()) {
            return "AI 服务未配置";
        }

        String systemPrompt = "你是 Bot 开发助手。" + strategy.promptStyle;
        return deepseekClient.chat(systemPrompt, query, Collections.emptyList());
    }

    /**
     * 降级生成
     */
    private String fallbackGenerate(String query, double[] queryEmbedding) {
        List<AiDoc> docs = hybridSearchService.hybridSearch(query, queryEmbedding, 5);
        if (docs.isEmpty()) {
            return generateWithoutContext(query, new DraftStrategy("FALLBACK", query, "", 5, 0));
        }

        List<String> contexts = docs.stream()
            .map(d -> d.getTitle() + ":\n" + d.getContent())
            .collect(Collectors.toList());

        return deepseekClient.chat(
            "你是 Bot 开发助手，基于参考文档回答问题。",
            query,
            contexts
        );
    }

    /**
     * 提取关键词（用于简洁策略）
     */
    private String extractKeywords(String query) {
        // 简化实现：移除常见停用词
        String[] stopWords = {"的", "是", "什么", "如何", "怎么", "为什么", "请", "帮", "我"};
        String result = query;
        for (String sw : stopWords) {
            result = result.replace(sw, " ");
        }
        return result.replaceAll("\\s+", " ").trim();
    }

    private String truncate(String text, int maxLen) {
        if (text == null) return "";
        return text.length() > maxLen ? text.substring(0, maxLen) + "..." : text;
    }

    // ===== 内部类 =====

    public static class DraftStrategy {
        public String name;
        public String searchQuery;
        public String promptStyle;
        public int topK;
        public double weight;

        public DraftStrategy(String name, String searchQuery, String promptStyle, int topK, double weight) {
            this.name = name;
            this.searchQuery = searchQuery;
            this.promptStyle = promptStyle;
            this.topK = topK;
            this.weight = weight;
        }
    }

    public static class Draft {
        public DraftStrategy strategy;
        public String answer;
        public List<AiDoc> docs;
    }

    public static class ScoredDraft {
        public Draft draft;
        public double score;

        public ScoredDraft(Draft draft, double score) {
            this.draft = draft;
            this.score = score;
        }
    }

    public static class SpeculativeRAGResult {
        public String query;
        public String answer;
        public String selectedStrategy;
        public double selectedScore;
        public List<Draft> drafts;
        public List<ScoredDraft> scoredDrafts;
        public List<AiDoc> finalDocs;
        public long latencyMs;

        public Map<String, Object> toMap() {
            Map<String, Object> map = new HashMap<>();
            map.put("query", query);
            map.put("answer", answer);
            map.put("selectedStrategy", selectedStrategy);
            map.put("selectedScore", selectedScore);
            map.put("latencyMs", latencyMs);

            if (scoredDrafts != null) {
                List<Map<String, Object>> draftMaps = new ArrayList<>();
                for (ScoredDraft sd : scoredDrafts) {
                    Map<String, Object> dm = new HashMap<>();
                    dm.put("strategy", sd.draft.strategy.name);
                    dm.put("score", sd.score);
                    dm.put("answerPreview", truncateStatic(sd.draft.answer, 100));
                    draftMaps.add(dm);
                }
                map.put("drafts", draftMaps);
            }

            return map;
        }

        private static String truncateStatic(String text, int maxLen) {
            if (text == null) return "";
            return text.length() > maxLen ? text.substring(0, maxLen) + "..." : text;
        }
    }
}
