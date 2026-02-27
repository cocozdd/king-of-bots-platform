package com.kob.backend.service.impl.ai.crag;

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
 * CRAG (Corrective RAG) 服务
 * 
 * 论文: Corrective Retrieval Augmented Generation (2024)
 * 
 * 核心思想:
 * 1. 检索后评估文档相关性
 * 2. 相关性低时触发修正策略
 * 3. 三种动作: Correct(直接使用), Incorrect(重新检索/Web搜索), Ambiguous(补充检索)
 * 
 * 面试要点:
 * - 解决传统RAG"垃圾进垃圾出"问题
 * - 引入知识精炼(Knowledge Refinement)
 * - 自适应决策比固定流程更鲁棒
 */
@Service
public class CRAGService {
    
    private static final Logger log = LoggerFactory.getLogger(CRAGService.class);
    
    // 相关性阈值
    private static final double CORRECT_THRESHOLD = 0.7;    // 高于此值: 直接使用
    private static final double INCORRECT_THRESHOLD = 0.3;  // 低于此值: 需要修正
    // 介于两者之间: Ambiguous，需要补充检索
    
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
        log.info("CRAG Service 初始化完成");
    }
    
    /**
     * CRAG 主流程
     */
    public CRAGResult process(String query, double[] queryEmbedding) {
        long startTime = System.currentTimeMillis();
        CRAGResult result = new CRAGResult();
        result.query = query;
        result.steps = new ArrayList<>();

        // ===== 鲁棒性检查：提前拦截问题查询 =====
        PromptSecurityService.SecurityCheckResult securityCheck = securityService.performFullSecurityCheck(query);
        if (!securityCheck.isPassed()) {
            log.warn("[CRAG] 安全检查拦截: type={}, query={}", securityCheck.getRejectType(), truncate(query, 50));
            result.action = CRAGAction.INCORRECT;
            result.answer = securityCheck.getRejectReason();
            result.finalDocs = Collections.emptyList();
            result.steps.add(new CRAGStep("SECURITY", "安全拦截", securityCheck.getRejectType()));
            result.latencyMs = System.currentTimeMillis() - startTime;
            return result;
        }

        // Step 1: 初始检索
        log.info("[CRAG] Step 1: 初始检索 - query: {}", truncate(query, 50));
        List<AiDoc> initialDocs = hybridSearchService.hybridSearch(query, queryEmbedding, 5);
        result.steps.add(new CRAGStep("RETRIEVE", "初始检索", 
                String.format("检索到 %d 篇文档", initialDocs.size())));
        
        if (initialDocs.isEmpty()) {
            // 无检索结果，直接使用 LLM
            result.action = CRAGAction.INCORRECT;
            result.steps.add(new CRAGStep("EVALUATE", "无检索结果", "触发直接生成"));
            result.answer = generateWithoutContext(query);
            result.finalDocs = Collections.emptyList();
            result.latencyMs = System.currentTimeMillis() - startTime;
            return result;
        }
        
        // Step 2: 评估文档相关性
        log.info("[CRAG] Step 2: 评估文档相关性");
        List<RelevanceScore> scores = evaluateRelevance(query, initialDocs);
        double avgScore = scores.stream().mapToDouble(s -> s.score).average().orElse(0);
        result.relevanceScores = scores;
        result.avgRelevance = avgScore;
        
        result.steps.add(new CRAGStep("EVALUATE", "相关性评估", 
                String.format("平均相关性: %.2f", avgScore)));
        
        // Step 3: 决策 - Correct / Incorrect / Ambiguous
        if (avgScore >= CORRECT_THRESHOLD) {
            // CORRECT: 文档质量高，直接使用
            result.action = CRAGAction.CORRECT;
            result.steps.add(new CRAGStep("DECIDE", "CORRECT", 
                    "文档质量高，直接使用"));
            result.finalDocs = initialDocs;
            
        } else if (avgScore < INCORRECT_THRESHOLD) {
            // INCORRECT: 文档质量低，需要修正
            result.action = CRAGAction.INCORRECT;
            result.steps.add(new CRAGStep("DECIDE", "INCORRECT", 
                    "文档质量低，触发知识精炼"));
            
            // 知识精炼: 提取有用片段 + 补充生成
            List<AiDoc> refinedDocs = refineKnowledge(query, initialDocs, scores);
            result.finalDocs = refinedDocs;
            result.steps.add(new CRAGStep("REFINE", "知识精炼", 
                    String.format("精炼后保留 %d 篇", refinedDocs.size())));
            
        } else {
            // AMBIGUOUS: 不确定，补充检索
            result.action = CRAGAction.AMBIGUOUS;
            result.steps.add(new CRAGStep("DECIDE", "AMBIGUOUS", 
                    "相关性模糊，补充检索"));
            
            // 查询改写后重新检索
            String rewrittenQuery = rewriteQuery(query);
            List<AiDoc> additionalDocs = hybridSearchService.hybridSearch(
                    rewrittenQuery, queryEmbedding, 3);
            
            // 合并去重
            Set<String> existingIds = initialDocs.stream()
                    .map(AiDoc::getId).collect(Collectors.toSet());
            List<AiDoc> newDocs = additionalDocs.stream()
                    .filter(d -> !existingIds.contains(d.getId()))
                    .collect(Collectors.toList());
            
            List<AiDoc> combined = new ArrayList<>(initialDocs);
            combined.addAll(newDocs);
            result.finalDocs = combined;
            
            result.steps.add(new CRAGStep("SUPPLEMENT", "补充检索", 
                    String.format("新增 %d 篇文档", newDocs.size())));
        }
        
        // Step 4: 生成最终答案
        log.info("[CRAG] Step 4: 生成答案 - 使用 {} 篇文档", result.finalDocs.size());
        List<String> contexts = result.finalDocs.stream()
                .map(AiDoc::getContent)
                .collect(Collectors.toList());
        result.answer = generateAnswer(query, contexts);
        result.steps.add(new CRAGStep("GENERATE", "生成答案", "完成"));
        
        result.latencyMs = System.currentTimeMillis() - startTime;
        log.info("[CRAG] 完成 - action={}, 耗时={}ms", result.action, result.latencyMs);
        
        return result;
    }
    
    /**
     * 评估文档相关性
     * 使用 LLM 作为评估器
     */
    private List<RelevanceScore> evaluateRelevance(String query, List<AiDoc> docs) {
        List<RelevanceScore> scores = new ArrayList<>();
        
        for (AiDoc doc : docs) {
            double score = calculateRelevance(query, doc);
            scores.add(new RelevanceScore(doc.getId(), doc.getTitle(), score));
        }
        
        return scores;
    }
    
    /**
     * 计算单个文档的相关性分数
     * 简化版: 基于关键词重叠 + 语义相似度估计
     */
    private double calculateRelevance(String query, AiDoc doc) {
        String content = doc.getContent().toLowerCase();
        String queryLower = query.toLowerCase();
        
        // 1. 关键词重叠率
        String[] queryTerms = queryLower.split("\\s+");
        long matchCount = Arrays.stream(queryTerms)
                .filter(term -> term.length() > 1 && content.contains(term))
                .count();
        double keywordScore = queryTerms.length > 0 ? 
                (double) matchCount / queryTerms.length : 0;
        
        // 2. 标题匹配加分
        double titleBonus = 0;
        if (doc.getTitle() != null) {
            String titleLower = doc.getTitle().toLowerCase();
            long titleMatch = Arrays.stream(queryTerms)
                    .filter(term -> titleLower.contains(term))
                    .count();
            titleBonus = queryTerms.length > 0 ? 
                    0.3 * titleMatch / queryTerms.length : 0;
        }
        
        // 3. 内容长度因子 (太短的文档可能信息不足)
        double lengthFactor = Math.min(1.0, doc.getContent().length() / 200.0);
        
        return Math.min(1.0, keywordScore * 0.5 + titleBonus + lengthFactor * 0.2);
    }
    
    /**
     * 知识精炼 - 从低质量文档中提取有用片段
     */
    private List<AiDoc> refineKnowledge(String query, List<AiDoc> docs, 
                                         List<RelevanceScore> scores) {
        // 只保留相关性高于最低阈值的文档
        Map<String, Double> scoreMap = scores.stream()
                .collect(Collectors.toMap(s -> s.docId, s -> s.score));
        
        return docs.stream()
                .filter(doc -> scoreMap.getOrDefault(doc.getId(), 0.0) > 0.2)
                .collect(Collectors.toList());
    }
    
    /**
     * 查询改写 - 用于补充检索
     */
    private String rewriteQuery(String query) {
        // 简化版: 添加同义词扩展
        if (query.contains("优化")) {
            return query + " 改进 提升 性能";
        }
        if (query.contains("错误") || query.contains("bug")) {
            return query + " 问题 修复 解决";
        }
        return query + " 方法 策略 实现";
    }
    
    /**
     * 无上下文直接生成
     */
    private String generateWithoutContext(String query) {
        if (!deepseekClient.enabled()) {
            return "AI 服务未配置";
        }
        String systemPrompt = """
            你是 Bot 开发助手，直接回答问题。
            
            格式要求：
            - 使用 Markdown 格式输出
            - 段落之间用空行分隔
            - 使用 ## 作为二级标题，### 作为三级标题
            - 代码必须用 ```语言名 和 ``` 包裹，每行代码单独一行
            - 列表项之间适当换行
            """;
        return deepseekClient.chat(systemPrompt, query, Collections.emptyList());
    }
    
    /**
     * 基于上下文生成答案
     */
    private String generateAnswer(String query, List<String> contexts) {
        if (!deepseekClient.enabled()) {
            return "AI 服务未配置";
        }
        
        String systemPrompt = """
            你是 Bot 开发助手，基于提供的参考文档回答问题。
            
            格式要求：
            - 使用 Markdown 格式输出
            - 段落之间用空行分隔
            - 使用 ## 作为二级标题，### 作为三级标题
            - 代码必须用 ```语言名 和 ``` 包裹，每行代码单独一行
            - 列表项之间适当换行
            
            内容要求：
            - 回答要简洁准确
            - 如果知识片段不相关，可以结合你的知识回答
            - 不知道就说不知道
            """;
        
        return deepseekClient.chat(systemPrompt, query, contexts);
    }
    
    private String truncate(String text, int maxLen) {
        if (text == null) return "";
        return text.length() > maxLen ? text.substring(0, maxLen) + "..." : text;
    }
    
    // ===== 结果类 =====
    
    public enum CRAGAction {
        CORRECT,    // 直接使用检索结果
        INCORRECT,  // 需要修正
        AMBIGUOUS   // 补充检索
    }
    
    public static class CRAGResult {
        public String query;
        public CRAGAction action;
        public String answer;
        public List<AiDoc> finalDocs;
        public List<RelevanceScore> relevanceScores;
        public double avgRelevance;
        public List<CRAGStep> steps;
        public long latencyMs;
    }
    
    public static class CRAGStep {
        public String phase;
        public String action;
        public String detail;
        
        public CRAGStep(String phase, String action, String detail) {
            this.phase = phase;
            this.action = action;
            this.detail = detail;
        }
    }
    
    public static class RelevanceScore {
        public String docId;
        public String title;
        public double score;
        
        public RelevanceScore(String docId, String title, double score) {
            this.docId = docId;
            this.title = title;
            this.score = score;
        }
    }
}
