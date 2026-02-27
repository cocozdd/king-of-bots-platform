package com.kob.backend.service.impl.ai.selfrag;

import com.kob.backend.service.impl.ai.DeepseekClient;
import com.kob.backend.service.impl.ai.AiMetricsService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.util.*;

/**
 * Self-RAG 服务 - 自我反思的检索增强生成
 * 
 * 功能：
 * - 自适应检索：判断是否需要检索
 * - 检索质量评估：评估文档相关性
 * - 答案自我评估：检查生成质量
 * - 迭代优化：质量不佳时重新检索
 * 
 * 面试要点：
 * - Self-RAG vs 传统 RAG：加入反思 token，自主决策
 * - 四个特殊 token：[Retrieve]、[IsRel]、[IsSup]、[IsUse]
 * - 减少不必要检索，提升生成质量
 * 
 * 论文：Self-RAG: Learning to Retrieve, Generate, and Critique (2023)
 */
@Service
public class SelfRAGService {
    
    private static final Logger log = LoggerFactory.getLogger(SelfRAGService.class);
    
    @Autowired(required = false)
    private AiMetricsService metricsService;
    
    private DeepseekClient deepseekClient;
    
    // 阈值配置
    private static final double RELEVANCE_THRESHOLD = 0.6;
    private static final double SUPPORT_THRESHOLD = 0.7;
    private static final double USEFULNESS_THRESHOLD = 0.5;
    private static final int MAX_RETRIEVAL_ATTEMPTS = 3;
    
    @PostConstruct
    public void init() {
        this.deepseekClient = new DeepseekClient(metricsService);
    }
    
    /**
     * Self-RAG 主流程
     */
    public SelfRAGResult generate(String query, List<String> knowledgeBase) {
        log.info("Self-RAG 开始处理: {}", query);
        
        SelfRAGResult result = new SelfRAGResult();
        result.query = query;
        result.steps = new ArrayList<>();
        
        // Step 1: 判断是否需要检索 [Retrieve]
        boolean needRetrieval = assessRetrievalNeed(query);
        result.steps.add(new ReflectionStep("Retrieve", needRetrieval, 
                needRetrieval ? "问题需要外部知识" : "问题可直接回答"));
        
        String answer;
        List<String> usedDocs = new ArrayList<>();
        
        if (!needRetrieval) {
            // 不需要检索，直接生成
            answer = generateWithoutRetrieval(query);
        } else {
            // 需要检索，进入 Self-RAG 循环
            answer = selfRAGLoop(query, knowledgeBase, result, usedDocs);
        }
        
        result.answer = answer;
        result.usedDocuments = usedDocs;
        result.totalSteps = result.steps.size();
        
        log.info("Self-RAG 完成: {} 步反思, 使用 {} 个文档", 
                result.totalSteps, usedDocs.size());
        
        return result;
    }
    
    /**
     * Self-RAG 循环
     */
    private String selfRAGLoop(String query, List<String> knowledgeBase,
                                SelfRAGResult result, List<String> usedDocs) {
        
        for (int attempt = 0; attempt < MAX_RETRIEVAL_ATTEMPTS; attempt++) {
            log.debug("Self-RAG 第 {} 次尝试", attempt + 1);
            
            // Step 2: 检索文档
            List<RetrievedDoc> retrieved = retrieveDocuments(query, knowledgeBase, attempt);
            
            // Step 3: 评估检索相关性 [IsRel]
            List<RetrievedDoc> relevantDocs = new ArrayList<>();
            for (RetrievedDoc doc : retrieved) {
                double relevance = assessRelevance(query, doc.content);
                doc.relevanceScore = relevance;
                
                boolean isRelevant = relevance >= RELEVANCE_THRESHOLD;
                result.steps.add(new ReflectionStep("IsRel", isRelevant,
                        String.format("文档'%s' 相关性: %.2f", 
                                truncate(doc.content, 30), relevance)));
                
                if (isRelevant) {
                    relevantDocs.add(doc);
                }
            }
            
            if (relevantDocs.isEmpty()) {
                result.steps.add(new ReflectionStep("Retry", true, 
                        "无相关文档，尝试重新检索"));
                continue;
            }
            
            // Step 4: 生成答案
            String context = buildContext(relevantDocs);
            String answer = generateWithContext(query, context);
            
            // Step 5: 评估答案是否有支撑 [IsSup]
            double supportScore = assessSupport(answer, context);
            boolean isSupported = supportScore >= SUPPORT_THRESHOLD;
            result.steps.add(new ReflectionStep("IsSup", isSupported,
                    String.format("答案支撑度: %.2f", supportScore)));
            
            if (!isSupported) {
                result.steps.add(new ReflectionStep("Retry", true, 
                        "答案缺乏支撑，重新检索"));
                continue;
            }
            
            // Step 6: 评估答案有用性 [IsUse]
            double usefulnessScore = assessUsefulness(query, answer);
            boolean isUseful = usefulnessScore >= USEFULNESS_THRESHOLD;
            result.steps.add(new ReflectionStep("IsUse", isUseful,
                    String.format("答案有用性: %.2f", usefulnessScore)));
            
            if (isUseful) {
                // 记录使用的文档
                for (RetrievedDoc doc : relevantDocs) {
                    usedDocs.add(doc.content);
                }
                return answer;
            }
        }
        
        // 达到最大尝试次数，返回最佳努力答案
        result.steps.add(new ReflectionStep("Fallback", true, 
                "达到最大尝试次数，返回当前最佳答案"));
        return generateWithoutRetrieval(query);
    }
    
    /**
     * 判断是否需要检索 [Retrieve]
     */
    private boolean assessRetrievalNeed(String query) {
        // 简化实现：基于关键词判断
        // 生产环境可用分类器或 LLM 判断
        
        // 事实性问题需要检索
        String[] factKeywords = {"怎么", "如何", "什么是", "为什么", "区别", 
                "原理", "方法", "步骤", "代码", "算法"};
        
        // 简单问候不需要检索
        String[] greetingKeywords = {"你好", "谢谢", "再见", "好的"};
        
        String queryLower = query.toLowerCase();
        
        for (String keyword : greetingKeywords) {
            if (queryLower.contains(keyword)) {
                return false;
            }
        }
        
        for (String keyword : factKeywords) {
            if (queryLower.contains(keyword)) {
                return true;
            }
        }
        
        // 默认需要检索
        return query.length() > 10;
    }
    
    /**
     * 检索文档
     */
    private List<RetrievedDoc> retrieveDocuments(String query, 
            List<String> knowledgeBase, int attempt) {
        
        List<RetrievedDoc> results = new ArrayList<>();
        
        // 简化实现：关键词匹配
        // 生产环境使用向量检索
        String[] queryTerms = query.toLowerCase().split("\\s+");
        
        for (int i = 0; i < knowledgeBase.size(); i++) {
            String doc = knowledgeBase.get(i);
            String docLower = doc.toLowerCase();
            
            int matchCount = 0;
            for (String term : queryTerms) {
                if (term.length() > 1 && docLower.contains(term)) {
                    matchCount++;
                }
            }
            
            if (matchCount > 0) {
                RetrievedDoc retrieved = new RetrievedDoc();
                retrieved.id = "doc_" + i;
                retrieved.content = doc;
                retrieved.initialScore = (double) matchCount / queryTerms.length;
                results.add(retrieved);
            }
        }
        
        // 按初始分数排序
        results.sort((a, b) -> Double.compare(b.initialScore, a.initialScore));
        
        // 返回 Top-3
        return results.subList(0, Math.min(3, results.size()));
    }
    
    /**
     * 评估文档相关性 [IsRel]
     */
    private double assessRelevance(String query, String document) {
        // 简化实现：基于关键词重叠
        // 生产环境可用 Cross-Encoder 或 LLM
        
        Set<String> queryTerms = new HashSet<>(Arrays.asList(
                query.toLowerCase().split("\\s+")));
        Set<String> docTerms = new HashSet<>(Arrays.asList(
                document.toLowerCase().split("\\s+")));
        
        // 计算 Jaccard 相似度
        Set<String> intersection = new HashSet<>(queryTerms);
        intersection.retainAll(docTerms);
        
        Set<String> union = new HashSet<>(queryTerms);
        union.addAll(docTerms);
        
        if (union.isEmpty()) return 0.0;
        
        return (double) intersection.size() / union.size();
    }
    
    /**
     * 评估答案支撑度 [IsSup]
     */
    private double assessSupport(String answer, String context) {
        // 简化实现：检查答案中的关键词是否在上下文中
        // 生产环境可用 NLI 模型
        
        String[] answerSentences = answer.split("[。.!?！？]");
        int supportedCount = 0;
        
        for (String sentence : answerSentences) {
            if (sentence.trim().length() < 5) continue;
            
            String[] words = sentence.split("\\s+");
            int matched = 0;
            for (String word : words) {
                if (word.length() > 2 && context.contains(word)) {
                    matched++;
                }
            }
            
            if (words.length > 0 && (double) matched / words.length > 0.3) {
                supportedCount++;
            }
        }
        
        return answerSentences.length > 0 ? 
                (double) supportedCount / answerSentences.length : 0;
    }
    
    /**
     * 评估答案有用性 [IsUse]
     */
    private double assessUsefulness(String query, String answer) {
        // 简化实现：基于长度和关键词覆盖
        // 生产环境可用 LLM 评估
        
        if (answer == null || answer.length() < 20) {
            return 0.2; // 过短的答案不太有用
        }
        
        // 检查是否回答了问题中的关键词
        String[] queryTerms = query.toLowerCase().split("\\s+");
        String answerLower = answer.toLowerCase();
        
        int covered = 0;
        for (String term : queryTerms) {
            if (term.length() > 2 && answerLower.contains(term)) {
                covered++;
            }
        }
        
        double coverage = queryTerms.length > 0 ? 
                (double) covered / queryTerms.length : 0;
        
        // 综合评分
        double lengthScore = Math.min(1.0, answer.length() / 200.0);
        
        return coverage * 0.6 + lengthScore * 0.4;
    }
    
    /**
     * 无检索生成
     */
    private String generateWithoutRetrieval(String query) {
        try {
            String systemPrompt = """
                    你是 KOB 平台的 AI 助手，直接回答用户问题。
                    
                    格式要求：
                    - 使用 Markdown 格式输出
                    - 段落之间用空行分隔
                    - 使用 ## 作为二级标题，### 作为三级标题
                    - 代码必须用 ```语言名 和 ``` 包裹，每行代码单独一行
                    - 列表项之间适当换行
                    """;
            return deepseekClient.chat(systemPrompt, query, List.of());
        } catch (Exception e) {
            log.error("生成失败", e);
            return "抱歉，我暂时无法回答这个问题。";
        }
    }
    
    /**
     * 基于上下文生成
     */
    private String generateWithContext(String query, String context) {
        String systemPrompt = String.format("""
            你是 KOB 平台的 AI 助手。基于以下知识回答用户问题。
            只使用提供的知识，不要编造信息。
            
            格式要求：
            - 使用 Markdown 格式输出
            - 段落之间用空行分隔
            - 使用 ## 作为二级标题，### 作为三级标题
            - 代码必须用 ```语言名 和 ``` 包裹，每行代码单独一行
            - 列表项之间适当换行
            
            知识：
            %s
            """, context);
        
        try {
            return deepseekClient.chat(systemPrompt,
                    query,
                    List.of());
        } catch (Exception e) {
            log.error("生成失败", e);
            return "抱歉，生成答案时出现错误。";
        }
    }
    
    /**
     * 构建上下文
     */
    private String buildContext(List<RetrievedDoc> docs) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < docs.size(); i++) {
            sb.append(String.format("[%d] %s\n\n", i + 1, docs.get(i).content));
        }
        return sb.toString();
    }
    
    /**
     * 截断文本
     */
    private String truncate(String text, int maxLen) {
        if (text.length() <= maxLen) return text;
        return text.substring(0, maxLen) + "...";
    }
    
    // 内部类
    public static class SelfRAGResult {
        public String query;
        public String answer;
        public List<String> usedDocuments;
        public List<ReflectionStep> steps;
        public int totalSteps;
    }
    
    public static class ReflectionStep {
        public String token;      // Retrieve, IsRel, IsSup, IsUse
        public boolean passed;
        public String reason;
        
        public ReflectionStep(String token, boolean passed, String reason) {
            this.token = token;
            this.passed = passed;
            this.reason = reason;
        }
    }
    
    private static class RetrievedDoc {
        String id;
        String content;
        double initialScore;
        double relevanceScore;
    }
}
