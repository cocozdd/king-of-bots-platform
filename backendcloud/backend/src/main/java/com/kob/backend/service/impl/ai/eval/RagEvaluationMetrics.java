package com.kob.backend.service.impl.ai.eval;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * RAG 评测指标服务
 * 
 * 功能：
 * - 检索质量评估（Precision/Recall/MRR/NDCG）
 * - 生成质量评估（Faithfulness/Relevance）
 * - 端到端评估（Answer Correctness）
 * 
 * 面试要点：
 * - RAGAS 框架：Context Precision, Context Recall, Faithfulness, Answer Relevancy
 * - 离线评测 vs 在线评测
 * - Golden Dataset 构建
 */
@Service
public class RagEvaluationMetrics {
    
    private static final Logger log = LoggerFactory.getLogger(RagEvaluationMetrics.class);
    
    // 评测结果存储
    private final Map<String, EvaluationResult> evaluationHistory = new ConcurrentHashMap<>();
    private final AtomicLong totalQueries = new AtomicLong(0);
    private final AtomicInteger relevantRetrievals = new AtomicInteger(0);
    
    // 累计指标
    private double totalPrecision = 0.0;
    private double totalRecall = 0.0;
    private double totalMRR = 0.0;
    private double totalNDCG = 0.0;
    private int evaluationCount = 0;
    
    /**
     * 评估检索质量
     * 
     * @param queryId 查询ID
     * @param retrievedDocs 检索到的文档ID列表
     * @param relevantDocs 相关文档ID列表（Ground Truth）
     * @return 评估结果
     */
    public synchronized RetrievalMetrics evaluateRetrieval(
            String queryId,
            List<String> retrievedDocs,
            Set<String> relevantDocs) {
        
        totalQueries.incrementAndGet();
        
        // 计算 Precision@K
        int k = retrievedDocs.size();
        int relevantCount = 0;
        double dcg = 0.0;
        double idcg = 0.0;
        int firstRelevantRank = -1;
        
        for (int i = 0; i < k; i++) {
            boolean isRelevant = relevantDocs.contains(retrievedDocs.get(i));
            if (isRelevant) {
                relevantCount++;
                dcg += 1.0 / Math.log(i + 2);  // log2(rank + 1)
                if (firstRelevantRank == -1) {
                    firstRelevantRank = i + 1;
                }
            }
        }
        
        // 计算 IDCG（理想情况）
        int idealRelevant = Math.min(k, relevantDocs.size());
        for (int i = 0; i < idealRelevant; i++) {
            idcg += 1.0 / Math.log(i + 2);
        }
        
        double precision = k > 0 ? (double) relevantCount / k : 0;
        double recall = relevantDocs.size() > 0 ? 
                (double) relevantCount / relevantDocs.size() : 0;
        double mrr = firstRelevantRank > 0 ? 1.0 / firstRelevantRank : 0;
        double ndcg = idcg > 0 ? dcg / idcg : 0;
        
        // 更新累计指标
        totalPrecision += precision;
        totalRecall += recall;
        totalMRR += mrr;
        totalNDCG += ndcg;
        evaluationCount++;
        
        if (relevantCount > 0) {
            relevantRetrievals.incrementAndGet();
        }
        
        RetrievalMetrics metrics = new RetrievalMetrics(
                precision, recall, mrr, ndcg, k, relevantCount);
        
        log.info("检索评估 [{}]: P@{}={:.3f}, R={:.3f}, MRR={:.3f}, NDCG={:.3f}",
                queryId, k, precision, recall, mrr, ndcg);
        
        return metrics;
    }
    
    /**
     * 评估生成质量 - Faithfulness（忠实度）
     * 检查生成的答案是否基于检索到的上下文
     * 
     * @param answer 生成的答案
     * @param context 检索到的上下文
     * @return 忠实度得分 (0-1)
     */
    public double evaluateFaithfulness(String answer, String context) {
        if (answer == null || answer.isEmpty() || context == null || context.isEmpty()) {
            return 0.0;
        }
        
        // 简化实现：检查答案中的关键信息是否出现在上下文中
        // 生产环境可使用 NLI 模型或 LLM 评估
        String[] answerSentences = answer.split("[。.!?！？]");
        int supportedSentences = 0;
        
        for (String sentence : answerSentences) {
            sentence = sentence.trim();
            if (sentence.length() < 5) continue;
            
            // 检查句子中的关键词是否在上下文中
            String[] words = sentence.split("\\s+");
            int matchedWords = 0;
            for (String word : words) {
                if (word.length() > 2 && context.contains(word)) {
                    matchedWords++;
                }
            }
            
            if (words.length > 0 && (double) matchedWords / words.length > 0.3) {
                supportedSentences++;
            }
        }
        
        double faithfulness = answerSentences.length > 0 ? 
                (double) supportedSentences / answerSentences.length : 0;
        
        log.debug("忠实度评估: {:.3f} ({}/{} 句子有支撑)", 
                faithfulness, supportedSentences, answerSentences.length);
        
        return faithfulness;
    }
    
    /**
     * 评估生成质量 - Answer Relevancy（答案相关性）
     * 检查答案是否回答了问题
     * 
     * @param question 用户问题
     * @param answer 生成的答案
     * @return 相关性得分 (0-1)
     */
    public double evaluateAnswerRelevancy(String question, String answer) {
        if (question == null || answer == null) {
            return 0.0;
        }
        
        // 简化实现：检查问题关键词是否在答案中被提及
        Set<String> questionKeywords = extractKeywords(question);
        Set<String> answerKeywords = extractKeywords(answer);
        
        if (questionKeywords.isEmpty()) {
            return 0.5; // 无法评估
        }
        
        int matchedKeywords = 0;
        for (String keyword : questionKeywords) {
            if (answerKeywords.contains(keyword) || answer.toLowerCase().contains(keyword)) {
                matchedKeywords++;
            }
        }
        
        double relevancy = (double) matchedKeywords / questionKeywords.size();
        
        // 惩罚过短的答案
        if (answer.length() < 20) {
            relevancy *= 0.5;
        }
        
        return Math.min(1.0, relevancy);
    }
    
    /**
     * 提取关键词
     */
    private Set<String> extractKeywords(String text) {
        Set<String> keywords = new HashSet<>();
        String[] stopWords = {"的", "是", "在", "和", "了", "有", "我", "你", "这", "那",
                "a", "the", "is", "are", "to", "for", "of", "and", "how", "what", "why"};
        Set<String> stopSet = new HashSet<>(Arrays.asList(stopWords));
        
        String[] words = text.toLowerCase().split("[\\s，。？！,.?!]+");
        for (String word : words) {
            if (word.length() > 1 && !stopSet.contains(word)) {
                keywords.add(word);
            }
        }
        
        return keywords;
    }
    
    /**
     * 端到端评估
     */
    public EvaluationResult evaluateEndToEnd(
            String queryId,
            String question,
            String answer,
            String context,
            List<String> retrievedDocs,
            Set<String> relevantDocs) {
        
        // 检索质量
        RetrievalMetrics retrievalMetrics = evaluateRetrieval(queryId, retrievedDocs, relevantDocs);
        
        // 生成质量
        double faithfulness = evaluateFaithfulness(answer, context);
        double relevancy = evaluateAnswerRelevancy(question, answer);
        
        // 综合得分
        double overallScore = (retrievalMetrics.precision() * 0.3 +
                retrievalMetrics.recall() * 0.2 +
                faithfulness * 0.25 +
                relevancy * 0.25);
        
        EvaluationResult result = new EvaluationResult(
                queryId,
                retrievalMetrics,
                faithfulness,
                relevancy,
                overallScore,
                LocalDateTime.now()
        );
        
        evaluationHistory.put(queryId, result);
        
        log.info("端到端评估 [{}]: 综合得分={:.3f}, 忠实度={:.3f}, 相关性={:.3f}",
                queryId, overallScore, faithfulness, relevancy);
        
        return result;
    }
    
    /**
     * 获取累计指标
     */
    public Map<String, Object> getAggregatedMetrics() {
        Map<String, Object> metrics = new HashMap<>();
        
        if (evaluationCount > 0) {
            metrics.put("avgPrecision", totalPrecision / evaluationCount);
            metrics.put("avgRecall", totalRecall / evaluationCount);
            metrics.put("avgMRR", totalMRR / evaluationCount);
            metrics.put("avgNDCG", totalNDCG / evaluationCount);
        }
        
        metrics.put("totalQueries", totalQueries.get());
        metrics.put("evaluationCount", evaluationCount);
        metrics.put("hitRate", totalQueries.get() > 0 ? 
                (double) relevantRetrievals.get() / totalQueries.get() : 0);
        
        return metrics;
    }
    
    /**
     * 获取评估历史
     */
    public List<EvaluationResult> getRecentEvaluations(int limit) {
        return evaluationHistory.values().stream()
                .sorted(Comparator.comparing(EvaluationResult::timestamp).reversed())
                .limit(limit)
                .toList();
    }
    
    /**
     * 重置指标
     */
    public void resetMetrics() {
        totalPrecision = 0;
        totalRecall = 0;
        totalMRR = 0;
        totalNDCG = 0;
        evaluationCount = 0;
        totalQueries.set(0);
        relevantRetrievals.set(0);
        evaluationHistory.clear();
        log.info("评测指标已重置");
    }
    
    // 记录类
    public record RetrievalMetrics(
            double precision,
            double recall,
            double mrr,
            double ndcg,
            int k,
            int relevantCount
    ) {}
    
    public record EvaluationResult(
            String queryId,
            RetrievalMetrics retrievalMetrics,
            double faithfulness,
            double answerRelevancy,
            double overallScore,
            LocalDateTime timestamp
    ) {}
}
