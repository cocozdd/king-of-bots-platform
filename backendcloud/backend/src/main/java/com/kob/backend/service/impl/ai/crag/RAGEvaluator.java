package com.kob.backend.service.impl.ai.crag;

import com.kob.backend.controller.ai.dto.AiDoc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

/**
 * RAG 评估器 - 量化检索和生成质量
 * 
 * 面试要点:
 * - RAG 系统需要可量化的评估指标
 * - 检索质量: Recall@K, MRR, NDCG
 * - 生成质量: Faithfulness(忠实度), Relevance(相关性)
 * - 幻觉检测: 答案是否与上下文矛盾
 * 
 * 评估维度:
 * 1. Context Relevance - 检索文档与问题的相关性
 * 2. Answer Relevance - 答案与问题的相关性
 * 3. Faithfulness - 答案是否基于检索内容 (非幻觉)
 * 4. Context Utilization - 上下文利用率
 */
@Service
public class RAGEvaluator {
    
    private static final Logger log = LoggerFactory.getLogger(RAGEvaluator.class);
    
    /**
     * 综合评估 RAG 结果
     */
    public EvaluationResult evaluate(String query, String answer, List<AiDoc> retrievedDocs) {
        log.info("[Evaluator] 评估 RAG 结果 - query长度={}, 文档数={}", 
                query.length(), retrievedDocs.size());
        
        EvaluationResult result = new EvaluationResult();
        result.query = query;
        result.answer = answer;
        result.docCount = retrievedDocs.size();
        
        // 1. 评估检索相关性
        result.contextRelevance = evaluateContextRelevance(query, retrievedDocs);
        
        // 2. 评估答案相关性
        result.answerRelevance = evaluateAnswerRelevance(query, answer);
        
        // 3. 评估忠实度 (非幻觉)
        result.faithfulness = evaluateFaithfulness(answer, retrievedDocs);
        
        // 4. 评估上下文利用率
        result.contextUtilization = evaluateContextUtilization(answer, retrievedDocs);
        
        // 5. 计算综合得分
        result.overallScore = calculateOverallScore(result);
        
        // 6. 生成评估摘要
        result.summary = generateSummary(result);
        
        log.info("[Evaluator] 评估完成 - overall={:.2f}", result.overallScore);
        return result;
    }
    
    /**
     * 评估检索上下文与问题的相关性
     * Context Relevance = 相关文档数 / 总文档数
     */
    private double evaluateContextRelevance(String query, List<AiDoc> docs) {
        if (docs.isEmpty()) return 0;
        
        String queryLower = query.toLowerCase();
        String[] queryTerms = queryLower.split("\\s+");
        
        int relevantCount = 0;
        for (AiDoc doc : docs) {
            String content = (doc.getTitle() + " " + doc.getContent()).toLowerCase();
            long matchCount = Arrays.stream(queryTerms)
                    .filter(term -> term.length() > 1 && content.contains(term))
                    .count();
            
            // 超过 30% 的词匹配认为相关
            if (matchCount >= Math.max(1, queryTerms.length * 0.3)) {
                relevantCount++;
            }
        }
        
        return (double) relevantCount / docs.size();
    }
    
    /**
     * 评估答案与问题的相关性
     */
    private double evaluateAnswerRelevance(String query, String answer) {
        if (answer == null || answer.isEmpty()) return 0;
        
        String queryLower = query.toLowerCase();
        String answerLower = answer.toLowerCase();
        
        // 1. 关键词覆盖率
        String[] queryTerms = queryLower.split("\\s+");
        long coveredTerms = Arrays.stream(queryTerms)
                .filter(term -> term.length() > 1 && answerLower.contains(term))
                .count();
        double keywordCoverage = queryTerms.length > 0 ? 
                (double) coveredTerms / queryTerms.length : 0;
        
        // 2. 答案长度因子 (太短可能信息不足)
        double lengthFactor = Math.min(1.0, answer.length() / 100.0);
        
        // 3. 结构化因子 (包含要点/步骤等)
        double structureFactor = 0;
        if (answer.contains("1.") || answer.contains("1、") || 
            answer.contains("首先") || answer.contains("步骤")) {
            structureFactor = 0.2;
        }
        
        return Math.min(1.0, keywordCoverage * 0.5 + lengthFactor * 0.3 + structureFactor);
    }
    
    /**
     * 评估忠实度 - 答案是否基于检索内容
     * Faithfulness = 可追溯的陈述数 / 总陈述数
     */
    private double evaluateFaithfulness(String answer, List<AiDoc> docs) {
        if (answer == null || answer.isEmpty() || docs.isEmpty()) return 0;
        
        // 提取答案中的关键陈述
        String[] sentences = answer.split("[。！？.!?]");
        if (sentences.length == 0) return 0;
        
        // 合并所有文档内容
        String allContent = docs.stream()
                .map(d -> d.getTitle() + " " + d.getContent())
                .collect(Collectors.joining(" "))
                .toLowerCase();
        
        int groundedCount = 0;
        for (String sentence : sentences) {
            if (sentence.trim().isEmpty()) continue;
            
            // 检查句子中的关键词是否能在文档中找到支撑
            String[] words = sentence.toLowerCase().split("\\s+");
            long supportedWords = Arrays.stream(words)
                    .filter(w -> w.length() > 2 && allContent.contains(w))
                    .count();
            
            if (supportedWords >= Math.max(1, words.length * 0.3)) {
                groundedCount++;
            }
        }
        
        return (double) groundedCount / sentences.length;
    }
    
    /**
     * 评估上下文利用率 - 检索内容被答案使用的比例
     */
    private double evaluateContextUtilization(String answer, List<AiDoc> docs) {
        if (answer == null || answer.isEmpty() || docs.isEmpty()) return 0;
        
        String answerLower = answer.toLowerCase();
        
        int usedDocs = 0;
        for (AiDoc doc : docs) {
            // 提取文档的关键词
            String[] docWords = doc.getContent().toLowerCase().split("\\s+");
            Set<String> significantWords = Arrays.stream(docWords)
                    .filter(w -> w.length() > 3)
                    .limit(10)
                    .collect(Collectors.toSet());
            
            // 检查这些词是否出现在答案中
            long usedWords = significantWords.stream()
                    .filter(answerLower::contains)
                    .count();
            
            if (usedWords >= Math.max(1, significantWords.size() * 0.2)) {
                usedDocs++;
            }
        }
        
        return (double) usedDocs / docs.size();
    }
    
    /**
     * 计算综合得分 (加权平均)
     */
    private double calculateOverallScore(EvaluationResult result) {
        // 权重: 忠实度最重要 (防幻觉), 其次是答案相关性
        return result.faithfulness * 0.35 +
               result.answerRelevance * 0.30 +
               result.contextRelevance * 0.20 +
               result.contextUtilization * 0.15;
    }
    
    /**
     * 生成评估摘要
     */
    private String generateSummary(EvaluationResult result) {
        StringBuilder sb = new StringBuilder();
        
        if (result.overallScore >= 0.8) {
            sb.append("✅ 优秀 - ");
        } else if (result.overallScore >= 0.6) {
            sb.append("⚠️ 良好 - ");
        } else {
            sb.append("❌ 需改进 - ");
        }
        
        List<String> issues = new ArrayList<>();
        if (result.faithfulness < 0.5) issues.add("可能存在幻觉");
        if (result.contextRelevance < 0.5) issues.add("检索相关性低");
        if (result.answerRelevance < 0.5) issues.add("答案相关性低");
        if (result.contextUtilization < 0.3) issues.add("上下文利用不足");
        
        if (issues.isEmpty()) {
            sb.append("回答质量良好");
        } else {
            sb.append(String.join(", ", issues));
        }
        
        return sb.toString();
    }
    
    // ===== 结果类 =====
    
    public static class EvaluationResult {
        public String query;
        public String answer;
        public int docCount;
        
        // 各维度得分 (0-1)
        public double contextRelevance;     // 检索相关性
        public double answerRelevance;      // 答案相关性
        public double faithfulness;         // 忠实度 (非幻觉)
        public double contextUtilization;   // 上下文利用率
        
        public double overallScore;         // 综合得分
        public String summary;              // 评估摘要
        
        public Map<String, Double> toScoreMap() {
            Map<String, Double> map = new LinkedHashMap<>();
            map.put("contextRelevance", contextRelevance);
            map.put("answerRelevance", answerRelevance);
            map.put("faithfulness", faithfulness);
            map.put("contextUtilization", contextUtilization);
            map.put("overall", overallScore);
            return map;
        }
    }
}
