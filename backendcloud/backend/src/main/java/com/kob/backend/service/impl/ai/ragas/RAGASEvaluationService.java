package com.kob.backend.service.impl.ai.ragas;

import com.kob.backend.controller.ai.dto.AiDoc;
import com.kob.backend.service.impl.ai.AiMetricsService;
import com.kob.backend.service.impl.ai.DeepseekClient;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.util.*;
import java.util.stream.Collectors;

/**
 * RAGAS (Retrieval Augmented Generation Assessment) 评估服务
 * 
 * 实现 RAGAS 框架的核心评估指标：
 * 1. Faithfulness (忠实度) - 答案是否基于检索内容
 * 2. Answer Relevancy (答案相关性) - 答案是否回答了问题
 * 3. Context Precision (上下文精确率) - 检索内容是否相关
 * 4. Context Recall (上下文召回率) - 是否检索到所有相关内容
 * 
 * 论文: RAGAS: Automated Evaluation of Retrieval Augmented Generation (2023)
 * 
 * 面试要点：
 * - 为什么需要 RAGAS：传统指标（BLEU/ROUGE）不适合评估 RAG
 * - 核心思想：使用 LLM 作为评估器
 * - 四个指标的计算方法和意义
 * - 如何平衡评估成本和准确性
 */
@Service
public class RAGASEvaluationService {
    
    private static final Logger log = LoggerFactory.getLogger(RAGASEvaluationService.class);
    
    @Autowired(required = false)
    private AiMetricsService metricsService;
    
    private DeepseekClient deepseekClient;
    
    @PostConstruct
    public void init() {
        this.deepseekClient = new DeepseekClient(metricsService);
        log.info("RAGAS Evaluation Service 初始化完成");
    }
    
    /**
     * 完整 RAGAS 评估
     * 
     * @param question 用户问题
     * @param answer RAG 生成的答案
     * @param contexts 检索到的上下文
     * @param groundTruth 标准答案（可选，用于计算召回率）
     * @return 评估结果
     */
    public RAGASResult evaluate(
            String question, 
            String answer, 
            List<String> contexts,
            String groundTruth) {
        
        long startTime = System.currentTimeMillis();
        RAGASResult result = new RAGASResult();
        result.question = question;
        result.answer = answer;
        result.contextCount = contexts != null ? contexts.size() : 0;
        
        log.info("[RAGAS] 开始评估: question='{}'", truncate(question, 50));
        
        // 1. Faithfulness 评估
        result.faithfulness = evaluateFaithfulness(answer, contexts);
        log.debug("[RAGAS] Faithfulness = {:.2f}", result.faithfulness);
        
        // 2. Answer Relevancy 评估
        result.answerRelevancy = evaluateAnswerRelevancy(question, answer);
        log.debug("[RAGAS] Answer Relevancy = {:.2f}", result.answerRelevancy);
        
        // 3. Context Precision 评估
        result.contextPrecision = evaluateContextPrecision(question, contexts);
        log.debug("[RAGAS] Context Precision = {:.2f}", result.contextPrecision);
        
        // 4. Context Recall 评估（需要 ground truth）
        if (groundTruth != null && !groundTruth.isEmpty()) {
            result.contextRecall = evaluateContextRecall(groundTruth, contexts);
            log.debug("[RAGAS] Context Recall = {:.2f}", result.contextRecall);
        }
        
        // 计算综合分数
        result.overallScore = calculateOverallScore(result);
        
        result.latencyMs = System.currentTimeMillis() - startTime;
        
        log.info("[RAGAS] 评估完成: overall={:.2f}, faithfulness={:.2f}, relevancy={:.2f}, precision={:.2f}, {}ms",
                result.overallScore, result.faithfulness, result.answerRelevancy, 
                result.contextPrecision, result.latencyMs);
        
        return result;
    }
    
    /**
     * 快速评估（仅计算关键指标）
     */
    public RAGASResult quickEvaluate(String question, String answer, List<String> contexts) {
        return evaluate(question, answer, contexts, null);
    }
    
    /**
     * 1. Faithfulness 评估
     * 
     * 衡量答案中的陈述是否都可以从检索内容中推断出来
     * 
     * 计算方法：
     * 1. 将答案分解为原子陈述
     * 2. 判断每个陈述是否被上下文支持
     * 3. Faithfulness = 支持的陈述数 / 总陈述数
     */
    private double evaluateFaithfulness(String answer, List<String> contexts) {
        if (answer == null || answer.isEmpty()) return 0.0;
        if (contexts == null || contexts.isEmpty()) return 0.0;
        
        if (!deepseekClient.enabled()) {
            return estimateFaithfulnessHeuristic(answer, contexts);
        }
        
        String systemPrompt = """
            你是一个事实核查专家。评估答案中的陈述是否被上下文支持。
            
            任务：
            1. 将答案分解为独立的事实陈述
            2. 判断每个陈述是否可以从上下文中推断
            3. 计算支持率
            
            返回 JSON 格式：
            {
                "statements": ["陈述1", "陈述2", ...],
                "supported": [true, false, ...],
                "faithfulness": 0.75,
                "reasoning": "分析过程"
            }
            """;
        
        String contextStr = contexts.stream()
                .limit(3)
                .collect(Collectors.joining("\n---\n"));
        
        String userMessage = String.format(
                "上下文:\n%s\n\n答案:\n%s\n\n请评估答案的忠实度（返回 JSON）：",
                contextStr, answer);
        
        try {
            String response = deepseekClient.chat(systemPrompt, userMessage, Collections.emptyList());
            return parseScoreFromResponse(response, "faithfulness", 0.5);
        } catch (Exception e) {
            log.warn("[RAGAS] Faithfulness 评估失败: {}", e.getMessage());
            return estimateFaithfulnessHeuristic(answer, contexts);
        }
    }
    
    /**
     * 2. Answer Relevancy 评估
     * 
     * 衡量答案与问题的相关程度
     * 
     * 计算方法：
     * 1. 从答案反向生成可能的问题
     * 2. 计算生成问题与原问题的相似度
     * 3. Answer Relevancy = 平均相似度
     */
    private double evaluateAnswerRelevancy(String question, String answer) {
        if (question == null || answer == null) return 0.0;
        if (question.isEmpty() || answer.isEmpty()) return 0.0;
        
        if (!deepseekClient.enabled()) {
            return estimateRelevancyHeuristic(question, answer);
        }
        
        String systemPrompt = """
            你是一个答案质量评估专家。评估答案与问题的相关程度。
            
            评估标准：
            - 答案是否直接回答了问题
            - 答案是否包含问题所需的信息
            - 答案是否简洁且不跑题
            
            返回 JSON 格式：
            {
                "relevancy": 0.85,
                "directlyAnswers": true,
                "containsRequired": true,
                "concise": true,
                "reasoning": "评估说明"
            }
            """;
        
        String userMessage = String.format(
                "问题: %s\n\n答案: %s\n\n请评估答案相关性（返回 JSON）：",
                question, answer);
        
        try {
            String response = deepseekClient.chat(systemPrompt, userMessage, Collections.emptyList());
            return parseScoreFromResponse(response, "relevancy", 0.5);
        } catch (Exception e) {
            log.warn("[RAGAS] Answer Relevancy 评估失败: {}", e.getMessage());
            return estimateRelevancyHeuristic(question, answer);
        }
    }
    
    /**
     * 3. Context Precision 评估
     * 
     * 衡量检索到的上下文是否与问题相关
     * 
     * 计算方法：
     * 1. 判断每个上下文块是否与问题相关
     * 2. 按排名加权（前面的更重要）
     * 3. Context Precision = Σ (相关性 × 位置权重) / Σ 位置权重
     */
    private double evaluateContextPrecision(String question, List<String> contexts) {
        if (question == null || contexts == null || contexts.isEmpty()) return 0.0;
        
        if (!deepseekClient.enabled()) {
            return estimatePrecisionHeuristic(question, contexts);
        }
        
        String systemPrompt = """
            你是一个检索质量评估专家。判断每个检索结果是否与问题相关。
            
            评估标准：
            - 相关 (1): 包含回答问题所需的信息
            - 部分相关 (0.5): 包含一些相关信息但不充分
            - 不相关 (0): 与问题无关
            
            返回 JSON 格式：
            {
                "relevanceScores": [1.0, 0.5, 0, ...],
                "precision": 0.75,
                "reasoning": "评估说明"
            }
            """;
        
        StringBuilder contextList = new StringBuilder();
        for (int i = 0; i < Math.min(contexts.size(), 5); i++) {
            contextList.append(String.format("[%d] %s\n\n", i + 1, truncate(contexts.get(i), 200)));
        }
        
        String userMessage = String.format(
                "问题: %s\n\n检索结果:\n%s\n请评估每个结果的相关性（返回 JSON）：",
                question, contextList);
        
        try {
            String response = deepseekClient.chat(systemPrompt, userMessage, Collections.emptyList());
            return parseScoreFromResponse(response, "precision", 0.5);
        } catch (Exception e) {
            log.warn("[RAGAS] Context Precision 评估失败: {}", e.getMessage());
            return estimatePrecisionHeuristic(question, contexts);
        }
    }
    
    /**
     * 4. Context Recall 评估
     * 
     * 衡量检索结果是否覆盖了回答问题所需的所有信息
     * 
     * 计算方法：
     * 1. 将标准答案分解为关键信息点
     * 2. 检查每个信息点是否在上下文中出现
     * 3. Context Recall = 覆盖的信息点数 / 总信息点数
     */
    private double evaluateContextRecall(String groundTruth, List<String> contexts) {
        if (groundTruth == null || groundTruth.isEmpty()) return -1.0; // 未计算
        if (contexts == null || contexts.isEmpty()) return 0.0;
        
        if (!deepseekClient.enabled()) {
            return estimateRecallHeuristic(groundTruth, contexts);
        }
        
        String systemPrompt = """
            你是一个信息完整性评估专家。检查上下文是否包含标准答案所需的所有信息。
            
            任务：
            1. 从标准答案中提取关键信息点
            2. 检查每个信息点是否在上下文中出现
            3. 计算召回率
            
            返回 JSON 格式：
            {
                "keyPoints": ["要点1", "要点2", ...],
                "covered": [true, false, ...],
                "recall": 0.80,
                "reasoning": "评估说明"
            }
            """;
        
        String contextStr = contexts.stream()
                .limit(5)
                .collect(Collectors.joining("\n---\n"));
        
        String userMessage = String.format(
                "标准答案:\n%s\n\n上下文:\n%s\n\n请评估上下文的覆盖率（返回 JSON）：",
                groundTruth, contextStr);
        
        try {
            String response = deepseekClient.chat(systemPrompt, userMessage, Collections.emptyList());
            return parseScoreFromResponse(response, "recall", 0.5);
        } catch (Exception e) {
            log.warn("[RAGAS] Context Recall 评估失败: {}", e.getMessage());
            return estimateRecallHeuristic(groundTruth, contexts);
        }
    }
    
    /**
     * 从 LLM 响应中解析分数
     */
    private double parseScoreFromResponse(String response, String key, double defaultValue) {
        try {
            int start = response.indexOf("{");
            int end = response.lastIndexOf("}");
            if (start >= 0 && end > start) {
                String jsonStr = response.substring(start, end + 1);
                JSONObject json = JSON.parseObject(jsonStr);
                
                // 尝试多个可能的 key
                String[] possibleKeys = {key, key + "Score", key.toLowerCase()};
                for (String k : possibleKeys) {
                    if (json.containsKey(k)) {
                        return Math.max(0, Math.min(1, json.getDoubleValue(k)));
                    }
                }
            }
        } catch (Exception e) {
            log.debug("[RAGAS] 解析失败: {}", e.getMessage());
        }
        return defaultValue;
    }
    
    /**
     * 计算综合分数
     */
    private double calculateOverallScore(RAGASResult result) {
        double sum = 0;
        int count = 0;
        
        if (result.faithfulness >= 0) { sum += result.faithfulness; count++; }
        if (result.answerRelevancy >= 0) { sum += result.answerRelevancy; count++; }
        if (result.contextPrecision >= 0) { sum += result.contextPrecision; count++; }
        if (result.contextRecall >= 0) { sum += result.contextRecall; count++; }
        
        return count > 0 ? sum / count : 0;
    }
    
    // ===== 启发式估算方法（无 LLM 时的降级） =====
    
    private double estimateFaithfulnessHeuristic(String answer, List<String> contexts) {
        String combinedContext = String.join(" ", contexts).toLowerCase();
        String[] words = answer.toLowerCase().split("\\s+");
        
        int matched = 0;
        for (String word : words) {
            if (word.length() > 2 && combinedContext.contains(word)) {
                matched++;
            }
        }
        
        return words.length > 0 ? (double) matched / words.length : 0;
    }
    
    private double estimateRelevancyHeuristic(String question, String answer) {
        String[] questionTerms = question.toLowerCase().split("\\s+");
        String answerLower = answer.toLowerCase();
        
        int matched = 0;
        for (String term : questionTerms) {
            if (term.length() > 1 && answerLower.contains(term)) {
                matched++;
            }
        }
        
        return questionTerms.length > 0 ? (double) matched / questionTerms.length : 0;
    }
    
    private double estimatePrecisionHeuristic(String question, List<String> contexts) {
        String questionLower = question.toLowerCase();
        String[] terms = questionLower.split("\\s+");
        
        double totalScore = 0;
        for (String context : contexts) {
            String contextLower = context.toLowerCase();
            int matched = 0;
            for (String term : terms) {
                if (term.length() > 1 && contextLower.contains(term)) {
                    matched++;
                }
            }
            totalScore += terms.length > 0 ? (double) matched / terms.length : 0;
        }
        
        return contexts.isEmpty() ? 0 : totalScore / contexts.size();
    }
    
    private double estimateRecallHeuristic(String groundTruth, List<String> contexts) {
        String[] keyTerms = groundTruth.toLowerCase().split("\\s+");
        String combinedContext = String.join(" ", contexts).toLowerCase();
        
        int covered = 0;
        for (String term : keyTerms) {
            if (term.length() > 2 && combinedContext.contains(term)) {
                covered++;
            }
        }
        
        return keyTerms.length > 0 ? (double) covered / keyTerms.length : 0;
    }
    
    private String truncate(String text, int maxLen) {
        if (text == null) return "";
        return text.length() > maxLen ? text.substring(0, maxLen) + "..." : text;
    }
    
    // ===== 结果类 =====
    
    public static class RAGASResult {
        public String question;
        public String answer;
        public int contextCount;
        
        // RAGAS 四大指标
        public double faithfulness = -1;      // 忠实度
        public double answerRelevancy = -1;   // 答案相关性
        public double contextPrecision = -1;  // 上下文精确率
        public double contextRecall = -1;     // 上下文召回率
        
        public double overallScore;           // 综合分数
        public long latencyMs;
        
        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("question", question);
            map.put("answerPreview", truncate(answer, 100));
            map.put("contextCount", contextCount);
            
            Map<String, Object> metrics = new LinkedHashMap<>();
            metrics.put("faithfulness", formatScore(faithfulness));
            metrics.put("answerRelevancy", formatScore(answerRelevancy));
            metrics.put("contextPrecision", formatScore(contextPrecision));
            metrics.put("contextRecall", formatScore(contextRecall));
            metrics.put("overall", formatScore(overallScore));
            map.put("metrics", metrics);
            
            map.put("latencyMs", latencyMs);
            
            return map;
        }
        
        private String formatScore(double score) {
            if (score < 0) return "N/A";
            return String.format("%.2f", score);
        }
        
        private String truncate(String text, int maxLen) {
            if (text == null) return "";
            return text.length() > maxLen ? text.substring(0, maxLen) + "..." : text;
        }
        
        /**
         * 获取评级
         */
        public String getGrade() {
            if (overallScore >= 0.8) return "A";
            if (overallScore >= 0.6) return "B";
            if (overallScore >= 0.4) return "C";
            return "D";
        }
    }
}
