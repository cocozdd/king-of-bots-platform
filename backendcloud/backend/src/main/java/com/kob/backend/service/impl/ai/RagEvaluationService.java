package com.kob.backend.service.impl.ai;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * RAG系统评测服务
 * 用于量化评估RAG问答系统的准确率和性能
 *
 * 面试要点：
 * 1. 评测指标：准确率、召回率、P95延迟
 * 2. 评测方法：关键词命中率 + 语义相似度
 * 3. 优化方向：Query Rewriting、Hybrid Search、Re-ranking
 */
@Service
public class RagEvaluationService {

    private static final Logger logger = LoggerFactory.getLogger(RagEvaluationService.class);

    @Autowired(required = false)
    private AiHintServiceImpl aiHintService;

    private final ObjectMapper objectMapper = new ObjectMapper();

    // 评测结果缓存
    private final Map<String, EvaluationResult> resultCache = new ConcurrentHashMap<>();

    /**
     * 运行完整评测
     */
    public EvaluationResult runFullEvaluation() {
        List<TestCase> testCases = loadTestCases();
        if (testCases.isEmpty()) {
            return EvaluationResult.empty("No test cases loaded");
        }

        List<TestResult> results = new ArrayList<>();
        List<Long> latencies = new ArrayList<>();
        int hits = 0;

        for (TestCase testCase : testCases) {
            long startTime = System.currentTimeMillis();

            try {
                // 调用RAG服务获取答案
                String answer = queryRag(testCase.question);
                long latency = System.currentTimeMillis() - startTime;
                latencies.add(latency);

                // 评估答案质量
                double score = evaluateAnswer(answer, testCase.expectedKeywords);
                boolean isHit = score >= 0.5; // 命中阈值
                if (isHit) hits++;

                results.add(new TestResult(
                    testCase.id,
                    testCase.question,
                    answer,
                    score,
                    latency,
                    isHit
                ));

                logger.info("Evaluated {}: score={}, latency={}ms",
                    testCase.id, String.format("%.2f", score), latency);

            } catch (Exception e) {
                logger.error("Failed to evaluate {}: {}", testCase.id, e.getMessage());
                results.add(new TestResult(
                    testCase.id, testCase.question, "ERROR: " + e.getMessage(),
                    0.0, System.currentTimeMillis() - startTime, false
                ));
            }
        }

        // 计算统计指标
        double accuracy = (double) hits / testCases.size();
        double avgLatency = latencies.stream().mapToLong(Long::longValue).average().orElse(0);
        long p95Latency = calculateP95(latencies);

        EvaluationResult result = new EvaluationResult(
            accuracy,
            avgLatency,
            p95Latency,
            testCases.size(),
            hits,
            results,
            System.currentTimeMillis()
        );

        // 缓存结果
        resultCache.put("latest", result);

        return result;
    }

    /**
     * 异步运行评测
     */
    public CompletableFuture<EvaluationResult> runEvaluationAsync() {
        return CompletableFuture.supplyAsync(this::runFullEvaluation);
    }

    /**
     * 获取最近的评测结果
     */
    public EvaluationResult getLatestResult() {
        return resultCache.getOrDefault("latest", EvaluationResult.empty("No evaluation run yet"));
    }

    /**
     * 加载测试用例
     */
    private List<TestCase> loadTestCases() {
        try {
            // 从classpath加载，或从外部文件加载
            InputStream is = getClass().getResourceAsStream("/rag-eval-dataset.json");
            if (is == null) {
                // 尝试从项目根目录加载
                ClassPathResource resource = new ClassPathResource("rag-eval-dataset.json");
                if (resource.exists()) {
                    is = resource.getInputStream();
                }
            }

            if (is == null) {
                logger.warn("Evaluation dataset not found, using built-in test cases");
                return getBuiltInTestCases();
            }

            JsonNode root = objectMapper.readTree(is);
            JsonNode testCasesNode = root.get("test_cases");

            List<TestCase> testCases = new ArrayList<>();
            for (JsonNode node : testCasesNode) {
                List<String> keywords = new ArrayList<>();
                for (JsonNode kw : node.get("expected_keywords")) {
                    keywords.add(kw.asText());
                }
                testCases.add(new TestCase(
                    node.get("id").asText(),
                    node.get("question").asText(),
                    keywords,
                    node.get("category").asText(),
                    node.get("difficulty").asText()
                ));
            }
            return testCases;

        } catch (IOException e) {
            logger.error("Failed to load test cases: {}", e.getMessage());
            return getBuiltInTestCases();
        }
    }

    /**
     * 内置测试用例（备用）
     */
    private List<TestCase> getBuiltInTestCases() {
        return Arrays.asList(
            new TestCase("qa_001", "蛇怎么移动？",
                Arrays.asList("方向", "上", "右", "下", "左"), "game_rules", "easy"),
            new TestCase("qa_002", "什么情况下会输？",
                Arrays.asList("撞墙", "碰撞", "身体"), "game_rules", "easy"),
            new TestCase("qa_003", "如何实现BFS寻路？",
                Arrays.asList("队列", "广度优先", "最短路径"), "algorithm", "medium"),
            new TestCase("qa_004", "怎么避开障碍物？",
                Arrays.asList("检测", "方向", "安全"), "strategy", "medium"),
            new TestCase("qa_005", "MCTS算法怎么用？",
                Arrays.asList("模拟", "选择", "扩展", "UCB"), "algorithm", "hard")
        );
    }

    /**
     * 调用RAG服务
     */
    private String queryRag(String question) {
        if (aiHintService != null) {
            try {
                // 使用同步方式获取答案
                return aiHintService.getHintSync(question);
            } catch (Exception e) {
                logger.warn("RAG service call failed: {}", e.getMessage());
            }
        }
        // 降级：返回模拟答案
        return "RAG服务未启用，这是模拟答案。关于 " + question + " 的回答...";
    }

    /**
     * 评估答案质量（关键词命中率）
     */
    private double evaluateAnswer(String answer, List<String> expectedKeywords) {
        if (answer == null || answer.isEmpty() || expectedKeywords.isEmpty()) {
            return 0.0;
        }

        String lowerAnswer = answer.toLowerCase();
        long hits = expectedKeywords.stream()
            .filter(kw -> lowerAnswer.contains(kw.toLowerCase()))
            .count();

        return (double) hits / expectedKeywords.size();
    }

    /**
     * 计算P95延迟
     */
    private long calculateP95(List<Long> latencies) {
        if (latencies.isEmpty()) return 0;

        List<Long> sorted = latencies.stream().sorted().collect(Collectors.toList());
        int p95Index = (int) Math.ceil(sorted.size() * 0.95) - 1;
        return sorted.get(Math.max(0, p95Index));
    }

    // ==================== 数据类 ====================

    public static class TestCase {
        public final String id;
        public final String question;
        public final List<String> expectedKeywords;
        public final String category;
        public final String difficulty;

        public TestCase(String id, String question, List<String> expectedKeywords,
                       String category, String difficulty) {
            this.id = id;
            this.question = question;
            this.expectedKeywords = expectedKeywords;
            this.category = category;
            this.difficulty = difficulty;
        }
    }

    public static class TestResult {
        public final String id;
        public final String question;
        public final String answer;
        public final double score;
        public final long latencyMs;
        public final boolean hit;

        public TestResult(String id, String question, String answer,
                         double score, long latencyMs, boolean hit) {
            this.id = id;
            this.question = question;
            this.answer = answer;
            this.score = score;
            this.latencyMs = latencyMs;
            this.hit = hit;
        }
    }

    public static class EvaluationResult {
        public final double accuracy;
        public final double avgLatencyMs;
        public final long p95LatencyMs;
        public final int totalCases;
        public final int hitCases;
        public final List<TestResult> details;
        public final long timestamp;
        public final String error;

        public EvaluationResult(double accuracy, double avgLatencyMs, long p95LatencyMs,
                               int totalCases, int hitCases, List<TestResult> details,
                               long timestamp) {
            this.accuracy = accuracy;
            this.avgLatencyMs = avgLatencyMs;
            this.p95LatencyMs = p95LatencyMs;
            this.totalCases = totalCases;
            this.hitCases = hitCases;
            this.details = details;
            this.timestamp = timestamp;
            this.error = null;
        }

        private EvaluationResult(String error) {
            this.accuracy = 0;
            this.avgLatencyMs = 0;
            this.p95LatencyMs = 0;
            this.totalCases = 0;
            this.hitCases = 0;
            this.details = Collections.emptyList();
            this.timestamp = System.currentTimeMillis();
            this.error = error;
        }

        public static EvaluationResult empty(String error) {
            return new EvaluationResult(error);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("accuracy", String.format("%.2f%%", accuracy * 100));
            map.put("avgLatencyMs", String.format("%.0f", avgLatencyMs));
            map.put("p95LatencyMs", p95LatencyMs);
            map.put("totalCases", totalCases);
            map.put("hitCases", hitCases);
            map.put("timestamp", timestamp);
            if (error != null) {
                map.put("error", error);
            }
            return map;
        }
    }
}
