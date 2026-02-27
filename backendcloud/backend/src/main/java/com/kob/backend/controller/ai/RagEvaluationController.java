package com.kob.backend.controller.ai;

import com.kob.backend.service.impl.ai.RagEvaluationService;
import com.kob.backend.service.impl.ai.RagEvaluationService.EvaluationResult;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * RAG评测API
 *
 * 接口说明：
 * - GET /ai/eval/run - 运行评测（同步，可能耗时较长）
 * - GET /ai/eval/run/async - 异步运行评测
 * - GET /ai/eval/result - 获取最近评测结果
 * - GET /ai/eval/status - 获取评测状态
 */
@RestController
@RequestMapping("/ai/eval")
public class RagEvaluationController {

    @Autowired
    private RagEvaluationService evaluationService;

    private CompletableFuture<EvaluationResult> runningTask = null;

    /**
     * 运行评测（同步）
     */
    @GetMapping("/run")
    public Map<String, Object> runEvaluation() {
        EvaluationResult result = evaluationService.runFullEvaluation();
        return result.toMap();
    }

    /**
     * 异步运行评测
     */
    @PostMapping("/run/async")
    public Map<String, Object> runEvaluationAsync() {
        if (runningTask != null && !runningTask.isDone()) {
            return Map.of(
                "status", "running",
                "message", "Evaluation already in progress"
            );
        }

        runningTask = evaluationService.runEvaluationAsync();
        return Map.of(
            "status", "started",
            "message", "Evaluation started, check /ai/eval/result for results"
        );
    }

    /**
     * 获取最近评测结果
     */
    @GetMapping("/result")
    public Map<String, Object> getResult() {
        EvaluationResult result = evaluationService.getLatestResult();
        Map<String, Object> response = new java.util.LinkedHashMap<>(result.toMap());

        // 添加详细结果（可选）
        if (!result.details.isEmpty()) {
            response.put("details", result.details.stream()
                .map(d -> Map.of(
                    "id", d.id,
                    "question", d.question,
                    "score", String.format("%.2f", d.score),
                    "latencyMs", d.latencyMs,
                    "hit", d.hit
                ))
                .toList());
        }

        return response;
    }

    /**
     * 获取评测状态
     */
    @GetMapping("/status")
    public Map<String, Object> getStatus() {
        boolean isRunning = runningTask != null && !runningTask.isDone();
        EvaluationResult latest = evaluationService.getLatestResult();

        return Map.of(
            "isRunning", isRunning,
            "hasResult", latest.error == null,
            "lastRunTimestamp", latest.timestamp,
            "lastAccuracy", latest.error == null ?
                String.format("%.2f%%", latest.accuracy * 100) : "N/A"
        );
    }
}
