package com.kob.backend.controller.ai;

import com.kob.backend.controller.ai.dto.AiHintRequest;
import com.kob.backend.controller.ai.dto.AiHintResponse;
import com.kob.backend.controller.ai.dto.AiDoc;
import com.kob.backend.repository.AiCorpusRepository;
import com.kob.backend.service.ai.AiHintService;
import com.kob.backend.service.impl.ai.*;
import com.kob.backend.service.impl.ai.search.HybridSearchService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import javax.annotation.PostConstruct;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

/**
 * AI 提示控制器
 * 
 * 提供基于 RAG 的 AI 辅助功能
 * 使用异步处理避免阻塞，提升用户体验
 */
@RestController
public class AiHintController {

    private static final Logger log = LoggerFactory.getLogger(AiHintController.class);

    @Autowired
    private AiHintService aiHintService;
    
    @Autowired
    private AiMetricsService metricsService;
    
    @Autowired
    private PromptSecurityService securityService;
    
    @Autowired
    private EmbeddingCacheService cacheService;
    
    @Autowired
    private AiCorpusRepository corpusRepository;
    
    @Autowired
    private HybridSearchService hybridSearchService;
    
    @Value("${dashscope.api.key:}")
    private String dashscopeApiKey;
    
    private DashscopeEmbeddingClient dashscopeClient;
    private DeepseekClient deepseekClient;
    private final ExecutorService executor = Executors.newFixedThreadPool(4);
    
    @PostConstruct
    public void init() {
        String key = dashscopeApiKey;
        if (key == null || key.isEmpty()) {
            key = System.getenv("DASHSCOPE_API_KEY");
        }
        dashscopeClient = new DashscopeEmbeddingClient(key, metricsService, cacheService);
        deepseekClient = new DeepseekClient(metricsService);
    }

    /**
     * 获取 AI 提示（异步 + 安全验证）
     * 
     * @param request 包含用户问题、代码片段等
     * @return AI 生成的回答和引用来源
     */
    @PostMapping("/ai/hint")
    public CompletableFuture<ResponseEntity<AiHintResponse>> hint(@RequestBody AiHintRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                // 1. 安全验证 - 问题
                PromptSecurityService.ValidationResult questionResult = 
                    securityService.validateQuestion(request.getQuestion());
                if (!questionResult.isSafe()) {
                    securityService.logSuspiciousRequest(request.getQuestion(), questionResult.getReason());
                    return ResponseEntity.ok(AiHintResponse.error("输入不合法: " + questionResult.getReason()));
                }
                
                // 2. 安全验证 - 代码片段
                PromptSecurityService.ValidationResult codeResult = 
                    securityService.validateCodeSnippet(request.getCodeSnippet());
                if (!codeResult.isSafe()) {
                    securityService.logSuspiciousRequest(request.getCodeSnippet(), codeResult.getReason());
                    return ResponseEntity.ok(AiHintResponse.error("输入不合法: " + codeResult.getReason()));
                }
                
                // 3. 安全验证 - 错误日志
                PromptSecurityService.ValidationResult errorResult = 
                    securityService.validateErrorLog(request.getErrorLog());
                if (!errorResult.isSafe()) {
                    securityService.logSuspiciousRequest(request.getErrorLog(), errorResult.getReason());
                    return ResponseEntity.ok(AiHintResponse.error("输入不合法: " + errorResult.getReason()));
                }
                
                // 4. 使用清洗后的输入
                request.setQuestion(questionResult.getSanitizedInput());
                request.setCodeSnippet(codeResult.getSanitizedInput());
                request.setErrorLog(errorResult.getSanitizedInput());
                
                // 5. 调用 AI 服务
                AiHintResponse response = aiHintService.hint(request);
                return ResponseEntity.ok(response);
            } catch (Exception e) {
                log.error("AI hint failed for question: {}", request.getQuestion(), e);
                return ResponseEntity.ok(AiHintResponse.error("AI 服务暂时不可用，请稍后再试"));
            }
        });
    }
    
    /**
     * 流式 AI 提示 (SSE)
     */
    @GetMapping(value = "/ai/hint/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamHint(@RequestParam String question) {
        SseEmitter emitter = new SseEmitter(120000L);  // 增加到120秒
        
        executor.submit(() -> {
            try {
                // 安全验证
                PromptSecurityService.ValidationResult validation = 
                        securityService.validateQuestion(question);
                if (!validation.isSafe()) {
                    emitter.send(SseEmitter.event().name("error")
                            .data("{\"error\": \"输入不合法\"}"));
                    emitter.complete();
                    return;
                }
                
                String query = validation.getSanitizedInput();
                
                // 检查 DeepSeek 是否可用
                if (!deepseekClient.enabled()) {
                    emitter.send(SseEmitter.event().name("error")
                            .data("{\"error\": \"AI 服务未配置，请设置 DEEPSEEK_API_KEY\"}"));
                    emitter.complete();
                    return;
                }
                
                // 尝试检索文档
                List<String> contexts = new java.util.ArrayList<>();
                try {
                    if (dashscopeClient.enabled()) {
                        double[] embedding = dashscopeClient.embed(query);
                        List<AiDoc> docs = hybridSearchService.hybridSearch(query, embedding, 5);
                        contexts = docs.stream()
                                .map(AiDoc::getContent)
                                .collect(Collectors.toList());
                    }
                } catch (Exception e) {
                    log.warn("检索文档失败，将直接回答: {}", e.getMessage());
                }
                
                // 流式调用 DeepSeek
                final SseEmitter emitterRef = emitter;
                final List<String> finalContexts = contexts;
                
                // KOB 项目专用的 System Prompt，包含项目上下文
                String kobSystemPrompt = """
                    你是 KOB（King of Bots）贪吃蛇对战游戏的 Bot 开发助手。
                    
                    ## 项目背景
                    - 这是一个双人贪吃蛇对战游戏，两条蛇在 13x14 的网格地图上对战
                    - 用户编写 Java Bot 代码，系统会编译并执行用户的代码
                    - **每回合 Bot 必须在 2 秒内返回移动方向（0-上, 1-右, 2-下, 3-左）**
                    - 超时会导致 Bot 返回默认方向 0，可能导致撞墙失败
                    
                    ## Bot 代码结构
                    用户的 Bot 代码需要实现 `Supplier<Integer>` 接口：
                    ```java
                    public class Bot implements java.util.function.Supplier<Integer> {
                        @Override
                        public Integer get() {
                            // 从 input.txt 读取游戏状态
                            // 返回方向 0-3
                        }
                    }
                    ```
                    
                    ## 常见问题及解决方案
                    
                    ### Bot 超时问题
                    超时的真正原因通常是：
                    1. **寻路算法效率低** - 使用 A* 或优化的 BFS 替代朴素 DFS
                    2. **重复解析地图** - 预处理障碍物，避免每次重新解析 input
                    3. **搜索深度过大** - 限制搜索深度，保证 2 秒内返回
                    4. **未使用剪枝** - 提前排除死路
                    
                    ### 性能优化建议
                    - 使用 BFS 而非 DFS（BFS 能找最短路径且不会栈溢出）
                    - 位运算优化地图状态存储
                    - 缓存计算结果，避免重复计算
                    - 使用 `ArrayDeque` 而非 `LinkedList` 作为队列
                    
                    ## 回答要求
                    - 结合项目实际给出具体可操作的建议
                    - 提供示例代码时使用 Java，并添加中文注释
                    - 使用 Markdown 格式，代码块用 ```java 包裹
                    - 如果问题与 Bot 开发无关，简洁回答即可
                    """;
                
                deepseekClient.streamChat(
                        kobSystemPrompt,
                        query,
                        finalContexts,
                        token -> {
                            try {
                                // 对换行符进行编码，避免 SSE 传输时丢失
                                String encodedToken = token.replace("\n", "___NEWLINE___");
                                emitterRef.send(SseEmitter.event()
                                        .name("chunk").data(encodedToken));
                            } catch (Exception ignored) {}
                        },
                        () -> {
                            try {
                                emitterRef.send(SseEmitter.event()
                                        .name("done").data("{\"status\": \"completed\"}"));
                                emitterRef.complete();
                            } catch (Exception ignored) {}
                        },
                        error -> {
                            try {
                                emitterRef.send(SseEmitter.event()
                                        .name("error").data("{\"error\": \"生成失败\"}"));
                                emitterRef.complete();
                            } catch (Exception ignored) {}
                        }
                );
                
            } catch (Exception e) {
                log.error("流式提示失败: {}", e.getMessage());
                try {
                    emitter.send(SseEmitter.event().name("error")
                            .data("{\"error\": \"" + e.getMessage() + "\"}"));
                } catch (Exception ignored) {}
                emitter.completeWithError(e);
            }
        });
        
        return emitter;
    }
    
    /**
     * 获取 AI 成本和使用统计
     * 
     * @return 包含 Token 使用量、成本、调用次数等指标
     */
    @GetMapping("/ai/metrics")
    public ResponseEntity<AiMetricsService.MetricsSummary> getMetrics() {
        AiMetricsService.MetricsSummary summary = metricsService.getSummary();
        log.info("查询AI指标: {}", summary);
        return ResponseEntity.ok(summary);
    }
    
    /**
     * 重置统计数据（仅用于测试）
     * 
     * @return 成功消息
     */
    @PostMapping("/ai/metrics/reset")
    public ResponseEntity<String> resetMetrics() {
        metricsService.reset();
        log.info("AI指标已重置");
        return ResponseEntity.ok("指标已重置");
    }
    
    /**
     * 获取 Embedding 缓存统计
     * 
     * @return 缓存命中率、节省成本等指标
     */
    @GetMapping("/ai/cache")
    public ResponseEntity<EmbeddingCacheService.CacheSummary> getCacheStats() {
        EmbeddingCacheService.CacheSummary summary = cacheService.getSummary();
        log.info("查询缓存统计: {}", summary);
        return ResponseEntity.ok(summary);
    }
    
    /**
     * 清空缓存（仅用于测试）
     * 
     * @return 成功消息
     */
    @PostMapping("/ai/cache/clear")
    public ResponseEntity<String> clearCache() {
        cacheService.clear();
        log.info("Embedding缓存已清空");
        return ResponseEntity.ok("缓存已清空");
    }
}
