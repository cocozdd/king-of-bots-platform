package com.kob.backend.controller.ai;

import com.kob.backend.controller.ai.dto.AiDoc;
import com.kob.backend.controller.ai.dto.AiHintRequest;
import com.kob.backend.controller.ai.dto.AiHintResponse;
import com.kob.backend.config.AiServiceProperties;
import com.kob.backend.pojo.PythonChatRequest;
import com.kob.backend.repository.AiCorpusRepository;
import com.kob.backend.service.impl.ai.*;
import com.kob.backend.service.impl.ai.agentic.AgenticRAGService;
import com.kob.backend.service.impl.ai.message.AIMessage;
import com.kob.backend.service.impl.ai.message.BaseMessage;
import com.kob.backend.service.impl.ai.message.HumanMessage;
import com.kob.backend.service.impl.ai.message.SystemMessage;
import com.kob.backend.service.impl.ai.memory.ConversationMemoryService;
import com.kob.backend.service.impl.ai.message.BaseMessage;
import com.kob.backend.service.impl.ai.message.HumanMessage;
import com.kob.backend.service.impl.ai.message.SystemMessage;
import com.kob.backend.service.impl.ai.search.HybridSearchService;
import com.kob.backend.service.impl.utils.UserDetailsImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;

import javax.annotation.PostConstruct;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

/**
 * AI Bot 助手控制器
 * 
 * 提供增强的 AI 功能：
 * - 智能问答（RAG 优化版）
 * - Bot 代码生成
 * - 代码分析与优化
 * - 对战分析
 * - 流式输出
 */
@RestController
@RequestMapping("/ai/bot")
public class AiBotController {
    
    private static final Logger log = LoggerFactory.getLogger(AiBotController.class);
    
    @Autowired
    private AiCorpusRepository corpusRepository;
    
    @Autowired
    private RerankService rerankService;
    
    @Autowired
    private HybridSearchService hybridSearchService;
    
    @Autowired
    private QueryRewriteService queryRewriteService;
    
    @Autowired
    private BotCodeGeneratorService codeGeneratorService;
    
    @Autowired
    private BattleAnalysisService battleAnalysisService;
    
    @Autowired
    private AiMetricsService metricsService;
    
    @Autowired
    private PromptSecurityService securityService;
    
    @Autowired
    private EmbeddingCacheService cacheService;

    @Autowired
    private ConversationMemoryService memoryService;

    @Autowired
    private PythonAiServiceClient pythonAiServiceClient;

    @Autowired(required = false)
    private AgenticRAGService agenticRAGService;

    @Autowired
    private AiServiceProperties aiServiceProperties;

    @Autowired
    private RestTemplate restTemplate;

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
        log.info("AI Bot Controller 初始化完成");
    }
    
    /**
     * 增强版智能问答 - 使用 RAG 优化
     * 
     * 优化策略：
     * 1. Query 改写（HyDE/Multi-Query）
     * 2. 混合检索（Vector + BM25）
     * 3. Rerank 精排
     */
    @PostMapping("/ask")
    public CompletableFuture<ResponseEntity<Map<String, Object>>> ask(@RequestBody AiHintRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            Map<String, Object> result = new HashMap<>();
            long startTime = System.currentTimeMillis();

            // 处理 sessionId（支持多轮对话）
            String sessionId = request.getSessionId();
            Long userId = resolveUserId(request.getUserId());

            try {
                // 1. 安全验证
                PromptSecurityService.ValidationResult validation =
                        securityService.validateQuestion(request.getQuestion());
                if (!validation.isSafe()) {
                    result.put("success", false);
                    result.put("error", "输入不合法: " + validation.getReason());
                    return ResponseEntity.ok(result);
                }

                String query = validation.getSanitizedInput();

                // 1.5 如果提供了 sessionId，记录用户消息并获取上下文
                List<Map<String, String>> conversationHistory = new java.util.ArrayList<>();
                if (sessionId != null && !sessionId.isEmpty()) {
                    memoryService.addUserMessage(sessionId, userId, query);
                    conversationHistory = memoryService.getConversationContext(sessionId);

                    // 自动生成会话标题（第一条消息时）
                    if (conversationHistory.size() <= 2) {
                        memoryService.autoGenerateTitle(sessionId, query);
                    }
                }

                // 2. Query 改写
                QueryRewriteService.QueryRewriteResult rewriteResult = 
                        queryRewriteService.smartRewrite(query, deepseekClient, "auto");
                String searchQuery = rewriteResult.getPrimaryQuery();
                
                // 3. 生成 Embedding
                if (!dashscopeClient.enabled()) {
                    result.put("success", false);
                    result.put("error", "Embedding 服务未配置");
                    return ResponseEntity.ok(result);
                }
                double[] embedding = dashscopeClient.embed(searchQuery);

                // 3.5 复用 AgenticRAG 的拒答逻辑（如果可用）
                String rejection = agenticRAGService != null
                    ? agenticRAGService.checkAndGetRejection(query, embedding)
                    : null;
                if (rejection != null) {
                    if (sessionId != null && !sessionId.isEmpty()) {
                        memoryService.addAssistantMessage(sessionId, rejection);
                    }

                    long latency = System.currentTimeMillis() - startTime;
                    result.put("success", true);
                    result.put("answer", rejection);
                    result.put("sources", List.of());
                    result.put("rewriteStrategy", rewriteResult.strategy);
                    result.put("latencyMs", latency);
                    result.put("sessionId", sessionId);
                    return ResponseEntity.ok(result);
                }
                
                // 4. 混合检索
                List<AiDoc> candidates = hybridSearchService.hybridSearch(query, embedding, 20);
                // 6. 生成回答
                if (!deepseekClient.enabled()) {
                    result.put("success", false);
                    result.put("error", "Chat 服务未配置");
                    return ResponseEntity.ok(result);
                }
                
                // 5. Rerank 精排
                List<AiDoc> topDocs = rerankService.rerank(query, candidates, 5);
                
                // 6. 生成回答 - 通过 Python 服务（支持会话管理）
                List<String> contexts = topDocs.stream()
                        .map(AiDoc::getContent)
                        .collect(Collectors.toList());
                String ctx = contexts.stream()
                        .map(s -> s.length() > 600 ? s.substring(0, 600) + "..." : s)
                        .collect(Collectors.joining("\n\n"));
                
                String systemPrompt = "你是 King of Bots (KOB) 贪吃蛇对战平台的 Bot 开发助手, 基于提供的参考文档回答问题。\n\n" +
                    "【重要】本项目的游戏规则 (必须遵守, 不要使用通用贪吃蛇知识):\n" +
                    "1. 这是一个双人贪吃蛇对战游戏, 两条蛇在 13x14 的地图上对战\n" +
                    "2. 本游戏【没有食物】! 蛇的长度是自动增长的:\n" +
                    "   - 前10步每步增长1格\n" +
                    "   - 之后每3步增长1格 (step % 3 == 1 时增长)\n\n" +
                    "格式要求: 使用 Markdown 格式, 代码用 ```java 包裹。\n\n" +
                    "【参考文档】\n" + ctx + "\n\n" +
                    "内容要求:\n" +
                    "- 回答要简洁准确, 基于本项目的实际规则\n" +
                    "- 引用相关文档";
                
                String answer = deepseekClient.chat(systemPrompt, query, contexts);

                // 7. 如果有 sessionId，保存助手回复
                if (sessionId != null && !sessionId.isEmpty()) {
                    memoryService.addAssistantMessage(sessionId, answer);
                }

                // 注意：Python 服务已自动保存会话历史，这里无需再调用 memoryService

                // 8. 构建响应
                List<Map<String, String>> sources = topDocs.stream()
                        .map(doc -> {
                            Map<String, String> source = new HashMap<>();
                            source.put("id", doc.getId());
                            source.put("title", doc.getTitle());
                            source.put("category", doc.getCategory());
                            return source;
                        })
                        .collect(Collectors.toList());

                long latency = System.currentTimeMillis() - startTime;

                result.put("success", true);
                result.put("answer", answer);
                result.put("sources", sources);
                result.put("rewriteStrategy", rewriteResult.strategy);
                result.put("latencyMs", latency);
                result.put("sessionId", sessionId);

                log.info("智能问答完成: query长度={}, 检索{}篇, 耗时{}ms, sessionId={}",
                        query.length(), topDocs.size(), latency, sessionId);
                
            } catch (Exception e) {
                log.error("智能问答失败: {}", e.getMessage(), e);
                result.put("success", false);
                result.put("error", "服务暂时不可用: " + e.getMessage());
            }
            
            return ResponseEntity.ok(result);
        }, executor);
    }
    
    /**
     * Bot 代码生成
     * 支持普通模式和 Speculative 模式（并行生成多种策略）
     */
    @PostMapping("/generate")
    public CompletableFuture<ResponseEntity<Map<String, Object>>> generateCode(
            @RequestBody Map<String, Object> request) {
        return CompletableFuture.supplyAsync(() -> {
            Map<String, Object> result = new HashMap<>();

            String description = (String) request.get("description");
            Boolean speculative = (Boolean) request.getOrDefault("speculative", false);

            if (description == null || description.trim().isEmpty()) {
                result.put("success", false);
                result.put("error", "请提供策略描述");
                return ResponseEntity.ok(result);
            }

            // 2. 本地降级逻辑 (保持原有)
            // 安全验证
            PromptSecurityService.ValidationResult validation =
                    securityService.validateQuestion(description);
            if (!validation.isSafe()) {
                result.put("success", false);
                result.put("error", "输入不合法");
                return ResponseEntity.ok(result);
            }

            String sanitizedInput = validation.getSanitizedInput();

            if (Boolean.TRUE.equals(speculative)) {
                // Speculative 模式：并行生成三种策略
                try {
                    CompletableFuture<BotCodeGeneratorService.CodeGenResult> aggressiveFuture =
                        CompletableFuture.supplyAsync(() ->
                            codeGeneratorService.generateBotCode(sanitizedInput + " (策略风格：激进进攻型，优先追击对手，主动出击)", deepseekClient), executor);

                    CompletableFuture<BotCodeGeneratorService.CodeGenResult> balancedFuture =
                        CompletableFuture.supplyAsync(() ->
                            codeGeneratorService.generateBotCode(sanitizedInput + " (策略风格：攻守兼备型，平衡进攻和防守)", deepseekClient), executor);

                    CompletableFuture<BotCodeGeneratorService.CodeGenResult> conservativeFuture =
                        CompletableFuture.supplyAsync(() ->
                            codeGeneratorService.generateBotCode(sanitizedInput + " (策略风格：保守生存型，优先保证存活，避免风险)", deepseekClient), executor);

                    // 等待所有完成
                    CompletableFuture.allOf(aggressiveFuture, balancedFuture, conservativeFuture).join();

                    BotCodeGeneratorService.CodeGenResult aggressive = aggressiveFuture.get();
                    BotCodeGeneratorService.CodeGenResult balanced = balancedFuture.get();
                    BotCodeGeneratorService.CodeGenResult conservative = conservativeFuture.get();

                    result.put("success", true);
                    result.put("aggressive", Map.of(
                        "fullCode", aggressive.isSuccess() ? aggressive.getFullCode() : "",
                        "explanation", aggressive.isSuccess() ? aggressive.getExplanation() : aggressive.getError()
                    ));
                    result.put("balanced", Map.of(
                        "fullCode", balanced.isSuccess() ? balanced.getFullCode() : "",
                        "explanation", balanced.isSuccess() ? balanced.getExplanation() : balanced.getError()
                    ));
                    result.put("conservative", Map.of(
                        "fullCode", conservative.isSuccess() ? conservative.getFullCode() : "",
                        "explanation", conservative.isSuccess() ? conservative.getExplanation() : conservative.getError()
                    ));
                } catch (Exception e) {
                    log.error("[API] Speculative 代码生成错误: {}", e.getMessage());
                    result.put("success", false);
                    result.put("error", "生成失败: " + e.getMessage());
                }
            } else {
                // 普通模式
                BotCodeGeneratorService.CodeGenResult genResult =
                        codeGeneratorService.generateBotCode(sanitizedInput, deepseekClient);

                if (genResult.isSuccess()) {
                    result.put("success", true);
                    result.put("fullCode", genResult.getFullCode());
                    result.put("explanation", genResult.getExplanation());
                } else {
                    result.put("success", false);
                    result.put("error", genResult.getError());
                }
            }

            return ResponseEntity.ok(result);
        }, executor);
    }

    /**
     * 公平性检查：用户是否提供了自己的思路
     * 
     * 规则：
     * - 用户提供思路（算法、策略描述）→ 可以给代码
     * - 用户只说"给我厉害的AI" → 先引导思考
     */
    private boolean userHasIdea(String description) {
        if (description == null) return false;
        String lower = description.toLowerCase();
        
        // 有思路的标志
        String[] ideaIndicators = {
            "bfs", "dfs", "广度", "深度", "a*", "dijkstra", "动态规划",
            "队列", "栈", "递归", "遍历", "计算",
            "我想", "我的思路", "我打算", "我觉得",
            "先判断", "然后", "如果", "检测", "评估",
            "安全", "空间", "距离", "方向"
        };
        
        for (String indicator : ideaIndicators) {
            if (lower.contains(indicator)) return true;
        }
        return false;
    }
    
    private boolean isLazyRequest(String description) {
        if (description == null) return false;
        String lower = description.toLowerCase();
        
        String[] lazyPatterns = {
            "给我一个厉害", "给我一个强", "给我最强", "直接给我",
            "帮我写一个", "写一个能赢", "给我代码", "要必胜",
            "给我bot", "给我ai", "最强bot", "最强ai"
        };
        
        for (String pattern : lazyPatterns) {
            if (lower.contains(pattern)) return true;
        }
        return false;
    }
    
    /**
     * 流式代码生成 (SSE)
     * 
     * 公平性设计：
     * - 用户提供思路 → 生成代码
     * - 用户无思路 → 引导思考
     */
    @GetMapping(value = "/generate/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamGenerateCode(@RequestParam String description) {
        SseEmitter emitter = new SseEmitter(120000L); // 2分钟超时
        
        executor.submit(() -> {
            try {
                // 安全验证
                PromptSecurityService.ValidationResult validation = 
                        securityService.validateQuestion(description);
                if (!validation.isSafe()) {
                    emitter.send(SseEmitter.event().name("error").data("输入不合法"));
                    emitter.complete();
                    return;
                }
                
                String safeDescription = validation.getSanitizedInput();
                
                // === 公平性检查 ===
                if (isLazyRequest(safeDescription) && !userHasIdea(safeDescription)) {
                    // 伸手党，没有思路 -> 引导思考
                    String guidance = "我很乐意帮助你! 但为了公平和真正帮助你学习, 我需要先了解你的想法:\n\n" +
                        "**1. 你想实现什么策略?**\n" +
                        "- 安全优先? 追着敌人? 还是占领空间?\n\n" +
                        "**2. 你了解哪些算法?**\n" +
                        "- BFS (广度优先) 可以找最短路径\n" +
                        "- 连通区域计算可以评估空间大小\n\n" +
                        "**3. 你目前的思路是什么?**\n" +
                        "- 哪怕只是大概想法也可以\n\n" +
                        "请告诉我你的思路, 我会帮你实现!\n\n" +
                        "**示例描述**:\n" +
                        "- \"我想用 BFS 找到离对手最远的安全位置\"\n" +
                        "- \"我的思路是先计算四个方向的可用空间\"\n" +
                        "- \"我想实现一个能预测对手移动的策略\"";
                    
                    emitter.send(SseEmitter.event().name("guidance").data(guidance.replace("\n", "___NEWLINE___")));
                    emitter.send(SseEmitter.event().name("done").data("needs_idea"));
                    emitter.complete();
                    return;
                }
                
                // 发送开始事件
                emitter.send(SseEmitter.event().name("start").data("开始生成代码..."));
                
                // 构建提示
                String systemPrompt = "你是一个专业的贪吃蛇Bot代码生成器。\n" +
                    "根据用户描述生成Java策略代码片段。\n" +
                    "只输出核心策略代码, 不要完整类结构。\n" +
                    "代码应该返回0-3的方向值(0:上,1:右,2:下,3:左)。";
                
                // 流式调用 DeepSeek
                deepseekClient.streamChat(systemPrompt, safeDescription, List.of(),
                    token -> {
                        try {
                            // 对换行符进行编码，避免 SSE 传输时丢失
                            String encodedToken = token.replace("\n", "___NEWLINE___");
                            emitter.send(SseEmitter.event().name("chunk").data(encodedToken));
                        } catch (Exception e) {
                            log.warn("发送chunk失败: {}", e.getMessage());
                        }
                    },
                    () -> {
                        try {
                            emitter.send(SseEmitter.event().name("done").data("完成"));
                            emitter.complete();
                        } catch (Exception e) {
                            log.warn("完成事件发送失败: {}", e.getMessage());
                        }
                    },
                    e -> {
                        try {
                            emitter.send(SseEmitter.event().name("error").data(e.getMessage()));
                            emitter.completeWithError(e);
                        } catch (Exception ex) {
                            log.warn("错误事件发送失败: {}", ex.getMessage());
                        }
                    }
                );
                
            } catch (Exception e) {
                log.error("流式代码生成失败", e);
                try {
                    emitter.send(SseEmitter.event().name("error").data(e.getMessage()));
                    emitter.completeWithError(e);
                } catch (Exception ex) {
                    log.warn("错误处理失败: {}", ex.getMessage());
                }
            }
        });
        
        return emitter;
    }

    /**
     * HITL 聊天代理（通过 Java 后端注入 userId）
     */
    @PostMapping("/chat/hitl")
    public ResponseEntity<Map<String, Object>> chatWithHitl(@RequestBody Map<String, Object> request) {
        Map<String, Object> payload = new HashMap<>();
        if (request != null) {
            payload.putAll(request);
        }

        Long resolvedUserId = resolveUserId(extractUserId(payload.get("userId")));
        if (resolvedUserId != null && resolvedUserId > 0) {
            payload.put("userId", resolvedUserId);
        } else {
            payload.remove("userId");
        }

        if (aiServiceProperties == null || !aiServiceProperties.isEnabled()) {
            Map<String, Object> error = new HashMap<>();
            error.put("success", false);
            error.put("error", "Python AI Service is disabled");
            return ResponseEntity.ok(error);
        }

        String baseUrl = normalizeBaseUrl(aiServiceProperties.getBaseUrl());
        try {
            ResponseEntity<Map> response = restTemplate.postForEntity(
                    baseUrl + "/api/bot/chat/hitl",
                    payload,
                    Map.class
            );
            Map<String, Object> body = response.getBody();
            return ResponseEntity.status(response.getStatusCode())
                    .body(body != null ? body : Map.of("success", false, "error", "Empty response from Python"));
        } catch (RestClientException e) {
            Map<String, Object> error = new HashMap<>();
            error.put("success", false);
            error.put("error", "Python AI Service call failed: " + e.getMessage());
            return ResponseEntity.ok(error);
        }
    }
    
    /**
     * 代码分析
     */
    @PostMapping("/analyze")
    public CompletableFuture<ResponseEntity<Map<String, Object>>> analyzeCode(
            @RequestBody Map<String, String> request) {
        return CompletableFuture.supplyAsync(() -> {
            Map<String, Object> result = new HashMap<>();
            
            String code = request.get("code");
            if (code == null || code.trim().isEmpty()) {
                result.put("success", false);
                result.put("error", "请提供代码");
                return ResponseEntity.ok(result);
            }
            
            // 2. 本地降级
            BotCodeGeneratorService.CodeAnalysisResult analysisResult = 
                    codeGeneratorService.analyzeCode(code, deepseekClient);
            
            if (analysisResult.isSuccess()) {
                result.put("success", true);
                result.put("analysis", analysisResult.getAnalysis());
            } else {
                result.put("success", false);
                result.put("error", analysisResult.getError());
            }
            
            return ResponseEntity.ok(result);
        }, executor);
    }
    
    /**
     * 代码修复
     */
    @PostMapping("/fix")
    public CompletableFuture<ResponseEntity<Map<String, Object>>> fixCode(
            @RequestBody Map<String, String> request) {
        return CompletableFuture.supplyAsync(() -> {
            Map<String, Object> result = new HashMap<>();
            
            String code = request.get("code");
            String errorLog = request.get("errorLog");
            
            if (code == null || code.trim().isEmpty()) {
                result.put("success", false);
                result.put("error", "请提供代码");
                return ResponseEntity.ok(result);
            }
            
            // 2. 本地降级
            BotCodeGeneratorService.CodeGenResult fixResult = 
                    codeGeneratorService.fixCode(code, errorLog != null ? errorLog : "", deepseekClient);
            
            if (fixResult.isSuccess()) {
                result.put("success", true);
                result.put("fixedCode", fixResult.getFullCode());
                result.put("explanation", fixResult.getExplanation());
            } else {
                result.put("success", false);
                result.put("error", fixResult.getError());
            }
            
            return ResponseEntity.ok(result);
        }, executor);
    }
    
    /**
     * 代码解释
     */
    @PostMapping("/explain")
    public CompletableFuture<ResponseEntity<Map<String, Object>>> explainCode(
            @RequestBody Map<String, String> request) {
        return CompletableFuture.supplyAsync(() -> {
            Map<String, Object> result = new HashMap<>();
            
            String code = request.get("code");
            if (code == null || code.trim().isEmpty()) {
                result.put("success", false);
                result.put("error", "请提供代码");
                return ResponseEntity.ok(result);
            }
            
            String explanation = codeGeneratorService.explainCode(code, deepseekClient);
            result.put("success", true);
            result.put("explanation", explanation);
            
            return ResponseEntity.ok(result);
        }, executor);
    }
    
    /**
     * 对战分析
     */
    @PostMapping("/battle/analyze")
    public CompletableFuture<ResponseEntity<Map<String, Object>>> analyzeBattle(
            @RequestBody Map<String, String> request) {
        return CompletableFuture.supplyAsync(() -> {
            Map<String, Object> result = new HashMap<>();
            
            String mapData = request.get("mapData");
            String stepsA = request.get("stepsA");
            String stepsB = request.get("stepsB");
            String loser = request.get("loser");
            
            if (mapData == null || mapData.isEmpty()) {
                result.put("success", false);
                result.put("error", "请提供地图数据");
                return ResponseEntity.ok(result);
            }
            
            BattleAnalysisService.BattleAnalysisResult analysisResult = 
                    battleAnalysisService.analyzeBattle(mapData, stepsA, stepsB, loser, deepseekClient);
            
            result.put("success", true);
            result.put("stats", analysisResult.getStats());
            result.put("keyMoments", analysisResult.getKeyMoments());
            result.put("aiAnalysis", analysisResult.getAiAnalysis());
            
            return ResponseEntity.ok(result);
        }, executor);
    }
    
    /**
     * 流式问答 (SSE) - 支持多轮对话和RAG增强
     */
    @GetMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamAsk(
            @RequestParam String question,
            @RequestParam(required = false) String sessionId,
            @RequestParam(required = false) Long userId) {
        SseEmitter emitter = new SseEmitter(60000L); // 60秒超时

        executor.submit(() -> {
            try {
                // 发送开始事件
                emitter.send(SseEmitter.event()
                        .name("start")
                        .data("{\"status\": \"processing\"}"));

                // 安全验证
                PromptSecurityService.ValidationResult validation =
                        securityService.validateQuestion(question);
                if (!validation.isSafe()) {
                    emitter.send(SseEmitter.event()
                            .name("error")
                            .data("{\"error\": \"输入不合法\"}"));
                    emitter.complete();
                    return;
                }

                String query = validation.getSanitizedInput();

                // 会话管理：保存用户消息并获取上下文
                if (sessionId != null && !sessionId.isEmpty()) {
                    log.info("[Session] 保存用户消息 sessionId={}, query={}", sessionId, query.substring(0, Math.min(query.length(), 50)));
                    Long userIdValue = resolveUserId(userId);
                    memoryService.addUserMessage(sessionId, userIdValue, query);

                    // 自动生成会话标题（第一条消息时）
                    List<Map<String, String>> conversationHistory = memoryService.getConversationContext(sessionId);
                    log.info("[Session] 获取会话上下文 sessionId={}, 消息数={}", sessionId, conversationHistory.size());
                    if (conversationHistory.size() <= 2) {
                        memoryService.autoGenerateTitle(sessionId, query);
                    }
                } else {
                    log.warn("[Session] sessionId 为空，跳过会话保存");
                }

                // 发送检索状态
                emitter.send(SseEmitter.event()
                        .name("status")
                        .data("{\"step\": \"searching\", \"message\": \"正在检索相关文档...\"}"));

                // 检索文档
                if (dashscopeClient.enabled()) {
                    double[] embedding = dashscopeClient.embed(query);

                    // 复用 AgenticRAG 的拒答逻辑（如果可用）
                    String rejection = agenticRAGService != null
                        ? agenticRAGService.checkAndGetRejection(query, embedding)
                        : null;
                    if (rejection != null) {
                        // 保存拒答消息到会话
                        if (sessionId != null && !sessionId.isEmpty()) {
                            memoryService.addAssistantMessage(sessionId, rejection);
                        }

                        String encoded = rejection.replace("\n", "___NEWLINE___");
                        emitter.send(SseEmitter.event()
                                .name("chunk")
                                .data(encoded));
                        emitter.send(SseEmitter.event()
                                .name("done")
                                .data("{\"status\": \"completed\"}"));
                        emitter.complete();
                        return;
                    }
                    List<AiDoc> docs = hybridSearchService.hybridSearch(query, embedding, 5);

                    // 发送检索结果
                    emitter.send(SseEmitter.event()
                            .name("status")
                            .data(String.format("{\"step\": \"found\", \"count\": %d}", docs.size())));

                    // 发送生成状态
                    emitter.send(SseEmitter.event()
                            .name("status")
                            .data("{\"step\": \"generating\", \"message\": \"正在生成回答...\"}"));

                    // 构建包含 RAG 上下文的 System Prompt
                    List<String> contexts = docs.stream()
                            .map(AiDoc::getContent)
                            .collect(Collectors.toList());

                    String ctx = contexts.stream()
                            .map(s -> s.length() > 600 ? s.substring(0, 600) + "..." : s)
                            .collect(Collectors.joining("\n\n"));

                    String systemPrompt = "你是 King of Bots (KOB) 贪吃蛇对战平台的 Bot 开发助手, 基于提供的参考文档回答问题。\n\n" +
                            "【重要】本项目的游戏规则 (必须遵守, 不要使用通用贪吃蛇知识):\n" +
                            "1. 这是一个双人贪吃蛇对战游戏, 两条蛇在 13x14 的地图上对战\n" +
                            "2. 本游戏【没有食物】! 蛇的长度是自动增长的:\n" +
                            "   - 前10步每步增长1格\n" +
                            "   - 之后每3步增长1格 (step % 3 == 1 时增长)\n" +
                            "3. 获胜条件: 让对手撞墙或撞到蛇身 (自己或对手的)\n" +
                            "4. 移动方向: 0=上, 1=右, 2=下, 3=左\n" +
                            "5. 地图坐标从(0,0)开始, 有固定的障碍物墙壁\n\n" +
                            "格式要求: 使用 Markdown 格式, 代码用 ```java 包裹。\n\n" +
                            "【参考文档】\n" + ctx + "\n\n" +
                            "内容要求:\n" +
                            "- 回答要简洁准确, 基于本项目的实际规则\n" +
                            "- 引用相关文档\n" +
                            "- 不要提及\"食物\"、\"吃食物\"等概念, 本游戏没有食物\n" +
                            "- 不知道就说不知道";

                    // 累积完整答案（用于会话保存）
                    final StringBuilder fullResponse = new StringBuilder();
                    final SseEmitter emitterRef = emitter;
                    final String finalSessionId = sessionId;

                    // 优先使用 Python 服务流式接口
                    if (pythonAiServiceClient.enabled()) {
                        PythonChatRequest request = new PythonChatRequest();
                        request.setSessionId(sessionId);
                        request.setQuestion(query);
                        request.setSystemPrompt(systemPrompt);
                        request.setStream(true);
                        request.setUserId(resolveUserId(userId));

                        pythonAiServiceClient.streamChat(
                                request,
                                // onDelta: 每收到一个 token 立即发送
                                token -> {
                                    try {
                                        fullResponse.append(token);
                                        String encoded = token.replace("\n", "___NEWLINE___");
                                        emitterRef.send(SseEmitter.event()
                                                .name("chunk")
                                                .data(encoded));
                                    } catch (Exception e) {
                                        log.error("发送 token 失败: {}", e.getMessage());
                                    }
                                },
                                // onError: 处理错误
                                error -> {
                                    try {
                                        emitterRef.send(SseEmitter.event()
                                                .name("error")
                                                .data("{\"error\": \"" + error.getMessage() + "\"}"));
                                        emitterRef.complete();
                                    } catch (Exception ignored) {}
                                },
                                // onComplete
                                () -> {
                                    try {
                                        // 保存完整答案到会话
                                        if (finalSessionId != null && !finalSessionId.isEmpty()) {
                                            String response = fullResponse.toString();
                                            log.info("[Session] 保存助手消息 sessionId={}, 长度={}", finalSessionId, response.length());
                                            memoryService.addAssistantMessage(finalSessionId, response);
                                            log.info("[Session] 助手消息保存成功");
                                        } else {
                                            log.warn("[Session] onComplete: sessionId 为空，跳过会话保存");
                                        }

                                        emitterRef.send(SseEmitter.event()
                                                .name("done")
                                                .data("{\"status\": \"completed\"}"));
                                        emitterRef.complete();
                                    } catch (Exception e) {
                                        log.error("完成事件处理失败: {}", e.getMessage(), e);
                                    }
                                }
                        );
                        return; // 流式处理会自己完成
                    }

                    // 降级到本地 DeepSeek（也支持会话保存）
                    if (deepseekClient.enabled()) {
                        deepseekClient.streamChat(
                                systemPrompt,
                                query,
                                contexts,
                                // onToken: 每收到一个 token 立即发送
                                token -> {
                                    try {
                                        fullResponse.append(token);
                                        String encodedToken = token.replace("\n", "___NEWLINE___");
                                        emitterRef.send(SseEmitter.event()
                                                .name("chunk")
                                                .data(encodedToken));
                                    } catch (Exception e) {
                                        log.warn("发送失败: {}", e.getMessage());
                                    }
                                },
                                // onComplete
                                () -> {
                                    try {
                                        // 保存完整答案到会话
                                        if (finalSessionId != null && !finalSessionId.isEmpty()) {
                                            memoryService.addAssistantMessage(finalSessionId, fullResponse.toString());
                                        }

                                        emitterRef.send(SseEmitter.event()
                                                .name("done")
                                                .data("{\"status\": \"completed\"}"));
                                        emitterRef.complete();
                                    } catch (Exception ignored) {}
                                },
                                // onError
                                error -> {
                                    log.error("流式生成失败: {}", error.getMessage());
                                    try {
                                        emitterRef.send(SseEmitter.event()
                                                .name("error")
                                                .data("{\"error\": \"生成失败\"}"));
                                        emitterRef.complete();
                                    } catch (Exception ignored) {}
                                }
                        );
                        return; // 流式处理会自己完成
                    }
                }
                
                // 非流式情况的完成事件
                emitter.send(SseEmitter.event()
                        .name("done")
                        .data("{\"status\": \"completed\"}"));
                emitter.complete();
                
            } catch (Exception e) {
                log.error("流式问答失败: {}", e.getMessage());
                try {
                    emitter.send(SseEmitter.event()
                            .name("error")
                            .data("{\"error\": \"" + e.getMessage() + "\"}"));
                } catch (Exception ignored) {}
                emitter.completeWithError(e);
            }
        });
        
        return emitter;
    }
    
    /**
     * 获取 AI 服务状态
     */
    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> getStatus() {
        Map<String, Object> status = new HashMap<>();
        status.put("embeddingEnabled", dashscopeClient != null && dashscopeClient.enabled());
        status.put("chatEnabled", deepseekClient != null && deepseekClient.enabled());
        status.put("pythonEnabled", pythonAiServiceClient != null && pythonAiServiceClient.enabled());
        status.put("metrics", metricsService.getSummary());
        return ResponseEntity.ok(status);
    }

    /**
     * 降级方法：当 Python 服务不可用时，直接调用本地 DeepSeek
     */
    private String fallbackToLocalChat(String systemPrompt, String query) {
        if (!deepseekClient.enabled()) {
            return "Chat 服务暂时不可用";
        }
        List<BaseMessage> messages = new java.util.ArrayList<>();
        messages.add(new SystemMessage(systemPrompt));
        messages.add(new HumanMessage(query));
        return deepseekClient.chat(messages);
    }

    private Long resolveUserId(Long requestUserId) {
        Object authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication instanceof UsernamePasswordAuthenticationToken) {
            UsernamePasswordAuthenticationToken authenticationToken =
                    (UsernamePasswordAuthenticationToken) authentication;
            if (authenticationToken.getPrincipal() instanceof UserDetailsImpl) {
                UserDetailsImpl loginUser = (UserDetailsImpl) authenticationToken.getPrincipal();
                Integer id = loginUser.getUser().getId();
                if (id != null) {
                    return id.longValue();
                }
            }
        }
        return requestUserId != null ? requestUserId : 0L;
    }

    private Long extractUserId(Object rawUserId) {
        if (rawUserId instanceof Number) {
            return ((Number) rawUserId).longValue();
        }
        if (rawUserId instanceof String) {
            String value = ((String) rawUserId).trim();
            if (!value.isEmpty()) {
                try {
                    return Long.parseLong(value);
                } catch (NumberFormatException ignored) {}
            }
        }
        return null;
    }

    private String normalizeBaseUrl(String baseUrl) {
        if (baseUrl == null || baseUrl.isBlank()) {
            return "http://127.0.0.1:3003";
        }
        return baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
    }
}
