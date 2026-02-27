package com.kob.backend.service.impl.ai.streaming;

import com.kob.backend.service.impl.ai.AiMetricsService;
import com.kob.backend.service.impl.ai.DeepseekClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import javax.annotation.PostConstruct;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

/**
 * 流式 Agent 服务 - SSE 实时输出推理过程
 * 
 * 功能：
 * - 实时流式输出 Agent 思考过程
 * - ReAct 循环的可视化
 * - 工具调用过程展示
 * 
 * 面试要点：
 * - SSE vs WebSocket：SSE 单向、轻量，适合流式输出
 * - 背压处理：客户端消费慢时的处理策略
 * - 超时管理：长时间推理的连接保持
 */
@Service
public class StreamingAgentService {
    
    private static final Logger log = LoggerFactory.getLogger(StreamingAgentService.class);
    private static final int MAX_ITERATIONS = 5;
    
    private final ExecutorService executor = Executors.newCachedThreadPool();
    
    @Autowired(required = false)
    private AiMetricsService metricsService;
    
    private DeepseekClient deepseekClient;
    
    // 内置工具定义
    private final Map<String, ToolDefinition> tools = new HashMap<>();
    
    @PostConstruct
    public void init() {
        this.deepseekClient = new DeepseekClient(metricsService);
        registerBuiltinTools();
    }
    
    /**
     * 创建流式 Agent 执行器
     */
    public SseEmitter executeStreaming(String query, Long userId) {
        SseEmitter emitter = new SseEmitter(120000L); // 2分钟超时
        
        executor.submit(() -> {
            try {
                runReActLoop(query, userId, emitter);
            } catch (Exception e) {
                log.error("Streaming agent error", e);
                sendEvent(emitter, "error", Map.of("message", e.getMessage()));
                emitter.completeWithError(e);
            }
        });
        
        emitter.onCompletion(() -> log.info("SSE completed for user {}", userId));
        emitter.onTimeout(() -> log.warn("SSE timeout for user {}", userId));
        
        return emitter;
    }
    
    /**
     * ReAct 循环 - 流式版本
     */
    private void runReActLoop(String query, Long userId, SseEmitter emitter) throws IOException {
        List<Map<String, String>> messages = new ArrayList<>();
        
        // System Prompt
        String systemPrompt = buildSystemPrompt();
        messages.add(Map.of("role", "system", "content", systemPrompt));
        messages.add(Map.of("role", "user", "content", query));
        
        // 发送开始事件
        sendEvent(emitter, "start", Map.of(
            "query", query,
            "timestamp", System.currentTimeMillis()
        ));
        
        for (int i = 0; i < MAX_ITERATIONS; i++) {
            // 发送思考开始
            sendEvent(emitter, "thinking", Map.of(
                "iteration", i + 1,
                "status", "开始推理..."
            ));
            
            // 调用 LLM（流式）
            StringBuilder response = new StringBuilder();
            streamLLMCall(messages, token -> {
                response.append(token);
                sendEvent(emitter, "token", Map.of("content", token));
            });
            
            String fullResponse = response.toString();
            messages.add(Map.of("role", "assistant", "content", fullResponse));
            
            // 解析响应
            AgentResponse parsed = parseAgentResponse(fullResponse);
            
            // 发送思考内容
            if (parsed.thought != null) {
                sendEvent(emitter, "thought", Map.of(
                    "content", parsed.thought,
                    "iteration", i + 1
                ));
            }
            
            // 检查是否有最终答案
            if (parsed.finalAnswer != null) {
                sendEvent(emitter, "answer", Map.of(
                    "content", parsed.finalAnswer,
                    "iterations", i + 1
                ));
                emitter.complete();
                return;
            }
            
            // 执行工具调用
            if (parsed.action != null) {
                sendEvent(emitter, "action", Map.of(
                    "tool", parsed.action,
                    "input", parsed.actionInput != null ? parsed.actionInput : ""
                ));
                
                String observation = executeTool(parsed.action, parsed.actionInput);
                
                sendEvent(emitter, "observation", Map.of(
                    "content", observation,
                    "tool", parsed.action
                ));
                
                // 添加观察结果到消息历史
                messages.add(Map.of("role", "user", 
                    "content", "Observation: " + observation));
            }
        }
        
        // 达到最大迭代
        sendEvent(emitter, "max_iterations", Map.of(
            "message", "达到最大推理次数，请简化问题后重试"
        ));
        emitter.complete();
    }
    
    /**
     * 流式 LLM 调用
     */
    private void streamLLMCall(List<Map<String, String>> messages, Consumer<String> onToken) {
        // 简化实现：模拟流式输出
        // 生产环境应使用真正的 SSE API
        try {
            String systemPrompt = messages.get(0).get("content");
            String question = messages.size() > 1 ? messages.get(1).get("content") : "";
            String response = deepseekClient.chat(systemPrompt, question, List.of());
            
            // 模拟流式输出（每10字符一个token）
            for (int i = 0; i < response.length(); i += 10) {
                int end = Math.min(i + 10, response.length());
                onToken.accept(response.substring(i, end));
                Thread.sleep(50); // 模拟延迟
            }
        } catch (Exception e) {
            log.error("LLM call failed", e);
            onToken.accept("[Error: " + e.getMessage() + "]");
        }
    }
    
    /**
     * 发送 SSE 事件
     */
    private void sendEvent(SseEmitter emitter, String eventType, Map<String, Object> data) {
        try {
            Map<String, Object> event = new HashMap<>(data);
            event.put("type", eventType);
            event.put("timestamp", System.currentTimeMillis());
            
            emitter.send(SseEmitter.event()
                    .name(eventType)
                    .data(event));
        } catch (IOException e) {
            log.warn("Failed to send SSE event: {}", eventType);
        }
    }
    
    /**
     * 构建 System Prompt
     */
    private String buildSystemPrompt() {
        StringBuilder sb = new StringBuilder();
        sb.append("你是 KOB 平台的 AI 助手，使用 ReAct 模式回答用户问题。\n\n");
        sb.append("可用工具：\n");
        
        for (ToolDefinition tool : tools.values()) {
            sb.append(String.format("- %s: %s\n", tool.name, tool.description));
        }
        
        sb.append("\n响应格式要求：\n");
        sb.append("Thought: 你的思考过程\n");
        sb.append("Action: 工具名称\n");
        sb.append("Action Input: 工具输入\n");
        sb.append("或者直接输出：\n");
        sb.append("Thought: 最终思考\n");
        sb.append("Final Answer: 最终答案\n");
        sb.append("\n最终答案格式要求：\n");
        sb.append("- 使用 Markdown 格式输出\n");
        sb.append("- 段落之间用空行分隔\n");
        sb.append("- 使用 ## 作为二级标题，### 作为三级标题\n");
        sb.append("- 代码必须用 ```语言名 和 ``` 包裹，每行代码单独一行\n");
        sb.append("- 列表项之间适当换行\n");
        
        return sb.toString();
    }
    
    /**
     * 解析 Agent 响应
     */
    private AgentResponse parseAgentResponse(String response) {
        AgentResponse result = new AgentResponse();
        
        // 提取 Thought
        int thoughtIdx = response.indexOf("Thought:");
        if (thoughtIdx >= 0) {
            int endIdx = response.indexOf("\n", thoughtIdx);
            if (endIdx < 0) endIdx = response.length();
            result.thought = response.substring(thoughtIdx + 8, endIdx).trim();
        }
        
        // 提取 Final Answer
        int answerIdx = response.indexOf("Final Answer:");
        if (answerIdx >= 0) {
            result.finalAnswer = response.substring(answerIdx + 13).trim();
            return result;
        }
        
        // 提取 Action
        int actionIdx = response.indexOf("Action:");
        if (actionIdx >= 0) {
            int endIdx = response.indexOf("\n", actionIdx);
            if (endIdx < 0) endIdx = response.length();
            result.action = response.substring(actionIdx + 7, endIdx).trim();
        }
        
        // 提取 Action Input
        int inputIdx = response.indexOf("Action Input:");
        if (inputIdx >= 0) {
            int endIdx = response.indexOf("\n", inputIdx);
            if (endIdx < 0) endIdx = response.length();
            result.actionInput = response.substring(inputIdx + 13, endIdx).trim();
        }
        
        return result;
    }
    
    /**
     * 执行工具
     */
    private String executeTool(String toolName, String input) {
        ToolDefinition tool = tools.get(toolName.toLowerCase());
        if (tool == null) {
            return "未知工具: " + toolName;
        }
        
        try {
            return tool.executor.apply(input);
        } catch (Exception e) {
            return "工具执行错误: " + e.getMessage();
        }
    }
    
    /**
     * 注册内置工具
     */
    private void registerBuiltinTools() {
        tools.put("search_knowledge", new ToolDefinition(
            "search_knowledge",
            "搜索 Bot 开发知识库",
            input -> {
                // 简化实现
                if (input.contains("BFS")) {
                    return "BFS（广度优先搜索）适合寻找最短路径，时间复杂度O(V+E)";
                } else if (input.contains("策略")) {
                    return "Bot策略包括：贪心策略、防守策略、进攻策略";
                }
                return "未找到相关知识: " + input;
            }
        ));
        
        tools.put("analyze_code", new ToolDefinition(
            "analyze_code",
            "分析 Bot 代码并给出建议",
            input -> "代码分析完成：建议优化移动逻辑，避免死角"
        ));
        
        tools.put("get_examples", new ToolDefinition(
            "get_examples",
            "获取代码示例",
            input -> """
                // BFS 移动示例
                Queue<int[]> queue = new LinkedList<>();
                queue.offer(new int[]{startX, startY});
                while (!queue.isEmpty()) {
                    int[] pos = queue.poll();
                    // 处理四个方向...
                }
                """
        ));
        
        tools.put("calculate", new ToolDefinition(
            "calculate",
            "执行数学计算",
            input -> {
                try {
                    // 简化计算器
                    if (input.contains("+")) {
                        String[] parts = input.split("\\+");
                        int result = Integer.parseInt(parts[0].trim()) + 
                                   Integer.parseInt(parts[1].trim());
                        return String.valueOf(result);
                    }
                    return "计算: " + input;
                } catch (Exception e) {
                    return "计算错误";
                }
            }
        ));
    }
    
    // 内部类
    private static class AgentResponse {
        String thought;
        String action;
        String actionInput;
        String finalAnswer;
    }
    
    private record ToolDefinition(
        String name,
        String description,
        java.util.function.Function<String, String> executor
    ) {}
}
