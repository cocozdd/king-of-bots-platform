package com.kob.backend.controller.ai;

import com.kob.backend.service.impl.ai.DeepseekClient;
import com.kob.backend.service.impl.ai.AiMetricsService;
import com.kob.backend.service.impl.ai.PromptSecurityService;
import com.kob.backend.service.impl.ai.agent.AgentExecutor;
import com.kob.backend.service.impl.ai.agent.AgentRouter;
import com.kob.backend.service.impl.ai.agent.AgentTool;
import com.kob.backend.service.impl.ai.agent.ToolRegistry;
import com.kob.backend.service.impl.ai.mcp.McpServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.annotation.PostConstruct;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

/**
 * AI Agent 和 MCP 控制器
 * 
 * 提供：
 * - Agent 任务执行（ReAct 模式）
 * - MCP 协议端点
 * - 工具管理
 */
@RestController
@RequestMapping("/ai/agent")
public class AgentController {
    
    private static final Logger log = LoggerFactory.getLogger(AgentController.class);
    
    @Autowired
    private AgentExecutor agentExecutor;
    
    @Autowired
    private AgentRouter agentRouter;
    
    @Autowired
    private ToolRegistry toolRegistry;
    
    @Autowired
    private McpServer mcpServer;
    
    @Autowired
    private AiMetricsService metricsService;
    
    @Autowired
    private PromptSecurityService securityService;
    
    private DeepseekClient deepseekClient;
    private final ExecutorService executor = Executors.newFixedThreadPool(4);
    
    @PostConstruct
    public void init() {
        deepseekClient = new DeepseekClient(metricsService);
        log.info("Agent Controller 初始化完成");
    }
    
    // ==================== Agent 接口 ====================
    
    /**
     * 执行 Agent 任务
     * 
     * 使用 ReAct 模式进行多步推理
     */
    @PostMapping("/execute")
    public CompletableFuture<ResponseEntity<Map<String, Object>>> executeAgent(
            @RequestBody Map<String, Object> request) {
        return CompletableFuture.supplyAsync(() -> {
            Map<String, Object> result = new HashMap<>();
            long startTime = System.currentTimeMillis();
            
            try {
                String task = (String) request.get("task");
                if (task == null || task.trim().isEmpty()) {
                    result.put("success", false);
                    result.put("error", "请提供任务描述");
                    return ResponseEntity.ok(result);
                }
                
                // 安全验证
                PromptSecurityService.ValidationResult validation = 
                        securityService.validateQuestion(task);
                if (!validation.isSafe()) {
                    result.put("success", false);
                    result.put("error", "输入不合法: " + validation.getReason());
                    return ResponseEntity.ok(result);
                }
                
                // 提取上下文
                @SuppressWarnings("unchecked")
                Map<String, Object> context = (Map<String, Object>) request.getOrDefault("context", new HashMap<>());
                
                // 执行 Agent（通过路由器选择 Java 或 Python 后端）
                AgentRouter.AgentResult agentResult = 
                        agentRouter.execute(validation.getSanitizedInput(), deepseekClient, context);
                
                long latency = System.currentTimeMillis() - startTime;
                
                if (agentResult.isSuccess()) {
                    result.put("success", true);
                    result.put("answer", agentResult.getAnswer());
                    result.put("thoughtChain", agentResult.getThoughtChain());
                    result.put("steps", agentResult.getSteps());
                } else {
                    result.put("success", false);
                    result.put("error", agentResult.getError());
                    result.put("thoughtChain", agentResult.getThoughtChain());
                }
                result.put("latencyMs", latency);
                result.put("backend", agentRouter.usePython() ? "python" : "java");
                
                log.info("Agent 任务完成: {} 步, 耗时 {}ms, 后端: {}", 
                        agentResult.getSteps(), latency, agentRouter.usePython() ? "python" : "java");
                
            } catch (Exception e) {
                log.error("Agent 执行失败: {}", e.getMessage(), e);
                result.put("success", false);
                result.put("error", "执行失败: " + e.getMessage());
            }
            
            return ResponseEntity.ok(result);
        }, executor);
    }
    
    /**
     * 获取可用工具列表
     */
    @GetMapping("/tools")
    public ResponseEntity<Map<String, Object>> getTools() {
        Map<String, Object> result = new HashMap<>();

        if (agentRouter.usePython()) {
            List<Map<String, String>> tools = agentRouter.getTools();
            result.put("tools", tools);
            result.put("count", tools.size());
            result.put("backend", "python");
            return ResponseEntity.ok(result);
        }

        List<Map<String, Object>> tools = toolRegistry.getAll().stream()
            .map(tool -> {
                Map<String, Object> toolInfo = new HashMap<>();
                toolInfo.put("name", tool.getName());
                toolInfo.put("description", tool.getDescription());
                toolInfo.put("parameters", tool.getParameterSchema());
                return toolInfo;
            })
            .collect(Collectors.toList());

        result.put("tools", tools);
        result.put("count", tools.size());
        result.put("backend", "java");
        return ResponseEntity.ok(result);
    }
    
    /**
     * 直接调用工具
     */
    @PostMapping("/tools/call")
    public ResponseEntity<Map<String, Object>> callTool(@RequestBody Map<String, Object> request) {
        Map<String, Object> result = new HashMap<>();

        if (agentRouter.usePython()) {
            result.put("success", false);
            result.put("error", "Python Agent 模式下不支持 Java 侧直接工具调用，请通过 /ai/agent/execute 调用");
            return ResponseEntity.ok(result);
        }
        
        String toolName = (String) request.get("tool");
        @SuppressWarnings("unchecked")
        Map<String, Object> parameters = (Map<String, Object>) request.getOrDefault("parameters", new HashMap<>());
        
        if (toolName == null || toolName.isEmpty()) {
            result.put("success", false);
            result.put("error", "请指定工具名称");
            return ResponseEntity.ok(result);
        }
        
        AgentTool tool = toolRegistry.get(toolName);
        if (tool == null) {
            result.put("success", false);
            result.put("error", "工具不存在: " + toolName);
            return ResponseEntity.ok(result);
        }
        
        try {
            AgentTool.ToolResult toolResult = tool.execute(parameters);
            result.put("success", toolResult.isSuccess());
            result.put("output", toolResult.getOutput());
            if (toolResult.getData() != null) {
                result.put("data", toolResult.getData());
            }
        } catch (Exception e) {
            result.put("success", false);
            result.put("error", "工具执行失败: " + e.getMessage());
        }
        
        return ResponseEntity.ok(result);
    }
    
    // ==================== MCP 接口 ====================
    
    /**
     * MCP JSON-RPC 端点
     * 
     * 符合 MCP 协议规范的统一入口
     */
    @PostMapping("/mcp")
    public ResponseEntity<String> mcpEndpoint(@RequestBody String jsonRequest) {
        log.debug("MCP 请求: {}", jsonRequest);
        McpServer.McpResponse response = mcpServer.handleRequest(jsonRequest);
        return ResponseEntity.ok(response.toJson());
    }
    
    /**
     * MCP 初始化（便捷接口）
     */
    @PostMapping("/mcp/initialize")
    public ResponseEntity<Map<String, Object>> mcpInitialize() {
        String request = "{\"jsonrpc\":\"2.0\",\"method\":\"initialize\",\"params\":{},\"id\":1}";
        McpServer.McpResponse response = mcpServer.handleRequest(request);
        
        Map<String, Object> result = new HashMap<>();
        result.put("success", response.getError() == null);
        result.put("result", response.getResult());
        if (response.getError() != null) {
            result.put("error", response.getError());
        }
        return ResponseEntity.ok(result);
    }
    
    /**
     * MCP 工具列表（便捷接口）
     */
    @GetMapping("/mcp/tools")
    public ResponseEntity<Map<String, Object>> mcpToolsList() {
        String request = "{\"jsonrpc\":\"2.0\",\"method\":\"tools/list\",\"params\":{},\"id\":1}";
        McpServer.McpResponse response = mcpServer.handleRequest(request);
        
        Map<String, Object> result = new HashMap<>();
        result.put("success", response.getError() == null);
        result.put("result", response.getResult());
        return ResponseEntity.ok(result);
    }
    
    /**
     * MCP 资源列表（便捷接口）
     */
    @GetMapping("/mcp/resources")
    public ResponseEntity<Map<String, Object>> mcpResourcesList() {
        String request = "{\"jsonrpc\":\"2.0\",\"method\":\"resources/list\",\"params\":{},\"id\":1}";
        McpServer.McpResponse response = mcpServer.handleRequest(request);
        
        Map<String, Object> result = new HashMap<>();
        result.put("success", response.getError() == null);
        result.put("result", response.getResult());
        return ResponseEntity.ok(result);
    }
    
    /**
     * MCP 提示列表（便捷接口）
     */
    @GetMapping("/mcp/prompts")
    public ResponseEntity<Map<String, Object>> mcpPromptsList() {
        String request = "{\"jsonrpc\":\"2.0\",\"method\":\"prompts/list\",\"params\":{},\"id\":1}";
        McpServer.McpResponse response = mcpServer.handleRequest(request);
        
        Map<String, Object> result = new HashMap<>();
        result.put("success", response.getError() == null);
        result.put("result", response.getResult());
        return ResponseEntity.ok(result);
    }
    
    /**
     * 获取 Agent 和 MCP 状态
     */
    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> getStatus() {
        Map<String, Object> status = new HashMap<>();
        status.put("agentEnabled", true);
        status.put("mcpEnabled", true);
        if (agentRouter.usePython()) {
            status.put("toolCount", agentRouter.getTools().size());
        } else {
            status.put("toolCount", toolRegistry.getAll().size());
        }
        status.put("llmEnabled", deepseekClient != null && deepseekClient.enabled());
        return ResponseEntity.ok(status);
    }
}
