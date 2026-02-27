package com.kob.backend.service.impl.ai.mcp;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * MCP (Model Context Protocol) 服务端实现
 * 
 * MCP 是 Anthropic 提出的标准化 AI 工具协议，核心概念：
 * 1. Resources: 静态数据资源（如文档、配置）
 * 2. Tools: 可执行的工具/函数
 * 3. Prompts: 预定义的提示模板
 * 4. Sampling: 请求 LLM 采样
 * 
 * 面试亮点：
 * - 遵循 MCP 2024 规范
 * - JSON-RPC 2.0 消息格式
 * - 支持工具发现和调用
 * - 可扩展的资源管理
 */
@Component
public class McpServer {
    
    private static final Logger log = LoggerFactory.getLogger(McpServer.class);
    private static final String MCP_VERSION = "2024-11-05";
    
    private final Map<String, McpTool> tools = new ConcurrentHashMap<>();
    private final Map<String, McpResource> resources = new ConcurrentHashMap<>();
    private final Map<String, McpPrompt> prompts = new ConcurrentHashMap<>();
    
    private String serverName = "kob-ai-server";
    private String serverVersion = "1.0.0";
    
    @PostConstruct
    public void init() {
        registerBuiltinTools();
        registerBuiltinResources();
        registerBuiltinPrompts();
        log.info("MCP Server 初始化完成: {} tools, {} resources, {} prompts",
                tools.size(), resources.size(), prompts.size());
    }
    
    // ==================== JSON-RPC 消息处理 ====================
    
    /**
     * 处理 JSON-RPC 请求
     */
    public McpResponse handleRequest(String jsonRequest) {
        try {
            JSONObject request = JSON.parseObject(jsonRequest);
            String method = request.getString("method");
            Object params = request.get("params");
            Object id = request.get("id");
            
            Object result = dispatch(method, params);
            return McpResponse.success(id, result);
            
        } catch (McpException e) {
            return McpResponse.error(null, e.getCode(), e.getMessage());
        } catch (Exception e) {
            log.error("MCP 请求处理失败: {}", e.getMessage());
            return McpResponse.error(null, -32603, "Internal error: " + e.getMessage());
        }
    }
    
    private Object dispatch(String method, Object params) {
        return switch (method) {
            case "initialize" -> handleInitialize(params);
            case "tools/list" -> handleToolsList();
            case "tools/call" -> handleToolsCall(params);
            case "resources/list" -> handleResourcesList();
            case "resources/read" -> handleResourcesRead(params);
            case "prompts/list" -> handlePromptsList();
            case "prompts/get" -> handlePromptsGet(params);
            default -> throw new McpException(-32601, "Method not found: " + method);
        };
    }
    
    // ==================== 协议方法实现 ====================
    
    private Map<String, Object> handleInitialize(Object params) {
        Map<String, Object> result = new HashMap<>();
        result.put("protocolVersion", MCP_VERSION);
        
        Map<String, Object> serverInfo = new HashMap<>();
        serverInfo.put("name", serverName);
        serverInfo.put("version", serverVersion);
        result.put("serverInfo", serverInfo);
        
        Map<String, Object> capabilities = new HashMap<>();
        capabilities.put("tools", Map.of("listChanged", true));
        capabilities.put("resources", Map.of("subscribe", false, "listChanged", true));
        capabilities.put("prompts", Map.of("listChanged", true));
        result.put("capabilities", capabilities);
        
        return result;
    }
    
    private Map<String, Object> handleToolsList() {
        List<Map<String, Object>> toolList = new ArrayList<>();
        for (McpTool tool : tools.values()) {
            toolList.add(tool.toSchema());
        }
        return Map.of("tools", toolList);
    }
    
    private Map<String, Object> handleToolsCall(Object params) {
        if (!(params instanceof Map)) {
            throw new McpException(-32602, "Invalid params");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> p = (Map<String, Object>) params;
        
        String name = (String) p.get("name");
        @SuppressWarnings("unchecked")
        Map<String, Object> arguments = (Map<String, Object>) p.getOrDefault("arguments", new HashMap<>());
        
        McpTool tool = tools.get(name);
        if (tool == null) {
            throw new McpException(-32602, "Unknown tool: " + name);
        }
        
        try {
            McpToolResult result = tool.execute(arguments);
            List<Map<String, Object>> content = new ArrayList<>();
            content.add(Map.of(
                    "type", "text",
                    "text", result.getContent()
            ));
            return Map.of(
                    "content", content,
                    "isError", !result.isSuccess()
            );
        } catch (Exception e) {
            return Map.of(
                    "content", List.of(Map.of("type", "text", "text", "Error: " + e.getMessage())),
                    "isError", true
            );
        }
    }
    
    private Map<String, Object> handleResourcesList() {
        List<Map<String, Object>> resourceList = new ArrayList<>();
        for (McpResource resource : resources.values()) {
            resourceList.add(resource.toSchema());
        }
        return Map.of("resources", resourceList);
    }
    
    private Map<String, Object> handleResourcesRead(Object params) {
        if (!(params instanceof Map)) {
            throw new McpException(-32602, "Invalid params");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> p = (Map<String, Object>) params;
        
        String uri = (String) p.get("uri");
        McpResource resource = resources.get(uri);
        if (resource == null) {
            throw new McpException(-32602, "Unknown resource: " + uri);
        }
        
        List<Map<String, Object>> contents = new ArrayList<>();
        contents.add(Map.of(
                "uri", uri,
                "mimeType", resource.getMimeType(),
                "text", resource.getContent()
        ));
        return Map.of("contents", contents);
    }
    
    private Map<String, Object> handlePromptsList() {
        List<Map<String, Object>> promptList = new ArrayList<>();
        for (McpPrompt prompt : prompts.values()) {
            promptList.add(prompt.toSchema());
        }
        return Map.of("prompts", promptList);
    }
    
    private Map<String, Object> handlePromptsGet(Object params) {
        if (!(params instanceof Map)) {
            throw new McpException(-32602, "Invalid params");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> p = (Map<String, Object>) params;
        
        String name = (String) p.get("name");
        @SuppressWarnings("unchecked")
        Map<String, String> arguments = (Map<String, String>) p.getOrDefault("arguments", new HashMap<>());
        
        McpPrompt prompt = prompts.get(name);
        if (prompt == null) {
            throw new McpException(-32602, "Unknown prompt: " + name);
        }
        
        List<Map<String, Object>> messages = prompt.render(arguments);
        return Map.of("messages", messages);
    }
    
    // ==================== 注册方法 ====================
    
    public void registerTool(McpTool tool) {
        tools.put(tool.getName(), tool);
    }
    
    public void registerResource(McpResource resource) {
        resources.put(resource.getUri(), resource);
    }
    
    public void registerPrompt(McpPrompt prompt) {
        prompts.put(prompt.getName(), prompt);
    }
    
    private void registerBuiltinTools() {
        // Bot 代码生成工具
        registerTool(new McpTool(
                "generate_bot_code",
                "根据策略描述生成 Bot 代码",
                Map.of(
                        "type", "object",
                        "properties", Map.of(
                                "description", Map.of(
                                        "type", "string",
                                        "description", "Bot 策略描述"
                                ),
                                "style", Map.of(
                                        "type", "string",
                                        "enum", List.of("aggressive", "defensive", "balanced"),
                                        "description", "代码风格"
                                )
                        ),
                        "required", List.of("description")
                ),
                args -> {
                    String desc = (String) args.get("description");
                    return McpToolResult.success("生成的 Bot 代码框架:\n```java\npublic class Bot { ... }\n```");
                }
        ));
        
        // 知识库查询工具
        registerTool(new McpTool(
                "search_knowledge",
                "搜索 Bot 开发知识库",
                Map.of(
                        "type", "object",
                        "properties", Map.of(
                                "query", Map.of(
                                        "type", "string",
                                        "description", "搜索关键词"
                                ),
                                "limit", Map.of(
                                        "type", "integer",
                                        "description", "返回结果数量",
                                        "default", 5
                                )
                        ),
                        "required", List.of("query")
                ),
                args -> {
                    String query = (String) args.get("query");
                    return McpToolResult.success("找到 3 篇关于 '" + query + "' 的文档");
                }
        ));
        
        // 对战分析工具
        registerTool(new McpTool(
                "analyze_battle",
                "分析对战回放数据",
                Map.of(
                        "type", "object",
                        "properties", Map.of(
                                "battleId", Map.of(
                                        "type", "string",
                                        "description", "对战 ID"
                                )
                        ),
                        "required", List.of("battleId")
                ),
                args -> {
                    String battleId = (String) args.get("battleId");
                    return McpToolResult.success("对战 " + battleId + " 分析完成：玩家 A 胜率更高");
                }
        ));
    }
    
    private void registerBuiltinResources() {
        // 游戏规则文档
        registerResource(new McpResource(
                "kob://docs/game-rules",
                "游戏规则文档",
                "text/markdown",
                """
                # 贪吃蛇对战规则
                
                1. 双方蛇在 13x14 的地图上移动
                2. 每回合同时移动一格
                3. 撞墙、撞身体或撞对手则失败
                4. 每 3 回合蛇身增长一格
                """
        ));
        
        // API 文档
        registerResource(new McpResource(
                "kob://docs/bot-api",
                "Bot API 文档",
                "text/markdown",
                """
                # Bot API 参考
                
                ## 输入格式
                - 地图数据: 13*14 的 0/1 矩阵
                - 蛇 A 位置: (ax, ay)
                - 蛇 B 位置: (bx, by)
                
                ## 输出格式
                - 0: 上  1: 右  2: 下  3: 左
                """
        ));
    }
    
    private void registerBuiltinPrompts() {
        // Bot 开发助手提示
        registerPrompt(new McpPrompt(
                "bot_assistant",
                "Bot 开发助手提示模板",
                List.of(
                        new McpPrompt.Argument("task", "任务描述", true),
                        new McpPrompt.Argument("style", "回答风格", false)
                ),
                args -> {
                    String task = args.getOrDefault("task", "");
                    String style = args.getOrDefault("style", "详细");
                    return List.of(
                            Map.of(
                                    "role", "system",
                                    "content", Map.of(
                                            "type", "text",
                                            "text", "你是 Bot 开发助手，回答风格: " + style
                                    )
                            ),
                            Map.of(
                                    "role", "user",
                                    "content", Map.of(
                                            "type", "text",
                                            "text", task
                                    )
                            )
                    );
                }
        ));
        
        // 代码审查提示
        registerPrompt(new McpPrompt(
                "code_review",
                "代码审查提示模板",
                List.of(
                        new McpPrompt.Argument("code", "待审查的代码", true)
                ),
                args -> {
                    String code = args.getOrDefault("code", "");
                    return List.of(
                            Map.of(
                                    "role", "system",
                                    "content", Map.of(
                                            "type", "text",
                                            "text", "你是代码审查专家，请审查以下 Bot 代码"
                                    )
                            ),
                            Map.of(
                                    "role", "user",
                                    "content", Map.of(
                                            "type", "text",
                                            "text", "请审查这段代码:\n```\n" + code + "\n```"
                                    )
                            )
                    );
                }
        ));
    }
    
    // ==================== 内部类定义 ====================
    
    public static class McpException extends RuntimeException {
        private final int code;
        
        public McpException(int code, String message) {
            super(message);
            this.code = code;
        }
        
        public int getCode() { return code; }
    }
    
    public static class McpResponse {
        private final String jsonrpc = "2.0";
        private final Object id;
        private final Object result;
        private final Map<String, Object> error;
        
        private McpResponse(Object id, Object result, Map<String, Object> error) {
            this.id = id;
            this.result = result;
            this.error = error;
        }
        
        public static McpResponse success(Object id, Object result) {
            return new McpResponse(id, result, null);
        }
        
        public static McpResponse error(Object id, int code, String message) {
            Map<String, Object> error = new HashMap<>();
            error.put("code", code);
            error.put("message", message);
            return new McpResponse(id, null, error);
        }
        
        public String toJson() {
            Map<String, Object> map = new HashMap<>();
            map.put("jsonrpc", jsonrpc);
            map.put("id", id);
            if (result != null) map.put("result", result);
            if (error != null) map.put("error", error);
            return JSON.toJSONString(map);
        }
        
        public Object getResult() { return result; }
        public Map<String, Object> getError() { return error; }
    }
}
