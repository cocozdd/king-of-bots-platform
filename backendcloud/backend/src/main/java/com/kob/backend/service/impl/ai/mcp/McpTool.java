package com.kob.backend.service.impl.ai.mcp;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * MCP 工具定义
 * 
 * 符合 MCP 规范的工具结构
 */
public class McpTool {
    
    private final String name;
    private final String description;
    private final Map<String, Object> inputSchema;
    private final Function<Map<String, Object>, McpToolResult> handler;
    
    public McpTool(String name, String description, Map<String, Object> inputSchema,
                   Function<Map<String, Object>, McpToolResult> handler) {
        this.name = name;
        this.description = description;
        this.inputSchema = inputSchema;
        this.handler = handler;
    }
    
    public String getName() { return name; }
    public String getDescription() { return description; }
    public Map<String, Object> getInputSchema() { return inputSchema; }
    
    public McpToolResult execute(Map<String, Object> arguments) {
        return handler.apply(arguments);
    }
    
    /**
     * 生成符合 MCP 规范的 Schema
     */
    public Map<String, Object> toSchema() {
        Map<String, Object> schema = new HashMap<>();
        schema.put("name", name);
        schema.put("description", description);
        schema.put("inputSchema", inputSchema);
        return schema;
    }
}
