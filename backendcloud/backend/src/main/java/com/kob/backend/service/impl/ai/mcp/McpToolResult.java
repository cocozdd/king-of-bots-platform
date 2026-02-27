package com.kob.backend.service.impl.ai.mcp;

/**
 * MCP 工具执行结果
 */
public class McpToolResult {
    
    private final boolean success;
    private final String content;
    private final Object data;
    
    private McpToolResult(boolean success, String content, Object data) {
        this.success = success;
        this.content = content;
        this.data = data;
    }
    
    public static McpToolResult success(String content) {
        return new McpToolResult(true, content, null);
    }
    
    public static McpToolResult success(String content, Object data) {
        return new McpToolResult(true, content, data);
    }
    
    public static McpToolResult error(String message) {
        return new McpToolResult(false, message, null);
    }
    
    public boolean isSuccess() { return success; }
    public String getContent() { return content; }
    public Object getData() { return data; }
}
