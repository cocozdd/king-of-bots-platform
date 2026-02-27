package com.kob.backend.service.impl.ai.agent;

import java.util.Map;

/**
 * Agent 工具接口
 * 
 * 定义 Agent 可调用的工具规范
 */
public interface AgentTool {
    
    /**
     * 工具名称
     */
    String getName();
    
    /**
     * 工具描述（供 LLM 理解）
     */
    String getDescription();
    
    /**
     * 参数 Schema（JSON Schema 格式）
     */
    Map<String, Object> getParameterSchema();
    
    /**
     * 执行工具
     * @param parameters 工具参数
     * @return 执行结果
     */
    ToolResult execute(Map<String, Object> parameters);
    
    /**
     * 工具执行结果
     */
    class ToolResult {
        private final boolean success;
        private final String output;
        private final Object data;
        
        private ToolResult(boolean success, String output, Object data) {
            this.success = success;
            this.output = output;
            this.data = data;
        }
        
        public static ToolResult success(String output) {
            return new ToolResult(true, output, null);
        }
        
        public static ToolResult success(String output, Object data) {
            return new ToolResult(true, output, data);
        }
        
        public static ToolResult error(String error) {
            return new ToolResult(false, error, null);
        }
        
        public boolean isSuccess() { return success; }
        public String getOutput() { return output; }
        public Object getData() { return data; }
    }
}
