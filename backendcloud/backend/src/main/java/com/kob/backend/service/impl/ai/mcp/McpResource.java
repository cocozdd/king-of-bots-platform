package com.kob.backend.service.impl.ai.mcp;

import java.util.HashMap;
import java.util.Map;

/**
 * MCP 资源定义
 * 
 * 资源是静态数据，如文档、配置等
 */
public class McpResource {
    
    private final String uri;
    private final String name;
    private final String mimeType;
    private final String content;
    private final String description;
    
    public McpResource(String uri, String name, String mimeType, String content) {
        this(uri, name, mimeType, content, null);
    }
    
    public McpResource(String uri, String name, String mimeType, String content, String description) {
        this.uri = uri;
        this.name = name;
        this.mimeType = mimeType;
        this.content = content;
        this.description = description;
    }
    
    public String getUri() { return uri; }
    public String getName() { return name; }
    public String getMimeType() { return mimeType; }
    public String getContent() { return content; }
    public String getDescription() { return description; }
    
    /**
     * 生成符合 MCP 规范的 Schema
     */
    public Map<String, Object> toSchema() {
        Map<String, Object> schema = new HashMap<>();
        schema.put("uri", uri);
        schema.put("name", name);
        schema.put("mimeType", mimeType);
        if (description != null) {
            schema.put("description", description);
        }
        return schema;
    }
}
