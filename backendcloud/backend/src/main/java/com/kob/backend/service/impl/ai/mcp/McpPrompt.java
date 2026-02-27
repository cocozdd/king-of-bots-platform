package com.kob.backend.service.impl.ai.mcp;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * MCP 提示模板定义
 * 
 * 预定义的提示模板，支持参数化
 */
public class McpPrompt {
    
    private final String name;
    private final String description;
    private final List<Argument> arguments;
    private final Function<Map<String, String>, List<Map<String, Object>>> renderer;
    
    public McpPrompt(String name, String description, List<Argument> arguments,
                     Function<Map<String, String>, List<Map<String, Object>>> renderer) {
        this.name = name;
        this.description = description;
        this.arguments = arguments;
        this.renderer = renderer;
    }
    
    public String getName() { return name; }
    public String getDescription() { return description; }
    public List<Argument> getArguments() { return arguments; }
    
    public List<Map<String, Object>> render(Map<String, String> args) {
        return renderer.apply(args);
    }
    
    /**
     * 生成符合 MCP 规范的 Schema
     */
    public Map<String, Object> toSchema() {
        Map<String, Object> schema = new HashMap<>();
        schema.put("name", name);
        schema.put("description", description);
        
        List<Map<String, Object>> argList = new ArrayList<>();
        for (Argument arg : arguments) {
            Map<String, Object> argMap = new HashMap<>();
            argMap.put("name", arg.name);
            argMap.put("description", arg.description);
            argMap.put("required", arg.required);
            argList.add(argMap);
        }
        schema.put("arguments", argList);
        
        return schema;
    }
    
    /**
     * 提示参数定义
     */
    public static class Argument {
        private final String name;
        private final String description;
        private final boolean required;
        
        public Argument(String name, String description, boolean required) {
            this.name = name;
            this.description = description;
            this.required = required;
        }
        
        public String getName() { return name; }
        public String getDescription() { return description; }
        public boolean isRequired() { return required; }
    }
}
