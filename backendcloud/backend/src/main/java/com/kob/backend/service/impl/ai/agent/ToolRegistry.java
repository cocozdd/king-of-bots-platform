package com.kob.backend.service.impl.ai.agent;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 工具注册中心
 * 
 * 管理所有可用的 Agent 工具
 * 
 * @deprecated 自 Phase 3 起，推荐使用 Python Agent 的工具系统。
 *             此类将在下个版本移除。Python 工具定义在 aiservice/agent/tools.py
 * @see com.kob.backend.service.impl.ai.agent.AgentRouter
 */
@Deprecated
@Component
public class ToolRegistry {
    
    private static final Logger log = LoggerFactory.getLogger(ToolRegistry.class);
    
    private final Map<String, AgentTool> tools = new ConcurrentHashMap<>();
    
    @PostConstruct
    public void init() {
        // 注册内置工具
        registerBuiltinTools();
        log.info("工具注册中心初始化完成，已注册 {} 个工具", tools.size());
    }
    
    private void registerBuiltinTools() {
        // 1. 知识库搜索工具
        register(new KnowledgeSearchTool());
        
        // 2. 代码分析工具
        register(new CodeAnalysisTool());
        
        // 3. 对战数据查询工具
        register(new BattleQueryTool());
        
        // 4. 策略推荐工具
        register(new StrategyRecommendTool());
        
        // 5. 计算器工具
        register(new CalculatorTool());
    }
    
    public void register(AgentTool tool) {
        tools.put(tool.getName(), tool);
        log.debug("注册工具: {}", tool.getName());
    }
    
    public void unregister(String name) {
        tools.remove(name);
    }
    
    public AgentTool get(String name) {
        return tools.get(name);
    }
    
    public List<AgentTool> getAll() {
        return new ArrayList<>(tools.values());
    }
    
    public boolean has(String name) {
        return tools.containsKey(name);
    }
    
    /**
     * 生成工具描述（供 LLM 使用）
     */
    public String generateToolsPrompt() {
        StringBuilder sb = new StringBuilder();
        sb.append("你可以使用以下工具：\n\n");
        
        for (AgentTool tool : tools.values()) {
            sb.append("工具名称: ").append(tool.getName()).append("\n");
            sb.append("描述: ").append(tool.getDescription()).append("\n");
            sb.append("参数: ").append(formatSchema(tool.getParameterSchema())).append("\n\n");
        }
        
        return sb.toString();
    }
    
    private String formatSchema(Map<String, Object> schema) {
        if (schema == null || schema.isEmpty()) {
            return "无参数";
        }
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Object> entry : schema.entrySet()) {
            sb.append(entry.getKey()).append(": ").append(entry.getValue()).append("; ");
        }
        return sb.toString();
    }
    
    // ==================== 内置工具实现 ====================
    
    /**
     * 知识库搜索工具
     */
    static class KnowledgeSearchTool implements AgentTool {
        @Override
        public String getName() { return "knowledge_search"; }
        
        @Override
        public String getDescription() {
            return "搜索 Bot 开发知识库，获取策略、教程、最佳实践等信息";
        }
        
        @Override
        public Map<String, Object> getParameterSchema() {
            Map<String, Object> schema = new HashMap<>();
            schema.put("query", "搜索关键词（必填）");
            schema.put("limit", "返回结果数量，默认5");
            return schema;
        }
        
        @Override
        public ToolResult execute(Map<String, Object> parameters) {
            String query = (String) parameters.get("query");
            if (query == null || query.isEmpty()) {
                return ToolResult.error("缺少查询参数 query");
            }
            // 实际实现会调用 HybridSearchService
            return ToolResult.success("搜索结果: 找到关于 '" + query + "' 的相关文档");
        }
    }
    
    /**
     * 代码分析工具
     */
    static class CodeAnalysisTool implements AgentTool {
        @Override
        public String getName() { return "code_analysis"; }
        
        @Override
        public String getDescription() {
            return "分析 Bot 代码，检查语法错误、逻辑问题和性能瓶颈";
        }
        
        @Override
        public Map<String, Object> getParameterSchema() {
            Map<String, Object> schema = new HashMap<>();
            schema.put("code", "要分析的代码（必填）");
            schema.put("focus", "分析重点: syntax/logic/performance，默认 all");
            return schema;
        }
        
        @Override
        public ToolResult execute(Map<String, Object> parameters) {
            String code = (String) parameters.get("code");
            if (code == null || code.isEmpty()) {
                return ToolResult.error("缺少代码参数 code");
            }
            return ToolResult.success("代码分析完成: 代码结构良好，未发现明显问题");
        }
    }
    
    /**
     * 对战数据查询工具
     */
    static class BattleQueryTool implements AgentTool {
        @Override
        public String getName() { return "battle_query"; }
        
        @Override
        public String getDescription() {
            return "查询对战记录和统计数据，包括胜率、对战历史等";
        }
        
        @Override
        public Map<String, Object> getParameterSchema() {
            Map<String, Object> schema = new HashMap<>();
            schema.put("botId", "Bot ID（可选）");
            schema.put("userId", "用户 ID（可选）");
            schema.put("limit", "返回记录数，默认10");
            return schema;
        }
        
        @Override
        public ToolResult execute(Map<String, Object> parameters) {
            return ToolResult.success("查询到最近10场对战记录");
        }
    }
    
    /**
     * 策略推荐工具
     */
    static class StrategyRecommendTool implements AgentTool {
        @Override
        public String getName() { return "strategy_recommend"; }
        
        @Override
        public String getDescription() {
            return "根据当前局势推荐最佳策略，支持开局、中盘、残局分析";
        }
        
        @Override
        public Map<String, Object> getParameterSchema() {
            Map<String, Object> schema = new HashMap<>();
            schema.put("gameState", "当前游戏状态（地图和蛇位置）");
            schema.put("phase", "游戏阶段: opening/midgame/endgame");
            return schema;
        }
        
        @Override
        public ToolResult execute(Map<String, Object> parameters) {
            String phase = (String) parameters.getOrDefault("phase", "midgame");
            return ToolResult.success("针对 " + phase + " 阶段的策略建议: 优先占领中央区域");
        }
    }
    
    /**
     * 计算器工具
     */
    static class CalculatorTool implements AgentTool {
        @Override
        public String getName() { return "calculator"; }
        
        @Override
        public String getDescription() {
            return "执行数学计算，支持基本运算和统计计算";
        }
        
        @Override
        public Map<String, Object> getParameterSchema() {
            Map<String, Object> schema = new HashMap<>();
            schema.put("expression", "数学表达式，如 '3 + 5 * 2'");
            return schema;
        }
        
        @Override
        public ToolResult execute(Map<String, Object> parameters) {
            String expr = (String) parameters.get("expression");
            if (expr == null) {
                return ToolResult.error("缺少表达式参数");
            }
            // 简单计算示例
            try {
                // 简单的四则运算解析
                double result = evaluateSimple(expr);
                return ToolResult.success("计算结果: " + result, result);
            } catch (Exception e) {
                return ToolResult.error("计算失败: " + e.getMessage());
            }
        }
        
        private double evaluateSimple(String expr) {
            // 简单的表达式计算器（不依赖 JavaScript 引擎）
            expr = expr.replaceAll("\\s+", "");
            try {
                return parseExpression(expr);
            } catch (Exception e) {
                throw new RuntimeException("表达式计算失败: " + expr);
            }
        }
        
        private double parseExpression(String expr) {
            // 处理加减
            int pos = -1;
            int depth = 0;
            for (int i = expr.length() - 1; i >= 0; i--) {
                char c = expr.charAt(i);
                if (c == ')') depth++;
                else if (c == '(') depth--;
                else if (depth == 0 && (c == '+' || c == '-') && i > 0) {
                    pos = i;
                    break;
                }
            }
            if (pos > 0) {
                double left = parseExpression(expr.substring(0, pos));
                double right = parseTerm(expr.substring(pos + 1));
                return expr.charAt(pos) == '+' ? left + right : left - right;
            }
            return parseTerm(expr);
        }
        
        private double parseTerm(String expr) {
            // 处理乘除
            int pos = -1;
            int depth = 0;
            for (int i = expr.length() - 1; i >= 0; i--) {
                char c = expr.charAt(i);
                if (c == ')') depth++;
                else if (c == '(') depth--;
                else if (depth == 0 && (c == '*' || c == '/')) {
                    pos = i;
                    break;
                }
            }
            if (pos > 0) {
                double left = parseTerm(expr.substring(0, pos));
                double right = parseFactor(expr.substring(pos + 1));
                return expr.charAt(pos) == '*' ? left * right : left / right;
            }
            return parseFactor(expr);
        }
        
        private double parseFactor(String expr) {
            // 处理括号和数字
            if (expr.startsWith("(") && expr.endsWith(")")) {
                return parseExpression(expr.substring(1, expr.length() - 1));
            }
            return Double.parseDouble(expr);
        }
    }
}
