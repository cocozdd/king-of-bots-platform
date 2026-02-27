package com.kob.backend.service.impl.ai.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.kob.backend.config.AiServiceProperties;
import com.kob.backend.service.impl.ai.ABTestRouter;
import com.kob.backend.service.impl.ai.AiMetricsCollector;
import com.kob.backend.service.impl.ai.DeepseekClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.*;

/**
 * Agent 路由器 - 根据配置决定调用 Java AgentExecutor 或 Python Agent
 * 
 * 支持：
 * - Agent 执行（execute）
 * - 工具列表（tools）
 * 
 * 配置：
 * - ai.agent.backend=python 使用 Python 实现（LangGraph）
 * - ai.agent.backend=java 使用 Java 实现（默认）
 * 
 * Phase 3 增强：
 * - 支持 A/B 测试路由
 * - 集成监控指标收集
 * - 自动降级机制
 */
@Service
public class AgentRouter {
    
    private static final Logger log = LoggerFactory.getLogger(AgentRouter.class);
    
    private final RestTemplate restTemplate;
    private final AiServiceProperties properties;
    private final AgentExecutor javaAgentExecutor;
    private final ABTestRouter abTestRouter;
    private final AiMetricsCollector metricsCollector;
    private final ObjectMapper mapper = new ObjectMapper();
    
    @Value("${ai.agent.backend:java}")
    private String agentBackend;

    @Value("${ai.agent.allow-java-fallback:false}")
    private boolean allowJavaFallback;
    
    public AgentRouter(
            RestTemplate restTemplate,
            AiServiceProperties properties,
            AgentExecutor javaAgentExecutor,
            ABTestRouter abTestRouter,
            AiMetricsCollector metricsCollector) {
        this.restTemplate = restTemplate;
        this.properties = properties;
        this.javaAgentExecutor = javaAgentExecutor;
        this.abTestRouter = abTestRouter;
        this.metricsCollector = metricsCollector;
    }
    
    /**
     * 判断是否使用 Python 后端
     */
    public boolean usePython() {
        return "python".equalsIgnoreCase(agentBackend) && properties.isEnabled();
    }
    
    /**
     * 执行 Agent 任务（支持 A/B 测试）
     * 
     * @param task 任务描述
     * @param llmClient LLM 客户端（Java 后端使用）
     * @param context 上下文信息
     * @param userId 用户 ID（用于 A/B 测试路由）
     * @return Agent 执行结果
     */
    public AgentResult execute(String task, DeepseekClient llmClient, Map<String, Object> context, Integer userId) {
        long startTime = System.currentTimeMillis();
        boolean success = false;
        String backend = "java";
        
        try {
            // A/B 测试路由决策
            ABTestRouter.RouteDecision decision = abTestRouter.routeAgent(userId);
            backend = decision.getValue();
            
            AgentResult result;
            if (decision == ABTestRouter.RouteDecision.PYTHON && properties.isEnabled()) {
                result = pythonExecuteWithFallback(task, context, llmClient);
            } else {
                AgentExecutor.AgentResult javaResult = javaAgentExecutor.execute(task, llmClient, context);
                result = convertJavaResult(javaResult);
            }
            
            success = result.isSuccess();
            return result;
        } finally {
            long latency = System.currentTimeMillis() - startTime;
            metricsCollector.recordAgentRequest(backend, latency, success);
        }
    }
    
    /**
     * 执行 Agent 任务（兼容旧接口）
     */
    public AgentResult execute(String task, DeepseekClient llmClient, Map<String, Object> context) {
        return execute(task, llmClient, context, null);
    }
    
    /**
     * Python Agent 执行带自动降级
     */
    private AgentResult pythonExecuteWithFallback(String task, Map<String, Object> context, DeepseekClient llmClient) {
        AgentResult result = pythonExecute(task, context);
        if (result != null) {
            return result;
        }

        if (!allowJavaFallback) {
            log.error("Python Agent 调用失败，且 Java 回退已禁用。");
            metricsCollector.recordFallback("agent", "python", "none", "Python execution failed and java fallback disabled");
            abTestRouter.recordFallback("agent", "Python execution failed and java fallback disabled");
            return new AgentResult(
                false,
                "",
                new ArrayList<>(),
                0,
                "Python Agent 不可用，且 Java 回退已禁用"
            );
        }
        
        // Python 失败，降级到 Java
        log.warn("Python Agent 失败，降级到 Java Agent");
        metricsCollector.recordFallback("agent", "python", "java", "Python execution failed");
        abTestRouter.recordFallback("agent", "Python execution failed");
        
        AgentExecutor.AgentResult javaResult = javaAgentExecutor.execute(task, llmClient, context);
        return convertJavaResult(javaResult);
    }
    
    /**
     * 调用 Python Agent
     */
    private AgentResult pythonExecute(String task, Map<String, Object> context) {
        try {
            String url = properties.getBaseUrl() + "/api/agent/execute";
            
            Map<String, Object> request = new HashMap<>();
            request.put("task", task);
            request.put("context", context != null ? context : new HashMap<>());
            request.put("trace_id", UUID.randomUUID().toString());
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<Map<String, Object>> entity = new HttpEntity<>(request, headers);
            
            ResponseEntity<String> response = restTemplate.exchange(
                url, HttpMethod.POST, entity, String.class);
            
            JsonNode root = mapper.readTree(response.getBody());
            
            boolean success = root.path("success").asBoolean(false);
            String answer = root.path("answer").asText("");
            int steps = root.path("steps").asInt(0);
            String error = root.path("error").asText(null);
            
            List<Map<String, Object>> thoughtChain = new ArrayList<>();
            JsonNode chainNode = root.path("thought_chain");
            if (chainNode.isArray()) {
                for (JsonNode node : chainNode) {
                    Map<String, Object> step = new HashMap<>();
                    step.put("step", node.path("step").asInt(0));
                    step.put("thought", node.path("thought").asText(""));
                    step.put("action", node.path("action").asText(null));
                    step.put("observation", node.path("observation").asText(null));
                    thoughtChain.add(step);
                }
            }
            
            log.info("Python Agent 执行完成: success={}, steps={}", success, steps);
            return new AgentResult(success, answer, thoughtChain, steps, error);
        } catch (Exception e) {
            log.error("Python Agent 调用失败: {}", e.getMessage());
            return null;
        }
    }
    
    /**
     * 转换 Java Agent 结果
     */
    private AgentResult convertJavaResult(AgentExecutor.AgentResult javaResult) {
        return new AgentResult(
            javaResult.isSuccess(),
            javaResult.getAnswer(),
            javaResult.getThoughtChainMaps(),
            javaResult.getThoughtChain().size(),
            javaResult.getError()
        );
    }
    
    /**
     * 获取工具列表
     */
    public List<Map<String, String>> getTools() {
        if (usePython()) {
            List<Map<String, String>> tools = pythonGetTools();
            if (tools != null) {
                return tools;
            }

            if (!allowJavaFallback) {
                log.warn("Python Tools 获取失败，且 Java 回退已禁用。");
                return List.of(Map.of(
                    "name", "python_tools_unavailable",
                    "description", "Python tools unavailable and java fallback disabled"
                ));
            }
        }
        
        // 返回 Java 工具列表
        return javaAgentExecutor.getToolDescriptions();
    }
    
    /**
     * 调用 Python 获取工具列表
     */
    private List<Map<String, String>> pythonGetTools() {
        try {
            String url = properties.getBaseUrl() + "/api/agent/tools";
            
            ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);
            JsonNode root = mapper.readTree(response.getBody());
            
            if (!root.path("success").asBoolean(false)) {
                return null;
            }
            
            List<Map<String, String>> tools = new ArrayList<>();
            JsonNode toolsNode = root.path("tools");
            if (toolsNode.isArray()) {
                for (JsonNode node : toolsNode) {
                    Map<String, String> tool = new HashMap<>();
                    tool.put("name", node.path("name").asText());
                    tool.put("description", node.path("description").asText());
                    tools.add(tool);
                }
            }
            
            return tools;
        } catch (Exception e) {
            log.warn("Python Tools 调用失败: {}", e.getMessage());
            return null;
        }
    }
    
    /**
     * Agent 执行结果
     */
    public static class AgentResult {
        private final boolean success;
        private final String answer;
        private final List<Map<String, Object>> thoughtChain;
        private final int steps;
        private final String error;
        
        public AgentResult(boolean success, String answer, 
                          List<Map<String, Object>> thoughtChain, int steps, String error) {
            this.success = success;
            this.answer = answer;
            this.thoughtChain = thoughtChain;
            this.steps = steps;
            this.error = error;
        }
        
        public boolean isSuccess() { return success; }
        public String getAnswer() { return answer; }
        public List<Map<String, Object>> getThoughtChain() { return thoughtChain; }
        public int getSteps() { return steps; }
        public String getError() { return error; }
    }
}
