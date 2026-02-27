package com.kob.backend.service.impl.ai.agent;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.kob.backend.service.impl.ai.DeepseekClient;
import com.kob.backend.service.impl.ai.message.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * AI Agent 执行器
 * 
 * 实现 ReAct (Reason-Act-Observe) 模式的多步推理：
 * 1. Reason: LLM 分析任务，决定下一步行动
 * 2. Act: 调用工具执行操作
 * 3. Observe: 获取工具结果，更新上下文
 * 4. 循环直到任务完成或达到最大步数
 * 
 * 面试亮点：
 * - ReAct 模式实现多步推理
 * - 工具调用和结果观察
 * - 循环检测和超时控制
 * - 思维链（Chain-of-Thought）追踪
 * 
 * @deprecated 自 Phase 3 起，推荐使用 Python LangGraph Agent 实现。
 *             此类将在下个版本移除。请通过 AgentRouter 路由到 Python 后端。
 *             配置: ai.agent.backend=python
 * @see com.kob.backend.service.impl.ai.agent.AgentRouter
 */
@Deprecated
@Service
public class AgentExecutor {
    
    private static final Logger log = LoggerFactory.getLogger(AgentExecutor.class);
    private static final int MAX_ITERATIONS = 10;
    private static final Pattern TOOL_CALL_PATTERN = Pattern.compile(
            "\\[TOOL_CALL\\]\\s*\\{([^}]+)\\}",
            Pattern.CASE_INSENSITIVE | Pattern.DOTALL
    );
    private static final Pattern FINAL_ANSWER_PATTERN = Pattern.compile(
            "\\[FINAL_ANSWER\\]\\s*(.+)",
            Pattern.CASE_INSENSITIVE | Pattern.DOTALL
    );
    
    @Autowired
    private ToolRegistry toolRegistry;
    
    /**
     * 执行 Agent 任务
     */
    public AgentResult execute(String task, DeepseekClient llmClient) {
        return execute(task, llmClient, new HashMap<>());
    }
    
    /**
     * 带上下文执行 Agent 任务 - 2026年重构版本
     *
     * 核心改进：
     * 1. ❌ 废弃 StringBuilder scratchpad（字符串拼接）
     * 2. ✅ 使用 List<BaseMessage>（结构化消息）
     *
     * 为什么这样改？（参考文档2.5.4节）
     * - 旧方式：scratchpad.append("\nThought: " + ...) 容易混乱
     * - 新方式：messages.add(new AIMessage(...)) 清晰、类型安全
     */
    public AgentResult execute(String task, DeepseekClient llmClient, Map<String, Object> context) {
        log.info("Agent 开始执行任务: {}", task.length() > 100 ? task.substring(0, 100) + "..." : task);

        List<ThoughtStep> thoughtChain = new ArrayList<>();

        // ========== 核心改变：使用结构化消息列表 ==========
        List<BaseMessage> messages = new ArrayList<>();

        // 第一条：System Prompt
        messages.add(new SystemMessage(buildSystemPrompt()));

        // 第二条：用户任务（只在第一轮添加）
        String initialUserMessage = buildUserMessage(task, "", context);
        messages.add(new HumanMessage(initialUserMessage));

        for (int i = 0; i < MAX_ITERATIONS; i++) {
            log.debug("Agent 迭代 {}/{}", i + 1, MAX_ITERATIONS);

            // ========== 调用 LLM（使用结构化消息）==========
            String response;
            try {
                response = llmClient.chat(messages);  // 新方法！
            } catch (Exception e) {
                log.error("LLM 调用失败: {}", e.getMessage());
                return AgentResult.error("LLM 调用失败: " + e.getMessage(), thoughtChain);
            }

            // 解析响应
            ParsedResponse parsed = parseResponse(response);

            // 记录思考步骤
            ThoughtStep step = new ThoughtStep(i + 1, parsed.thought, parsed.action, parsed.actionInput);
            thoughtChain.add(step);

            // ========== 将LLM的响应加入消息历史 ==========
            messages.add(new AIMessage(response));

            // 检查是否完成
            if (parsed.isFinalAnswer) {
                log.info("Agent 完成任务，共 {} 步", i + 1);
                return AgentResult.success(parsed.finalAnswer, thoughtChain);
            }

            // 执行工具调用
            if (parsed.action != null && !parsed.action.isEmpty()) {
                AgentTool tool = toolRegistry.get(parsed.action);
                String observation;

                if (tool == null) {
                    observation = "错误: 工具 '" + parsed.action + "' 不存在";
                    step.setObservation(observation);
                } else {
                    try {
                        AgentTool.ToolResult result = tool.execute(parsed.actionInput);
                        observation = result.isSuccess() ? result.getOutput() : "错误: " + result.getOutput();
                        step.setObservation(observation);
                    } catch (Exception e) {
                        observation = "工具执行失败: " + e.getMessage();
                        step.setObservation(observation);
                    }
                }

                // ========== 核心改进：用HumanMessage返回工具结果 ==========
                // Phase 3会改为ToolMessage，现在先用HumanMessage临时替代
                String observationMessage = String.format(
                    "Observation: %s\n\n请根据上述观察结果继续思考。",
                    observation
                );
                messages.add(new HumanMessage(observationMessage));
            }
        }

        log.warn("Agent 达到最大迭代次数");
        return AgentResult.error("达到最大迭代次数，任务未完成", thoughtChain);
    }
    
    private String buildSystemPrompt() {
        return """
            你是一个智能 Agent，负责帮助用户完成 Bot 开发相关任务。
            
            你的工作方式是 ReAct 模式：
            1. Thought: 分析当前情况，思考下一步
            2. Action: 决定使用哪个工具
            3. Action Input: 工具的输入参数
            4. Observation: 观察工具返回结果
            5. 重复以上步骤直到完成任务
            
            """ + toolRegistry.generateToolsPrompt() + """
            
            响应格式：
            - 如果需要使用工具：
              Thought: [你的思考过程]
              Action: [工具名称]
              Action Input: {"param": "value"}
            
            - 如果已完成任务：
              Thought: [最终思考]
              [FINAL_ANSWER] [最终答案]
            
            最终答案格式要求：
            - 使用 Markdown 格式输出
            - 段落之间用空行分隔
            - 使用 ## 作为二级标题，### 作为三级标题
            - 代码必须用 ```语言名 和 ``` 包裹，每行代码单独一行
            - 列表项之间适当换行
            
            注意：
            - 每次只能调用一个工具
            - 不要编造工具不存在的结果
            - 如果无法完成，说明原因
            """;
    }
    
    private String buildUserMessage(String task, String scratchpad, Map<String, Object> context) {
        StringBuilder sb = new StringBuilder();
        sb.append("任务: ").append(task).append("\n");
        
        if (!context.isEmpty()) {
            sb.append("\n上下文:\n");
            for (Map.Entry<String, Object> entry : context.entrySet()) {
                sb.append("- ").append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
            }
        }
        
        if (!scratchpad.isEmpty()) {
            sb.append("\n已执行的步骤:").append(scratchpad);
        }
        
        sb.append("\n\n请继续执行任务：");
        return sb.toString();
    }
    
    private ParsedResponse parseResponse(String response) {
        ParsedResponse parsed = new ParsedResponse();
        
        // 提取 Thought
        int thoughtIdx = response.indexOf("Thought:");
        if (thoughtIdx >= 0) {
            int endIdx = response.indexOf("\n", thoughtIdx + 8);
            if (endIdx < 0) endIdx = response.length();
            parsed.thought = response.substring(thoughtIdx + 8, endIdx).trim();
        }
        
        // 检查是否是最终答案
        Matcher finalMatcher = FINAL_ANSWER_PATTERN.matcher(response);
        if (finalMatcher.find()) {
            parsed.isFinalAnswer = true;
            parsed.finalAnswer = finalMatcher.group(1).trim();
            return parsed;
        }
        
        // 提取 Action
        int actionIdx = response.indexOf("Action:");
        if (actionIdx >= 0) {
            int endIdx = response.indexOf("\n", actionIdx + 7);
            if (endIdx < 0) endIdx = response.length();
            parsed.action = response.substring(actionIdx + 7, endIdx).trim();
        }
        
        // 提取 Action Input
        int inputIdx = response.indexOf("Action Input:");
        if (inputIdx >= 0) {
            int endIdx = response.indexOf("\n", inputIdx + 13);
            if (endIdx < 0) endIdx = response.length();
            String inputStr = response.substring(inputIdx + 13, endIdx).trim();
            try {
                if (inputStr.startsWith("{")) {
                    parsed.actionInput = JSON.parseObject(inputStr).getInnerMap();
                }
            } catch (Exception e) {
                parsed.actionInput = Map.of("input", inputStr);
            }
        }
        
        return parsed;
    }
    
    private static class ParsedResponse {
        String thought = "";
        String action = "";
        Map<String, Object> actionInput = new HashMap<>();
        boolean isFinalAnswer = false;
        String finalAnswer = "";
    }
    
    /**
     * 思考步骤
     */
    public static class ThoughtStep {
        private final int step;
        private final String thought;
        private final String action;
        private final Map<String, Object> actionInput;
        private String observation;
        
        public ThoughtStep(int step, String thought, String action, Map<String, Object> actionInput) {
            this.step = step;
            this.thought = thought;
            this.action = action;
            this.actionInput = actionInput;
        }
        
        public void setObservation(String observation) {
            this.observation = observation;
        }
        
        public int getStep() { return step; }
        public String getThought() { return thought; }
        public String getAction() { return action; }
        public Map<String, Object> getActionInput() { return actionInput; }
        public String getObservation() { return observation; }
        
        public Map<String, Object> toMap() {
            Map<String, Object> map = new HashMap<>();
            map.put("step", step);
            map.put("thought", thought);
            map.put("action", action);
            map.put("actionInput", actionInput);
            map.put("observation", observation);
            return map;
        }
    }
    
    /**
     * Agent 执行结果
     */
    public static class AgentResult {
        private final boolean success;
        private final String answer;
        private final String error;
        private final List<ThoughtStep> thoughtChain;
        
        private AgentResult(boolean success, String answer, String error, List<ThoughtStep> thoughtChain) {
            this.success = success;
            this.answer = answer;
            this.error = error;
            this.thoughtChain = thoughtChain;
        }
        
        public static AgentResult success(String answer, List<ThoughtStep> thoughtChain) {
            return new AgentResult(true, answer, null, thoughtChain);
        }
        
        public static AgentResult error(String error, List<ThoughtStep> thoughtChain) {
            return new AgentResult(false, null, error, thoughtChain);
        }
        
        public boolean isSuccess() { return success; }
        public String getAnswer() { return answer; }
        public String getError() { return error; }
        public List<ThoughtStep> getThoughtChain() { return thoughtChain; }
        
        public List<Map<String, Object>> getThoughtChainMaps() {
            List<Map<String, Object>> list = new ArrayList<>();
            for (ThoughtStep step : thoughtChain) {
                list.add(step.toMap());
            }
            return list;
        }
    }
    
    /**
     * 获取工具描述列表（供 AgentRouter 使用）
     */
    public List<Map<String, String>> getToolDescriptions() {
        List<Map<String, String>> descriptions = new ArrayList<>();
        for (AgentTool tool : toolRegistry.getAll()) {
            Map<String, String> desc = new HashMap<>();
            desc.put("name", tool.getName());
            desc.put("description", tool.getDescription());
            descriptions.add(desc);
        }
        return descriptions;
    }
}
