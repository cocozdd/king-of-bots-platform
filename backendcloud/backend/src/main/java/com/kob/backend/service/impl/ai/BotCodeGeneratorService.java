package com.kob.backend.service.impl.ai;

import com.kob.backend.service.impl.ai.prompt.PromptTemplateRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.util.List;
import java.util.Map;

/**
 * Bot 代码生成服务
 * 
 * 功能：
 * - 根据用户描述生成 Bot 代码框架
 * - 分析现有代码提供优化建议
 * - 解释代码逻辑
 * - 修复代码错误
 * 
 * 技术：
 * - 使用 DeepSeek-Coder 或 DeepSeek-Chat
 * - 支持流式输出
 */
@Service
public class BotCodeGeneratorService {
    
    private static final Logger log = LoggerFactory.getLogger(BotCodeGeneratorService.class);

    @Autowired
    private AiMetricsService metricsService;

    @Autowired(required = false)
    private PromptTemplateRegistry promptRegistry;

    @Autowired
    private PromptSecurityService securityService;
    
    // Bot 代码模板
    private static final String BOT_CODE_TEMPLATE = """
        package com.kob.botrunningsystem.utils;
        
        import java.util.ArrayList;
        import java.util.List;
        
        public class Bot implements BotInterface {
            
            // 方向数组: 0-上, 1-右, 2-下, 3-左
            private static final int[] dx = {-1, 0, 1, 0};
            private static final int[] dy = {0, 1, 0, -1};
            
            @Override
            public Integer nextMove(String input) {
                // 解析输入
                String[] parts = input.split("#");
                String map = parts[0];
                int rows = 13, cols = 14;
                int[][] g = parseMap(map, rows, cols);
                
                int mySx = Integer.parseInt(parts[1]);
                int mySy = Integer.parseInt(parts[2]);
                String mySteps = parts[3].substring(1, parts[3].length() - 1);
                
                int opSx = Integer.parseInt(parts[4]);
                int opSy = Integer.parseInt(parts[5]);
                String opSteps = parts[6].substring(1, parts[6].length() - 1);
                
                // 计算当前位置
                int[] myPos = calculatePosition(mySx, mySy, mySteps);
                int[] opPos = calculatePosition(opSx, opSy, opSteps);
                
                // TODO: 在这里实现你的策略
                %s
                
                return 0; // 默认向上
            }
            
            private int[][] parseMap(String map, int rows, int cols) {
                int[][] g = new int[rows][cols];
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        g[i][j] = map.charAt(i * cols + j) - '0';
                    }
                }
                return g;
            }
            
            private int[] calculatePosition(int sx, int sy, String steps) {
                int x = sx, y = sy;
                for (char c : steps.toCharArray()) {
                    int d = c - '0';
                    x += dx[d];
                    y += dy[d];
                }
                return new int[]{x, y};
            }
        }
        """;
    
    /**
     * 根据描述生成 Bot 代码
     * 
     * @param description 用户的策略描述
     * @param deepseekClient DeepSeek 客户端
     * @return 生成的代码
     */
    public CodeGenResult generateBotCode(String description, DeepseekClient deepseekClient) {
        if (deepseekClient == null || !deepseekClient.enabled()) {
            return CodeGenResult.error("AI 服务不可用");
        }

        // 安全验证
        PromptSecurityService.ValidationResult validation = securityService.validateQuestion(description);
        if (!validation.isSafe()) {
            log.warn("[BotCodeGen] 输入验证失败: {}", validation.getReason());
            return CodeGenResult.error(validation.getReason());
        }

        // 检查是否包含注入攻击模式
        if (securityService.isPromptInjection(description)) {
            log.warn("[BotCodeGen] 检测到提示注入: {}", truncate(description, 50));
            return CodeGenResult.error("输入包含不允许的内容");
        }

        long startTime = System.currentTimeMillis();

        // 使用 PromptTemplate（如果可用）或回退到硬编码
        String systemPrompt;
        if (promptRegistry != null && promptRegistry.get("bot_code_generation") != null) {
            systemPrompt = promptRegistry.applyAsText("bot_code_generation", Map.of(
                    "description", description,
                    "style", "简洁清晰，添加必要注释"
            ));
        } else {
            // Fallback: 硬编码 prompt（向后兼容）
            systemPrompt = """
                你是一个专业的 Java 游戏 Bot 开发专家，专注于 King of Bots (KOB) 贪吃蛇对战平台。
                用户会描述他们想要的 Bot 策略，你需要生成对应的 Java 代码片段。

                【重要】本游戏规则（必须遵守）：
                - 这是一个双人贪吃蛇对战游戏，地图是 13x14 的网格
                - 【没有食物】蛇的长度是自动增长的：前10步每步增长1格，之后每3步增长1格
                - 1表示障碍物（墙），0表示可通行
                - 两条蛇从对角出发，每回合同时移动
                - 获胜条件：让对手撞墙或撞到蛇身（自己或对手的）
                - 不要提及"食物"、"吃食物"等概念，本游戏没有食物

                代码要求：
                - 只输出策略逻辑代码，不要完整类
                - 可用变量：g(地图), myPos(我的位置), opPos(对手位置), dx/dy(方向数组)
                - 返回方向：0-上, 1-右, 2-下, 3-左
                - 代码简洁清晰，添加必要注释
                """;
        }

        String userPrompt = "请为以下策略生成 Bot 代码：\n" + description;
        
        try {
            String response = deepseekClient.chat(systemPrompt, userPrompt, List.of());
            
            // 提取代码块
            String code = extractCode(response);
            
            // 生成完整代码
            String fullCode = String.format(BOT_CODE_TEMPLATE, code);
            
            long latency = System.currentTimeMillis() - startTime;
            
            if (metricsService != null) {
                int inputTokens = metricsService.estimateTokens(systemPrompt + userPrompt);
                int outputTokens = metricsService.estimateTokens(response);
                metricsService.recordCodeGenCall("generate", inputTokens, outputTokens, latency);
            }
            
            log.info("Bot代码生成完成: 描述长度={}, 代码长度={}, 耗时{}ms",
                    description.length(), fullCode.length(), latency);
            
            return CodeGenResult.success(fullCode, code, response);
            
        } catch (Exception e) {
            log.error("Bot代码生成失败: {}", e.getMessage());
            return CodeGenResult.error("代码生成失败: " + e.getMessage());
        }
    }
    
    /**
     * 分析现有代码并提供优化建议
     * 
     * @param code 用户的代码
     * @param deepseekClient DeepSeek 客户端
     * @return 分析结果
     */
    public CodeAnalysisResult analyzeCode(String code, DeepseekClient deepseekClient) {
        if (deepseekClient == null || !deepseekClient.enabled()) {
            return CodeAnalysisResult.error("AI 服务不可用");
        }

        // 安全验证
        PromptSecurityService.ValidationResult validation = securityService.validateCodeSnippet(code);
        if (!validation.isSafe()) {
            log.warn("[BotCodeGen] 代码验证失败: {}", validation.getReason());
            return CodeAnalysisResult.error(validation.getReason());
        }

        long startTime = System.currentTimeMillis();

        // 使用 PromptTemplate（如果可用）
        String systemPrompt;
        if (promptRegistry != null && promptRegistry.get("code_analysis") != null) {
            systemPrompt = promptRegistry.applyAsText("code_analysis", Map.of("code", code));
        } else {
            // Fallback: 硬编码 prompt
            systemPrompt = """
                你是一个专业的代码审查专家，专注于 King of Bots (KOB) 贪吃蛇对战游戏 Bot 开发。

                【重要】本游戏规则：
                - 双人贪吃蛇对战，13x14 地图
                - 【没有食物】蛇自动增长：前10步每步增长，之后每3步增长1格
                - 获胜条件：让对手撞墙或撞蛇身
                - 不要提及"食物"等概念

                请分析用户的 Bot 代码，提供以下方面的建议：

                1. **策略分析**：当前策略的优缺点
                2. **性能优化**：算法效率、内存使用
                3. **Bug 风险**：潜在的问题和边界情况
                4. **改进建议**：具体的优化方向

                输出格式：
                ## 策略分析
                ...

                ## 性能评估
                ...

                ## 潜在问题
                ...

                ## 改进建议
                ...
                """;
        }

        String userPrompt = "请分析以下 Bot 代码：\n```java\n" + code + "\n```";
        
        try {
            String analysis = deepseekClient.chat(systemPrompt, userPrompt, List.of());
            
            long latency = System.currentTimeMillis() - startTime;
            
            if (metricsService != null) {
                int inputTokens = metricsService.estimateTokens(systemPrompt + userPrompt);
                int outputTokens = metricsService.estimateTokens(analysis);
                metricsService.recordCodeGenCall("analyze", inputTokens, outputTokens, latency);
            }
            
            log.info("代码分析完成: 代码长度={}, 分析长度={}, 耗时{}ms",
                    code.length(), analysis.length(), latency);
            
            return CodeAnalysisResult.success(analysis);
            
        } catch (Exception e) {
            log.error("代码分析失败: {}", e.getMessage());
            return CodeAnalysisResult.error("分析失败: " + e.getMessage());
        }
    }
    
    /**
     * 修复代码错误
     * 
     * @param code 有问题的代码
     * @param errorLog 错误日志
     * @param deepseekClient DeepSeek 客户端
     * @return 修复后的代码
     */
    public CodeGenResult fixCode(String code, String errorLog, DeepseekClient deepseekClient) {
        if (deepseekClient == null || !deepseekClient.enabled()) {
            return CodeGenResult.error("AI 服务不可用");
        }

        // 安全验证
        PromptSecurityService.ValidationResult codeValidation = securityService.validateCodeSnippet(code);
        if (!codeValidation.isSafe()) {
            log.warn("[BotCodeGen] 代码验证失败: {}", codeValidation.getReason());
            return CodeGenResult.error(codeValidation.getReason());
        }

        PromptSecurityService.ValidationResult logValidation = securityService.validateErrorLog(errorLog);
        if (!logValidation.isSafe()) {
            log.warn("[BotCodeGen] 错误日志验证失败: {}", logValidation.getReason());
            return CodeGenResult.error(logValidation.getReason());
        }

        long startTime = System.currentTimeMillis();

        // 使用 PromptTemplate（如果可用）
        String systemPrompt;
        if (promptRegistry != null && promptRegistry.get("code_fix") != null) {
            systemPrompt = promptRegistry.applyAsText("code_fix", Map.of(
                    "code", code,
                    "error", errorLog != null ? errorLog : ""
            ));
        } else {
            // Fallback: 硬编码 prompt
            systemPrompt = """
                你是一个专业的 Java 调试专家，专注于 King of Bots (KOB) 贪吃蛇对战游戏 Bot 开发。
                用户会提供有 Bug 的代码和错误日志，请帮助修复。

                【重要】本游戏规则：
                - 双人贪吃蛇对战，13x14 地图，【没有食物】
                - 蛇自动增长：前10步每步增长，之后每3步增长1格
                - 获胜条件：让对手撞墙或撞蛇身

                要求：
                1. 分析错误原因
                2. 提供修复后的完整代码
                3. 解释修改的地方

                输出格式：
                ## 错误分析
                ...

                ## 修复后代码
                ```java
                ...
                ```

                ## 修改说明
                ...
                """;
        }

        String userPrompt = "代码：\n```java\n" + code + "\n```\n\n错误日志：\n" + errorLog;
        
        try {
            String response = deepseekClient.chat(systemPrompt, userPrompt, List.of());
            String fixedCode = extractCode(response);
            
            long latency = System.currentTimeMillis() - startTime;
            
            if (metricsService != null) {
                int inputTokens = metricsService.estimateTokens(systemPrompt + userPrompt);
                int outputTokens = metricsService.estimateTokens(response);
                metricsService.recordCodeGenCall("fix", inputTokens, outputTokens, latency);
            }
            
            log.info("代码修复完成: 耗时{}ms", latency);
            
            return CodeGenResult.success(fixedCode, fixedCode, response);
            
        } catch (Exception e) {
            log.error("代码修复失败: {}", e.getMessage());
            return CodeGenResult.error("修复失败: " + e.getMessage());
        }
    }
    
    /**
     * 解释代码逻辑
     * 
     * @param code 要解释的代码
     * @param deepseekClient DeepSeek 客户端
     * @return 代码解释
     */
    public String explainCode(String code, DeepseekClient deepseekClient) {
        if (deepseekClient == null || !deepseekClient.enabled()) {
            return "AI 服务不可用";
        }
        
        String systemPrompt = """
            你是一个耐心的编程导师。
            请用简单易懂的语言解释用户的代码逻辑。
            - 逐步解释每个部分的作用
            - 说明整体策略思路
            - 如果有复杂算法，用简单例子说明
            """;
        
        String userPrompt = "请解释这段 Bot 代码的逻辑：\n```java\n" + code + "\n```";
        
        try {
            return deepseekClient.chat(systemPrompt, userPrompt, List.of());
        } catch (Exception e) {
            return "解释失败: " + e.getMessage();
        }
    }
    
    private String truncate(String text, int maxLen) {
        if (text == null) return "";
        return text.length() > maxLen ? text.substring(0, maxLen) + "..." : text;
    }

    /**
     * 从 LLM 响应中提取代码块
     */
    private String extractCode(String response) {
        // 尝试提取 ```java ... ``` 代码块
        int start = response.indexOf("```java");
        if (start != -1) {
            start += 7;
            int end = response.indexOf("```", start);
            if (end != -1) {
                return response.substring(start, end).trim();
            }
        }
        
        // 尝试提取 ``` ... ``` 代码块
        start = response.indexOf("```");
        if (start != -1) {
            start += 3;
            // 跳过可能的语言标识
            int lineEnd = response.indexOf("\n", start);
            if (lineEnd != -1 && lineEnd - start < 20) {
                start = lineEnd + 1;
            }
            int end = response.indexOf("```", start);
            if (end != -1) {
                return response.substring(start, end).trim();
            }
        }
        
        // 没有代码块，返回原始响应
        return response;
    }
    
    /**
     * 代码生成结果
     */
    public static class CodeGenResult {
        private boolean success;
        private String fullCode;      // 完整代码
        private String strategyCode;  // 策略部分代码
        private String explanation;   // 解释
        private String error;
        
        public static CodeGenResult success(String fullCode, String strategyCode, String explanation) {
            CodeGenResult result = new CodeGenResult();
            result.success = true;
            result.fullCode = fullCode;
            result.strategyCode = strategyCode;
            result.explanation = explanation;
            return result;
        }
        
        public static CodeGenResult error(String error) {
            CodeGenResult result = new CodeGenResult();
            result.success = false;
            result.error = error;
            return result;
        }
        
        // Getters
        public boolean isSuccess() { return success; }
        public String getFullCode() { return fullCode; }
        public String getStrategyCode() { return strategyCode; }
        public String getExplanation() { return explanation; }
        public String getError() { return error; }
    }
    
    /**
     * 代码分析结果
     */
    public static class CodeAnalysisResult {
        private boolean success;
        private String analysis;
        private String error;
        
        public static CodeAnalysisResult success(String analysis) {
            CodeAnalysisResult result = new CodeAnalysisResult();
            result.success = true;
            result.analysis = analysis;
            return result;
        }
        
        public static CodeAnalysisResult error(String error) {
            CodeAnalysisResult result = new CodeAnalysisResult();
            result.success = false;
            result.error = error;
            return result;
        }
        
        // Getters
        public boolean isSuccess() { return success; }
        public String getAnalysis() { return analysis; }
        public String getError() { return error; }
    }
}
