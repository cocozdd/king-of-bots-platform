package com.kob.backend.service.impl.ai;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.kob.backend.service.impl.ai.message.BaseMessage;
import com.kob.backend.service.impl.ai.message.SystemMessage;
import com.kob.backend.service.impl.ai.message.HumanMessage;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/**
 * DeepSeek 客户端
 *
 * 功能：
 * - 调用 DeepSeek Chat API
 * - Token 使用统计和成本监控
 * - 性能指标记录
 */
public class DeepseekClient {

    private static final String BASE = "https://api.deepseek.com/v1";
    private static final String EMBED_MODEL = "deepseek-embedding";
    private static final String CHAT_MODEL = "deepseek-chat";
    private final String apiKey;
    private final HttpClient httpClient;
    private final AiMetricsService metricsService;

    public DeepseekClient(AiMetricsService metricsService) {
        // 优先使用环境变量，否则使用系统属性 (从 application.properties 读取)
        String envKey = System.getenv("DEEPSEEK_API_KEY");
        String propKey = System.getProperty("deepseek.api.key");
        this.apiKey = (envKey != null && !envKey.isEmpty()) ? envKey : propKey;
        this.metricsService = metricsService;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();
    }

    public boolean enabled() {
        return apiKey != null && !apiKey.isEmpty();
    }

    public double[] embed(String text) {
        JSONObject payload = new JSONObject();
        payload.put("model", EMBED_MODEL);
        payload.put("input", text);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(BASE + "/embeddings"))
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(15))
                .POST(HttpRequest.BodyPublishers.ofString(payload.toJSONString(), StandardCharsets.UTF_8))
                .build();
        try {
            HttpResponse<String> resp = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            System.out.println("[DeepseekClient] embed status=" + resp.statusCode() + " body=" + resp.body());
            if (resp.statusCode() != 200) {
                throw new RuntimeException("embed failed: status=" + resp.statusCode() + ", body=" + resp.body());
            }
            JSONObject body = JSON.parseObject(resp.body());
            JSONArray arr = body.getJSONArray("data").getJSONObject(0).getJSONArray("embedding");
            double[] vec = new double[arr.size()];
            for (int i = 0; i < arr.size(); i++) {
                vec[i] = arr.getDoubleValue(i);
            }
            return vec;
        } catch (Exception e) {
            throw new RuntimeException("embed failed: " + e.getMessage(), e);
        }
    }

    /**
     * 聊天API - 旧版本（兼容保留）
     *
     * @deprecated 请使用 {@link #chat(List)} 方法，使用结构化消息
     */
    @Deprecated
    public String chat(String systemPrompt, String question, List<String> contexts) {
        // 转换为结构化消息
        List<BaseMessage> messages = new ArrayList<>();

        if (systemPrompt != null && !systemPrompt.isEmpty()) {
            // 注入 CoT 指令
            String enhancedPrompt = systemPrompt + "\n\n" +
                    "IMPORTANT: Before answering, you must output your thinking process wrapped in <think> tags. " +
                    "For example: <think>First, I will analyze the user's request...</think>\n" +
                    "Then provide the final answer.";
            messages.add(new SystemMessage(enhancedPrompt));
        }

        // 构建用户消息（包含contexts）
        String ctx = contexts.stream()
                .limit(5)
                .map(s -> s.length() > 600 ? s.substring(0, 600) + "..." : s)
                .collect(Collectors.joining("\n"));
        String userContent = ctx + "\n问题: " + question;
        messages.add(new HumanMessage(userContent));

        // 调用新的chat方法
        return chat(messages);
    }

    /**
     * 聊天API - 新版本（2026年标准）
     *
     * 使用结构化消息体系（List<BaseMessage>）替代字符串拼接
     *
     * 优势：
     * 1. 类型安全：role由类型保证，不会写错
     * 2. 代码清晰：IDE有完整提示
     * 3. 易于扩展：支持多模态、工具调用等
     *
     * 参考：LangChain_深度解析与实战.md 第2.5.4节
     *
     * @param messages 消息列表（SystemMessage, HumanMessage, AIMessage等）
     * @return LLM的回复
     */
    public String chat(List<BaseMessage> messages) {
        long startTime = System.currentTimeMillis();

        // ========== 转换为JSONArray ==========
        JSONArray messagesArray = new JSONArray();
        for (BaseMessage msg : messages) {
            messagesArray.add(msg.toJSON());
        }

        // ========== 估算Token ==========
        int inputTokens = 0;
        if (metricsService != null) {
            String allContent = messages.stream()
                .map(BaseMessage::getContent)
                .filter(c -> c != null)
                .collect(Collectors.joining("\n"));
            inputTokens = metricsService.estimateTokens(allContent);
        }

        // ========== 构建请求 ==========
        JSONObject payload = new JSONObject();
        payload.put("model", CHAT_MODEL);
        payload.put("messages", messagesArray);
        payload.put("temperature", 0.2);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(BASE + "/chat/completions"))
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(120))
                .POST(HttpRequest.BodyPublishers.ofString(payload.toJSONString(), StandardCharsets.UTF_8))
                .build();

        // ========== 发送请求 ==========
        try {
            HttpResponse<String> resp = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            System.out.println("[DeepseekClient] chat status=" + resp.statusCode() + " body=" + resp.body());
            if (resp.statusCode() != 200) {
                throw new RuntimeException("chat failed: status=" + resp.statusCode() + ", body=" + resp.body());
            }
            JSONObject body = JSON.parseObject(resp.body());
            String answer = body.getJSONArray("choices")
                    .getJSONObject(0)
                    .getJSONObject("message")
                    .getString("content");

            // 记录指标
            if (metricsService != null) {
                int outputTokens = metricsService.estimateTokens(answer);
                long latency = System.currentTimeMillis() - startTime;
                metricsService.recordChatCall(CHAT_MODEL, inputTokens, outputTokens, latency);
            }

            // 后处理：修复 Markdown 格式
            return formatMarkdownResponse(answer);
        } catch (Exception e) {
            throw new RuntimeException("chat failed: " + e.getMessage(), e);
        }
    }

    /**
     * 流式聊天 API - 真正的 SSE 流式输出
     */
    public void streamChat(String systemPrompt, String question, List<String> contexts,
                           Consumer<String> onToken, Runnable onComplete, Consumer<Exception> onError) {
        JSONArray messages = new JSONArray();
        if (systemPrompt != null && !systemPrompt.isEmpty()) {
            JSONObject sys = new JSONObject();
            sys.put("role", "system");
            // 不在流式输出中使用 <think> 标签，避免输出到前端
            sys.put("content", systemPrompt);
            messages.add(sys);
        }
        
        JSONObject user = new JSONObject();
        String ctx = contexts != null && !contexts.isEmpty() ? contexts.stream()
                .limit(5)
                .map(s -> s.length() > 600 ? s.substring(0, 600) + "..." : s)
                .collect(Collectors.joining("\n")) : "";
        String userContent = ctx.isEmpty() ? question : ctx + "\n问题: " + question;
        user.put("role", "user");
        user.put("content", userContent);
        messages.add(user);

        JSONObject payload = new JSONObject();
        payload.put("model", CHAT_MODEL);
        payload.put("messages", messages);
        payload.put("temperature", 0.2);
        payload.put("stream", true);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(BASE + "/chat/completions"))
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(60))
                .POST(HttpRequest.BodyPublishers.ofString(payload.toJSONString(), StandardCharsets.UTF_8))
                .build();

        try {
            HttpResponse<java.io.InputStream> resp = httpClient.send(request, 
                    HttpResponse.BodyHandlers.ofInputStream());
            
            if (resp.statusCode() != 200) {
                String errorBody = new String(resp.body().readAllBytes(), StandardCharsets.UTF_8);
                onError.accept(new RuntimeException("status=" + resp.statusCode() + ", body=" + errorBody));
                return;
            }

            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(resp.body(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.isEmpty() || !line.startsWith("data: ")) continue;
                    String data = line.substring(6).trim();
                    if (data.equals("[DONE]")) break;
                    
                    try {
                        JSONObject chunk = JSON.parseObject(data);
                        JSONArray choices = chunk.getJSONArray("choices");
                        if (choices != null && !choices.isEmpty()) {
                            JSONObject delta = choices.getJSONObject(0).getJSONObject("delta");
                            if (delta != null && delta.containsKey("content")) {
                                String content = delta.getString("content");
                                if (content != null && !content.isEmpty()) {
                                    // 过滤 <think> 标签内容（DeepSeek可能自带思考过程）
                                    onToken.accept(filterThinkTags(content));
                                }
                            }
                        }
                    } catch (Exception ignored) {}
                }
            }
            onComplete.run();
        } catch (Exception e) {
            onError.accept(e);
        }
    }
    
    /**
     * 简单过滤：直接返回内容，<think>标签由前端处理
     * 因为流式输出时标签可能跨多个chunk，后端难以完美处理
     */
    private String filterThinkTags(String content) {
        // 直接返回，让前端处理 think 标签
        return content;
    }
    
    /**
     * 后处理：修复 Markdown 格式问题
     * 智能检测和修复缺失的换行符
     */
    private String formatMarkdownResponse(String text) {
        if (text == null || text.isEmpty()) {
            return text;
        }
        
        String result = text;
        
        // ========== 第一阶段：保护已有的代码块 ==========
        java.util.List<String> codeBlocks = new java.util.ArrayList<>();
        java.util.regex.Pattern codePattern = java.util.regex.Pattern.compile("```[\\s\\S]*?```");
        java.util.regex.Matcher codeMatcher = codePattern.matcher(result);
        StringBuffer sb = new StringBuffer();
        int codeIndex = 0;
        while (codeMatcher.find()) {
            String codeBlock = codeMatcher.group();
            // 修复代码块内部格式
            codeBlock = formatCodeBlock(codeBlock);
            codeBlocks.add(codeBlock);
            codeMatcher.appendReplacement(sb, "___CODE_BLOCK_" + codeIndex + "___");
            codeIndex++;
        }
        codeMatcher.appendTail(sb);
        result = sb.toString();
        
        // ========== 第二阶段：修复标题格式 ==========
        // 检测连在一起的标题 (如 "文字##标题" -> "文字\n\n## 标题")
        result = result.replaceAll("([^#\\n])(#{1,6})([^#\\s])", "$1\n\n$2 $3");
        result = result.replaceAll("([^#\\n])(#{1,6}\\s)", "$1\n\n$2");
        // 确保 # 后有空格
        result = result.replaceAll("(#{1,6})([^#\\s\\n])", "$1 $2");
        // 标题后添加换行
        result = result.replaceAll("(#{1,6}\\s[^\\n]+?)([。！？\\.!?])([^\\n])", "$1$2\n\n$3");
        
        // ========== 第三阶段：修复段落格式 ==========
        // 中文句号后添加换行（如果后面紧跟文字）
        result = result.replaceAll("([。！？])([^\\u201c\\u2018\\u300d\\uff09\\s\\n])", "$1\n\n$2");
        // 英文句号+空格后，如果是大写字母开头，添加换行
        result = result.replaceAll("([.!?])\\s+([A-Z])", "$1\n\n$2");
        
        // ========== 第四阶段：修复列表格式 ==========
        // 数字列表 (1. 2. 3.)
        result = result.replaceAll("([^\\n\\d])(\\d+\\.)\\s*([^\\d])", "$1\n$2 $3");
        // 破折号列表
        result = result.replaceAll("([^\\n-])(-\\s+)", "$1\n$2");
        // 星号列表
        result = result.replaceAll("([^\\n*])(\\*\\s+)", "$1\n$2");
        
        // ========== 第五阶段：修复特定Markdown关键词 ==========
        // 常见的段落开头词
        String[] sectionKeywords = {"注意", "总结", "示例", "说明", "步骤", "方法", "原理", "特点", 
                                     "优点", "缺点", "实现", "代码", "算法", "复杂度", "时间复杂度", "空间复杂度"};
        for (String keyword : sectionKeywords) {
            result = result.replaceAll("([^\\n])(" + keyword + "[：:])\\s*", "$1\n\n$2 ");
        }
        
        // ========== 第六阶段：还原代码块 ==========
        for (int i = 0; i < codeBlocks.size(); i++) {
            result = result.replace("___CODE_BLOCK_" + i + "___", "\n\n" + codeBlocks.get(i) + "\n\n");
        }
        
        // ========== 第七阶段：清理多余换行 ==========
        result = result.replaceAll("\\n{3,}", "\n\n");
        result = result.replaceAll("^\\n+", "");
        
        return result.trim();
    }
    
    /**
     * 格式化代码块内容
     * 核心策略：保留AI原始输出的换行，只在明显缺失时才修复
     */
    private String formatCodeBlock(String codeBlock) {
        // 提取语言标识和代码内容
        java.util.regex.Pattern p = java.util.regex.Pattern.compile("```(\\w*)([\\s\\S]*)```", java.util.regex.Pattern.DOTALL);
        java.util.regex.Matcher m = p.matcher(codeBlock);
        
        if (!m.matches()) {
            return codeBlock;
        }
        
        String lang = m.group(1);
        String code = m.group(2);
        
        // ========== 关键判断：如果代码已有换行，直接返回原样 ==========
        // 模仿 BotCodeGeneratorService.extractCode() 的做法
        // AI 模型返回的代码块通常已经格式正确，不需要处理
        int lineCount = code.split("\n").length;
        if (lineCount >= 3) {
            // 代码已有足够换行，保持原样
            return codeBlock;
        }
        
        // ========== 只有当代码完全没有换行时才进行修复 ==========
        // 这种情况很少见，通常是 AI 输出异常
        
        // 在关键字前添加换行
        String[] keywords = {"def ", "class ", "if ", "elif ", "for ", "while ", "return ", "import ", "from "};
        for (String kw : keywords) {
            code = code.replaceAll("([^\\n])(" + kw + ")", "$1\n$2");
        }
        
        // 在赋值语句边界添加换行
        code = code.replaceAll("([)\\]\\}'\">0-9])\\s+([a-zA-Z_][a-zA-Z0-9_]*)\\s*=", "$1\n$2 =");
        
        // 在 ): 后添加换行
        code = code.replaceAll("\\):([^\\n])", "):\n    $1");
        
        // 清理
        if (!code.startsWith("\n")) code = "\n" + code;
        if (!code.endsWith("\n")) code = code + "\n";
        
        return "```" + lang + code + "```";
    }
}
