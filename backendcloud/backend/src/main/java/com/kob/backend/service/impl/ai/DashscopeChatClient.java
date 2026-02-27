package com.kob.backend.service.impl.ai;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.stream.Collectors;

/**
 * DashScope Chat 客户端（使用 Qwen 模型）
 * 
 * 通过 curl 调用 DashScope API，避免 Java HttpClient TLS 问题
 */
public class DashscopeChatClient {

    private static final Logger log = LoggerFactory.getLogger(DashscopeChatClient.class);
    private static final String CHAT_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation";
    private static final String MODEL = "qwen-turbo";
    
    private final String apiKey;
    private final AiMetricsService metricsService;

    public DashscopeChatClient(String apiKey, AiMetricsService metricsService) {
        this.apiKey = apiKey;
        this.metricsService = metricsService;
    }

    public boolean enabled() {
        return apiKey != null && !apiKey.isEmpty();
    }

    public String chat(String systemPrompt, String question, List<String> contexts) {
        long startTime = System.currentTimeMillis();
        
        JSONArray messages = new JSONArray();
        
        // System message
        if (systemPrompt != null && !systemPrompt.isEmpty()) {
            JSONObject sys = new JSONObject();
            sys.put("role", "system");
            sys.put("content", systemPrompt);
            messages.add(sys);
        }
        
        // User message with context
        JSONObject user = new JSONObject();
        String ctx = contexts != null ? contexts.stream()
                .limit(5)
                .map(s -> s.length() > 600 ? s.substring(0, 600) + "..." : s)
                .collect(Collectors.joining("\n")) : "";
        String userContent = ctx.isEmpty() ? question : ctx + "\n问题: " + question;
        user.put("role", "user");
        user.put("content", userContent);
        messages.add(user);

        // Build request payload
        JSONObject input = new JSONObject();
        input.put("messages", messages);
        
        JSONObject parameters = new JSONObject();
        parameters.put("result_format", "message");
        parameters.put("temperature", 0.7);
        
        JSONObject payload = new JSONObject();
        payload.put("model", MODEL);
        payload.put("input", input);
        payload.put("parameters", parameters);

        try {
            String response = sendWithCurl(payload.toJSONString());
            
            JSONObject body = JSON.parseObject(response);
            
            // Check for errors
            if (body.containsKey("code")) {
                String errorCode = body.getString("code");
                String errorMsg = body.getString("message");
                throw new RuntimeException("DashScope error: " + errorCode + " - " + errorMsg);
            }
            
            // Extract answer: output.choices[0].message.content
            String answer = body.getJSONObject("output")
                    .getJSONArray("choices")
                    .getJSONObject(0)
                    .getJSONObject("message")
                    .getString("content");
            
            // Record metrics
            if (metricsService != null) {
                int inputTokens = metricsService.estimateTokens(userContent);
                int outputTokens = metricsService.estimateTokens(answer);
                long latency = System.currentTimeMillis() - startTime;
                metricsService.recordChatCall(MODEL, inputTokens, outputTokens, latency);
            }
            
            log.info("[DashscopeChatClient] Chat success, answer length: {}", answer.length());
            return answer;
            
        } catch (Exception e) {
            log.error("[DashscopeChatClient] Chat failed: {}", e.getMessage());
            throw new RuntimeException("DashScope chat failed: " + e.getMessage(), e);
        }
    }

    /**
     * 使用 curl 发送请求（避免 Java HttpClient TLS 问题）
     */
    private String sendWithCurl(String payload) throws Exception {
        java.io.File tempFile = java.io.File.createTempFile("dashscope_chat_", ".json");
        try {
            java.nio.file.Files.writeString(tempFile.toPath(), payload);
            
            ProcessBuilder pb = new ProcessBuilder(
                "curl", "-s", "-X", "POST", CHAT_URL,
                "-H", "Authorization: Bearer " + apiKey,
                "-H", "Content-Type: application/json",
                "-d", "@" + tempFile.getAbsolutePath()
            );
            pb.redirectErrorStream(true);
            
            Process process = pb.start();
            String response;
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                response = reader.lines().collect(Collectors.joining("\n"));
            }
            
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                throw new RuntimeException("curl failed with exit code " + exitCode + ": " + response);
            }
            
            return response;
        } finally {
            tempFile.delete();
        }
    }
}
