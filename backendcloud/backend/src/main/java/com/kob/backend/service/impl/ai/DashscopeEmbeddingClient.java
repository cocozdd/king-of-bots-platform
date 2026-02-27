package com.kob.backend.service.impl.ai;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.stream.Collectors;
import javax.net.ssl.SSLHandshakeException;
import javax.net.ssl.SSLParameters;

/**
 * DashScope Embedding 客户端
 * 
 * 功能：
 * - 调用 DashScope text-embedding-v2 API
 * - Token 使用统计和成本监控
 * - 性能指标记录
 * - Embedding 缓存（避免重复调用 API）
 */
public class DashscopeEmbeddingClient {
    private static final Logger log = LoggerFactory.getLogger(DashscopeEmbeddingClient.class);
    private static final String EMBEDDING_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding";
    private static final String MODEL = "text-embedding-v2";
    private final String apiKey;
    private final HttpClient httpClient;
    private final HttpClient compatHttpClient;
    private final AiMetricsService metricsService;
    private final EmbeddingCacheService cacheService;

    public DashscopeEmbeddingClient(String apiKey, AiMetricsService metricsService, EmbeddingCacheService cacheService) {
        this.apiKey = apiKey;
        this.metricsService = metricsService;
        this.cacheService = cacheService;
        this.httpClient = buildDefaultClient();
        this.compatHttpClient = buildCompatClient();
    }

    private static HttpClient buildDefaultClient() {
        return HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();
    }

    private static HttpClient buildCompatClient() {
        SSLParameters sslParameters = new SSLParameters();
        sslParameters.setProtocols(new String[]{"TLSv1.2"});
        return HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .sslParameters(sslParameters)
                .version(HttpClient.Version.HTTP_1_1)
                .build();
    }

    private HttpResponse<String> sendWithFallback(HttpRequest request) throws Exception {
        try {
            return httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        } catch (SSLHandshakeException e) {
            log.warn("DashScope TLS handshake failed, retrying with TLSv1.2/HTTP1.1", e);
            try {
                return compatHttpClient.send(request, HttpResponse.BodyHandlers.ofString());
            } catch (Exception e2) {
                log.warn("TLSv1.2 fallback also failed, will use HttpURLConnection", e2);
                throw e2; // Let caller handle with HttpURLConnection fallback
            }
        }
    }
    
    /**
     * 使用 curl 命令作为最终降级方案（已验证可用）
     */
    private String sendWithCurl(String payload) throws Exception {
        // 将 payload 写入临时文件避免命令行转义问题
        java.io.File tempFile = java.io.File.createTempFile("dashscope_", ".json");
        try {
            java.nio.file.Files.writeString(tempFile.toPath(), payload);
            
            ProcessBuilder pb = new ProcessBuilder(
                "curl", "-s", "-X", "POST", EMBEDDING_URL,
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

    public boolean enabled() {
        return apiKey != null && !apiKey.isEmpty();
    }

    public double[] embed(String text) {
        // 1. 先检查缓存
        if (cacheService != null) {
            double[] cached = cacheService.get(text);
            if (cached != null) {
                log.info("Embedding 缓存命中，跳过 API 调用");
                return cached;
            }
        }
        
        // 2. 缓存未命中，调用 API
        long startTime = System.currentTimeMillis();
        
        // 估算输入 Token 数
        int inputTokens = metricsService != null ? metricsService.estimateTokens(text) : 0;
        
        JSONObject payload = new JSONObject();
        payload.put("model", MODEL);
        
        // input 格式: {"texts": [text]}
        JSONArray texts = new JSONArray();
        texts.add(text);
        JSONObject input = new JSONObject();
        input.put("texts", texts);
        payload.put("input", input);

        String payloadStr = payload.toJSONString();
        String responseBody = null;
        
        // 尝试使用 HttpClient（优先）
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(EMBEDDING_URL))
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(30))
                .POST(HttpRequest.BodyPublishers.ofString(payloadStr, StandardCharsets.UTF_8))
                .build();
        try {
            HttpResponse<String> resp = sendWithFallback(request);
            log.info("[DashscopeEmbeddingClient] HttpClient success, status={}", resp.statusCode());
            if (resp.statusCode() != 200) {
                throw new RuntimeException("status=" + resp.statusCode() + ", body=" + resp.body());
            }
            responseBody = resp.body();
        } catch (Exception e) {
            // HttpClient 失败，使用 curl 降级（最可靠的方案）
            log.warn("[DashscopeEmbeddingClient] HttpClient failed, falling back to curl: {}", e.getMessage());
            try {
                responseBody = sendWithCurl(payloadStr);
                log.info("[DashscopeEmbeddingClient] curl fallback success");
            } catch (Exception e2) {
                throw new RuntimeException("dashscope embed failed: " + e2.getMessage(), e2);
            }
        }
        
        try {
            JSONObject body = JSON.parseObject(responseBody);
            // 响应格式: {"output": {"embeddings": [{"embedding": [...]}]}}
            JSONArray arr = body.getJSONObject("output").getJSONArray("embeddings").getJSONObject(0).getJSONArray("embedding");
            double[] vec = new double[arr.size()];
            for (int i = 0; i < arr.size(); i++) {
                vec[i] = arr.getDoubleValue(i);
            }
            
            // 记录指标
            if (metricsService != null) {
                long latency = System.currentTimeMillis() - startTime;
                metricsService.recordEmbeddingCall(MODEL, inputTokens, arr.size(), latency);
            }
            
            // 3. 写入缓存
            if (cacheService != null) {
                cacheService.put(text, vec);
                log.info("Embedding 已缓存，下次相同查询将从缓存获取");
            }
            
            return vec;
        } catch (Exception e) {
            throw new RuntimeException("dashscope embed parse failed: " + e.getMessage(), e);
        }
    }
}
