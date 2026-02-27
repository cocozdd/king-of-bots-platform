package com.kob.backend.service.impl.ai;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.kob.backend.config.AiServiceProperties;
import com.kob.backend.config.GlobalExceptionHandler.AiServiceException;
import com.kob.backend.pojo.PythonChatRequest;
import com.kob.backend.pojo.PythonChatResponse;
import com.kob.backend.service.impl.ai.message.BaseMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.HttpStatusCodeException;
import org.springframework.web.client.RestTemplate;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;

@Service
public class PythonAiServiceClient {

    private static final Logger log = LoggerFactory.getLogger(PythonAiServiceClient.class);
    private final RestTemplate restTemplate;
    private final AiServiceProperties properties;
    private final String legacyBaseUrl;
    private final ObjectMapper mapper = new ObjectMapper();

    public PythonAiServiceClient(
            RestTemplate restTemplate,
            AiServiceProperties properties,
            @Value("${ai.python.url:}") String legacyBaseUrl) {
        this.restTemplate = restTemplate;
        this.properties = properties;
        this.legacyBaseUrl = legacyBaseUrl;
    }

    public boolean enabled() {
        return properties.isEnabled();
    }

    public Map<String, Object> health() {
        if (!enabled()) {
            throw new AiServiceException("Python AI Service is disabled");
        }
        try {
            return restTemplate.getForObject(baseUrl() + "/health", Map.class);
        } catch (RestClientException e) {
            log.error("Python AI Service health check failed: {}", e.getMessage(), e);
            throw new AiServiceException("Python AI Service health check failed", e);
        }
    }

    public PythonChatResponse chat(PythonChatRequest request) {
        if (!enabled()) {
            return PythonChatResponse.error("Python AI Service is disabled");
        }
        if (request == null) {
            throw new IllegalArgumentException("request is required");
        }
        String traceId = prepareRequest(request);
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.set("X-Trace-Id", traceId);
            HttpEntity<PythonChatRequest> entity = new HttpEntity<>(request, headers);
            ResponseEntity<PythonChatResponse> response = restTemplate.exchange(
                baseUrl() + "/api/chat",
                HttpMethod.POST,
                entity,
                PythonChatResponse.class
            );
            return response.getBody();
        } catch (HttpStatusCodeException e) {
            PythonChatResponse errorResponse = parseErrorResponse(e.getResponseBodyAsString(), traceId);
            // Phase 4: 区分安全拦截和普通错误
            if (errorResponse.getError() != null && errorResponse.getError().isSecurityBlock()) {
                log.info("[Guardrails] Security block: code={}, traceId={}", 
                    errorResponse.getError().getCode(), traceId);
            } else {
                log.warn("Python AI Service returned error: status={}, body={}", 
                    e.getStatusCode(), e.getResponseBodyAsString());
            }
            return errorResponse;
        } catch (RestClientException e) {
            log.error("Python AI Service call failed: {}", e.getMessage(), e);
            return PythonChatResponse.error("PYTHON_UNAVAILABLE", "Python AI Service call failed");
        }
    }

    public PythonChatResponse chat(List<BaseMessage> messages) {
        PythonChatRequest request = buildRequest(messages, null);
        return chat(request);
    }

    public PythonChatResponse chat(List<BaseMessage> messages, String systemPrompt) {
        PythonChatRequest request = buildRequest(messages, systemPrompt);
        return chat(request);
    }

    public List<String> chatStream(List<BaseMessage> messages) {
        if (!enabled()) {
            return List.of();
        }
        PythonChatRequest request = buildRequest(messages, null);
        prepareRequest(request);
        List<String> deltas = new ArrayList<>();
        try {
            restTemplate.execute(
                baseUrl() + "/api/chat/stream",
                HttpMethod.POST,
                httpRequest -> {
                    HttpHeaders headers = httpRequest.getHeaders();
                    headers.setContentType(MediaType.APPLICATION_JSON);
                    if (request.getTraceId() != null && !request.getTraceId().isBlank()) {
                        headers.set("X-Trace-Id", request.getTraceId());
                    }
                    byte[] body = mapper.writeValueAsBytes(request);
                    httpRequest.getBody().write(body);
                },
                response -> {
                    try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(response.getBody(), StandardCharsets.UTF_8))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            if (!line.startsWith("data:")) {
                                continue;
                            }
                            String payload = line.substring(5).trim();
                            if (payload.isEmpty()) {
                                continue;
                            }
                            JsonNode node = mapper.readTree(payload);
                            if (node.has("delta")) {
                                deltas.add(node.get("delta").asText());
                            } else if (node.has("event") && "error".equals(node.get("event").asText())) {
                                throw new AiServiceException("Python AI Service stream error: " + node);
                            }
                        }
                    }
                    return null;
                }
            );
        } catch (Exception e) {
            log.error("Python AI Service stream failed: {}", e.getMessage(), e);
            throw new AiServiceException("Python AI Service stream failed", e);
        }
        return deltas;
    }

    public void streamChat(
        PythonChatRequest request,
        Consumer<String> onDelta,
        Consumer<PythonChatResponse.ErrorInfo> onError,
        Runnable onComplete
    ) {
        if (!enabled()) {
            PythonChatResponse.ErrorInfo error = new PythonChatResponse.ErrorInfo();
            error.setCode("PYTHON_DISABLED");
            error.setMessage("Python AI Service is disabled");
            onError.accept(error);
            return;
        }
        if (request == null) {
            throw new IllegalArgumentException("request is required");
        }

        prepareRequest(request);

        try {
            restTemplate.execute(
                baseUrl() + "/api/chat/stream",
                HttpMethod.POST,
                httpRequest -> {
                    HttpHeaders headers = httpRequest.getHeaders();
                    headers.setContentType(MediaType.APPLICATION_JSON);
                    headers.setAccept(List.of(MediaType.TEXT_EVENT_STREAM));
                    if (request.getTraceId() != null && !request.getTraceId().isBlank()) {
                        headers.set("X-Trace-Id", request.getTraceId());
                    }
                    byte[] body = mapper.writeValueAsBytes(request);
                    httpRequest.getBody().write(body);
                },
                response -> {
                    boolean completed = false;
                    try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(response.getBody(), StandardCharsets.UTF_8))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            if (!line.startsWith("data:")) {
                                continue;
                            }
                            String payload = line.substring(5).trim();
                            if (payload.isEmpty()) {
                                continue;
                            }
                            JsonNode node = mapper.readTree(payload);
                            if (node.has("delta")) {
                                onDelta.accept(node.get("delta").asText());
                            } else if (node.has("event")) {
                                String event = node.get("event").asText();
                                if ("done".equals(event)) {
                                    completed = true;
                                    onComplete.run();
                                } else if ("error".equals(event)) {
                                    PythonChatResponse.ErrorInfo error = parseErrorNode(node);
                                    onError.accept(error);
                                    completed = true;
                                    break;
                                }
                            } else if (node.has("error")) {
                                onError.accept(parseErrorNode(node));
                                completed = true;
                                break;
                            }
                        }
                    } catch (Exception e) {
                        PythonChatResponse.ErrorInfo error = new PythonChatResponse.ErrorInfo();
                        error.setCode("PYTHON_STREAM_ERROR");
                        error.setMessage("Python stream parse failed: " + e.getMessage());
                        onError.accept(error);
                        completed = true;
                    } finally {
                        if (!completed) {
                            onComplete.run();
                        }
                    }
                    return null;
                }
            );
        } catch (Exception e) {
            PythonChatResponse.ErrorInfo error = new PythonChatResponse.ErrorInfo();
            error.setCode("PYTHON_STREAM_ERROR");
            error.setMessage("Python AI Service stream failed: " + e.getMessage());
            onError.accept(error);
        }
    }


    private PythonChatRequest buildRequest(List<BaseMessage> messages, String systemPrompt) {
        PythonChatRequest request = new PythonChatRequest();
        request.setSystemPrompt(systemPrompt);
        if (messages == null || messages.isEmpty()) {
            return request;
        }
        List<PythonChatRequest.ChatMessage> converted = new ArrayList<>();
        for (BaseMessage message : messages) {
            if (message == null) {
                continue;
            }
            converted.add(new PythonChatRequest.ChatMessage(message.getRole(), message.getContent()));
        }
        request.setMessages(converted);
        return request;
    }

    private String prepareRequest(PythonChatRequest request) {
        if (request.getVersion() == null || request.getVersion().isBlank()) {
            request.setVersion("v1");
        }
        String traceId = request.getTraceId();
        if (traceId == null || traceId.isBlank()) {
            traceId = UUID.randomUUID().toString();
            request.setTraceId(traceId);
        }
        return traceId;
    }

    private PythonChatResponse parseErrorResponse(String body, String traceId) {
        if (body == null || body.isBlank()) {
            return PythonChatResponse.error("PYTHON_ERROR", "Python AI Service returned empty error");
        }
        try {
            PythonChatResponse parsed = mapper.readValue(body, PythonChatResponse.class);
            if (parsed.getTraceId() == null || parsed.getTraceId().isBlank()) {
                parsed.setTraceId(traceId);
            }
            return parsed;
        } catch (Exception e) {
            PythonChatResponse fallback = PythonChatResponse.error("PYTHON_ERROR", "Python AI Service call failed");
            fallback.setTraceId(traceId);
            return fallback;
        }
    }

    private PythonChatResponse.ErrorInfo parseErrorNode(JsonNode node) {
        PythonChatResponse.ErrorInfo error = new PythonChatResponse.ErrorInfo();
        JsonNode errorNode = node.get("error");
        if (errorNode != null && errorNode.isObject()) {
            if (errorNode.has("code")) {
                error.setCode(errorNode.get("code").asText());
            }
            if (errorNode.has("message")) {
                error.setMessage(errorNode.get("message").asText());
            }
        } else if (node.has("message")) {
            error.setMessage(node.get("message").asText());
        }
        if (error.getCode() == null || error.getCode().isBlank()) {
            error.setCode("PYTHON_STREAM_ERROR");
        }
        if (error.getMessage() == null || error.getMessage().isBlank()) {
            error.setMessage("Python AI Service stream error");
        }
        return error;
    }

    private String baseUrl() {
        String base = properties.getBaseUrl();
        if ((base == null || base.isBlank()) && legacyBaseUrl != null && !legacyBaseUrl.isBlank()) {
            base = legacyBaseUrl;
        }
        if (base == null || base.isBlank()) {
            base = "http://127.0.0.1:3003";
        }
        return base.endsWith("/") ? base.substring(0, base.length() - 1) : base;
    }
}
