package com.kob.backend.controller.ai;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.kob.backend.pojo.PythonChatRequest;
import com.kob.backend.pojo.PythonChatResponse;
import com.kob.backend.service.impl.ai.PythonAiServiceClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

@RestController
@RequestMapping("/ai/python")
public class PythonAiController {

    private final PythonAiServiceClient aiServiceClient;
    private final ExecutorService executor = Executors.newFixedThreadPool(4);
    private final ObjectMapper mapper = new ObjectMapper();

    @Autowired
    public PythonAiController(PythonAiServiceClient aiServiceClient) {
        this.aiServiceClient = aiServiceClient;
    }

    @PostMapping("/chat")
    public ResponseEntity<PythonChatResponse> chat(@RequestBody PythonChatRequest request) {
        PythonChatResponse response = aiServiceClient.chat(request);
        if (response == null) {
            return ResponseEntity.ok(PythonChatResponse.error("Empty response from Python AI Service"));
        }
        if (response.getError() == null) {
            response.setSuccess(true);
        }
        return ResponseEntity.ok(response);
    }

    @GetMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter stream(@RequestParam(required = false) String message,
                             @RequestParam(required = false) String question,
                             @RequestParam(required = false) String systemPrompt) {
        SseEmitter emitter = new SseEmitter(60000L);
        AtomicBoolean finished = new AtomicBoolean(false);

        executor.submit(() -> {
            if (!aiServiceClient.enabled()) {
                sendError(emitter, finished, "PYTHON_DISABLED", "Python AI Service is disabled", null);
                return;
            }

            String content = (message != null && !message.isBlank()) ? message : question;
            if (content == null || content.isBlank()) {
                sendError(emitter, finished, "INVALID_REQUEST", "message or question is required", null);
                return;
            }

            PythonChatRequest request = new PythonChatRequest();
            request.setMessage(content);
            request.setSystemPrompt(systemPrompt);
            String traceId = UUID.randomUUID().toString();
            request.setTraceId(traceId);
            request.setVersion("v1");

            try {
                emitter.send(SseEmitter.event()
                    .name("start")
                    .data(mapper.writeValueAsString(Map.of("trace_id", traceId, "version", "v1"))));
            } catch (Exception ignored) {}

            aiServiceClient.streamChat(
                request,
                token -> {
                    if (finished.get()) {
                        return;
                    }
                    try {
                        String encoded = token.replace("\n", "___NEWLINE___");
                        emitter.send(SseEmitter.event()
                            .name("chunk")
                            .data(encoded));
                    } catch (Exception ignored) {}
                },
                error -> sendError(emitter, finished, error.getCode(), error.getMessage(), traceId),
                () -> completeEmitter(emitter, finished)
            );
        });

        return emitter;
    }

    private void completeEmitter(SseEmitter emitter, AtomicBoolean finished) {
        if (!finished.compareAndSet(false, true)) {
            return;
        }
        try {
            emitter.send(SseEmitter.event()
                .name("done")
                .data("{\"status\":\"completed\"}"));
        } catch (Exception ignored) {}
        emitter.complete();
    }

    private void sendError(SseEmitter emitter,
                           AtomicBoolean finished,
                           String code,
                           String message,
                           String traceId) {
        if (!finished.compareAndSet(false, true)) {
            return;
        }
        try {
            Map<String, String> data = new java.util.HashMap<>();
            data.put("code", code);
            data.put("message", message);
            if (traceId != null && !traceId.isBlank()) {
                data.put("trace_id", traceId);
            }
            String payload = mapper.writeValueAsString(data);
            emitter.send(SseEmitter.event()
                .name("error")
                .data(payload));
        } catch (Exception ignored) {}
        emitter.complete();
    }
}
