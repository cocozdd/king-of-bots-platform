package com.kob.backend.controller.ai;

import com.kob.backend.pojo.PythonChatRequest;
import com.kob.backend.pojo.PythonChatResponse;
import com.kob.backend.service.impl.ai.PythonAiServiceClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/ai/python")
public class PythonAiTestController {

    private final PythonAiServiceClient aiServiceClient;

    @Autowired
    public PythonAiTestController(PythonAiServiceClient aiServiceClient) {
        this.aiServiceClient = aiServiceClient;
    }

    @GetMapping("/test")
    public ResponseEntity<Map<String, Object>> test() {
        Map<String, Object> result = new HashMap<>();
        if (!aiServiceClient.enabled()) {
            result.put("success", false);
            result.put("error", "Python AI Service is disabled");
            return ResponseEntity.ok(result);
        }
        result.put("success", true);
        result.put("health", aiServiceClient.health());
        return ResponseEntity.ok(result);
    }

    @PostMapping("/chat-test")
    public ResponseEntity<PythonChatResponse> chatTest(@RequestBody(required = false) PythonChatRequest request) {
        if (request == null) {
            request = new PythonChatRequest();
        }
        if ((request.getMessage() == null || request.getMessage().isBlank())
            && (request.getMessages() == null || request.getMessages().isEmpty())
            && (request.getHistory() == null || request.getHistory().isEmpty())) {
            request.setMessage("hello");
        }
        PythonChatResponse response = aiServiceClient.chat(request);
        if (response == null) {
            return ResponseEntity.ok(PythonChatResponse.error("Empty response from Python AI Service"));
        }
        if (response.getError() == null) {
            response.setSuccess(true);
        }
        return ResponseEntity.ok(response);
    }
}
