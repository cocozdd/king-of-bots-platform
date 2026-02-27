package com.kob.backend.pojo;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.ArrayList;
import java.util.List;

@Data
@NoArgsConstructor
public class PythonChatRequest {
    private String version = "v1";
    @JsonProperty("trace_id")
    private String traceId;
    private String message;
    private String systemPrompt;
    @JsonProperty("session_id")
    private String sessionId;
    @JsonProperty("userId")
    private Long userId;
    private String question;
    private Boolean stream = false;
    private List<ChatMessage> messages = new ArrayList<>();
    private List<ChatMessage> history = new ArrayList<>();

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ChatMessage {
        private String role;
        private String content;
    }
}
