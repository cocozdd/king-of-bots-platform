package com.kob.backend.pojo;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
public class PythonChatResponse {
    private boolean success;
    private String version;
    @JsonProperty("trace_id")
    private String traceId;
    private String response;
    @JsonProperty("session_id")
    private String sessionId;
    private String model;
    private String thought;
    private ErrorInfo error;

    public static PythonChatResponse error(String message) {
        return error("PYTHON_UNAVAILABLE", message);
    }

    public static PythonChatResponse error(String code, String message) {
        PythonChatResponse response = new PythonChatResponse();
        response.setSuccess(false);
        ErrorInfo info = new ErrorInfo();
        info.setCode(code);
        info.setMessage(message);
        response.setError(info);
        return response;
    }

    @Data
    @NoArgsConstructor
    public static class ErrorInfo {
        private String code;
        private String message;
        
        /**
         * Phase 4: 判断是否为安全拦截错误
         * Guardrails 模块产生的错误码
         */
        public boolean isSecurityBlock() {
            if (code == null) return false;
            return code.equals("PROMPT_INJECTION_DETECTED") 
                || code.equals("RATE_LIMIT_EXCEEDED")
                || code.equals("SENSITIVE_CONTENT");
        }
        
        /**
         * 获取用户友好的错误消息
         */
        public String getUserFriendlyMessage() {
            if (code == null) return message;
            switch (code) {
                case "PROMPT_INJECTION_DETECTED":
                    return "检测到不安全的输入，请修改后重试";
                case "RATE_LIMIT_EXCEEDED":
                    return "请求过于频繁，请稍后再试";
                case "SENSITIVE_CONTENT":
                    return "输入包含敏感内容，请修改后重试";
                default:
                    return message;
            }
        }
    }
}
