package com.kob.backend.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * 全局异常处理器
 * 
 * 功能：
 * - 统一异常响应格式
 * - 异常日志记录
 * - 敏感信息脱敏
 * 
 * 面试要点：
 * - @ControllerAdvice 实现 AOP 切面
 * - 异常分类处理（业务异常 vs 系统异常）
 * - 统一响应格式便于前端处理
 */
@ControllerAdvice
public class GlobalExceptionHandler {
    
    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);
    
    /**
     * 业务异常 - 返回 400
     */
    @ExceptionHandler(IllegalArgumentException.class)
    @ResponseBody
    public ResponseEntity<Map<String, Object>> handleIllegalArgument(IllegalArgumentException e) {
        log.warn("业务参数异常: {}", e.getMessage());
        return buildErrorResponse(HttpStatus.BAD_REQUEST, "INVALID_ARGUMENT", e.getMessage());
    }
    
    /**
     * 状态异常 - 返回 409
     */
    @ExceptionHandler(IllegalStateException.class)
    @ResponseBody
    public ResponseEntity<Map<String, Object>> handleIllegalState(IllegalStateException e) {
        log.warn("业务状态异常: {}", e.getMessage());
        return buildErrorResponse(HttpStatus.CONFLICT, "INVALID_STATE", e.getMessage());
    }
    
    /**
     * 参数校验异常 - 返回 400
     */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    @ResponseBody
    public ResponseEntity<Map<String, Object>> handleValidation(MethodArgumentNotValidException e) {
        String message = e.getBindingResult().getFieldErrors().stream()
                .map(error -> error.getField() + ": " + error.getDefaultMessage())
                .findFirst()
                .orElse("参数校验失败");
        log.warn("参数校验异常: {}", message);
        return buildErrorResponse(HttpStatus.BAD_REQUEST, "VALIDATION_ERROR", message);
    }
    
    /**
     * AI 服务异常 - 返回 503
     */
    @ExceptionHandler(AiServiceException.class)
    @ResponseBody
    public ResponseEntity<Map<String, Object>> handleAiService(AiServiceException e) {
        log.error("AI服务异常: {}", e.getMessage());
        return buildErrorResponse(HttpStatus.SERVICE_UNAVAILABLE, "AI_SERVICE_ERROR", 
                "AI服务暂时不可用，请稍后重试");
    }
    
    /**
     * 未知异常 - 返回 500（脱敏处理）
     */
    @ExceptionHandler(Exception.class)
    @ResponseBody
    public ResponseEntity<Map<String, Object>> handleGeneral(Exception e) {
        log.error("系统异常: ", e);
        return buildErrorResponse(HttpStatus.INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", 
                "服务器内部错误，请稍后重试");
    }
    
    private ResponseEntity<Map<String, Object>> buildErrorResponse(
            HttpStatus status, String code, String message) {
        Map<String, Object> body = new HashMap<>();
        body.put("success", false);
        body.put("code", code);
        body.put("message", message);
        body.put("timestamp", LocalDateTime.now().toString());
        return ResponseEntity.status(status).body(body);
    }
    
    /**
     * 自定义 AI 服务异常
     */
    public static class AiServiceException extends RuntimeException {
        public AiServiceException(String message) {
            super(message);
        }
        
        public AiServiceException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
