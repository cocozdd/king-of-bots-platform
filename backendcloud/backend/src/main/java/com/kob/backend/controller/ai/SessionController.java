package com.kob.backend.controller.ai;

import com.kob.backend.service.impl.ai.memory.ConversationMemoryService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * AI 会话管理控制器
 *
 * 提供会话的 CRUD 操作：
 * - 创建会话
 * - 获取会话列表
 * - 获取会话历史
 * - 删除会话
 */
@RestController
@RequestMapping("/ai/session")
public class SessionController {

    private static final Logger log = LoggerFactory.getLogger(SessionController.class);

    @Autowired
    private ConversationMemoryService memoryService;

    /**
     * 创建新会话
     *
     * POST /ai/session/create
     * Body: { "userId": 123 }
     */
    @PostMapping("/create")
    public ResponseEntity<Map<String, Object>> createSession(@RequestBody Map<String, Object> request) {
        Map<String, Object> result = new HashMap<>();

        try {
            Long userId = extractUserId(request);
            if (userId == null) {
                // 没有用户ID时使用默认值（匿名用户）
                userId = 0L;
            }

            String sessionId = memoryService.createSession(userId);

            result.put("success", true);
            result.put("sessionId", sessionId);

            log.info("创建会话: sessionId={}, userId={}", sessionId, userId);
        } catch (Exception e) {
            log.error("创建会话失败: {}", e.getMessage());
            result.put("success", false);
            result.put("error", "创建会话失败: " + e.getMessage());
        }

        return ResponseEntity.ok(result);
    }

    /**
     * 获取用户的会话列表
     *
     * GET /ai/session/list?userId=123&limit=20
     */
    @GetMapping("/list")
    public ResponseEntity<Map<String, Object>> listSessions(
            @RequestParam(required = false, defaultValue = "0") Long userId,
            @RequestParam(required = false, defaultValue = "20") int limit) {
        Map<String, Object> result = new HashMap<>();

        try {
            List<Map<String, Object>> sessions = memoryService.getUserSessions(userId, limit);

            result.put("success", true);
            result.put("sessions", sessions);
            result.put("count", sessions.size());

        } catch (Exception e) {
            log.error("获取会话列表失败: {}", e.getMessage());
            result.put("success", false);
            result.put("error", "获取会话列表失败: " + e.getMessage());
        }

        return ResponseEntity.ok(result);
    }

    /**
     * 获取会话历史消息
     *
     * GET /ai/session/history?sessionId=xxx
     */
    @GetMapping("/history")
    public ResponseEntity<Map<String, Object>> getHistory(
            @RequestParam String sessionId) {
        Map<String, Object> result = new HashMap<>();

        try {
            List<Map<String, Object>> messages = memoryService.getSessionHistory(sessionId);
            Map<String, Object> stats = memoryService.getSessionStats(sessionId);

            result.put("success", true);
            result.put("messages", messages);
            result.put("stats", stats);

        } catch (Exception e) {
            log.error("获取会话历史失败: {}", e.getMessage());
            result.put("success", false);
            result.put("error", "获取会话历史失败: " + e.getMessage());
        }

        return ResponseEntity.ok(result);
    }

    /**
     * 删除会话
     *
     * DELETE /ai/session/delete?sessionId=xxx
     */
    @DeleteMapping("/delete")
    public ResponseEntity<Map<String, Object>> deleteSession(
            @RequestParam String sessionId) {
        Map<String, Object> result = new HashMap<>();

        try {
            memoryService.clearSession(sessionId);

            result.put("success", true);
            result.put("message", "会话已删除");

            log.info("删除会话: sessionId={}", sessionId);
        } catch (Exception e) {
            log.error("删除会话失败: {}", e.getMessage());
            result.put("success", false);
            result.put("error", "删除会话失败: " + e.getMessage());
        }

        return ResponseEntity.ok(result);
    }

    /**
     * 更新会话标题
     *
     * PUT /ai/session/title
     * Body: { "sessionId": "xxx", "title": "新标题" }
     */
    @PutMapping("/title")
    public ResponseEntity<Map<String, Object>> updateTitle(@RequestBody Map<String, String> request) {
        Map<String, Object> result = new HashMap<>();

        try {
            String sessionId = request.get("sessionId");
            String title = request.get("title");

            if (sessionId == null || title == null) {
                result.put("success", false);
                result.put("error", "sessionId 和 title 不能为空");
                return ResponseEntity.ok(result);
            }

            memoryService.updateSessionTitle(sessionId, title);

            result.put("success", true);
            result.put("message", "标题已更新");

        } catch (Exception e) {
            log.error("更新标题失败: {}", e.getMessage());
            result.put("success", false);
            result.put("error", "更新标题失败: " + e.getMessage());
        }

        return ResponseEntity.ok(result);
    }

    /**
     * 获取会话统计信息
     *
     * GET /ai/session/stats?sessionId=xxx
     */
    @GetMapping("/stats")
    public ResponseEntity<Map<String, Object>> getStats(
            @RequestParam String sessionId) {
        Map<String, Object> result = new HashMap<>();

        try {
            Map<String, Object> stats = memoryService.getSessionStats(sessionId);
            result.put("success", true);
            result.putAll(stats);

        } catch (Exception e) {
            log.error("获取会话统计失败: {}", e.getMessage());
            result.put("success", false);
            result.put("error", "获取会话统计失败: " + e.getMessage());
        }

        return ResponseEntity.ok(result);
    }

    /**
     * 获取记忆服务状态
     *
     * GET /ai/session/service/stats
     */
    @GetMapping("/service/stats")
    public ResponseEntity<Map<String, Object>> getServiceStats() {
        Map<String, Object> result = new HashMap<>();

        try {
            Map<String, Object> stats = memoryService.getStats();
            result.put("success", true);
            result.putAll(stats);

        } catch (Exception e) {
            log.error("获取服务统计失败: {}", e.getMessage());
            result.put("success", false);
            result.put("error", "获取服务统计失败: " + e.getMessage());
        }

        return ResponseEntity.ok(result);
    }

    private Long extractUserId(Map<String, Object> request) {
        Object userIdObj = request.get("userId");
        if (userIdObj == null) {
            return null;
        }
        if (userIdObj instanceof Number) {
            return ((Number) userIdObj).longValue();
        }
        try {
            return Long.parseLong(userIdObj.toString());
        } catch (NumberFormatException e) {
            return null;
        }
    }
}
