package com.kob.backend.service.impl.ai.memory;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.kob.backend.mapper.AiMessageMapper;
import com.kob.backend.mapper.AiSessionMapper;
import com.kob.backend.mapper.AiUserMemoryMapper;
import com.kob.backend.pojo.AiMessage;
import com.kob.backend.pojo.AiSession;
import com.kob.backend.pojo.AiUserMemory;
import com.kob.backend.service.impl.ai.DeepseekClient;
import com.kob.backend.service.impl.ai.langchain4j.Langchain4jMemoryAdapter;
import com.kob.backend.service.impl.ai.prompt.PromptTemplateRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 多轮对话记忆管理服务（带持久化）
 *
 * 功能：
 * - 短期记忆：当前会话上下文（内存 + LangChain4j ChatMemory）
 * - 长期记忆：用户历史偏好（PostgreSQL 持久化）
 * - 会话持久化：对话历史存储到数据库
 *
 * 面试要点：
 * - 记忆类型：Episodic（情景）、Semantic（语义）、Procedural（程序）
 * - 记忆压缩：超长对话时自动摘要
 * - Token 管理：控制上下文长度
 * - LangChain4j 集成：使用框架的 ChatMemory 抽象
 */
@Service
public class ConversationMemoryService {

    private static final Logger log = LoggerFactory.getLogger(ConversationMemoryService.class);

    // 配置
    private static final int MAX_SHORT_TERM_MESSAGES = 10;
    private static final int MAX_CONTEXT_TOKENS = 4000;
    private static final int SUMMARY_THRESHOLD = 8;

    // 内存缓存（用于快速访问，数据库为主存储）
    private final Map<String, SessionCache> sessionCaches = new ConcurrentHashMap<>();

    @Autowired(required = false)
    private AiSessionMapper sessionMapper;

    @Autowired(required = false)
    private AiMessageMapper messageMapper;

    @Autowired(required = false)
    private AiUserMemoryMapper userMemoryMapper;

    @Autowired(required = false)
    private Langchain4jMemoryAdapter langchain4jAdapter;

    @Autowired(required = false)
    private PromptTemplateRegistry promptRegistry;

    /**
     * 创建新会话
     */
    public String createSession(Long userId) {
        String sessionId = UUID.randomUUID().toString().replace("-", "").substring(0, 16);

        // 持久化到数据库
        if (sessionMapper != null) {
            AiSession session = new AiSession(sessionId, userId);
            sessionMapper.insert(session);
            log.info("创建新会话: sessionId={}, userId={}", sessionId, userId);
        }

        // 初始化内存缓存
        sessionCaches.put(sessionId, new SessionCache(sessionId, userId));

        // 初始化 LangChain4j ChatMemory
        if (langchain4jAdapter != null) {
            langchain4jAdapter.get(sessionId);
        }

        return sessionId;
    }

    /**
     * 添加用户消息到会话
     */
    public void addUserMessage(String sessionId, Long userId, String message) {
        // 确保会话存在
        SessionCache cache = getOrCreateSessionCache(sessionId, userId);
        cache.addMessage("user", message);

        // 持久化消息
        if (messageMapper != null) {
            AiMessage aiMessage = new AiMessage(sessionId, "user", message);
            messageMapper.insert(aiMessage);
            sessionMapper.incrementMessageCount(sessionId);
        }

        // 同步到 LangChain4j
        if (langchain4jAdapter != null) {
            langchain4jAdapter.addUserMessage(sessionId, message);
        }

        // 检查是否需要压缩
        if (cache.messages.size() > MAX_SHORT_TERM_MESSAGES) {
            compressMemory(cache);
        }

        // 提取偏好
        extractAndSavePreferences(sessionId, message);

        log.debug("添加用户消息到会话 {}: {} 条消息", sessionId, cache.messages.size());
    }

    /**
     * 添加助手回复到会话
     */
    public void addAssistantMessage(String sessionId, String message) {
        addAssistantMessage(sessionId, message, null);
    }

    /**
     * 添加助手回复到会话（带 token 统计）
     */
    public void addAssistantMessage(String sessionId, String message, Integer tokensUsed) {
        SessionCache cache = sessionCaches.get(sessionId);
        if (cache != null) {
            cache.addMessage("assistant", message);
        }

        // 持久化消息
        if (messageMapper != null) {
            AiMessage aiMessage = new AiMessage(sessionId, "assistant", message, tokensUsed);
            messageMapper.insert(aiMessage);
            sessionMapper.incrementMessageCount(sessionId);
        }

        // 同步到 LangChain4j
        if (langchain4jAdapter != null) {
            langchain4jAdapter.addAiMessage(sessionId, message);
        }
    }

    /**
     * 获取对话上下文（用于 LLM 调用）
     */
    public List<Map<String, String>> getConversationContext(String sessionId) {
        // 优先从 LangChain4j 获取
        if (langchain4jAdapter != null && langchain4jAdapter.hasSession(sessionId)) {
            return langchain4jAdapter.getMessagesAsMapList(sessionId);
        }

        // 回退到内存缓存
        SessionCache cache = sessionCaches.get(sessionId);
        if (cache == null) {
            // 尝试从数据库加载
            cache = loadSessionFromDB(sessionId);
            if (cache == null) {
                return new ArrayList<>();
            }
        }

        List<Map<String, String>> context = new ArrayList<>();

        // 1. 添加摘要（如果有）
        if (cache.summary != null && !cache.summary.isEmpty()) {
            context.add(Map.of(
                    "role", "system",
                    "content", "之前对话摘要: " + cache.summary
            ));
        }

        // 2. 添加用户长期记忆
        String userPrefs = getUserPreferences(cache.userId);
        if (userPrefs != null && !userPrefs.isEmpty()) {
            context.add(Map.of(
                    "role", "system",
                    "content", "用户偏好: " + userPrefs
            ));
        }

        // 3. 添加最近消息
        for (ChatMessage msg : cache.messages) {
            context.add(Map.of(
                    "role", msg.role,
                    "content", msg.content
            ));
        }

        return context;
    }

    /**
     * 获取会话历史（用于前端展示）
     */
    public List<Map<String, Object>> getSessionHistory(String sessionId) {
        if (messageMapper != null) {
            List<AiMessage> messages = messageMapper.selectBySessionId(sessionId);
            return messages.stream()
                    .map(msg -> {
                        Map<String, Object> map = new HashMap<>();
                        map.put("role", msg.getRole());
                        map.put("content", msg.getContent());
                        map.put("createdAt", msg.getCreatedAt().toString());
                        return map;
                    })
                    .collect(Collectors.toList());
        }

        // 回退到内存缓存
        SessionCache cache = sessionCaches.get(sessionId);
        if (cache == null) {
            return new ArrayList<>();
        }

        return cache.messages.stream()
                .map(msg -> {
                    Map<String, Object> map = new HashMap<>();
                    map.put("role", msg.role);
                    map.put("content", msg.content);
                    map.put("createdAt", msg.timestamp.toString());
                    return map;
                })
                .collect(Collectors.toList());
    }

    /**
     * 获取用户的所有会话列表
     */
    public List<Map<String, Object>> getUserSessions(Long userId, int limit) {
        if (sessionMapper != null) {
            List<AiSession> sessions = sessionMapper.selectRecentByUserId(userId, limit);
            return sessions.stream()
                    .map(s -> {
                        Map<String, Object> map = new HashMap<>();
                        map.put("sessionId", s.getSessionId());
                        map.put("title", s.getTitle() != null ? s.getTitle() : "新对话");
                        map.put("messageCount", s.getMessageCount());
                        map.put("status", s.getStatus());
                        map.put("lastActiveAt", s.getLastActiveAt().toString());
                        map.put("createdAt", s.getCreatedAt().toString());
                        return map;
                    })
                    .collect(Collectors.toList());
        }

        // 回退：从内存缓存获取
        return sessionCaches.values().stream()
                .filter(c -> c.userId.equals(userId))
                .sorted((a, b) -> b.createdAt.compareTo(a.createdAt))
                .limit(limit)
                .map(c -> {
                    Map<String, Object> map = new HashMap<>();
                    map.put("sessionId", c.sessionId);
                    map.put("title", "新对话");
                    map.put("messageCount", c.messages.size());
                    map.put("status", "active");
                    map.put("createdAt", c.createdAt.toString());
                    return map;
                })
                .collect(Collectors.toList());
    }

    /**
     * 更新会话标题
     */
    public void updateSessionTitle(String sessionId, String title) {
        if (sessionMapper != null) {
            sessionMapper.updateTitle(sessionId, title);
        }
    }

    /**
     * 自动生成会话标题（基于第一条消息）
     */
    public void autoGenerateTitle(String sessionId, String firstMessage) {
        String title = firstMessage.length() > 30
                ? firstMessage.substring(0, 30) + "..."
                : firstMessage;
        updateSessionTitle(sessionId, title);
    }

    /**
     * 压缩记忆 - 将旧消息转为摘要
     */
    private void compressMemory(SessionCache cache) {
        if (cache.messages.size() <= SUMMARY_THRESHOLD) {
            return;
        }

        // 提取要压缩的消息
        int compressCount = cache.messages.size() - SUMMARY_THRESHOLD / 2;
        List<ChatMessage> toCompress = new ArrayList<>(
                cache.messages.subList(0, compressCount));

        // 生成摘要
        String newSummary = generateSummary(toCompress);

        // 合并摘要
        if (cache.summary != null) {
            cache.summary = cache.summary + " | " + newSummary;
        } else {
            cache.summary = newSummary;
        }

        // 持久化摘要
        if (sessionMapper != null) {
            sessionMapper.updateSummary(cache.sessionId, cache.summary);
        }

        // 移除已压缩的消息
        cache.messages = new ArrayList<>(
                cache.messages.subList(compressCount, cache.messages.size()));

        log.info("压缩会话 {}: {} 条消息 → 摘要, 剩余 {} 条",
                cache.sessionId, compressCount, cache.messages.size());
    }

    /**
     * 生成对话摘要
     */
    private String generateSummary(List<ChatMessage> messages) {
        // 提取用户问题的关键词
        Set<String> topics = new HashSet<>();
        for (ChatMessage msg : messages) {
            if ("user".equals(msg.role)) {
                if (msg.content.contains("策略")) topics.add("策略");
                if (msg.content.contains("算法")) topics.add("算法");
                if (msg.content.contains("优化")) topics.add("优化");
                if (msg.content.contains("Bot")) topics.add("Bot开发");
                if (msg.content.contains("代码")) topics.add("代码问题");
                if (msg.content.contains("BFS")) topics.add("BFS寻路");
                if (msg.content.contains("DFS")) topics.add("DFS搜索");
                if (msg.content.contains("A*")) topics.add("A*算法");
            }
        }

        if (topics.isEmpty()) {
            return "用户进行了 " + messages.size() + " 轮对话";
        }

        return "讨论了: " + String.join("、", topics);
    }

    /**
     * 记录用户偏好（长期记忆）
     */
    public void recordUserPreference(Long userId, String preferenceType, String preferenceValue, double confidence) {
        if (userMemoryMapper == null) {
            log.debug("用户记忆持久化未启用，跳过偏好记录");
            return;
        }

        AiUserMemory memory = userMemoryMapper.selectByUserId(userId);
        if (memory == null) {
            memory = new AiUserMemory(userId);
            userMemoryMapper.insert(memory);
        }

        // 解析现有偏好
        JSONArray prefs;
        try {
            prefs = JSON.parseArray(memory.getPreferences());
            if (prefs == null) prefs = new JSONArray();
        } catch (Exception e) {
            prefs = new JSONArray();
        }

        // 添加新偏好
        JSONObject newPref = new JSONObject();
        newPref.put("type", preferenceType);
        newPref.put("value", preferenceValue);
        newPref.put("confidence", confidence);
        newPref.put("timestamp", LocalDateTime.now().toString());
        prefs.add(newPref);

        // 限制偏好数量
        while (prefs.size() > 20) {
            prefs.remove(0);
        }

        // 更新数据库
        userMemoryMapper.updatePreferences(userId, JSON.toJSONString(prefs));
        log.info("记录用户 {} 偏好: {}={} (confidence={})", userId, preferenceType, preferenceValue, confidence);
    }

    /**
     * 获取用户偏好（用于注入 System Prompt）
     */
    public String getUserPreferences(Long userId) {
        if (userMemoryMapper == null) {
            return "";
        }

        AiUserMemory memory = userMemoryMapper.selectByUserId(userId);
        if (memory == null || memory.getPreferences() == null) {
            return "";
        }

        try {
            JSONArray prefs = JSON.parseArray(memory.getPreferences());
            if (prefs == null || prefs.isEmpty()) {
                return "";
            }

            // 提取高置信度的偏好
            List<String> highConfPrefs = new ArrayList<>();
            for (int i = 0; i < prefs.size(); i++) {
                JSONObject pref = prefs.getJSONObject(i);
                double confidence = pref.getDoubleValue("confidence");
                if (confidence >= 0.6) {
                    highConfPrefs.add(pref.getString("value"));
                }
            }

            return String.join(", ", highConfPrefs);
        } catch (Exception e) {
            log.warn("解析用户偏好失败: {}", e.getMessage());
            return "";
        }
    }

    /**
     * 从对话中提取并保存用户偏好
     */
    public void extractAndSavePreferences(String sessionId, String userMessage) {
        SessionCache cache = sessionCaches.get(sessionId);
        if (cache == null) return;

        // 简单规则提取偏好（后续可替换为 LLM 提取）
        if (userMessage.contains("喜欢") || userMessage.contains("prefer") || userMessage.contains("偏好")) {
            extractPreferenceByRules(cache.userId, userMessage);
        }
    }

    private void extractPreferenceByRules(Long userId, String message) {
        if (message.contains("贪心")) recordUserPreference(userId, "strategy", "偏好贪心策略", 0.7);
        if (message.contains("防守")) recordUserPreference(userId, "strategy", "偏好防守策略", 0.7);
        if (message.contains("进攻")) recordUserPreference(userId, "strategy", "偏好进攻策略", 0.7);
        if (message.contains("简单")) recordUserPreference(userId, "code_style", "偏好简单实现", 0.6);
        if (message.contains("高效")) recordUserPreference(userId, "algorithm", "偏好高效算法", 0.6);
        if (message.contains("详细") || message.contains("注释"))
            recordUserPreference(userId, "code_style", "偏好详细注释", 0.6);
    }

    /**
     * 使用 LLM 提取偏好（高级版）
     */
    public void extractPreferencesWithLLM(String sessionId, DeepseekClient deepseekClient) {
        SessionCache cache = sessionCaches.get(sessionId);
        if (cache == null || cache.messages.size() < 3) {
            return;
        }

        if (promptRegistry == null || deepseekClient == null || !deepseekClient.enabled()) {
            log.debug("LLM 偏好提取未启用");
            return;
        }

        try {
            // 构建对话历史
            StringBuilder history = new StringBuilder();
            for (ChatMessage msg : cache.messages) {
                history.append(msg.role).append(": ").append(msg.content).append("\n");
            }

            String prompt = promptRegistry.applyAsText("preference_extraction",
                    Map.of("history", history.toString()));

            String response = deepseekClient.chat("你是一个偏好分析助手", prompt, List.of());

            // 解析 JSON 响应
            JSONObject result = JSON.parseObject(response);
            JSONArray prefs = result.getJSONArray("preferences");

            if (prefs != null) {
                for (int i = 0; i < prefs.size(); i++) {
                    JSONObject pref = prefs.getJSONObject(i);
                    recordUserPreference(
                            cache.userId,
                            pref.getString("type"),
                            pref.getString("value"),
                            pref.getDoubleValue("confidence")
                    );
                }
            }

            log.info("LLM 偏好提取完成: sessionId={}, 提取 {} 个偏好", sessionId, prefs != null ? prefs.size() : 0);
        } catch (Exception e) {
            log.warn("LLM 偏好提取失败: {}", e.getMessage());
        }
    }

    /**
     * 获取或创建会话缓存
     */
    private SessionCache getOrCreateSessionCache(String sessionId, Long userId) {
        return sessionCaches.computeIfAbsent(sessionId, k -> {
            // 尝试从数据库加载
            SessionCache cache = loadSessionFromDB(sessionId);
            if (cache != null) {
                return cache;
            }

            // 创建新缓存
            SessionCache newCache = new SessionCache(sessionId, userId);

            // 持久化新会话
            if (sessionMapper != null) {
                AiSession session = new AiSession(sessionId, userId);
                sessionMapper.insert(session);
            }

            return newCache;
        });
    }

    /**
     * 从数据库加载会话
     */
    private SessionCache loadSessionFromDB(String sessionId) {
        if (sessionMapper == null || messageMapper == null) {
            return null;
        }

        AiSession session = sessionMapper.selectBySessionId(sessionId);
        if (session == null) {
            return null;
        }

        SessionCache cache = new SessionCache(sessionId, session.getUserId());
        cache.summary = session.getSummary();

        // 加载最近消息
        List<AiMessage> messages = messageMapper.selectRecentBySessionId(sessionId, MAX_SHORT_TERM_MESSAGES);
        Collections.reverse(messages); // 恢复时间顺序

        for (AiMessage msg : messages) {
            cache.messages.add(new ChatMessage(msg.getRole(), msg.getContent(), msg.getCreatedAt()));
        }

        // 同步到 LangChain4j
        if (langchain4jAdapter != null) {
            langchain4jAdapter.importFromLegacy(sessionId, cache.messages.stream()
                    .map(m -> Map.of("role", m.role, "content", m.content))
                    .collect(Collectors.toList()));
        }

        log.info("从数据库加载会话 {}: {} 条消息", sessionId, cache.messages.size());
        return cache;
    }

    /**
     * 清除会话
     */
    public void clearSession(String sessionId) {
        sessionCaches.remove(sessionId);

        if (langchain4jAdapter != null) {
            langchain4jAdapter.clear(sessionId);
        }

        if (sessionMapper != null) {
            sessionMapper.updateStatus(sessionId, "deleted");
        }

        log.info("清除会话: {}", sessionId);
    }

    /**
     * 获取会话统计
     */
    public Map<String, Object> getSessionStats(String sessionId) {
        SessionCache cache = sessionCaches.get(sessionId);
        if (cache == null) {
            if (sessionMapper != null) {
                AiSession session = sessionMapper.selectBySessionId(sessionId);
                if (session != null) {
                    return Map.of(
                            "exists", true,
                            "messageCount", session.getMessageCount(),
                            "hasSummary", session.getSummary() != null,
                            "userId", session.getUserId(),
                            "createdAt", session.getCreatedAt().toString()
                    );
                }
            }
            return Map.of("exists", false);
        }

        return Map.of(
                "exists", true,
                "messageCount", cache.messages.size(),
                "hasSummary", cache.summary != null,
                "userId", cache.userId,
                "createdAt", cache.createdAt.toString()
        );
    }

    /**
     * 获取服务统计
     */
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("cachedSessions", sessionCaches.size());
        stats.put("totalCachedMessages", sessionCaches.values().stream()
                .mapToInt(s -> s.messages.size()).sum());

        if (langchain4jAdapter != null) {
            stats.put("langchain4jSessions", langchain4jAdapter.getActiveSessionCount());
        }

        return stats;
    }

    // 内部类
    private static class SessionCache {
        String sessionId;
        Long userId;
        List<ChatMessage> messages = new ArrayList<>();
        String summary;
        LocalDateTime createdAt = LocalDateTime.now();

        SessionCache(String sessionId, Long userId) {
            this.sessionId = sessionId;
            this.userId = userId;
        }

        void addMessage(String role, String content) {
            messages.add(new ChatMessage(role, content, LocalDateTime.now()));
        }
    }

    private record ChatMessage(String role, String content, LocalDateTime timestamp) {}
}
