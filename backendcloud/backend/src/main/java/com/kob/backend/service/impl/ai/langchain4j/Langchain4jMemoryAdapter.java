package com.kob.backend.service.impl.ai.langchain4j;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * LangChain4j ChatMemory 适配器
 *
 * 将 LangChain4j ChatMemory 与现有 ConversationMemoryService 桥接
 *
 * 面试亮点：
 * - 适配器模式：不破坏现有代码，平滑迁移
 * - 选择性集成：只用框架的 Memory 抽象，保留自研 RAG
 * - 双向同步：支持从现有服务导入历史，也支持导出到持久化存储
 */
@Component
public class Langchain4jMemoryAdapter implements ChatMemoryProvider {

    private static final Logger log = LoggerFactory.getLogger(Langchain4jMemoryAdapter.class);

    private static final int DEFAULT_MAX_MESSAGES = 10;

    // 存储每个会话的 ChatMemory 实例
    private final Map<Object, ChatMemory> memories = new ConcurrentHashMap<>();

    /**
     * 获取或创建指定会话的 ChatMemory
     */
    @Override
    public ChatMemory get(Object sessionId) {
        return memories.computeIfAbsent(sessionId, id -> {
            log.debug("创建新的 ChatMemory: sessionId={}", id);
            return MessageWindowChatMemory.builder()
                    .maxMessages(DEFAULT_MAX_MESSAGES)
                    .id(id)
                    .build();
        });
    }

    /**
     * 添加用户消息
     */
    public void addUserMessage(String sessionId, String content) {
        ChatMemory memory = get(sessionId);
        memory.add(UserMessage.from(content));
        log.debug("添加用户消息到会话 {}: {} 条消息", sessionId, memory.messages().size());
    }

    /**
     * 添加 AI 回复
     */
    public void addAiMessage(String sessionId, String content) {
        ChatMemory memory = get(sessionId);
        memory.add(AiMessage.from(content));
        log.debug("添加 AI 消息到会话 {}: {} 条消息", sessionId, memory.messages().size());
    }

    /**
     * 添加系统消息（如用户偏好、会话摘要等）
     */
    public void addSystemMessage(String sessionId, String content) {
        ChatMemory memory = get(sessionId);
        memory.add(SystemMessage.from(content));
        log.debug("添加系统消息到会话 {}", sessionId);
    }

    /**
     * 获取会话的所有消息（用于转换为 DeepSeek API 格式）
     */
    public List<ChatMessage> getMessages(String sessionId) {
        ChatMemory memory = memories.get(sessionId);
        if (memory == null) {
            return List.of();
        }
        return memory.messages();
    }

    /**
     * 将 LangChain4j 消息格式转换为简单的 Map 格式（兼容现有 DeepseekClient）
     */
    public List<Map<String, String>> getMessagesAsMapList(String sessionId) {
        return getMessages(sessionId).stream()
                .map(this::convertToMap)
                .toList();
    }

    /**
     * 清除指定会话的记忆
     */
    public void clear(String sessionId) {
        ChatMemory memory = memories.remove(sessionId);
        if (memory != null) {
            memory.clear();
            log.info("清除会话记忆: {}", sessionId);
        }
    }

    /**
     * 获取会话消息数量
     */
    public int getMessageCount(String sessionId) {
        ChatMemory memory = memories.get(sessionId);
        return memory == null ? 0 : memory.messages().size();
    }

    /**
     * 检查会话是否存在
     */
    public boolean hasSession(String sessionId) {
        return memories.containsKey(sessionId);
    }

    /**
     * 获取所有活跃会话数
     */
    public int getActiveSessionCount() {
        return memories.size();
    }

    /**
     * 从现有 ConversationMemoryService 导入历史消息
     * 用于迁移场景
     */
    public void importFromLegacy(String sessionId, List<Map<String, String>> legacyMessages) {
        ChatMemory memory = get(sessionId);
        memory.clear();

        for (Map<String, String> msg : legacyMessages) {
            String role = msg.get("role");
            String content = msg.get("content");

            switch (role) {
                case "user" -> memory.add(UserMessage.from(content));
                case "assistant" -> memory.add(AiMessage.from(content));
                case "system" -> memory.add(SystemMessage.from(content));
                default -> log.warn("未知消息角色: {}", role);
            }
        }

        log.info("从旧服务导入 {} 条消息到会话 {}", legacyMessages.size(), sessionId);
    }

    /**
     * 导出为可持久化的格式
     */
    public List<Map<String, Object>> exportForPersistence(String sessionId) {
        return getMessages(sessionId).stream()
                .map(msg -> {
                    Map<String, Object> map = new java.util.HashMap<>();
                    map.put("role", getRole(msg));
                    map.put("content", getContent(msg));
                    map.put("type", msg.type().name());
                    return map;
                })
                .toList();
    }

    // 辅助方法：转换消息为 Map
    private Map<String, String> convertToMap(ChatMessage message) {
        return Map.of(
                "role", getRole(message),
                "content", getContent(message)
        );
    }

    private String getRole(ChatMessage message) {
        return switch (message.type()) {
            case USER -> "user";
            case AI -> "assistant";
            case SYSTEM -> "system";
            default -> "unknown";
        };
    }

    private String getContent(ChatMessage message) {
        if (message instanceof UserMessage um) {
            return um.singleText();
        } else if (message instanceof AiMessage am) {
            return am.text();
        } else if (message instanceof SystemMessage sm) {
            return sm.text();
        }
        return "";
    }
}
