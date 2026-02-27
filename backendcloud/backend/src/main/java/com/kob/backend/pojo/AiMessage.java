package com.kob.backend.pojo;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * AI 对话消息实体
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@TableName("ai_message")
public class AiMessage {

    @TableId(type = IdType.AUTO)
    private Integer id;

    private String sessionId;

    private String role;

    private String content;

    private Integer tokensUsed;

    private String metadata;

    private LocalDateTime createdAt;

    public AiMessage(String sessionId, String role, String content) {
        this.sessionId = sessionId;
        this.role = role;
        this.content = content;
        this.createdAt = LocalDateTime.now();
    }

    public AiMessage(String sessionId, String role, String content, Integer tokensUsed) {
        this.sessionId = sessionId;
        this.role = role;
        this.content = content;
        this.tokensUsed = tokensUsed;
        this.createdAt = LocalDateTime.now();
    }
}
