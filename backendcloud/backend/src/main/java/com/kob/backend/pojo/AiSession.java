package com.kob.backend.pojo;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * AI 对话会话实体
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@TableName("ai_session")
public class AiSession {

    @TableId(type = IdType.AUTO)
    private Integer id;

    private String sessionId;

    private Long userId;

    private String title;

    private String summary;

    private String status;

    private Integer messageCount;

    private LocalDateTime createdAt;

    private LocalDateTime updatedAt;

    private LocalDateTime lastActiveAt;

    public AiSession(String sessionId, Long userId) {
        this.sessionId = sessionId;
        this.userId = userId;
        this.status = "active";
        this.messageCount = 0;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
        this.lastActiveAt = LocalDateTime.now();
    }
}
