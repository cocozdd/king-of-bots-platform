package com.kob.backend.pojo;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * AI 用户长期记忆实体
 *
 * 存储用户的偏好、话题历史和画像摘要
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@TableName("ai_user_memory")
public class AiUserMemory {

    @TableId(type = IdType.AUTO)
    private Integer id;

    private Long userId;

    private String preferences;

    private String topics;

    private String profileSummary;

    private LocalDateTime createdAt;

    private LocalDateTime updatedAt;

    public AiUserMemory(Long userId) {
        this.userId = userId;
        this.preferences = "[]";
        this.topics = "{}";
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
}
