package com.kob.backend.pojo;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Date;

@Data
@NoArgsConstructor
@AllArgsConstructor
@TableName("bot_action_log")
public class BotActionLog {
    @TableId(type = IdType.AUTO)
    private Long id;

    @TableField("bot_id")
    private Integer botId;

    @TableField("action_id")
    private String actionId;

    @TableField("action_type")
    private String actionType;

    @TableField("created_at")
    private Date createdAt;

    /**
     * 便捷构造器（不含 id 和 created_at，由数据库自动生成）
     */
    public BotActionLog(Integer botId, String actionId, String actionType) {
        this.botId = botId;
        this.actionId = actionId;
        this.actionType = actionType;
    }
}
