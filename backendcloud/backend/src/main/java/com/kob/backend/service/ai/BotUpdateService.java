package com.kob.backend.service.ai;

import com.kob.backend.mapper.BotActionLogMapper;
import com.kob.backend.mapper.BotMapper;
import com.kob.backend.pojo.Bot;
import com.kob.backend.pojo.BotActionLog;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Date;
import java.util.HashMap;
import java.util.Map;

/**
 * 幂等 Bot 代码更新服务
 *
 * 事务保证：action_log 插入 + bot 更新是原子的。
 * - 如果 action_log 插入冲突 → 返回 duplicate（不触发事务回滚）
 * - 如果 bot 更新失败（含影响行数 != 1）→ 事务回滚，action_log 也回滚 → 可安全重试
 */
@Service
public class BotUpdateService {

    @Autowired
    private BotMapper botMapper;

    @Autowired
    private BotActionLogMapper botActionLogMapper;

    @Transactional(rollbackFor = Exception.class)
    public Map<String, Object> idempotentUpdate(Bot bot, String content, String actionId) {
        Map<String, Object> result = new HashMap<>();

        // 1. 幂等判重（在事务内）
        if (actionId != null && !actionId.isEmpty()) {
            try {
                botActionLogMapper.insert(
                        new BotActionLog(bot.getId(), actionId, "code_update"));
            } catch (DuplicateKeyException e) {
                result.put("success", true);
                result.put("duplicate", true);
                result.put("message", "操作已执行 (幂等)");
                return result;  // 不触发回滚
            }
        }

        // 2. 更新 bot（与 action_log 同一事务）
        bot.setContent(content);
        bot.setModifytime(new Date());
        int updated = botMapper.updateById(bot);
        if (updated != 1) {
            throw new IllegalStateException("Bot update affected rows=" + updated);
        }

        result.put("success", true);
        result.put("duplicate", false);
        result.put("message", "Bot 代码更新成功");
        return result;
    }
}
