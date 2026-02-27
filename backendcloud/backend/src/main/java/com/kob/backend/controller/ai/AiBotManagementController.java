package com.kob.backend.controller.ai;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.kob.backend.mapper.BotMapper;
import com.kob.backend.pojo.Bot;
import com.kob.backend.service.ai.BotUpdateService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.*;

/**
 * AI Bot 管理控制器
 * 
 * 为 Python AI Service 的 HITL (Human-in-the-Loop) 功能提供后端支持
 * 
 * 端点：
 * - GET  /ai/bot/manage/list   获取用户的 Bot 列表
 * - GET  /ai/bot/manage/code   获取指定 Bot 的代码
 * - POST /ai/bot/manage/update 更新 Bot 代码
 * 
 * 2026 最佳实践：
 * - 所有端点都验证用户所有权
 * - 返回标准化的 JSON 响应
 */
@RestController
@RequestMapping("/ai/bot/manage")
public class AiBotManagementController {

    @Autowired
    private BotMapper botMapper;

    @Autowired
    private BotUpdateService botUpdateService;

    /**
     * 获取用户的 Bot 列表
     * 
     * @param userId 用户ID
     * @return 包含 Bot 列表的响应
     */
    @GetMapping("/list")
    public Map<String, Object> getUserBots(@RequestParam Long userId) {
        Map<String, Object> result = new HashMap<>();
        
        try {
            QueryWrapper<Bot> wrapper = new QueryWrapper<>();
            wrapper.eq("user_id", userId);
            wrapper.orderByDesc("modifytime");
            
            List<Bot> bots = botMapper.selectList(wrapper);
            
            // 转换为简化的响应格式
            List<Map<String, Object>> botList = new ArrayList<>();
            for (Bot bot : bots) {
                Map<String, Object> botInfo = new HashMap<>();
                botInfo.put("id", bot.getId());
                botInfo.put("title", bot.getTitle());
                botInfo.put("description", bot.getDescription());
                botInfo.put("modifytime", bot.getModifytime());
                botList.add(botInfo);
            }
            
            result.put("success", true);
            result.put("bots", botList);
            result.put("count", botList.size());
            
        } catch (Exception e) {
            result.put("success", false);
            result.put("error", "获取 Bot 列表失败: " + e.getMessage());
        }
        
        return result;
    }

    /**
     * 获取指定 Bot 的代码内容
     * 
     * @param botId  Bot ID
     * @param userId 用户ID（用于权限验证）
     * @return 包含 Bot 代码的响应
     */
    @GetMapping("/code")
    public Map<String, Object> getBotCode(
            @RequestParam Integer botId,
            @RequestParam Long userId
    ) {
        Map<String, Object> result = new HashMap<>();
        
        try {
            Bot bot = botMapper.selectById(botId);
            
            if (bot == null) {
                result.put("success", false);
                result.put("error", "Bot 不存在");
                return result;
            }
            
            // 验证所有权
            if (!bot.getUserId().equals(userId.intValue())) {
                result.put("success", false);
                result.put("error", "没有权限访问此 Bot");
                return result;
            }
            
            result.put("success", true);
            result.put("id", bot.getId());
            result.put("title", bot.getTitle());
            result.put("description", bot.getDescription());
            result.put("content", bot.getContent());
            result.put("modifytime", bot.getModifytime());
            
        } catch (Exception e) {
            result.put("success", false);
            result.put("error", "获取 Bot 代码失败: " + e.getMessage());
        }
        
        return result;
    }

    /**
     * 更新 Bot 代码（v4 - 支持幂等 actionId）
     * 
     * @param request 包含 botId, userId, content, actionId(可选) 的请求体
     * @return 更新结果
     */
    @PostMapping("/update")
    public Map<String, Object> updateBotCode(@RequestBody Map<String, Object> request) {
        Map<String, Object> result = new HashMap<>();
        
        try {
            Object botIdObj = request.get("botId");
            Object userIdObj = request.get("userId");
            Object contentObj = request.get("content");
            Object actionIdObj = request.get("actionId");
            
            if (botIdObj == null || userIdObj == null || contentObj == null) {
                result.put("success", false);
                result.put("error", "缺少必要参数");
                return result;
            }

            if (!(botIdObj instanceof Number) || !(userIdObj instanceof Number) || !(contentObj instanceof String)) {
                result.put("success", false);
                result.put("error", "参数类型错误");
                return result;
            }

            Integer botId = ((Number) botIdObj).intValue();
            Long userId = ((Number) userIdObj).longValue();
            String content = ((String) contentObj).trim();
            String actionId = actionIdObj == null ? null : String.valueOf(actionIdObj);
            
            Bot bot = botMapper.selectById(botId);
            
            if (bot == null) {
                result.put("success", false);
                result.put("error", "Bot 不存在");
                return result;
            }
            
            // 验证所有权
            if (!bot.getUserId().equals(userId.intValue())) {
                result.put("success", false);
                result.put("error", "没有权限修改此 Bot");
                return result;
            }
            
            // 代码长度检查（与现有 UpdateServiceImpl 一致）
            if (content.length() > 10000) {
                result.put("success", false);
                result.put("error", "代码长度不能超过 10000 字符");
                return result;
            }
            
            // 委托给 BotUpdateService（事务 + 幂等）
            result = botUpdateService.idempotentUpdate(bot, content, actionId);
            result.put("botId", botId);
            
        } catch (Exception e) {
            result.put("success", false);
            result.put("error", "更新 Bot 代码失败: " + e.getMessage());
        }
        
        return result;
    }
}
