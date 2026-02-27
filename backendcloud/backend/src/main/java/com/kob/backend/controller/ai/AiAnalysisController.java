package com.kob.backend.controller.ai;

import com.kob.backend.mapper.RecordMapper;
import com.kob.backend.pojo.Record;
import com.kob.backend.service.impl.ai.AiMetricsService;
import com.kob.backend.service.impl.ai.BattleAnalysisService;
import com.kob.backend.service.impl.ai.DeepseekClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.annotation.PostConstruct;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * AI 分析控制器
 *
 * 提供对战分析相关的 API 端点：
 * - 根据 recordId 分析对战
 * - 针对特定用户的失败原因分析
 *
 * 供 Python AI 工具调用
 */
@RestController
@RequestMapping("/ai/analysis")
public class AiAnalysisController {

    private static final Logger log = LoggerFactory.getLogger(AiAnalysisController.class);

    @Autowired
    private RecordMapper recordMapper;

    @Autowired
    private BattleAnalysisService battleAnalysisService;

    @Autowired
    private AiMetricsService metricsService;

    private DeepseekClient deepseekClient;

    @PostConstruct
    public void init() {
        deepseekClient = new DeepseekClient(metricsService);
        log.info("AI Analysis Controller 初始化完成");
    }

    /**
     * 根据对战记录 ID 分析对战
     * 供 Python AI 工具 battle_analysis 调用
     *
     * @param recordId 对战记录ID
     * @return 分析结果，包含 stats, key_moments, ai_analysis
     */
    @GetMapping("/record/{recordId}")
    public ResponseEntity<Map<String, Object>> analyzeByRecordId(@PathVariable Integer recordId) {
        log.info("收到对战分析请求: recordId={}", recordId);

        Map<String, Object> response = new HashMap<>();

        // 1. 从 Record 表查询对战数据
        Record record = recordMapper.selectById(recordId);
        if (record == null) {
            log.warn("对战记录不存在: recordId={}", recordId);
            return ResponseEntity.notFound().build();
        }

        try {
            // 2. 调用 BattleAnalysisService
            BattleAnalysisService.BattleAnalysisResult result = battleAnalysisService.analyzeBattle(
                record.getMap(),
                record.getASteps(),
                record.getBSteps(),
                record.getLoser(),
                deepseekClient
            );

            // 3. 构建响应
            response.put("stats", buildStatsMap(result.getStats()));
            response.put("key_moments", buildKeyMomentsMap(result.getKeyMoments()));
            response.put("ai_analysis", result.getAiAnalysis());

            log.info("对战分析完成: recordId={}", recordId);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("对战分析失败: recordId={}, error={}", recordId, e.getMessage(), e);
            response.put("error", "分析失败: " + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }

    /**
     * 专门分析用户失败原因
     * 从用户视角提供针对性分析
     *
     * @param recordId 对战记录ID
     * @param userId 用户ID（用于确定分析视角）
     * @return 针对该用户的失败原因分析
     */
    @GetMapping("/loss/{recordId}")
    public ResponseEntity<Map<String, Object>> analyzeLossReason(
            @PathVariable Integer recordId,
            @RequestParam Integer userId) {

        log.info("收到失败原因分析请求: recordId={}, userId={}", recordId, userId);

        Map<String, Object> response = new HashMap<>();

        // 1. 查询对战记录
        Record record = recordMapper.selectById(recordId);
        if (record == null) {
            log.warn("对战记录不存在: recordId={}", recordId);
            return ResponseEntity.notFound().build();
        }

        // 2. 确定用户角色（A 还是 B）
        boolean isPlayerA = record.getAId().equals(userId);
        boolean isPlayerB = record.getBId().equals(userId);

        if (!isPlayerA && !isPlayerB) {
            response.put("error", "用户未参与此对战");
            return ResponseEntity.badRequest().body(response);
        }

        try {
            // 3. 判断是否是输家
            String playerRole = isPlayerA ? "A" : "B";
            boolean isLoser = playerRole.equals(record.getLoser());

            // 4. 调用分析服务
            BattleAnalysisService.BattleAnalysisResult analysisResult = battleAnalysisService.analyzeBattle(
                record.getMap(),
                record.getASteps(),
                record.getBSteps(),
                record.getLoser(),
                deepseekClient
            );

            // 5. 构建用户视角的响应
            response.put("is_loser", isLoser);
            response.put("player_role", playerRole);
            response.put("record_id", recordId);

            // 获取策略信息
            BattleAnalysisService.BasicStats stats = analysisResult.getStats();
            response.put("your_strategy", isPlayerA ? stats.movementPatternA : stats.movementPatternB);
            response.put("opponent_strategy", isPlayerA ? stats.movementPatternB : stats.movementPatternA);
            response.put("your_control_area", isPlayerA ? stats.controlAreaA : stats.controlAreaB);
            response.put("opponent_control_area", isPlayerA ? stats.controlAreaB : stats.controlAreaA);

            // 提取关键失误（针对用户）
            List<Map<String, Object>> criticalMistakes = new ArrayList<>();
            for (BattleAnalysisService.KeyMoment moment : analysisResult.getKeyMoments()) {
                // 只保留与用户相关的危险时刻
                if (moment.type.contains(playerRole + " 危险") ||
                    (isLoser && moment.type.contains("决胜"))) {
                    Map<String, Object> mistake = new HashMap<>();
                    mistake.put("round", moment.round);
                    mistake.put("type", moment.type);
                    mistake.put("description", moment.description);
                    criticalMistakes.add(mistake);
                }
            }
            response.put("critical_mistakes", criticalMistakes);

            // 生成改进建议
            List<String> suggestions = generateSuggestions(isLoser, criticalMistakes, analysisResult);
            response.put("suggestions", suggestions);

            response.put("ai_analysis", analysisResult.getAiAnalysis());
            response.put("total_rounds", stats.totalRounds);
            response.put("result", stats.winner);

            log.info("失败原因分析完成: recordId={}, userId={}, isLoser={}", recordId, userId, isLoser);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("失败原因分析失败: recordId={}, userId={}, error={}", recordId, userId, e.getMessage(), e);
            response.put("error", "分析失败: " + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }

    /**
     * 生成针对性的改进建议
     */
    private List<String> generateSuggestions(boolean isLoser,
                                              List<Map<String, Object>> criticalMistakes,
                                              BattleAnalysisService.BattleAnalysisResult analysisResult) {
        List<String> suggestions = new ArrayList<>();

        if (isLoser) {
            // 针对输家的建议
            if (!criticalMistakes.isEmpty()) {
                suggestions.add("回顾关键失误时刻，思考当时还有哪些可选的移动方向");
            }

            BattleAnalysisService.BasicStats stats = analysisResult.getStats();

            // 根据移动模式给建议
            if ("单一方向".equals(stats.movementPatternA) || "单一方向".equals(stats.movementPatternB)) {
                suggestions.add("尝试更灵活的移动方式，避免被对手预判");
            }

            // 根据控制区域给建议
            if (stats.controlAreaA < stats.controlAreaB) {
                suggestions.add("尝试在开局占据更多空间，提高生存几率");
            }

            suggestions.add("分析对手的移动规律，预测其可能的行动");
            suggestions.add("在接近边界或障碍物时要格外小心");

        } else {
            // 针对赢家的建议（也可以提升）
            suggestions.add("这场表现不错！可以继续优化策略的稳定性");
            suggestions.add("尝试不同的开局策略，丰富战术库");
        }

        return suggestions;
    }

    /**
     * 构建统计信息 Map
     */
    private Map<String, Object> buildStatsMap(BattleAnalysisService.BasicStats stats) {
        Map<String, Object> map = new HashMap<>();
        map.put("totalRounds", stats.totalRounds);
        map.put("controlAreaA", stats.controlAreaA);
        map.put("controlAreaB", stats.controlAreaB);
        map.put("movementPatternA", stats.movementPatternA);
        map.put("movementPatternB", stats.movementPatternB);
        map.put("winner", stats.winner);
        return map;
    }

    /**
     * 构建关键时刻列表
     */
    private List<Map<String, Object>> buildKeyMomentsMap(List<BattleAnalysisService.KeyMoment> keyMoments) {
        List<Map<String, Object>> list = new ArrayList<>();
        for (BattleAnalysisService.KeyMoment moment : keyMoments) {
            Map<String, Object> map = new HashMap<>();
            map.put("round", moment.round);
            map.put("type", moment.type);
            map.put("description", moment.description);
            list.add(map);
        }
        return list;
    }
}
