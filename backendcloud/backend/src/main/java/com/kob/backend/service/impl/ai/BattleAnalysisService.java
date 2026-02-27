package com.kob.backend.service.impl.ai;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * 对战分析服务
 * 
 * 功能：
 * - 分析对战回放，解释关键决策点
 * - 评估双方策略优劣
 * - 提供改进建议
 * - 预测胜负走势
 */
@Service
public class BattleAnalysisService {
    
    private static final Logger log = LoggerFactory.getLogger(BattleAnalysisService.class);
    
    @Autowired
    private AiMetricsService metricsService;
    
    // 方向映射
    private static final String[] DIRECTION_NAMES = {"上", "右", "下", "左"};
    private static final int[] dx = {-1, 0, 1, 0};
    private static final int[] dy = {0, 1, 0, -1};
    
    /**
     * 分析对战记录
     * 
     * @param mapData 地图数据
     * @param stepsA A 玩家的步骤
     * @param stepsB B 玩家的步骤
     * @param loser 输家 ("A", "B", "all")
     * @param deepseekClient DeepSeek 客户端
     * @return 分析结果
     */
    public BattleAnalysisResult analyzeBattle(String mapData, String stepsA, String stepsB, 
                                               String loser, DeepseekClient deepseekClient) {
        long startTime = System.currentTimeMillis();
        
        // 1. 基础统计分析
        BasicStats stats = calculateBasicStats(mapData, stepsA, stepsB, loser);
        
        // 2. 关键回合识别
        List<KeyMoment> keyMoments = identifyKeyMoments(mapData, stepsA, stepsB, loser);
        
        // 3. AI 深度分析（如果可用）
        String aiAnalysis = null;
        if (deepseekClient != null && deepseekClient.enabled()) {
            aiAnalysis = generateAiAnalysis(mapData, stepsA, stepsB, loser, stats, keyMoments, deepseekClient);
        }
        
        long latency = System.currentTimeMillis() - startTime;
        log.info("对战分析完成: 回合数={}, 关键时刻={}, 耗时{}ms", 
                stats.totalRounds, keyMoments.size(), latency);
        
        return new BattleAnalysisResult(stats, keyMoments, aiAnalysis);
    }
    
    /**
     * 计算基础统计数据
     */
    private BasicStats calculateBasicStats(String mapData, String stepsA, String stepsB, String loser) {
        BasicStats stats = new BasicStats();
        
        // 回合数
        stats.totalRounds = Math.max(
                stepsA != null ? stepsA.length() : 0,
                stepsB != null ? stepsB.length() : 0
        );
        
        // 解析地图
        int rows = 13, cols = 14;
        int[][] map = parseMap(mapData, rows, cols);
        
        // 计算空间控制
        int[] posA = {rows - 2, 1};
        int[] posB = {1, cols - 2};
        
        // 模拟对战过程
        List<int[]> pathA = new ArrayList<>();
        List<int[]> pathB = new ArrayList<>();
        pathA.add(posA.clone());
        pathB.add(posB.clone());
        
        if (stepsA != null) {
            for (char c : stepsA.toCharArray()) {
                int d = c - '0';
                posA[0] += dx[d];
                posA[1] += dy[d];
                pathA.add(posA.clone());
            }
        }
        
        if (stepsB != null) {
            for (char c : stepsB.toCharArray()) {
                int d = c - '0';
                posB[0] += dx[d];
                posB[1] += dy[d];
                pathB.add(posB.clone());
            }
        }
        
        // 计算最终控制区域
        stats.controlAreaA = calculateControlArea(map, pathA, rows, cols);
        stats.controlAreaB = calculateControlArea(map, pathB, rows, cols);
        
        // 移动模式分析
        stats.movementPatternA = analyzeMovementPattern(stepsA);
        stats.movementPatternB = analyzeMovementPattern(stepsB);
        
        // 结果
        stats.winner = "all".equals(loser) ? "平局" : ("A".equals(loser) ? "B 获胜" : "A 获胜");
        
        return stats;
    }
    
    /**
     * 识别关键回合
     */
    private List<KeyMoment> identifyKeyMoments(String mapData, String stepsA, String stepsB, String loser) {
        List<KeyMoment> moments = new ArrayList<>();
        
        if (stepsA == null || stepsB == null) {
            return moments;
        }
        
        int rows = 13, cols = 14;
        int[][] map = parseMap(mapData, rows, cols);
        
        int[] posA = {rows - 2, 1};
        int[] posB = {1, cols - 2};
        
        int minLen = Math.min(stepsA.length(), stepsB.length());
        
        for (int round = 0; round < minLen; round++) {
            int dA = stepsA.charAt(round) - '0';
            int dB = stepsB.charAt(round) - '0';
            
            int[] newPosA = {posA[0] + dx[dA], posA[1] + dy[dA]};
            int[] newPosB = {posB[0] + dx[dB], posB[1] + dy[dB]};
            
            // 检测关键时刻
            
            // 1. 距离接近
            int distance = Math.abs(newPosA[0] - newPosB[0]) + Math.abs(newPosA[1] - newPosB[1]);
            if (distance <= 3 && round > 0) {
                moments.add(new KeyMoment(round + 1, "近距离对峙",
                        String.format("双方距离仅 %d 格，局势紧张", distance)));
            }
            
            // 2. 危险移动（接近边界或障碍）
            if (isDangerousMove(newPosA, map, rows, cols)) {
                moments.add(new KeyMoment(round + 1, "A 危险移动",
                        String.format("A 移动到危险位置 (%d, %d)", newPosA[0], newPosA[1])));
            }
            if (isDangerousMove(newPosB, map, rows, cols)) {
                moments.add(new KeyMoment(round + 1, "B 危险移动",
                        String.format("B 移动到危险位置 (%d, %d)", newPosB[0], newPosB[1])));
            }
            
            // 3. 最后一回合
            if (round == minLen - 1) {
                moments.add(new KeyMoment(round + 1, "决胜回合",
                        String.format("A向%s，B向%s，%s", 
                                DIRECTION_NAMES[dA], DIRECTION_NAMES[dB],
                                "all".equals(loser) ? "双方同归于尽" : 
                                ("A".equals(loser) ? "A 失误导致失败" : "B 失误导致失败"))));
            }
            
            posA = newPosA;
            posB = newPosB;
        }
        
        // 限制关键时刻数量
        if (moments.size() > 5) {
            // 保留最重要的
            List<KeyMoment> filtered = new ArrayList<>();
            for (KeyMoment m : moments) {
                if (m.type.contains("决胜") || m.type.contains("危险") || filtered.size() < 3) {
                    filtered.add(m);
                }
            }
            return filtered;
        }
        
        return moments;
    }
    
    /**
     * 生成 AI 深度分析
     */
    private String generateAiAnalysis(String mapData, String stepsA, String stepsB, String loser,
                                       BasicStats stats, List<KeyMoment> keyMoments,
                                       DeepseekClient deepseekClient) {
        String systemPrompt = """
            你是一个专业的贪吃蛇对战游戏分析师。
            请分析这场对战，给出专业点评。
            
            格式要求：
            - 使用 Markdown 格式输出
            - 段落之间用空行分隔
            - 使用 ## 作为二级标题，### 作为三级标题
            - 代码必须用 ```语言名 和 ``` 包裹，每行代码单独一行
            - 列表项之间适当换行
            
            分析要点：
            1. 整体策略评价
            2. 关键失误或亮点
            3. 给输家的改进建议
            4. 总结这场对战的经验教训
            
            语言风格：专业但易懂，像电竞解说一样有趣。
            """;
        
        StringBuilder userPrompt = new StringBuilder();
        userPrompt.append("## 对战数据\n");
        userPrompt.append(String.format("- 回合数: %d\n", stats.totalRounds));
        userPrompt.append(String.format("- 结果: %s\n", stats.winner));
        userPrompt.append(String.format("- A 控制区域: %d 格\n", stats.controlAreaA));
        userPrompt.append(String.format("- B 控制区域: %d 格\n", stats.controlAreaB));
        userPrompt.append(String.format("- A 移动模式: %s\n", stats.movementPatternA));
        userPrompt.append(String.format("- B 移动模式: %s\n", stats.movementPatternB));
        
        userPrompt.append("\n## 关键时刻\n");
        for (KeyMoment m : keyMoments) {
            userPrompt.append(String.format("- 第%d回合 [%s]: %s\n", m.round, m.type, m.description));
        }
        
        userPrompt.append("\n请给出专业分析：");
        
        try {
            String analysis = deepseekClient.chat(systemPrompt, userPrompt.toString(), List.of());
            
            if (metricsService != null) {
                int inputTokens = metricsService.estimateTokens(systemPrompt + userPrompt);
                int outputTokens = metricsService.estimateTokens(analysis);
                metricsService.recordCodeGenCall("battle-analysis", inputTokens, outputTokens, 0);
            }
            
            return analysis;
        } catch (Exception e) {
            log.warn("AI 分析生成失败: {}", e.getMessage());
            return null;
        }
    }
    
    // ========== 辅助方法 ==========
    
    private int[][] parseMap(String mapData, int rows, int cols) {
        int[][] map = new int[rows][cols];
        if (mapData == null || mapData.length() < rows * cols) {
            return map;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                map[i][j] = mapData.charAt(i * cols + j) - '0';
            }
        }
        return map;
    }
    
    private int calculateControlArea(int[][] map, List<int[]> path, int rows, int cols) {
        // 简化计算：统计路径覆盖的格子数
        boolean[][] visited = new boolean[rows][cols];
        int count = 0;
        for (int[] pos : path) {
            if (pos[0] >= 0 && pos[0] < rows && pos[1] >= 0 && pos[1] < cols) {
                if (!visited[pos[0]][pos[1]]) {
                    visited[pos[0]][pos[1]] = true;
                    count++;
                }
            }
        }
        return count;
    }
    
    private String analyzeMovementPattern(String steps) {
        if (steps == null || steps.isEmpty()) {
            return "无数据";
        }
        
        int[] counts = new int[4];
        for (char c : steps.toCharArray()) {
            int d = c - '0';
            if (d >= 0 && d < 4) counts[d]++;
        }
        
        int maxDir = 0;
        for (int i = 1; i < 4; i++) {
            if (counts[i] > counts[maxDir]) maxDir = i;
        }
        
        double diversity = 0;
        for (int count : counts) {
            if (count > 0) diversity++;
        }
        diversity /= 4.0;
        
        String pattern = DIRECTION_NAMES[maxDir] + "偏好";
        if (diversity > 0.7) pattern = "全方位移动";
        else if (diversity < 0.3) pattern = "单一方向";
        
        return pattern;
    }
    
    private boolean isDangerousMove(int[] pos, int[][] map, int rows, int cols) {
        // 检查是否接近边界
        if (pos[0] <= 1 || pos[0] >= rows - 2 || pos[1] <= 1 || pos[1] >= cols - 2) {
            return true;
        }
        // 检查周围障碍数量
        int obstacles = 0;
        for (int d = 0; d < 4; d++) {
            int nx = pos[0] + dx[d];
            int ny = pos[1] + dy[d];
            if (nx < 0 || nx >= rows || ny < 0 || ny >= cols || map[nx][ny] == 1) {
                obstacles++;
            }
        }
        return obstacles >= 2;
    }
    
    // ========== 数据类 ==========
    
    public static class BasicStats {
        public int totalRounds;
        public int controlAreaA;
        public int controlAreaB;
        public String movementPatternA;
        public String movementPatternB;
        public String winner;
    }
    
    public static class KeyMoment {
        public int round;
        public String type;
        public String description;
        
        public KeyMoment(int round, String type, String description) {
            this.round = round;
            this.type = type;
            this.description = description;
        }
    }
    
    public static class BattleAnalysisResult {
        private BasicStats stats;
        private List<KeyMoment> keyMoments;
        private String aiAnalysis;
        
        public BattleAnalysisResult(BasicStats stats, List<KeyMoment> keyMoments, String aiAnalysis) {
            this.stats = stats;
            this.keyMoments = keyMoments;
            this.aiAnalysis = aiAnalysis;
        }
        
        public BasicStats getStats() { return stats; }
        public List<KeyMoment> getKeyMoments() { return keyMoments; }
        public String getAiAnalysis() { return aiAnalysis; }
    }
}
