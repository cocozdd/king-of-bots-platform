package com.kob.backend.controller.ai;

import com.kob.backend.service.impl.ai.ABTestRouter;
import com.kob.backend.service.impl.ai.AiMetricsCollector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * A/B 测试管理控制器
 * 
 * 提供 A/B 测试和监控的管理接口：
 * - 查看 A/B 测试状态
 * - 动态调整流量比例
 * - 查看监控指标
 * - 对比 Java vs Python 性能
 */
@RestController
@RequestMapping("/ai/abtest")
public class ABTestController {
    
    private static final Logger log = LoggerFactory.getLogger(ABTestController.class);
    
    @Autowired
    private ABTestRouter abTestRouter;
    
    @Autowired
    private AiMetricsCollector metricsCollector;
    
    /**
     * 获取 A/B 测试状态
     */
    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> getStatus() {
        Map<String, Object> result = new HashMap<>();
        result.put("abtest", abTestRouter.getStats());
        result.put("metrics", metricsCollector.getAllMetrics());
        return ResponseEntity.ok(result);
    }
    
    /**
     * 获取 A/B 测试配置
     */
    @GetMapping("/config")
    public ResponseEntity<Map<String, Object>> getConfig() {
        return ResponseEntity.ok(Map.of(
            "enabled", abTestRouter.isAbTestEnabled(),
            "pythonTrafficPercentage", abTestRouter.getPythonTrafficPercentage(),
            "strategy", abTestRouter.getRoutingStrategy()
        ));
    }
    
    /**
     * 动态更新 Python 流量占比
     * 
     * 用于灰度发布：逐步提升 Python 流量
     * 
     * @param percentage 流量占比 (0-100)
     */
    @PostMapping("/traffic")
    public ResponseEntity<Map<String, Object>> updateTraffic(@RequestParam int percentage) {
        try {
            abTestRouter.updatePythonTrafficPercentage(percentage);
            log.info("[A/B Test] 流量调整: Python {}%", percentage);
            
            return ResponseEntity.ok(Map.of(
                "success", true,
                "message", "Python 流量已更新为 " + percentage + "%",
                "newConfig", Map.of(
                    "enabled", abTestRouter.isAbTestEnabled(),
                    "pythonTrafficPercentage", abTestRouter.getPythonTrafficPercentage()
                )
            ));
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(Map.of(
                "success", false,
                "error", e.getMessage()
            ));
        }
    }
    
    /**
     * 启用/禁用 A/B 测试
     */
    @PostMapping("/toggle")
    public ResponseEntity<Map<String, Object>> toggleABTest(@RequestParam boolean enabled) {
        abTestRouter.setAbTestEnabled(enabled);
        log.info("[A/B Test] 状态切换: {}", enabled ? "启用" : "禁用");
        
        return ResponseEntity.ok(Map.of(
            "success", true,
            "message", "A/B 测试已" + (enabled ? "启用" : "禁用"),
            "enabled", enabled
        ));
    }
    
    /**
     * 获取监控指标
     */
    @GetMapping("/metrics")
    public ResponseEntity<Map<String, Object>> getMetrics() {
        return ResponseEntity.ok(metricsCollector.getAllMetrics());
    }
    
    /**
     * 获取特定后端的指标
     */
    @GetMapping("/metrics/{backend}/{feature}")
    public ResponseEntity<Map<String, Object>> getBackendMetrics(
            @PathVariable String backend,
            @PathVariable String feature) {
        return ResponseEntity.ok(metricsCollector.getMetrics(backend, feature));
    }
    
    /**
     * 重置统计数据
     */
    @PostMapping("/reset")
    public ResponseEntity<Map<String, Object>> resetStats() {
        abTestRouter.resetStats();
        metricsCollector.resetMetrics();
        log.info("[A/B Test] 统计数据已重置");
        
        return ResponseEntity.ok(Map.of(
            "success", true,
            "message", "统计数据已重置"
        ));
    }
    
    /**
     * 灰度发布快捷操作
     * 
     * 支持的阶段：
     * - phase1: 20% Python
     * - phase2: 50% Python
     * - phase3: 80% Python
     * - full: 100% Python (全量)
     * - rollback: 0% Python (回滚)
     */
    @PostMapping("/rollout/{phase}")
    public ResponseEntity<Map<String, Object>> rollout(@PathVariable String phase) {
        int percentage;
        String description;
        
        switch (phase.toLowerCase()) {
            case "phase1":
                percentage = 20;
                description = "灰度阶段1: 20% 流量";
                break;
            case "phase2":
                percentage = 50;
                description = "灰度阶段2: 50% 流量";
                break;
            case "phase3":
                percentage = 80;
                description = "灰度阶段3: 80% 流量";
                break;
            case "full":
                percentage = 100;
                description = "全量发布: 100% 流量";
                break;
            case "rollback":
                percentage = 0;
                description = "回滚: 0% 流量（全部使用 Java）";
                break;
            default:
                return ResponseEntity.badRequest().body(Map.of(
                    "success", false,
                    "error", "未知的发布阶段: " + phase,
                    "validPhases", new String[]{"phase1", "phase2", "phase3", "full", "rollback"}
                ));
        }
        
        abTestRouter.setAbTestEnabled(percentage > 0 && percentage < 100);
        abTestRouter.updatePythonTrafficPercentage(percentage);
        
        log.info("[Rollout] {} - Python {}%", description, percentage);
        
        return ResponseEntity.ok(Map.of(
            "success", true,
            "phase", phase,
            "description", description,
            "pythonTrafficPercentage", percentage,
            "abTestEnabled", abTestRouter.isAbTestEnabled()
        ));
    }
}
