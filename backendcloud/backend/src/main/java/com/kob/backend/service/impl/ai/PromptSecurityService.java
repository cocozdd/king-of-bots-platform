package com.kob.backend.service.impl.ai;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Prompt 注入防护服务
 * 
 * 功能:
 * 1. 检测可疑的 Prompt Injection 模式
 * 2. 输入清洗和转义
 * 3. 长度限制
 * 4. 记录可疑请求
 * 
 * @author KOB Team
 */
@Service
public class PromptSecurityService {
    
    private static final Logger log = LoggerFactory.getLogger(PromptSecurityService.class);
    
    // 单个字段最大长度限制
    private static final int MAX_QUESTION_LENGTH = 2000;
    private static final int MAX_CODE_SNIPPET_LENGTH = 5000;
    private static final int MAX_ERROR_LOG_LENGTH = 3000;
    
    // 可疑模式列表（常见 Prompt Injection 攻击）
    private static final List<String> SUSPICIOUS_PATTERNS = Arrays.asList(
            // 直接指令覆盖
            "忽略之前", "ignore previous", "ignore above", "disregard",
            "forget everything", "忘记所有", "重新开始", "reset instructions",
            
            // 角色劫持
            "你现在是", "you are now", "act as", "pretend to be",
            "扮演", "assume the role",
            
            // 系统提示词泄露
            "显示系统提示", "show system prompt", "reveal instructions",
            "print instructions", "输出提示词", "泄露指令",
            
            // 输出格式劫持
            "返回json", "output as json", "format output",
            "以管理员身份", "as administrator", "with admin privileges",
            
            // 代码注入尝试
            "```system", "```assistant", "```user",
            "<|im_start|>", "<|im_end|>", "<system>", "</system>"
    );
    
    // 编译后的正则模式（性能优化）
    private static final List<Pattern> SUSPICIOUS_REGEX_PATTERNS = Arrays.asList(
            Pattern.compile("(?i)ignore\\s+(previous|above|all|everything)"),
            Pattern.compile("(?i)(act|pretend)\\s+as\\s+"),
            Pattern.compile("(?i)(system|admin|root)\\s+(prompt|instructions?)"),
            Pattern.compile("(?i)you\\s+are\\s+now\\s+"),
            Pattern.compile("(?i)forget\\s+(all|everything)"),
            Pattern.compile("(?i)输出.*(密码|api.*key|token|secret)"),
            Pattern.compile("(?i)show\\s+me\\s+(your|the)\\s+(prompt|instructions?)")
    );
    
    /**
     * 安全验证结果
     */
    public static class ValidationResult {
        private final boolean safe;
        private final String reason;
        private final String sanitizedInput;
        
        public ValidationResult(boolean safe, String reason, String sanitizedInput) {
            this.safe = safe;
            this.reason = reason;
            this.sanitizedInput = sanitizedInput;
        }
        
        public boolean isSafe() {
            return safe;
        }
        
        public String getReason() {
            return reason;
        }
        
        public String getSanitizedInput() {
            return sanitizedInput;
        }
        
        public static ValidationResult safe(String sanitizedInput) {
            return new ValidationResult(true, null, sanitizedInput);
        }
        
        public static ValidationResult unsafe(String reason) {
            return new ValidationResult(false, reason, null);
        }
    }
    
    /**
     * 验证问题输入
     * 
     * @param question 用户问题
     * @return 验证结果
     */
    public ValidationResult validateQuestion(String question) {
        if (question == null || question.trim().isEmpty()) {
            return ValidationResult.unsafe("问题不能为空");
        }
        
        // 长度检查
        if (question.length() > MAX_QUESTION_LENGTH) {
            log.warn("问题长度超限: {} > {}", question.length(), MAX_QUESTION_LENGTH);
            return ValidationResult.unsafe("问题长度不能超过 " + MAX_QUESTION_LENGTH + " 字符");
        }
        
        // 可疑模式检测
        String lowerQuestion = question.toLowerCase();
        for (String pattern : SUSPICIOUS_PATTERNS) {
            if (lowerQuestion.contains(pattern.toLowerCase())) {
                log.warn("检测到可疑模式: {} in question", pattern);
                return ValidationResult.unsafe("输入包含不允许的内容");
            }
        }
        
        // 正则模式检测
        for (Pattern pattern : SUSPICIOUS_REGEX_PATTERNS) {
            if (pattern.matcher(question).find()) {
                log.warn("检测到可疑正则模式: {}", pattern.pattern());
                return ValidationResult.unsafe("输入包含不允许的内容");
            }
        }
        
        // 清洗输入（移除多余空白、特殊字符）
        String sanitized = sanitizeInput(question);
        
        return ValidationResult.safe(sanitized);
    }
    
    /**
     * 验证代码片段
     * 
     * @param codeSnippet 代码片段
     * @return 验证结果
     */
    public ValidationResult validateCodeSnippet(String codeSnippet) {
        if (codeSnippet == null || codeSnippet.trim().isEmpty()) {
            // 代码片段可以为空
            return ValidationResult.safe("");
        }
        
        // 长度检查
        if (codeSnippet.length() > MAX_CODE_SNIPPET_LENGTH) {
            log.warn("代码片段长度超限: {} > {}", codeSnippet.length(), MAX_CODE_SNIPPET_LENGTH);
            return ValidationResult.unsafe("代码片段长度不能超过 " + MAX_CODE_SNIPPET_LENGTH + " 字符");
        }
        
        // 代码片段较宽松，只检查明显的系统指令
        String lower = codeSnippet.toLowerCase();
        if (lower.contains("system prompt") || lower.contains("系统提示")) {
            log.warn("代码片段包含可疑内容");
            return ValidationResult.unsafe("代码片段包含不允许的内容");
        }
        
        return ValidationResult.safe(codeSnippet.trim());
    }
    
    /**
     * 验证错误日志
     * 
     * @param errorLog 错误日志
     * @return 验证结果
     */
    public ValidationResult validateErrorLog(String errorLog) {
        if (errorLog == null || errorLog.trim().isEmpty()) {
            // 错误日志可以为空
            return ValidationResult.safe("");
        }
        
        // 长度检查
        if (errorLog.length() > MAX_ERROR_LOG_LENGTH) {
            log.warn("错误日志长度超限: {} > {}", errorLog.length(), MAX_ERROR_LOG_LENGTH);
            return ValidationResult.unsafe("错误日志长度不能超过 " + MAX_ERROR_LOG_LENGTH + " 字符");
        }
        
        return ValidationResult.safe(errorLog.trim());
    }
    
    /**
     * 清洗输入文本
     * 
     * @param input 原始输入
     * @return 清洗后的文本
     */
    private String sanitizeInput(String input) {
        if (input == null) {
            return "";
        }
        
        // 1. 去除前后空白
        String cleaned = input.trim();
        
        // 2. 规范化空白字符（多个空格/换行合并为单个）
        cleaned = cleaned.replaceAll("\\s+", " ");
        
        // 3. 移除控制字符（保留换行和制表符）
        cleaned = cleaned.replaceAll("[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]", "");
        
        // 4. 移除零宽字符（可能用于隐藏注入）
        cleaned = cleaned.replaceAll("[\u200B-\u200D\uFEFF]", "");
        
        return cleaned;
    }
    
    /**
     * 记录可疑请求（用于安全审计）
     *
     * @param input 可疑输入
     * @param reason 拦截原因
     */
    public void logSuspiciousRequest(String input, String reason) {
        log.warn("🚨 安全拦截 - 原因: {}, 输入: {}", reason,
                 input.length() > 100 ? input.substring(0, 100) + "..." : input);
    }

    // ========================================
    // 鲁棒性检测方法（统一供各 RAG 服务调用）
    // ========================================

    // KOB/Bot 开发相关关键词
    private static final List<String> DOMAIN_KEYWORDS = Arrays.asList(
            "bot", "蛇", "snake", "移动", "策略", "算法", "寻路", "代码",
            "java", "python", "游戏", "地图", "对战", "kob", "超时", "timeout",
            "战斗", "编程", "函数", "方法", "api", "接口", "逻辑", "调试",
            "bug", "错误", "配置", "运行", "编译", "bfs", "dfs", "路径",
            "坐标", "方向", "障碍", "墙", "身体", "长度", "回合", "步数"
    );

    // 明显的超域问题模式
    private static final List<String> OUT_OF_DOMAIN_PATTERNS = Arrays.asList(
            "红烧肉", "做菜", "食谱", "烹饪", "天气", "股票", "新闻",
            "电影", "音乐", "旅游", "购物", "价格", "怎么做饭", "菜谱",
            "明星", "八卦", "体育比赛", "足球", "篮球", "政治", "经济形势"
    );

    // 非技术领域指示词
    private static final List<String> NON_TECH_INDICATORS = Arrays.asList(
            "总统", "president", "价格", "天气", "股票", "新闻", "明星",
            "电影", "音乐", "体育", "政治", "经济", "历史人物", "哪一年"
    );

    // 多跳推理模式词
    private static final List<String> MULTI_HOP_PATTERNS = Arrays.asList(
            "那年", "当时", "同时", "之后", "之前", "期间", "发布时", "那时候"
    );

    /**
     * 检测是否为超域问题（Out of Domain）
     *
     * @param query 用户查询
     * @return true 如果是超域问题
     */
    public boolean isOutOfDomain(String query) {
        if (query == null || query.trim().isEmpty()) {
            return false;
        }

        String lowerQuery = query.toLowerCase();

        // 检查是否包含任何领域关键词
        boolean hasDomainKeyword = DOMAIN_KEYWORDS.stream()
                .anyMatch(kw -> lowerQuery.contains(kw.toLowerCase()));

        if (hasDomainKeyword) {
            return false; // 包含领域关键词，不是超域
        }

        // 检查是否包含明显的超域模式
        boolean hasOodPattern = OUT_OF_DOMAIN_PATTERNS.stream()
                .anyMatch(pattern -> lowerQuery.contains(pattern.toLowerCase()));

        if (hasOodPattern) {
            log.info("[OOD检测] 检测到超域模式: {}", truncate(query, 50));
            return true;
        }

        // 无领域关键词且查询长度较长，判定为超域
        if (query.length() > 10) {
            log.info("[OOD检测] 无领域关键词: {}", truncate(query, 50));
            return true;
        }

        return false;
    }

    /**
     * 检测是否为非技术性多跳推理问题
     *
     * @param query 用户查询
     * @return true 如果是非技术性多跳问题
     */
    public boolean isNonTechnicalMultiHop(String query) {
        if (query == null || query.trim().isEmpty()) {
            return false;
        }

        String lowerQuery = query.toLowerCase();

        boolean hasNonTechIndicator = NON_TECH_INDICATORS.stream()
                .anyMatch(indicator -> lowerQuery.contains(indicator.toLowerCase()));

        boolean hasMultiHopPattern = MULTI_HOP_PATTERNS.stream()
                .anyMatch(pattern -> lowerQuery.contains(pattern.toLowerCase()));

        if (hasNonTechIndicator && hasMultiHopPattern) {
            log.info("[多跳检测] 检测到非技术多跳推理: {}", truncate(query, 50));
            return true;
        }

        return false;
    }

    /**
     * 检测是否为提示注入攻击
     *
     * @param query 用户查询
     * @return true 如果检测到注入攻击
     */
    public boolean isPromptInjection(String query) {
        if (query == null || query.trim().isEmpty()) {
            return false;
        }

        String lowerQuery = query.toLowerCase();

        // 检查关键词模式
        String[] injectionPatterns = {
            "忽略", "ignore", "忘记", "forget", "密码", "password",
            "密钥", "secret", "api key", "token", "credential",
            "system prompt", "系统提示", "角色扮演", "假装你是",
            "jailbreak", "越狱", "绕过限制"
        };

        for (String pattern : injectionPatterns) {
            if (lowerQuery.contains(pattern.toLowerCase())) {
                log.warn("[注入检测] 检测到可疑模式: {} in query: {}", pattern, truncate(query, 50));
                return true;
            }
        }

        // 检查正则模式
        for (Pattern pattern : SUSPICIOUS_REGEX_PATTERNS) {
            if (pattern.matcher(query).find()) {
                log.warn("[注入检测] 正则匹配: {}", pattern.pattern());
                return true;
            }
        }

        return false;
    }

    /**
     * 综合安全检查（包含所有检测）
     *
     * @param query 用户查询
     * @return 安全检查结果
     */
    public SecurityCheckResult performFullSecurityCheck(String query) {
        // 1. 基础验证
        ValidationResult basicValidation = validateQuestion(query);
        if (!basicValidation.isSafe()) {
            return SecurityCheckResult.rejected(basicValidation.getReason(), "VALIDATION_FAILED");
        }

        // 2. 提示注入检测
        if (isPromptInjection(query)) {
            logSuspiciousRequest(query, "提示注入攻击");
            return SecurityCheckResult.rejected(
                "抱歉，我无法执行此类请求。我只能回答与 KOB Bot 开发相关的技术问题。",
                "PROMPT_INJECTION"
            );
        }

        // 3. 非技术多跳检测
        if (isNonTechnicalMultiHop(query)) {
            logSuspiciousRequest(query, "非技术多跳推理");
            return SecurityCheckResult.rejected(
                "抱歉，该问题超出了 KOB 技术问答的范围，我无法回答与 KOB/Bot 开发无关的问题。",
                "NON_TECH_MULTIHOP"
            );
        }

        // 4. 超域检测
        if (isOutOfDomain(query)) {
            logSuspiciousRequest(query, "超域问题");
            return SecurityCheckResult.rejected(
                "抱歉，我只能回答与 KOB Bot 开发相关的技术问题。如果您有关于 Bot 编写、游戏策略或平台使用的问题，欢迎提问！",
                "OUT_OF_DOMAIN"
            );
        }

        return SecurityCheckResult.passed(basicValidation.getSanitizedInput());
    }

    /**
     * 综合安全检查结果
     */
    public static class SecurityCheckResult {
        private final boolean passed;
        private final String rejectReason;
        private final String rejectType;
        private final String sanitizedInput;

        private SecurityCheckResult(boolean passed, String rejectReason, String rejectType, String sanitizedInput) {
            this.passed = passed;
            this.rejectReason = rejectReason;
            this.rejectType = rejectType;
            this.sanitizedInput = sanitizedInput;
        }

        public static SecurityCheckResult passed(String sanitizedInput) {
            return new SecurityCheckResult(true, null, null, sanitizedInput);
        }

        public static SecurityCheckResult rejected(String reason, String type) {
            return new SecurityCheckResult(false, reason, type, null);
        }

        public boolean isPassed() { return passed; }
        public String getRejectReason() { return rejectReason; }
        public String getRejectType() { return rejectType; }
        public String getSanitizedInput() { return sanitizedInput; }
    }

    private String truncate(String text, int maxLen) {
        if (text == null) return "";
        return text.length() > maxLen ? text.substring(0, maxLen) + "..." : text;
    }
}
