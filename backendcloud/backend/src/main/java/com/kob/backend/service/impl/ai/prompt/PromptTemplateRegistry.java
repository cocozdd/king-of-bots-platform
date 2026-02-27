package com.kob.backend.service.impl.ai.prompt;

import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Prompt 模板注册中心
 *
 * 使用 LangChain4j PromptTemplate 统一管理所有 Prompt 模板
 *
 * 面试亮点：
 * - 选择性集成 LangChain4j：只用框架的 PromptTemplate 抽象
 * - 集中管理：便于 A/B 测试和版本控制
 * - 模板化：支持变量替换，避免硬编码
 */
@Component
public class PromptTemplateRegistry {

    private static final Logger log = LoggerFactory.getLogger(PromptTemplateRegistry.class);

    private final Map<String, PromptTemplate> templates = new ConcurrentHashMap<>();

    @PostConstruct
    public void init() {
        // Bot 代码生成
        templates.put("bot_code_generation", PromptTemplate.from("""
            你是一个专业的 Java 游戏 Bot 开发专家，专注于 King of Bots (KOB) 贪吃蛇对战平台。
            用户会描述他们想要的 Bot 策略，你需要生成对应的 Java 代码片段。

            【重要】本游戏规则（必须遵守）：
            - 这是一个双人贪吃蛇对战游戏，地图是 13x14 的网格
            - 【没有食物】蛇的长度是自动增长的：前10步每步增长1格，之后每3步增长1格
            - 1表示障碍物（墙），0表示可通行
            - 两条蛇从对角出发，每回合同时移动
            - 获胜条件：让对手撞墙或撞到蛇身（自己或对手的）
            - 不要提及"食物"、"吃食物"等概念，本游戏没有食物

            代码要求：
            - 只输出策略逻辑代码，不要完整类
            - 可用变量：g(地图), myPos(我的位置), opPos(对手位置), dx/dy(方向数组)
            - 返回方向：0-上, 1-右, 2-下, 3-左
            - 代码风格：{{style}}

            用户需求: {{description}}
            请生成 Java 代码，用 ```java ``` 包裹。
            """));

        // RAG 问答
        templates.put("rag_qa", PromptTemplate.from("""
            你是 King of Bots (KOB) 贪吃蛇对战平台的 Bot 开发助手，基于提供的参考文档回答问题。

            【重要】本项目的游戏规则（必须遵守，不要使用通用贪吃蛇知识）：
            1. 这是一个双人贪吃蛇对战游戏，两条蛇在 13x14 的地图上对战
            2. 本游戏【没有食物】！蛇的长度是自动增长的：
               - 前10步每步增长1格
               - 之后每3步增长1格（step % 3 == 1 时增长）
            3. 获胜条件：让对手撞墙或撞到蛇身（自己或对手的）
            4. 移动方向：0=上, 1=右, 2=下, 3=左
            5. 地图坐标从(0,0)开始，有固定的障碍物墙壁

            参考文档:
            {{contexts}}

            用户问题: {{question}}

            内容要求：
            - 回答要简洁准确，基于本项目的实际规则
            - 引用相关文档
            - 不要提及"食物"、"吃食物"等概念
            - 不知道就说不知道
            - 使用 Markdown 格式输出
            """));

        // 对话摘要生成
        templates.put("conversation_summary", PromptTemplate.from("""
            请对以下对话进行摘要，提取关键信息。

            对话历史:
            {{history}}

            要求：
            1. 用一两句话概括对话主题
            2. 提取讨论的关键技术点
            3. 不要遗漏重要信息

            输出格式：简洁的中文摘要
            """));

        // 偏好提取
        templates.put("preference_extraction", PromptTemplate.from("""
            分析以下对话，提取用户的偏好信息。

            对话历史:
            {{history}}

            请识别并提取以下类型的偏好：
            1. 代码风格偏好（简洁/详细/注释丰富等）
            2. 策略偏好（进攻/防守/平衡等）
            3. 算法偏好（BFS/DFS/A*/贪心等）
            4. 交互偏好（详细解释/简洁回答等）

            返回 JSON 格式:
            {
                "preferences": [
                    {"type": "strategy", "value": "进攻型", "confidence": 0.8},
                    {"type": "code_style", "value": "注释丰富", "confidence": 0.6}
                ]
            }

            如果没有发现明确偏好，返回空数组。
            只返回 JSON，不要其他内容。
            """));

        // 代码分析
        templates.put("code_analysis", PromptTemplate.from("""
            你是一个专业的代码审查专家，专注于 King of Bots (KOB) 贪吃蛇对战游戏 Bot 开发。

            【重要】本游戏规则：
            - 双人贪吃蛇对战，13x14 地图
            - 【没有食物】蛇自动增长：前10步每步增长，之后每3步增长1格
            - 获胜条件：让对手撞墙或撞蛇身
            - 不要提及"食物"等概念

            请分析用户的 Bot 代码，提供以下方面的建议：

            1. **策略分析**：当前策略的优缺点
            2. **性能优化**：算法效率、内存使用
            3. **Bug 风险**：潜在的问题和边界情况
            4. **改进建议**：具体的优化方向

            用户代码:
            ```java
            {{code}}
            ```

            输出格式：
            ## 策略分析
            ...

            ## 性能评估
            ...

            ## 潜在问题
            ...

            ## 改进建议
            ...
            """));

        // 代码修复
        templates.put("code_fix", PromptTemplate.from("""
            你是一个专业的 Java 调试专家，专注于 King of Bots (KOB) 贪吃蛇对战游戏 Bot 开发。
            用户会提供有 Bug 的代码和错误日志，请帮助修复。

            【重要】本游戏规则：
            - 双人贪吃蛇对战，13x14 地图，【没有食物】
            - 蛇自动增长：前10步每步增长，之后每3步增长1格
            - 获胜条件：让对手撞墙或撞蛇身

            有问题的代码:
            ```java
            {{code}}
            ```

            错误信息:
            {{error}}

            要求：
            1. 分析错误原因
            2. 提供修复后的完整代码
            3. 解释修改的地方

            输出格式：
            ## 错误分析
            ...

            ## 修复后代码
            ```java
            ...
            ```

            ## 修改说明
            ...
            """));

        log.info("PromptTemplateRegistry 初始化完成: 注册 {} 个模板", templates.size());
    }

    /**
     * 获取模板
     */
    public PromptTemplate get(String name) {
        PromptTemplate template = templates.get(name);
        if (template == null) {
            log.warn("未找到 Prompt 模板: {}", name);
        }
        return template;
    }

    /**
     * 应用模板并生成 Prompt
     */
    public Prompt apply(String name, Map<String, Object> variables) {
        PromptTemplate template = get(name);
        if (template == null) {
            throw new IllegalArgumentException("Unknown template: " + name);
        }
        return template.apply(variables);
    }

    /**
     * 应用模板并返回文本
     */
    public String applyAsText(String name, Map<String, Object> variables) {
        return apply(name, variables).text();
    }

    /**
     * 注册新模板（支持运行时动态添加）
     */
    public void register(String name, String templateText) {
        templates.put(name, PromptTemplate.from(templateText));
        log.info("注册新 Prompt 模板: {}", name);
    }

    /**
     * 获取所有模板名称
     */
    public java.util.Set<String> getTemplateNames() {
        return templates.keySet();
    }
}
