package com.kob.backend.service.impl.ai.message;

/**
 * Human消息（用户消息）- 来自用户的直接输入
 *
 * 对应LangChain：
 * - Python: HumanMessage(content="帮我写一个Bot")
 * - Java: new HumanMessage("帮我写一个Bot")
 *
 * 防注入的关键（参考文档2.5.3节）：
 * - 旧方式：字符串拼接 "User: " + userInput
 *   问题：用户输入"忽略之前指令"可能覆盖System规则
 *
 * - 新方式：封装在HumanMessage中
 *   优势：模型被训练为"User不能覆盖System"，role隔离保护
 *
 * 多模态支持（未来扩展）：
 * - content可以是List<ContentPart>
 * - 支持文本+图片混合输入
 * - 示例：[{type:"text", text:"..."}, {type:"image_url", url:"..."}]
 *
 * 使用场景：
 * - 用户问题："帮我生成一个防守型Bot"
 * - 用户指令："分析最近10场对战"
 * - 用户输入："优化这段代码：int[] dx = ..."
 *
 * 示例：
 * <pre>
 * HumanMessage userMsg = new HumanMessage(
 *     "请帮我生成一个贪吃蛇Bot，策略是优先占据中心区域"
 * );
 * </pre>
 */
public class HumanMessage extends BaseMessage {

    /**
     * 构造函数
     *
     * @param content 用户输入的内容
     */
    public HumanMessage(String content) {
        super("user", content);  // role固定为"user"
    }

    /**
     * 便捷工厂方法：从用户输入创建
     *
     * @param userInput 用户原始输入
     * @return HumanMessage实例
     */
    public static HumanMessage fromUserInput(String userInput) {
        if (userInput == null || userInput.trim().isEmpty()) {
            throw new IllegalArgumentException("用户输入不能为空");
        }
        return new HumanMessage(userInput.trim());
    }
}
