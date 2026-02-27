package com.kob.backend.service.impl.ai.message;

/**
 * AI消息（Assistant消息）- 模型的思考和回复
 *
 * 对应LangChain：
 * - Python: AIMessage(content="我建议使用A*算法")
 * - Java: new AIMessage("我建议使用A*算法")
 *
 * 核心用途（参考文档2.5.2节）：
 * 1. 承载模型的思考过程（Thought）
 * 2. 承载模型的最终回复（Final Answer）
 * 3. 承载工具调用请求（Phase 3扩展）
 *
 * Phase 1 功能范围：
 * - ✅ 支持纯文本回复
 * - ❌ 暂不支持tool_calls（Phase 3实现）
 *
 * Phase 3 扩展预览：
 * <pre>
 * // 未来会添加：
 * private List<ToolCall> toolCalls;  // 工具调用请求列表
 * </pre>
 *
 * 使用场景：
 * - Agent的Thought: "用户想要一个防守型Bot，我需要先搜索相关策略"
 * - Agent的Action决策: "我决定调用knowledge_search工具"
 * - 最终答案: "这是生成的Bot代码：```java\n..."
 *
 * 示例：
 * <pre>
 * // LLM返回的思考
 * AIMessage thought = new AIMessage(
 *     "Thought: 用户需要Bot代码，我先搜索相关模板\n" +
 *     "Action: knowledge_search"
 * );
 *
 * // LLM返回的最终答案
 * AIMessage answer = new AIMessage(
 *     "这是为你生成的Bot代码：\n```java\n..."
 * );
 * </pre>
 */
public class AIMessage extends BaseMessage {

    /**
     * 构造函数
     *
     * @param content AI生成的内容
     */
    public AIMessage(String content) {
        super("assistant", content);  // role固定为"assistant"
    }

    /**
     * 便捷工厂方法：从LLM响应创建
     *
     * @param llmResponse LLM的原始响应
     * @return AIMessage实例
     */
    public static AIMessage fromLLMResponse(String llmResponse) {
        if (llmResponse == null) {
            llmResponse = "";
        }
        return new AIMessage(llmResponse.trim());
    }

    // ========== Phase 3 扩展预留 ==========
    // 未来会添加：
    // private List<ToolCall> toolCalls;
    // public List<ToolCall> getToolCalls() { ... }
    // public boolean hasToolCalls() { return toolCalls != null && !toolCalls.isEmpty(); }
}
