package com.kob.backend.service.impl.ai.message;

/**
 * Tool消息 - 工具执行结果的反馈
 *
 * 对应LangChain：
 * - Python: ToolMessage(content="查询结果：...", tool_call_id="call_123")
 * - Java: new ToolMessage("查询结果：...", "call_123")
 *
 * 核心机制（参考文档2.5.3节）：
 * 工具调用闭环流程：
 * 1. AIMessage包含tool_calls → [{"id": "call_123", "name": "search", "args": {...}}]
 * 2. 执行工具
 * 3. ToolMessage返回结果 → {"tool_call_id": "call_123", "content": "结果..."}
 * 4. 再次调用LLM，LLM根据结果继续推理
 *
 * tool_call_id的作用：
 * - 精准绑定：一个工具调用对应一个结果
 * - 支持并行调用：多个工具同时执行，通过ID区分
 * - 防止混乱：确保LLM知道哪个结果对应哪个工具
 *
 * Phase 1 状态：
 * - 类已创建，但暂未使用
 * - Phase 3 才会在Agent中真正使用
 *
 * Phase 3 使用示例（预览）：
 * <pre>
 * // LLM返回工具调用请求
 * AIMessage aiMsg = new AIMessage("");
 * aiMsg.addToolCall(new ToolCall("call_123", "knowledge_search", args));
 *
 * // 执行工具
 * String result = toolRegistry.execute("knowledge_search", args);
 *
 * // 返回工具结果
 * ToolMessage toolMsg = new ToolMessage(result, "call_123", "knowledge_search");
 *
 * // 将AIMessage和ToolMessage都加入消息列表
 * messages.add(aiMsg);
 * messages.add(toolMsg);
 *
 * // 再次调用LLM
 * AIMessage finalAnswer = llm.chat(messages);
 * </pre>
 */
public class ToolMessage extends BaseMessage {

    /**
     * 工具调用ID - 关键字段！
     * 必须与AIMessage中的tool_call_id匹配
     */
    private String toolCallId;

    /**
     * 工具名称（辅助字段，方便调试）
     */
    private String toolName;

    /**
     * 工具执行是否成功
     */
    private boolean success;

    /**
     * 构造函数
     *
     * @param content 工具返回的结果（可以是JSON字符串）
     * @param toolCallId 工具调用ID（必须与AIMessage的tool_call_id匹配）
     * @param toolName 工具名称
     */
    public ToolMessage(String content, String toolCallId, String toolName) {
        super("tool", content);  // role固定为"tool"
        this.toolCallId = toolCallId;
        this.toolName = toolName;
        this.success = true;  // 默认成功
    }

    /**
     * 创建成功的工具消息
     */
    public static ToolMessage success(String toolCallId, String toolName, String result) {
        return new ToolMessage(result, toolCallId, toolName);
    }

    /**
     * 创建失败的工具消息
     */
    public static ToolMessage error(String toolCallId, String toolName, String errorMessage) {
        ToolMessage msg = new ToolMessage(
            "{\"error\": \"" + errorMessage + "\"}",
            toolCallId,
            toolName
        );
        msg.success = false;
        return msg;
    }

    // ========== Getters ==========

    public String getToolCallId() {
        return toolCallId;
    }

    public String getToolName() {
        return toolName;
    }

    public boolean isSuccess() {
        return success;
    }

    @Override
    public String toString() {
        return String.format("[TOOL: %s, ID: %s, Success: %s] %s",
            toolName,
            toolCallId,
            success,
            getContent() != null && getContent().length() > 50
                ? getContent().substring(0, 50) + "..."
                : getContent()
        );
    }
}
