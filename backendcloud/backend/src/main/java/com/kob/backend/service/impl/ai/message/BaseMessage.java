package com.kob.backend.service.impl.ai.message;

import com.alibaba.fastjson.JSONObject;

/**
 * 结构化消息基类 - 符合2026年LangChain标准
 *
 * 设计理念（参考LangChain_深度解析与实战.md 第2.5.2节）：
 * 1. 类型安全：role由子类固定，编译时检查，避免写错
 * 2. 角色明确：System/User/Assistant/Tool清晰分离
 * 3. 防注入：User消息无法覆盖System规则（模型训练层面的保护）
 *
 * 为什么不用JSONObject？
 * - ❌ JSONObject: role可以随意修改，容易出错，IDE无提示
 * - ✅ BaseMessage: role由类型保证，IDE有完整提示
 *
 * 对应LangChain概念：
 * - Python: from langchain_core.messages import BaseMessage
 * - Java: 本类
 */
public abstract class BaseMessage {

    /**
     * 消息角色（system/user/assistant/tool）
     * 由子类构造函数传入，创建后不可修改
     */
    private final String role;

    /**
     * 消息内容
     * - 对于System/User/Assistant: 文本内容
     * - 对于Tool: 工具返回结果（可以是JSON字符串）
     * - 未来可扩展为Object以支持多模态（图片、音频等）
     */
    private String content;

    /**
     * 构造函数（protected，只允许子类调用）
     *
     * @param role 消息角色（由子类固定）
     * @param content 消息内容
     */
    protected BaseMessage(String role, String content) {
        if (role == null || role.isEmpty()) {
            throw new IllegalArgumentException("role不能为空");
        }
        this.role = role;
        this.content = content;
    }

    /**
     * 获取消息角色
     */
    public String getRole() {
        return role;
    }

    /**
     * 获取消息内容
     */
    public String getContent() {
        return content;
    }

    /**
     * 设置消息内容
     * 注意：role不可修改，只有content可以修改
     */
    public void setContent(String content) {
        this.content = content;
    }

    /**
     * 转换为DeepSeek API需要的JSONObject格式
     *
     * 输出格式：{"role": "xxx", "content": "xxx"}
     *
     * @return JSONObject
     */
    public JSONObject toJSON() {
        JSONObject json = new JSONObject();
        json.put("role", this.role);
        json.put("content", this.content);
        return json;
    }

    /**
     * 调试用：输出可读格式
     * 格式：[ROLE] content...
     */
    @Override
    public String toString() {
        String displayContent = content != null && content.length() > 100
            ? content.substring(0, 100) + "..."
            : content;
        return String.format("[%s] %s", role.toUpperCase(), displayContent);
    }
}
