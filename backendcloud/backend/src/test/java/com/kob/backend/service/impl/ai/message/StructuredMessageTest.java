package com.kob.backend.service.impl.ai.message;

import com.alibaba.fastjson.JSONObject;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 结构化消息体系测试
 *
 * 验证Phase 1的核心功能：
 * 1. BaseMessage类型安全
 * 2. 消息转JSON正确性
 * 3. 消息列表管理
 */
public class StructuredMessageTest {

    @Test
    public void testSystemMessage() {
        // 创建System消息
        SystemMessage sysMsg = new SystemMessage("你是AI助手");

        // 验证role固定
        assertEquals("system", sysMsg.getRole());
        assertEquals("你是AI助手", sysMsg.getContent());

        // 验证JSON转换
        JSONObject json = sysMsg.toJSON();
        assertEquals("system", json.getString("role"));
        assertEquals("你是AI助手", json.getString("content"));

        System.out.println("✅ SystemMessage测试通过: " + sysMsg);
    }

    @Test
    public void testHumanMessage() {
        // 创建Human消息
        HumanMessage humanMsg = new HumanMessage("帮我写Bot代码");

        // 验证role固定
        assertEquals("user", humanMsg.getRole());
        assertEquals("帮我写Bot代码", humanMsg.getContent());

        // 验证JSON转换
        JSONObject json = humanMsg.toJSON();
        assertEquals("user", json.getString("role"));

        System.out.println("✅ HumanMessage测试通过: " + humanMsg);
    }

    @Test
    public void testAIMessage() {
        // 创建AI消息
        AIMessage aiMsg = new AIMessage("我建议使用A*算法");

        // 验证role固定
        assertEquals("assistant", aiMsg.getRole());
        assertEquals("我建议使用A*算法", aiMsg.getContent());

        System.out.println("✅ AIMessage测试通过: " + aiMsg);
    }

    @Test
    public void testToolMessage() {
        // 创建Tool消息
        ToolMessage toolMsg = new ToolMessage(
            "{\"result\": \"搜索成功\"}",
            "call_123",
            "knowledge_search"
        );

        // 验证字段
        assertEquals("tool", toolMsg.getRole());
        assertEquals("call_123", toolMsg.getToolCallId());
        assertEquals("knowledge_search", toolMsg.getToolName());
        assertTrue(toolMsg.isSuccess());

        System.out.println("✅ ToolMessage测试通过: " + toolMsg);
    }

    @Test
    public void testMessageList() {
        // 模拟Agent的消息流程
        List<BaseMessage> messages = new ArrayList<>();

        // 1. System定义规则
        messages.add(new SystemMessage("你是KOB平台的AI助手"));

        // 2. User提问
        messages.add(new HumanMessage("帮我写一个贪吃蛇Bot"));

        // 3. AI思考
        messages.add(new AIMessage("我需要先搜索相关策略"));

        // 4. 工具返回结果
        messages.add(new ToolMessage(
            "找到3个相关策略",
            "call_001",
            "knowledge_search"
        ));

        // 5. AI最终回答
        messages.add(new AIMessage("根据策略，这是生成的Bot代码..."));

        // 验证消息列表
        assertEquals(5, messages.size());
        assertEquals("system", messages.get(0).getRole());
        assertEquals("user", messages.get(1).getRole());
        assertEquals("assistant", messages.get(2).getRole());
        assertEquals("tool", messages.get(3).getRole());
        assertEquals("assistant", messages.get(4).getRole());

        // 打印消息流
        System.out.println("\n✅ 消息列表测试通过:");
        for (int i = 0; i < messages.size(); i++) {
            System.out.println("  " + (i+1) + ". " + messages.get(i));
        }
    }

    @Test
    public void testMessageListIsTypeSafe() {
        List<BaseMessage> messages = new ArrayList<>();

        // ✅ 类型安全：IDE会提示，只能添加BaseMessage的子类
        messages.add(new SystemMessage("test"));
        messages.add(new HumanMessage("test"));
        messages.add(new AIMessage("test"));

        // ❌ 以下代码编译时会报错（类型安全保证）
        // messages.add("字符串");  // 编译错误！
        // messages.add(123);      // 编译错误！

        assertTrue(messages.get(0) instanceof SystemMessage);
        assertTrue(messages.get(1) instanceof HumanMessage);
        assertTrue(messages.get(2) instanceof AIMessage);

        System.out.println("✅ 类型安全测试通过");
    }

    @Test
    public void testRoleImmutable() {
        // 验证role不可修改
        SystemMessage msg = new SystemMessage("test");

        // role是final的，无法修改
        assertEquals("system", msg.getRole());

        // 只能修改content
        msg.setContent("new content");
        assertEquals("new content", msg.getContent());
        assertEquals("system", msg.getRole()); // role依然不变

        System.out.println("✅ Role不可变测试通过");
    }
}
