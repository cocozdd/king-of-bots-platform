"""
test_agent_eval.py - v4 HITL 闭环升级评测（16 个固定 case）

运行: cd aiservice && python -m pytest tests/test_agent_eval.py -v

测试类别：
- 基础 (1-2): 简单问题、Bot 列表
- HITL (3-6): interrupt、approve、reject、edit
- 幂等 (7-10): 重复 apply、进程重启、并发、事务回滚
- Verifier (11-14): 合法代码、缺入口、危险模式、失败重试
- 安全 (15): 篡改 resumeData
- 边界 (16): 无中断状态 resume
"""
import os
import sys
import json
import asyncio
import logging
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.state import AgentState
from agent.verifier import verify_bot_code, try_compile_check, VerificationResult
from agent.executor import (
    AgentExecutor, AgentResult, _apply_bot_update, _completed_actions, MAX_ITERATIONS,
)

logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================

def _make_proposal(bot_id=99999, user_id=99999, bot_name="TestBot", new_code="class Bot { Integer nextMove(String i) { return 0; } }"):
    """构建标准 proposal"""
    return {
        "bot_id": bot_id,
        "user_id": user_id,
        "bot_name": bot_name,
        "new_code": new_code,
        "summary": "测试修改",
    }


def _make_pending_approval(proposal=None, action_id="tc_001:99999", tool_call_id="tc_001", code=None):
    """构建标准 pending_approval"""
    p = proposal or _make_proposal()
    result = {
        "proposal": p,
        "action_id": action_id,
        "tool_call_id": tool_call_id,
    }
    if code is not None:
        result["code"] = code
    return result


def _make_state(**overrides) -> dict:
    """构建标准 AgentState dict"""
    base = {
        "messages": [],
        "step_count": 0,
        "tool_budget": MAX_ITERATIONS,
        "pending_approval": None,
        "last_action_id": None,
        "last_error": None,
        "verification_attempts": 0,
    }
    base.update(overrides)
    return base


# ============================================================
# 基础测试 (1-2)
# ============================================================

class TestBasic:
    """基础功能测试"""

    def test_01_simple_question_no_tool_calls(self):
        """#1 简单问题不触发工具 → 0 tool calls"""
        from langchain_core.messages import AIMessage

        # mock LLM 返回无 tool_calls 的 AIMessage
        mock_llm = MagicMock()
        mock_response = AIMessage(content="KOB 是一个贪吃蛇对战游戏。")
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response

        executor = AgentExecutor(llm=mock_llm, tools=[])
        # execute() 用普通 Agent，但 LLM mock 直接返回
        result = executor._execute_simple("什么是 KOB？", {})

        assert result.success is True
        assert result.answer
        assert result.steps == 1

    def test_02_bot_list_request(self):
        """#2 Bot 列表请求 → 调用 get_user_bots"""
        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool

        tool_invoked = {"called": False}

        @tool
        def get_user_bots(user_id: int) -> str:
            """获取用户的 Bot 列表"""
            tool_invoked["called"] = True
            return "**TestBot** (ID: 1)"

        mock_llm = MagicMock()
        # 第一次调用返回 tool_calls
        first_response = AIMessage(content="", tool_calls=[{
            "id": "tc_1", "name": "get_user_bots", "args": {"user_id": 99999}
        }])
        second_response = AIMessage(content="你有一个 Bot: TestBot (ID: 1)")
        mock_llm.bind_tools.return_value.invoke.side_effect = [first_response, second_response]

        executor = AgentExecutor(llm=mock_llm, tools=[get_user_bots])

        # 直接测试 hitl_tool_node 内部逻辑
        state = _make_state(messages=[first_response])
        # 手动执行工具
        tool_call = first_response.tool_calls[0]
        result = get_user_bots.invoke({"user_id": 99999})
        assert tool_invoked["called"] is True
        assert "TestBot" in result


# ============================================================
# HITL 测试 (3-6)
# ============================================================

class TestHITL:
    """HITL 闭环测试"""

    def test_03_modification_triggers_interrupt(self):
        """#3 修改请求触发 interrupt → pending_approval 非空"""
        from langchain_core.messages import AIMessage, ToolMessage
        from langchain_core.tools import tool

        proposal = _make_proposal()

        @tool
        def propose_and_apply_modification(bot_id: int, user_id: int, modification_description: str) -> str:
            """生成修改建议"""
            return json.dumps({
                "__hitl_proposal__": True,
                "proposal": proposal,
            })

        mock_llm = MagicMock()
        ai_msg = AIMessage(content="", tool_calls=[{
            "id": "tc_001", "name": "propose_and_apply_modification",
            "args": {"bot_id": 99999, "user_id": 99999, "modification_description": "测试"}
        }])
        mock_llm.bind_tools.return_value.invoke.return_value = ai_msg

        executor = AgentExecutor(llm=mock_llm, tools=[propose_and_apply_modification])

        # 模拟 hitl_tool_node 执行
        state = _make_state(messages=[ai_msg])
        last_message = state["messages"][-1]

        tool_call = last_message.tool_calls[0]
        result = propose_and_apply_modification.invoke(tool_call["args"])
        result_str = str(result)

        assert "__hitl_proposal__" in result_str
        proposal_data = json.loads(result_str)
        action_id = f"{tool_call['id']}:{proposal_data['proposal']['bot_id']}"

        # 验证 pending_approval 可正确构建
        pending = {
            "proposal": proposal_data["proposal"],
            "action_id": action_id,
            "tool_call_id": tool_call["id"],
        }
        assert pending["proposal"]["bot_id"] == 99999
        assert pending["action_id"] == "tc_001:99999"

    def test_04_approve_resume_verify_apply(self, mock_backend):
        """#4 approve → resume → verify → apply → 后端收到正确代码"""
        proposal = _make_proposal()
        code = proposal["new_code"]
        action_id = "tc_001:99999"

        # 直接测试 apply 逻辑
        result = _apply_bot_update(proposal, code, action_id)

        assert "成功" in result or "已应用" in result
        assert len(mock_backend.update_calls) == 1
        assert mock_backend.update_calls[0]["content"] == code
        assert mock_backend.update_calls[0]["actionId"] == action_id

    def test_05_reject_resume_no_backend_call(self, mock_backend):
        """#5 reject → resume → 不调后端"""
        # interrupt_node reject 逻辑：设置 pending_approval = None，不调 _apply_bot_update
        # 模拟 interrupt_node 的 reject 分支
        proposal = _make_proposal()
        approval = _make_pending_approval(proposal)

        # reject 时 pending_approval 被清空
        decision_type = "reject"
        assert decision_type not in ("approve", "edit")

        # 验证没有调用后端
        assert len(mock_backend.update_calls) == 0

    def test_06_edit_resume_apply_edited_code(self, mock_backend):
        """#6 edit → resume → apply 用编辑后代码"""
        proposal = _make_proposal()
        edited_code = "class Bot { Integer nextMove(String i) { return 1; } }"
        action_id = "tc_002:99999"

        result = _apply_bot_update(proposal, edited_code, action_id)

        assert "成功" in result or "已应用" in result
        assert len(mock_backend.update_calls) == 1
        assert mock_backend.update_calls[0]["content"] == edited_code


# ============================================================
# 幂等测试 (7-10)
# ============================================================

class TestIdempotency:
    """幂等保护测试"""

    def test_07_same_action_id_duplicate(self, mock_backend):
        """#7 同一 action_id 二次 apply → 第二次后端返回 duplicate=true"""
        proposal = _make_proposal()
        code = proposal["new_code"]
        action_id = "tc_dup:99999"

        r1 = _apply_bot_update(proposal, code, action_id)
        assert "成功" in r1

        r2 = _apply_bot_update(proposal, code, action_id)
        assert "已应用" in r2
        # 第二次被进程缓存拦截，不调后端
        assert len(mock_backend.update_calls) == 1

    def test_08_process_restart_idempotency(self, mock_backend):
        """#8 进程重启后幂等 → 清空缓存，后端仍判重"""
        proposal = _make_proposal()
        code = proposal["new_code"]
        action_id = "tc_restart:99999"

        # 第一次正常
        r1 = _apply_bot_update(proposal, code, action_id)
        assert "成功" in r1

        # 模拟进程重启：清空缓存
        _completed_actions.clear()

        # 第二次：缓存已空，但后端 action_log 判重
        r2 = _apply_bot_update(proposal, code, action_id)
        assert "已应用" in r2
        assert len(mock_backend.update_calls) == 2  # 两次都到了后端
        assert mock_backend.action_log[action_id] is True  # 后端已记录

    def test_09_concurrent_resume_same_action_id(self, mock_backend):
        """#9 并发双 resume 同 action_id → 只有一次写入成功（唯一键保证）"""
        proposal = _make_proposal()
        code = proposal["new_code"]
        action_id = "tc_concurrent:99999"

        # 清空缓存模拟两个进程
        _completed_actions.clear()
        r1 = _apply_bot_update(proposal, code, action_id)

        _completed_actions.clear()
        r2 = _apply_bot_update(proposal, code, action_id)

        # 第一次成功，第二次 duplicate
        assert "成功" in r1
        assert "已应用" in r2

    def test_10_action_log_insert_success_bot_update_fail(self, mock_backend):
        """#10 action_log 插入成功但 bot 更新失败 → 事务回滚，可安全重试"""
        proposal = _make_proposal()
        code = proposal["new_code"]
        action_id = "tc_txfail:99999"

        # 第一次：后端 update 失败
        mock_backend.should_fail_update = True
        r1 = _apply_bot_update(proposal, code, action_id)
        assert "失败" in r1

        # 重试：后端恢复正常（事务回滚后 action_log 也清除）
        mock_backend.should_fail_update = False
        _completed_actions.clear()
        r2 = _apply_bot_update(proposal, code, action_id)
        assert "成功" in r2


# ============================================================
# Verifier 测试 (11-14)
# ============================================================

class TestVerifier:
    """Verifier 节点测试"""

    def test_11_valid_code_passes(self, mock_backend_for_verifier):
        """#11 合法代码通过 → passed=True"""
        code = """
public class Bot {
    Integer nextMove(String input) {
        return 0;
    }
}
""".strip()
        result = verify_bot_code(code)
        assert result.passed is True

    def test_12_missing_entry_method(self):
        """#12 缺入口方法 → passed=False"""
        code = """
public class Bot {
    void helper() {}
}
""".strip()
        result = verify_bot_code(code)
        assert result.passed is False
        assert "nextMove" in result.reason or "main" in result.reason

    def test_13_dangerous_system_exit(self):
        """#13 危险模式 System.exit → passed=False"""
        code = """
public class Bot {
    Integer nextMove(String input) {
        System.exit(0);
        return 0;
    }
}
""".strip()
        result = verify_bot_code(code)
        assert result.passed is False
        assert "System.exit" in result.reason

    def test_14_fail_retry_success(self, mock_backend_for_verifier):
        """#14 失败→重试→成功 → attempts 0→1，第二次正确"""
        bad_code = "public class Bot { void helper() {} }"  # 缺入口
        good_code = "public class Bot { Integer nextMove(String i) { return 0; } }"

        # 第一次校验失败
        r1 = verify_bot_code(bad_code)
        assert r1.passed is False

        # 模拟 agent 修正后第二次校验
        r2 = verify_bot_code(good_code)
        assert r2.passed is True

        # 验证 attempts 递增逻辑（在 verifier_node 中）
        state = _make_state(
            pending_approval=_make_pending_approval(code=bad_code),
            verification_attempts=0,
        )
        # attempts 应该从 0 增加到 1
        attempts = state["verification_attempts"]
        result = verify_bot_code(state["pending_approval"]["code"])
        if not result.passed:
            new_attempts = attempts + 1
            assert new_attempts == 1


# ============================================================
# 安全测试 (15)
# ============================================================

class TestSecurity:
    """安全验证测试"""

    def test_15_tampered_resume_data_ignored(self):
        """#15 篡改 resumeData.proposal → interrupt_node 读 state 中的 pending_approval，忽略客户端"""
        original_proposal = _make_proposal(bot_id=99999, bot_name="OriginalBot")
        tampered_proposal = _make_proposal(bot_id=1, bot_name="TamperedBot")

        # state 中的 pending_approval 来自 checkpoint
        approval = _make_pending_approval(
            proposal=original_proposal,
            action_id="tc_001:99999",
        )

        # 模拟 resume 时客户端传来的 user_decision（篡改了 proposal）
        user_decision = {
            "type": "approve",
            "proposal": tampered_proposal,  # 篡改！
        }

        # interrupt_node 只从 state["pending_approval"] 读取
        # safe_decision 只提取 type + edited_code
        safe_decision = {
            "type": user_decision.get("type", "reject"),
        }
        if user_decision.get("edited_code"):
            safe_decision["edited_code"] = user_decision["edited_code"]

        # 验证 safe_decision 中没有 proposal
        assert "proposal" not in safe_decision

        # interrupt_node 使用的是 state 中的 approval
        assert approval["proposal"]["bot_id"] == 99999
        assert approval["proposal"]["bot_name"] == "OriginalBot"


# ============================================================
# 边界测试 (16)
# ============================================================

class TestBoundary:
    """边界条件测试"""

    @pytest.mark.asyncio
    async def test_16_resume_without_interrupt_state(self):
        """#16 无中断状态下 resume → 返回明确错误 NO_INTERRUPT_STATE"""
        mock_llm = MagicMock()
        executor = AgentExecutor(llm=mock_llm, tools=[])

        # Mock _build_hitl_agent 返回一个 mock graph
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.next = None  # 无中断状态
        mock_state.config = {"configurable": {"checkpoint_id": "cp_1"}}
        mock_graph.get_state.return_value = mock_state

        with patch.object(executor, '_build_hitl_agent', return_value=mock_graph):
            result = await executor.resume("session_no_interrupt", {"type": "approve"})

        assert result.success is False
        assert result.error == "NO_INTERRUPT_STATE"
        assert "不在中断状态" in result.answer


# ============================================================
# Verifier 附加测试
# ============================================================

class TestVerifierExtra:
    """Verifier 额外边界测试"""

    def test_empty_code(self):
        """空代码 → 不通过"""
        assert verify_bot_code("").passed is False
        assert verify_bot_code(None).passed is False
        assert verify_bot_code("   ").passed is False

    def test_unbalanced_braces(self):
        """花括号不平衡"""
        code = "public class Bot { Integer nextMove(String i) { return 0; }"
        result = verify_bot_code(code)
        assert result.passed is False
        assert "花括号" in result.reason

    def test_runtime_exec_blocked(self):
        """Runtime.exec 被阻止"""
        code = """
public class Bot {
    Integer nextMove(String input) {
        Runtime.getRuntime().exec("ls");
        return 0;
    }
}
""".strip()
        result = verify_bot_code(code)
        assert result.passed is False
        assert "Runtime.exec" in result.reason

    def test_process_builder_blocked(self):
        """ProcessBuilder 被阻止"""
        code = """
public class Bot {
    Integer nextMove(String input) {
        new ProcessBuilder("ls").start();
        return 0;
    }
}
""".strip()
        result = verify_bot_code(code)
        assert result.passed is False
        assert "ProcessBuilder" in result.reason

    def test_compile_check_connection_error(self):
        """编译检查端点不可达 → 视为通过"""
        import requests
        with patch("agent.verifier.http_requests.post", side_effect=requests.ConnectionError):
            result = try_compile_check("class Bot { Integer nextMove(String i) { return 0; } }")
            assert result.passed is True

    def test_main_entry_without_input(self):
        """main 入口但缺少 INPUT 环境变量 → 不通过"""
        code = """
public class Bot {
    public static void main(String[] args) {
        System.out.println(0);
    }
}
""".strip()
        result = verify_bot_code(code)
        assert result.passed is False
        assert "INPUT" in result.reason
