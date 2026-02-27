"""
Agent 状态定义

使用 TypedDict 定义 LangGraph 状态机的状态结构
"""
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
import operator

from langgraph.graph import add_messages


class ThoughtStep(TypedDict):
    """思考步骤"""
    step: int
    thought: str
    action: Optional[str]
    action_input: Optional[Dict[str, Any]]
    observation: Optional[str]


class AgentState(TypedDict):
    """
    Agent 状态 (v4)

    LangGraph 使用 TypedDict 定义状态结构。
    messages 使用 add_messages reducer（自动去重、追加）。
    新增字段支持工具预算、HITL 闭环、幂等保护和 verifier 重试。
    """
    # 消息历史（LangGraph reducer，自动去重追加）
    messages: Annotated[list, add_messages]

    # agent 节点每次 +1
    step_count: int

    # 工具调用次数上限（在 tools 节点扣减）
    tool_budget: int

    # HITL 中断时的 proposal（信任源，checkpoint 保存）
    pending_approval: Optional[Dict[str, Any]]

    # 最近一次幂等 action ID
    last_action_id: Optional[str]

    # 上次失败原因
    last_error: Optional[str]

    # verifier 重试次数
    verification_attempts: int


def create_initial_state(
    task: str,
    context: Dict[str, Any] = None,
) -> AgentState:
    """创建初始状态"""
    return AgentState(
        messages=[],
        step_count=0,
        tool_budget=10,
        pending_approval=None,
        last_action_id=None,
        last_error=None,
        verification_attempts=0,
    )


def add_thought_step(
    state: AgentState,
    thought: str,
    action: str = None,
    action_input: Dict[str, Any] = None,
    observation: str = None,
) -> ThoughtStep:
    """创建思考步骤"""
    return ThoughtStep(
        step=state["step_count"] + 1,
        thought=thought,
        action=action,
        action_input=action_input,
        observation=observation,
    )
