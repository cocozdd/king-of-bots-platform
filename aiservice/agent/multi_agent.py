"""
Multi-Agent 路由器（稳定版）

说明：
- 保留原模块对外接口：`multi_agent_invoke()`、`get_multi_agent_supervisor()`
- 采用轻量 Supervisor 路由（规则路由 + 工具调用）
- 避免复杂编排带来的运行时耦合，优先保证可用性与可维护性
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from .tools import (
    battle_analysis,
    battle_query,
    code_analysis,
    get_user_bots,
    knowledge_search,
    loss_reason_analyzer,
    strategy_recommend,
)

logger = logging.getLogger(__name__)


@dataclass
class MultiAgentResult:
    success: bool
    answer: str
    agent_used: str = "unknown"
    thought_process: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


def _extract_code_block(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"```(?:java)?\s*([\s\S]*?)```", text)
    if not match:
        return None
    return match.group(1).strip()


def _extract_record_id(text: str) -> Optional[int]:
    if not text:
        return None
    # 支持 record_id=123 / recordId:123 / 记录ID 123
    patterns = [
        r"record[_\s-]?id\s*[:=]\s*(\d+)",
        r"记录\s*ID\s*[:：]?\s*(\d+)",
        r"对战\s*ID\s*[:：]?\s*(\d+)",
    ]
    lower = text.lower()
    for pattern in patterns:
        m = re.search(pattern, lower)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def _invoke_tool(tool_obj: Any, **kwargs: Any) -> str:
    """兼容 LangChain @tool 和普通函数调用。"""
    try:
        if hasattr(tool_obj, "invoke"):
            return str(tool_obj.invoke(kwargs))
        return str(tool_obj(**kwargs))
    except Exception as e:
        logger.warning("工具调用失败 %s: %s", getattr(tool_obj, "name", tool_obj), e)
        return f"工具调用失败: {e}"


def _llm_fallback_answer(question: str, system_hint: str) -> str:
    """当工具结果不足时，使用 LLM 生成兜底回答。"""
    try:
        from llm_client import get_llm

        llm = get_llm()
        if llm is None:
            return "当前无法获取模型服务，请稍后重试。"

        prompt = (
            f"{system_hint}\n\n"
            f"用户问题：{question}\n"
            "请给出简洁、结构化回答；如果信息不足请明确说明不足点。"
        )
        result = llm.invoke([HumanMessage(content=prompt)])
        return getattr(result, "content", str(result))
    except Exception as e:
        logger.warning("LLM 兜底失败: %s", e)
        return f"暂时无法完成回答：{e}"


class MultiAgentSupervisor:
    """轻量 Supervisor：根据问题路由到专业处理器。"""

    def __init__(self, llm: Any = None, checkpointer: Any = None):
        self.llm = llm
        self.checkpointer = checkpointer

    def _route(self, question: str) -> str:
        q = (question or "").lower()

        # 视觉问题优先（当前 /api/multi-agent 仅文本输入，这里做语义路由）
        if any(k in q for k in ["截图", "图片", "image", "png", "jpg", "jpeg"]):
            return "vision_expert"

        if any(k in q for k in ["复盘", "对战", "失败", "输", "record", "胜率", "回放"]):
            return "analysis_expert"

        if any(k in q for k in ["优化", "学习", "思路", "复杂度", "改进", "提升"]):
            return "optimizer_expert"

        if any(k in q for k in ["代码", "bot", "java", "生成", "修复", "重构", "分析源码"]):
            return "code_expert"

        return "rag_expert"

    async def invoke(
        self,
        question: str,
        session_id: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> MultiAgentResult:
        if not question or not question.strip():
            return MultiAgentResult(
                success=False,
                answer="",
                agent_used="none",
                error="问题不能为空",
            )

        agent = self._route(question)
        thought_process: List[Dict[str, Any]] = [{"step": 1, "router": agent}]

        try:
            if agent == "rag_expert":
                answer = self._handle_rag(question, thought_process)
            elif agent == "code_expert":
                answer = self._handle_code(question, user_id, thought_process)
            elif agent == "analysis_expert":
                answer = self._handle_analysis(question, user_id, thought_process)
            elif agent == "optimizer_expert":
                answer = self._handle_optimizer(question, thought_process)
            else:
                answer = self._handle_vision(question, thought_process)

            return MultiAgentResult(
                success=True,
                answer=answer,
                agent_used=agent,
                thought_process=thought_process,
            )
        except Exception as e:
            logger.error("MultiAgent 执行失败: %s", e, exc_info=True)
            return MultiAgentResult(
                success=False,
                answer=f"执行失败: {e}",
                agent_used=agent,
                thought_process=thought_process,
                error=str(e),
            )

    def _handle_rag(self, question: str, thought: List[Dict[str, Any]]) -> str:
        thought.append({"step": 2, "action": "knowledge_search"})
        searched = _invoke_tool(knowledge_search, query=question, top_k=5)
        if searched and "工具调用失败" not in searched and "未找到" not in searched:
            return searched
        return _llm_fallback_answer(question, "你是 KOB 知识库专家")

    def _handle_code(self, question: str, user_id: Optional[int], thought: List[Dict[str, Any]]) -> str:
        code = _extract_code_block(question)
        if code:
            thought.append({"step": 2, "action": "code_analysis(review)"})
            return _invoke_tool(code_analysis, code=code, analysis_type="review")

        q = question.lower()
        if user_id and any(k in q for k in ["我的", "bot", "列表"]):
            thought.append({"step": 2, "action": "get_user_bots"})
            return _invoke_tool(get_user_bots, user_id=int(user_id))

        if any(k in q for k in ["生成", "写一个", "给我代码", "示例代码"]):
            return _llm_fallback_answer(
                question,
                "你是 KOB Bot 代码专家，请输出可运行的 Java 代码，使用 ```java 代码块返回。",
            )

        thought.append({"step": 2, "action": "code_analysis(explain)"})
        return _llm_fallback_answer(question, "你是 KOB 代码讲解专家")

    def _handle_analysis(self, question: str, user_id: Optional[int], thought: List[Dict[str, Any]]) -> str:
        record_id = _extract_record_id(question)
        if record_id and user_id:
            thought.append({"step": 2, "action": "loss_reason_analyzer"})
            return _invoke_tool(loss_reason_analyzer, record_id=record_id, user_id=int(user_id))

        if record_id:
            thought.append({"step": 2, "action": "battle_analysis"})
            return _invoke_tool(battle_analysis, record_id=record_id)

        if user_id:
            thought.append({"step": 2, "action": "battle_query"})
            return _invoke_tool(battle_query, user_id=int(user_id), limit=10)

        thought.append({"step": 2, "action": "strategy_recommend"})
        return _invoke_tool(strategy_recommend, scenario=question)

    def _handle_optimizer(self, question: str, thought: List[Dict[str, Any]]) -> str:
        code = _extract_code_block(question)
        if code:
            thought.append({"step": 2, "action": "code_analysis(optimize)"})
            return _invoke_tool(code_analysis, code=code, analysis_type="optimize")

        thought.append({"step": 2, "action": "strategy_recommend"})
        return _invoke_tool(strategy_recommend, scenario=question)

    def _handle_vision(self, question: str, thought: List[Dict[str, Any]]) -> str:
        thought.append({"step": 2, "action": "vision_fallback"})
        return (
            "当前 /api/multi-agent 仅接收文本。若需图片分析，请调用 /api/vision/analyze，"
            "并传入 imageData 与 analysisType。"
        )


_supervisor: Optional[MultiAgentSupervisor] = None


async def get_multi_agent_supervisor(checkpointer: Any = None) -> MultiAgentSupervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = MultiAgentSupervisor(checkpointer=checkpointer)
    return _supervisor


async def multi_agent_invoke(
    question: str,
    session_id: Optional[str] = None,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    supervisor = await get_multi_agent_supervisor()
    result = await supervisor.invoke(question, session_id, user_id)

    return {
        "success": result.success,
        "answer": result.answer,
        "agent_used": result.agent_used,
        "thought_process": result.thought_process,
        "sources": result.sources,
        "error": result.error,
    }
