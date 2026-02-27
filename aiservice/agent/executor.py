"""
Agent 执行器 - v4 HITL 闭环升级版 (LangGraph 1.0+)

核心特性：
- 使用 LangChain create_tool_calling_agent（兼容性更好）
- 支持 Human-in-the-Loop (HITL) 通过 LangGraph interrupt() 函数
- 5 节点图拓扑：agent → tools → interrupt → verifier → apply
- 副作用幂等化（确定性 action_id + 进程缓存 + 后端动作表）
- 支持 Time Travel 通过 checkpointer
- 支持异步执行和完整追踪
"""
import json
import logging
import os
import re
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import requests as http_requests
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .state import AgentState
from .tools import get_tools

# LangChain Agent imports
try:
    from langchain.agents import create_tool_calling_agent, AgentExecutor as LCAgentExecutor
    LANGCHAIN_AGENT_AVAILABLE = True
except ImportError:
    LANGCHAIN_AGENT_AVAILABLE = False

# LangGraph imports for HITL (1.0+ API)
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import interrupt, Command
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    
# Checkpointer imports for HITL
try:
    from langgraph.checkpoint.memory import MemorySaver
    MEMORY_SAVER_AVAILABLE = True
except ImportError:
    MEMORY_SAVER_AVAILABLE = False

logger = logging.getLogger(__name__)

# 最大迭代次数
MAX_ITERATIONS = int(os.getenv("AI_AGENT_MAX_ITERATIONS", "10"))

# Phase C: 幂等进程缓存（快路径，非唯一真相源）
_completed_actions: set = set()


def _apply_bot_update(proposal: dict, code: str, action_id: str) -> str:
    """执行副作用：写入 Bot 代码到后端数据库。
    
    层 1：进程内缓存快路径
    层 2：后端 bot_action_log 唯一约束（幂等真相源）
    """
    if action_id in _completed_actions:
        return f"修改已应用 (action_id={action_id})"

    try:
        resp = http_requests.post(
            "http://127.0.0.1:3000/ai/bot/manage/update",
            json={
                "botId": proposal.get("bot_id"),
                "userId": proposal.get("user_id"),
                "content": code,
                "actionId": action_id,
            },
            timeout=10,
        )
        data = resp.json()
        if data.get("success"):
            _completed_actions.add(action_id)
            if data.get("duplicate"):
                return f"修改已应用 (action_id={action_id})"
            return f"修改成功应用到 {proposal.get('bot_name', 'Bot')}"
        return f"更新失败: {data.get('error', f'HTTP {resp.status_code}')}"
    except Exception as e:
        logger.error("_apply_bot_update 失败: %s", e)
        return f"更新失败: {e}"


def _append_messages_from_chunk(result: Dict[str, Any], chunk: Dict[str, Any]) -> None:
    """Collect messages from stream chunks that may be nested by node name."""
    def _extend_messages(value: Any) -> None:
        if value is None:
            return
        messages = value if isinstance(value, list) else [value]
        result.setdefault("messages", []).extend(messages)

    for key, value in chunk.items():
        if key == "messages":
            _extend_messages(value)
            continue
        if isinstance(value, dict) and "messages" in value:
            _extend_messages(value.get("messages"))


def _get_messages_from_state(state: Any) -> List[Any]:
    """Best-effort extraction of messages from LangGraph state objects."""
    if state is None:
        return []

    # LangGraph state objects often expose values or channel_values.
    for attr in ("values", "channel_values"):
        values = getattr(state, attr, None)
        if isinstance(values, dict) and "messages" in values:
            return values.get("messages", []) or []

    if isinstance(state, dict):
        if "messages" in state:
            return state.get("messages", []) or []
        if "channel_values" in state and isinstance(state["channel_values"], dict):
            return state["channel_values"].get("messages", []) or []

    return []


@dataclass
class AgentResult:
    """Agent 执行结果"""
    success: bool
    answer: str
    thought_chain: List[Dict[str, Any]]
    steps: int
    error: Optional[str] = None
    interrupted: bool = False
    checkpoint_id: Optional[str] = None
    interrupt_data: Optional[Dict[str, Any]] = None


SYSTEM_PROMPT = """你是 KOB（King of Bots）游戏的 AI 助手，可以帮助用户管理和修改他们的 Bot 代码。

## 核心能力
1. **回答游戏问题**：关于规则、策略、算法的问题
2. **查看/分析用户的 Bot 代码**：当用户要求查看、分析、解释某个 Bot 时
3. **修改用户的 Bot 代码**：当用户要求修改某个 Bot 时

## 可用工具
- `get_user_bots(user_id, username)` - 获取用户的 Bot 列表。**支持双模查询**：
  - 方式1：传入 `user_id`（数字）
  - 方式2：传入 `username`（用户名字符串）
  - 如果 user_id 查询失败（用户不存在），请尝试使用 username 参数
- `get_bot_code(bot_id, user_id)` - 获取 Bot 的代码内容
- `code_analysis(code, analysis_type)` - 解释或审查 Bot 代码（analysis_type 可选: explain/explain_with_code/code_only/review/optimize）
- `propose_and_apply_modification(...)` - 生成修改建议并等待用户确认

## ⚠️ 关键规则 - 必须遵守

### 规则 1：查看/分析 Bot 的完整流程
当用户说"分析我的 XXX Bot"、"看看我的 Bot"、"解释这个 Bot 的逻辑"时：
1. **必须**先调用 `get_user_bots` 获取 Bot 列表，找到用户提到的 Bot 的 ID
2. **必须**调用 `get_bot_code` 获取该 Bot 的实际代码
3. **必须**调用 `code_analysis` 解释或展示代码：
   - 用户明确要求“展示代码/源码/给我代码/完整代码/代码原文” → `analysis_type=code_only`
   - 用户明确要求“结合代码/带上源码/具体内容/查看代码” → `analysis_type=explain_with_code`
   - 其他解释请求 → `analysis_type=explain`
4. 直接输出 `code_analysis` 结果，除非用户要求，否则不要追加额外扩展

### 规则 2：修改 Bot 的完整流程
当用户说"修改我的 XXX Bot"或"让我的 Bot 更强"时：
1. 先调用 `get_user_bots` 获取 Bot 列表，找到用户提到的 Bot
2. 调用 `get_bot_code` 获取该 Bot 的代码
3. 调用 `propose_and_apply_modification` 生成修改建议
4. 系统会暂停等待用户确认，用户确认后自动应用修改

### 规则 3：禁止猜测/幻觉
**严禁**在未调用 `get_bot_code` 获取实际代码的情况下：
- 基于 Bot 名称猜测代码内容
- 编造或假设代码实现
- 给出代码分析结果

如果你无法获取代码（工具调用失败），必须明确告知用户，而不是编造分析结果。

### 规则 4：禁止直接输出代码
当用户要求修改、优化、升级或改进 Bot 代码时，你绝对不能直接输出代码文本。
必须通过 `propose_and_apply_modification` 工具提交修改建议。
只有通过该工具，用户才能在 UI 中看到修改卡片并确认应用。

### 规则 5：用户查找策略
当需要查找用户的 Bot 时：
1. **优先**使用消息中提供的 `user_id`
2. 如果 `user_id` 查询失败（返回"用户不存在"），**尝试询问用户的用户名**
3. 如果用户提供了用户名，使用 `get_user_bots(username="用户名")` 进行查询
4. **不要反复询问** ID，主动提供解决方案

## 重要提示
- 用户 ID 会在消息中提供，格式：`当前用户ID: xxx`
- 如果用户没有指定 Bot 名称，先列出所有 Bot 让用户选择
- 如果 user_id 不可用，可以询问用户名并使用 username 参数查询

## 回答风格要求（必须遵守）
- 简洁回答，不要重复同一内容
- 只回答用户提出的问题，不要额外扩展优缺点/改进建议，除非用户明确要求
- 若调用 `code_analysis` 工具，优先直接使用其结果，最多补充一句总结
- 解释代码时先覆盖 package/import/类结构，再讲核心逻辑
- 用户要求“结合代码/带上源码”时，说明要比普通解释更细，必须引用具体标识符或条件
- 用户只问“策略”时，只给策略结论与必要的简短依据，避免额外段落
- 用户要求“展示代码/源码”时，不要追加解释或重复段落"""


class AgentExecutor:
    """
    Agent 执行器 - 使用 LangChain create_tool_calling_agent
    
    支持工具调用和 Bot 代码管理功能
    支持 HITL (Human-in-the-Loop) 通过 LangGraph interrupt()
    """
    
    def __init__(self, llm=None, tools=None, max_iterations: int = MAX_ITERATIONS):
        self.llm = llm
        self.tools = tools or get_tools()
        self.max_iterations = max_iterations
        self._agent = None
        self._agent_executor = None
        self._hitl_graph = None
        self._checkpointer = None

    def _inject_user_context(self, task: str, context: Optional[Dict[str, Any]]) -> str:
        """将用户身份注入到任务文本中，避免 LLM 猜测 user_id。"""
        if not context or not task:
            return task

        if "[当前用户ID:" in task:
            return task

        user_id = context.get("user_id")
        try:
            user_id_int = int(user_id)
        except (TypeError, ValueError):
            user_id_int = None

        if user_id_int and user_id_int > 0:
            return f"[当前用户ID: {user_id_int}]\n\n{task}"

        return task

    def _detect_code_request_type(self, task: str) -> Optional[str]:
        """识别是否为代码展示/解释请求，并返回对应 analysis_type。"""
        if not task:
            return None

        code_only_keywords = [
            "展示代码",
            "给我代码",
            "源码",
            "完整代码",
            "代码原文",
        ]
        explain_with_code_keywords = [
            "结合代码",
            "带上源码",
            "具体内容",
            "查看代码",
            "代码内容",
            "代码具体",
        ]

        for keyword in code_only_keywords:
            if keyword in task:
                return "code_only"

        for keyword in explain_with_code_keywords:
            if keyword in task:
                return "explain_with_code"

        return None

    def _extract_bot_id_from_text(self, text: str) -> Optional[int]:
        if not text:
            return None
        match = re.search(r"ID[:：]?\s*(\d+)", text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def _extract_bots_from_list(self, text: str) -> List[Dict[str, Any]]:
        bots = []
        if not text:
            return bots
        for name, bot_id in re.findall(r"\*\*(.+?)\*\*\s*\(ID[:：]?\s*(\d+)\)", text):
            try:
                bots.append({"name": name, "id": int(bot_id)})
            except ValueError:
                continue
        return bots

    def _find_last_bot_id_in_history(self, chat_history: Optional[List[Any]]) -> Optional[int]:
        if not chat_history:
            return None
        for msg in reversed(chat_history):
            content = getattr(msg, "content", "") or ""
            bot_id = self._extract_bot_id_from_text(content)
            if bot_id:
                return bot_id
        return None

    def _resolve_bot_id_for_request(
        self,
        task: str,
        user_id: int,
        chat_history: Optional[List[Any]],
    ) -> Optional[int]:
        explicit_id = self._extract_bot_id_from_text(task)
        if explicit_id:
            return explicit_id

        if user_id <= 0:
            return self._find_last_bot_id_in_history(chat_history)

        wants_first = "第一个" in task or "第1个" in task

        try:
            from .tools import get_user_bots
            bots_text = get_user_bots.invoke({"user_id": user_id})
        except Exception:
            bots_text = ""

        bots = self._extract_bots_from_list(bots_text)
        if bots:
            if wants_first:
                return bots[0]["id"]
            for bot in bots:
                if bot["name"] and bot["name"] in task:
                    return bot["id"]

        last_bot_id = self._find_last_bot_id_in_history(chat_history)
        if last_bot_id:
            return last_bot_id

        return None

    def _extract_code_from_bot_output(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = re.search(r"```(?:java)?\s*([\s\S]*?)```", text)
        if not match:
            return None
        return match.group(1).strip()

    def _try_direct_code_response(
        self,
        task: str,
        user_id: int,
        chat_history: Optional[List[Any]],
    ) -> Optional["AgentResult"]:
        analysis_type = self._detect_code_request_type(task)
        if not analysis_type:
            return None

        bot_id = self._resolve_bot_id_for_request(task, user_id, chat_history)
        if not bot_id:
            return None

        try:
            from .tools import get_bot_code, code_analysis
            bot_output = get_bot_code.invoke({"bot_id": bot_id, "user_id": user_id})
            if "没有权限" in bot_output or "不存在" in bot_output:
                return AgentResult(
                    success=False,
                    answer=bot_output,
                    thought_chain=[],
                    steps=1,
                    error=bot_output,
                )

            code = self._extract_code_from_bot_output(bot_output)
            if not code:
                return AgentResult(
                    success=True,
                    answer=bot_output,
                    thought_chain=[],
                    steps=1,
                )

            return AgentResult(
                success=True,
                answer=code_analysis.invoke({"code": code, "analysis_type": analysis_type}),
                thought_chain=[],
                steps=1,
            )
        except Exception as e:
            logger.warning("直接代码响应失败: %s", e)
            return None
    
    def _get_llm(self):
        """获取 LLM 实例"""
        if self.llm is not None:
            return self.llm
        
        try:
            from llm_client import build_llm
            self.llm = build_llm(streaming=False)
            return self.llm
        except Exception as e:
            logger.error("LLM 初始化失败: %s", e)
            return None
    
    def _build_agent(self):
        """使用 LangChain create_tool_calling_agent 构建 Agent"""
        if self._agent_executor is not None:
            return self._agent_executor
        
        llm = self._get_llm()
        if llm is None:
            logger.error("LLM 不可用，无法构建 Agent")
            return None
        
        if not LANGCHAIN_AGENT_AVAILABLE:
            logger.warning("LangChain Agent 不可用")
            return None
        
        try:
            # 构建 prompt 模板
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # 创建工具调用 Agent
            agent = create_tool_calling_agent(llm, self.tools, prompt)
            
            # 创建 AgentExecutor
            self._agent_executor = LCAgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=self.max_iterations,
                handle_parsing_errors=True,
            )
            
            logger.info("使用 create_tool_calling_agent 构建 Agent 成功")
            return self._agent_executor
        except Exception as e:
            logger.error("构建 Agent 失败: %s", e, exc_info=True)
            return None
    
    def execute(
        self,
        task: str,
        context: Dict[str, Any] = None,
    ) -> AgentResult:
        """
        执行 Agent 任务
        
        Args:
            task: 用户任务描述
            context: 上下文信息
            
        Returns:
            执行结果
        """
        logger.info("Agent 开始执行: %s", task[:100])
        
        context = context or {}
        task = self._inject_user_context(task, context)

        direct_result = self._try_direct_code_response(
            task,
            context.get("user_id") or 0,
            context.get("chat_history") or [],
        )
        if direct_result:
            return direct_result

        agent_executor = self._build_agent()
        
        if agent_executor is not None:
            return self._execute_with_agent(task, context, agent_executor)
        else:
            return self._execute_simple(task, context)
    
    def _execute_with_agent(
        self,
        task: str,
        context: Dict[str, Any],
        agent_executor,
    ) -> AgentResult:
        """使用 LangChain AgentExecutor 执行"""
        try:
            chat_history = []
            if context and context.get("chat_history"):
                chat_history = context.get("chat_history")

            # AgentExecutor 使用 input 作为输入
            result = agent_executor.invoke({
                "input": task,
                "chat_history": chat_history,
            })
            
            # 提取最终回答
            output = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])
            
            thought_chain = []
            for i, (action, observation) in enumerate(intermediate_steps):
                thought_chain.append({
                    "step": i + 1,
                    "action": action.tool if hasattr(action, 'tool') else str(action),
                    "action_input": action.tool_input if hasattr(action, 'tool_input') else None,
                    "observation": str(observation)[:500],
                })
            
            return AgentResult(
                success=True,
                answer=output,
                thought_chain=thought_chain,
                steps=len(intermediate_steps) + 1,
            )
        except Exception as e:
            logger.error("Agent 执行失败: %s", e, exc_info=True)
            return AgentResult(
                success=False,
                answer=f"执行失败: {e}",
                thought_chain=[],
                steps=0,
                error=str(e),
            )
    
    async def _get_checkpointer(self):
        """获取 Checkpointer 用于 HITL"""
        if self._checkpointer is not None:
            return self._checkpointer
        
        require_persistent = os.getenv(
            "HITL_REQUIRE_PERSISTENT_CHECKPOINTER", "0"
        ).lower() in ("1", "true", "yes", "on")

        try:
            from memory.langgraph_memory import get_langgraph_checkpointer
            self._checkpointer = await get_langgraph_checkpointer()
            return self._checkpointer
        except Exception as e:
            if require_persistent:
                logger.error("严格模式下无法获取持久化 checkpointer: %s", e)
                raise
            logger.warning("无法获取 PostgresSaver，降级为 MemorySaver: %s", e)
            if MEMORY_SAVER_AVAILABLE:
                self._checkpointer = MemorySaver()
                return self._checkpointer
            return None
    
    async def _build_hitl_agent(self):
        """
        构建支持 HITL 的 LangGraph Agent (v4 - 5 节点图拓扑)
        
        图拓扑：agent → tools → interrupt → verifier → apply → agent
        - agent:     调 LLM，step_count += 1
        - tools:     执行工具，proposal 写入 pending_approval（不调 interrupt）
        - interrupt: 读 state → interrupt() → 处理用户决策（resume 时只重跑此节点）
        - verifier:  校验代码质量
        - apply:     写 DB（唯一有副作用的节点，幂等保护）
        """
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph 不可用，HITL 功能禁用")
            return None
        
        if self._hitl_graph is not None:
            return self._hitl_graph
        
        llm = self._get_llm()
        if llm is None:
            return None
        
        checkpointer = await self._get_checkpointer()

        from .verifier import verify_bot_code, try_compile_check
        
        # ── 节点 1：agent（调 LLM）──
        def call_model(state: AgentState):
            """调用 LLM，step_count += 1"""
            messages = state["messages"]
            full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
            llm_with_tools = llm.bind_tools(self.tools)
            response = llm_with_tools.invoke(full_messages)
            return {
                "messages": [response],
                "step_count": state.get("step_count", 0) + 1,
            }
        
        # ── 节点 2：tools（执行工具，proposal 写入 state）──
        def hitl_tool_node(state: AgentState):
            """执行工具。检测 HITL proposal 响应并写入 pending_approval 状态。"""
            # 预算防御性检查
            if state.get("tool_budget", 0) <= 0:
                return {"last_error": "TOOL_BUDGET_EXHAUSTED"}

            last_message = state["messages"][-1]
            if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                return {"messages": []}
            
            results = []
            tool_calls_made = 0
            
            for tool_call in last_message.tool_calls:
                tool = next((t for t in self.tools if t.name == tool_call["name"]), None)
                if not tool:
                    continue

                tool_calls_made += 1
                try:
                    result = tool.invoke(tool_call["args"])
                    result_str = str(result)
                    
                    # 检测 HITL proposal 响应 → 写入 pending_approval
                    if "__hitl_proposal__" in result_str:
                        proposal_data = json.loads(result_str)
                        action_id = f"{tool_call['id']}:{proposal_data['proposal']['bot_id']}"
                        
                        return {
                            "messages": [ToolMessage(
                                content="修改建议已生成，等待用户确认。",
                                tool_call_id=tool_call["id"],
                            )],
                            "pending_approval": {
                                "proposal": proposal_data["proposal"],
                                "action_id": action_id,
                                "tool_call_id": tool_call["id"],
                            },
                            "tool_budget": state["tool_budget"] - tool_calls_made,
                        }
                    
                    results.append(ToolMessage(
                        content=result_str,
                        tool_call_id=tool_call["id"]
                    ))
                except Exception as e:
                    results.append(ToolMessage(
                        content=f"工具执行失败: {e}",
                        tool_call_id=tool_call["id"]
                    ))
            
            return {
                "messages": results,
                "tool_budget": state["tool_budget"] - tool_calls_made,
            }
        
        # ── 节点 3：interrupt（HITL 暂停/恢复）──
        def interrupt_node(state: AgentState):
            """从 state 读 proposal → interrupt() → 处理用户决策。
            
            关键：这个节点在 resume 时重跑，但它不调工具 —— 只读 state。
            proposal 来自 checkpoint 中的 pending_approval，不来自 tool.invoke() 重新执行。
            """
            approval = state.get("pending_approval")
            if not approval:
                return {}  # 无需中断，直接通过
            
            proposal = approval["proposal"]
            action_id = approval["action_id"]
            
            # interrupt()：首次暂停，resume 后返回 user_decision
            user_decision = interrupt({
                "__hitl_proposal__": True,
                "type": "bot_modification_proposal",
                "proposal": proposal,
                "action_id": action_id,
                "message": f"请确认是否应用对 {proposal.get('bot_name', 'Bot')} 的修改",
                "options": ["approve", "reject", "edit"],
            })
            
            # ↓ resume 后才执行 ↓
            decision_type = user_decision.get("type", "reject")
            
            if decision_type in ("approve", "edit"):
                code = (user_decision.get("edited_code")
                        if decision_type == "edit"
                        else proposal["new_code"])
                return {
                    "pending_approval": {
                        **approval,
                        "code": code,
                    },
                }
            else:
                return {
                    "messages": [AIMessage(
                        content=f"用户拒绝了对 {proposal.get('bot_name', 'Bot')} 的修改。代码保持不变。"
                    )],
                    "pending_approval": None,
                }
        
        # ── 节点 4：verifier（校验代码质量）──
        def verifier_node(state: AgentState):
            attempts = state.get("verification_attempts", 0)
            approval = state.get("pending_approval")
            if not approval or not approval.get("code"):
                return {"verification_attempts": 0, "last_error": None}
            
            code = approval["code"]
            result = verify_bot_code(code)
            if result.passed:
                compile_result = try_compile_check(code)
                if compile_result.passed:
                    return {"verification_attempts": 0, "last_error": None}
                result = compile_result
            
            MAX_VERIFICATION_ATTEMPTS = 2
            if attempts >= MAX_VERIFICATION_ATTEMPTS:
                logger.warning("校验失败 %d 次，带风险放行", attempts)
                return {
                    "messages": [AIMessage(
                        content=f"[VERIFIER] 已达到重试上限，带风险继续执行。原因: {result.reason}"
                    )],
                    "verification_attempts": 0,
                    "last_error": None,
                }
            
            return {
                "messages": [HumanMessage(
                    content=f"[VERIFIER] 待应用代码未通过校验: {result.reason}。"
                            f"{result.suggestions or '请修正后重试。'}"
                )],
                "verification_attempts": attempts + 1,
                "last_error": result.reason,
                "pending_approval": None,
            }
        
        # ── 节点 5：apply（唯一有副作用的节点）──
        def apply_node(state: AgentState):
            """执行副作用：写入 Bot 代码到数据库。"""
            approval = state.get("pending_approval")
            if not approval:
                return {}
            
            proposal = approval["proposal"]
            code = approval["code"]
            action_id = approval["action_id"]
            
            result = _apply_bot_update(proposal, code, action_id)
            
            return {
                "messages": [AIMessage(content=result)],
                "pending_approval": None,
                "last_action_id": action_id,
            }
        
        # ── 路由函数 ──
        def should_continue(state: AgentState):
            """主门禁：预算耗尽走 END，有 tool_calls 走 tools"""
            if state.get("tool_budget", 0) <= 0:
                return END
            last_message = state["messages"][-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            return END
        
        def after_interrupt(state: AgentState):
            """interrupt 后路由：approve/edit → verifier，reject → agent"""
            if state.get("pending_approval") and state["pending_approval"].get("code"):
                return "verifier"
            return "agent"
        
        def after_verify(state: AgentState):
            """verifier 后路由：校验通过 → apply，失败 → agent 重试"""
            if (state.get("pending_approval")
                    and state["pending_approval"].get("code")
                    and not state.get("last_error")):
                return "apply"
            return "agent"
        
        # ── 构建 Graph ──
        builder = StateGraph(AgentState)
        builder.add_node("agent", call_model)
        builder.add_node("tools", hitl_tool_node)
        builder.add_node("interrupt", interrupt_node)
        builder.add_node("verifier", verifier_node)
        builder.add_node("apply", apply_node)
        
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        builder.add_edge("tools", "interrupt")
        builder.add_conditional_edges("interrupt", after_interrupt,
                                      {"verifier": "verifier", "agent": "agent"})
        builder.add_conditional_edges("verifier", after_verify,
                                      {"apply": "apply", "agent": "agent"})
        builder.add_edge("apply", "agent")
        
        self._hitl_graph = builder.compile(checkpointer=checkpointer)
        logger.info("HITL Agent (v4 五节点) 构建成功")
        
        return self._hitl_graph
    
    async def execute_with_hitl(
        self, 
        task: str, 
        session_id: str, 
        user_id: int,
        chat_history: Optional[List[Any]] = None
    ) -> AgentResult:
        """
        支持 HITL 的执行（v4）
        
        当遇到需要确认的操作时会返回 interrupted=True
        """
        direct_result = self._try_direct_code_response(task, user_id, chat_history)
        if direct_result:
            return direct_result

        agent = await self._build_hitl_agent()
        
        if agent is None:
            # 降级为普通执行
            return self.execute(task, {"user_id": user_id, "chat_history": chat_history or []})
        
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            # 构建输入消息
            user_message = self._inject_user_context(task, {"user_id": user_id})
            messages = list(chat_history or [])
            messages.append(HumanMessage(content=user_message))
            
            # v4: AgentState 初始输入
            initial_input = {
                "messages": messages,
                "step_count": 0,
                "tool_budget": MAX_ITERATIONS,
                "pending_approval": None,
                "last_action_id": None,
                "last_error": None,
                "verification_attempts": 0,
            }
            
            # 关键兼容策略：
            # - interrupt() 在当前栈上对同步 graph.stream 更稳定
            # - AsyncPostgresSaver 需要避免在主线程直接走同步 checkpointer 调用
            # 因此将同步 stream/get_state 放到 worker thread 执行。
            def _run_sync_stream():
                local_result = {}
                local_interrupt_data = None
                for chunk in agent.stream(initial_input, config=config):
                    logger.debug("[HITL] Stream chunk keys: %s", list(chunk.keys()))

                    if "__interrupt__" in chunk:
                        interrupt_info = chunk["__interrupt__"]
                        if interrupt_info:
                            local_interrupt_data = (
                                interrupt_info[0].value
                                if hasattr(interrupt_info[0], "value")
                                else interrupt_info[0]
                            )
                            logger.info("[HITL] 检测到中断!")
                            break

                    _append_messages_from_chunk(local_result, chunk)
                    for key, value in chunk.items():
                        if key == "messages":
                            continue
                        if isinstance(value, dict) and "messages" in value:
                            continue
                        local_result[key] = value

                local_state = agent.get_state(config)
                return local_result, local_interrupt_data, local_state

            result, interrupt_data, state = await asyncio.to_thread(_run_sync_stream)
            checkpoint_id = state.config.get("configurable", {}).get("checkpoint_id")
            
            # 如果检测到中断
            if interrupt_data is not None:
                logger.info("[HITL] 返回中断结果，checkpoint_id: %s", checkpoint_id)
                return AgentResult(
                    success=True,
                    answer="",
                    thought_chain=[],
                    steps=0,
                    interrupted=True,
                    checkpoint_id=checkpoint_id,
                    interrupt_data=interrupt_data,
                )
            
            # 正常完成
            messages = result.get("messages", [])
            if not messages:
                messages = _get_messages_from_state(state)
            final_answer = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content and not getattr(msg, 'tool_calls', None):
                    final_answer = msg.content
                    break
            if not final_answer:
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        continue
                    if getattr(msg, "content", None):
                        final_answer = msg.content
                        break
            
            return AgentResult(
                success=True,
                answer=final_answer,
                thought_chain=[],
                steps=len(messages),
                checkpoint_id=checkpoint_id,
            )
            
        except Exception as e:
            logger.error("HITL 执行失败: %s", e, exc_info=True)
            return AgentResult(
                success=False,
                answer=f"执行失败: {e}",
                thought_chain=[],
                steps=0,
                error=str(e),
            )
    
    async def resume(self, session_id: str, user_decision: dict) -> AgentResult:
        """
        恢复被 interrupt 的执行（v4 - 真正的 LangGraph 闭环）
        
        使用 Command(resume=user_decision) 让图从断点恢复。
        只传 type + edited_code，不传 proposal（来自 checkpoint 的 pending_approval）。
        
        Args:
            session_id: 会话 ID
            user_decision: 用户决策 {"type": "approve|reject|edit", "edited_code": "..."}
        """
        try:
            agent = await self._build_hitl_agent()
            config = {"configurable": {"thread_id": session_id}}
            
            # 前置检查：确认图确实处于中断状态
            current_state = await asyncio.to_thread(agent.get_state, config)
            if not getattr(current_state, 'next', None):
                return AgentResult(
                    success=False,
                    answer="当前会话不在中断状态，无法恢复执行。",
                    thought_chain=[], steps=0,
                    error="NO_INTERRUPT_STATE",
                )
            
            # 只传 type + edited_code，不传 proposal（来自 checkpoint 的 pending_approval）
            safe_decision = {
                "type": user_decision.get("type", "reject"),
            }
            if user_decision.get("edited_code"):
                safe_decision["edited_code"] = user_decision["edited_code"]
            
            logger.info("[HITL Resume] Command(resume=%s), session=%s", safe_decision["type"], session_id)
            
            def _run_sync_resume():
                local_result = {}
                local_interrupt_data = None
                for chunk in agent.stream(Command(resume=safe_decision), config=config):
                    if "__interrupt__" in chunk:
                        interrupt_info = chunk["__interrupt__"]
                        if interrupt_info:
                            local_interrupt_data = (
                                interrupt_info[0].value
                                if hasattr(interrupt_info[0], "value")
                                else interrupt_info[0]
                            )
                        break
                    _append_messages_from_chunk(local_result, chunk)

                local_state = agent.get_state(config)
                return local_result, local_interrupt_data, local_state

            result, interrupt_data, state = await asyncio.to_thread(_run_sync_resume)
            checkpoint_id = state.config.get("configurable", {}).get("checkpoint_id")
            
            # 如果 resume 后又触发了新的中断
            if interrupt_data is not None:
                return AgentResult(
                    success=True,
                    answer="",
                    thought_chain=[],
                    steps=0,
                    interrupted=True,
                    checkpoint_id=checkpoint_id,
                    interrupt_data=interrupt_data,
                )
            
            # 提取最终回答
            messages = result.get("messages", [])
            if not messages:
                messages = _get_messages_from_state(state)
            final_answer = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content and not getattr(msg, 'tool_calls', None):
                    final_answer = msg.content
                    break
            if not final_answer:
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        continue
                    if getattr(msg, "content", None):
                        final_answer = msg.content
                        break
            
            return AgentResult(
                success=True,
                answer=final_answer,
                thought_chain=[],
                steps=len(messages),
                checkpoint_id=checkpoint_id,
            )
            
        except Exception as e:
            logger.error("恢复执行失败: %s", e, exc_info=True)
            return AgentResult(
                success=False,
                answer=f"恢复执行失败: {e}",
                thought_chain=[],
                steps=0,
                error=str(e),
            )
    
    def _execute_simple(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> AgentResult:
        """简化版执行（不使用 LangGraph）"""
        llm = self._get_llm()
        if llm is None:
            return AgentResult(
                success=False,
                answer="LLM 服务不可用",
                thought_chain=[],
                steps=0,
                error="LLM_UNAVAILABLE",
            )
        
        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=task),
            ]
            
            response = llm.invoke(messages)
            content = getattr(response, "content", str(response))
            
            return AgentResult(
                success=True,
                answer=content,
                thought_chain=[{
                    "step": 1,
                    "thought": content,
                    "action": None,
                    "observation": "直接回答",
                }],
                steps=1,
            )
        except Exception as e:
            logger.error("简化执行失败: %s", e)
            return AgentResult(
                success=False,
                answer=f"执行失败: {e}",
                thought_chain=[],
                steps=0,
                error=str(e),
            )


# 全局实例
_executor: Optional[AgentExecutor] = None


def get_executor() -> AgentExecutor:
    """获取全局执行器实例"""
    global _executor
    if _executor is None:
        _executor = AgentExecutor()
    return _executor


def execute_agent(
    task: str,
    context: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    便捷函数：执行 Agent 任务
    
    Args:
        task: 任务描述
        context: 上下文信息
        
    Returns:
        执行结果字典
    """
    executor = get_executor()
    result = executor.execute(task, context)
    
    return {
        "success": result.success,
        "answer": result.answer,
        "thought_chain": result.thought_chain,
        "steps": result.steps,
        "error": result.error,
    }
