"""
KOB AI Service v2 - 精简版 API

2026 年最佳实践：
- POST 方法用于流式请求（支持复杂参数）
- stream 参数控制流式/非流式
- session_id 支持多轮对话
- 统一的错误处理和安全检查

前端可见端点（6个）：
- GET  /health              健康检查
- POST /api/bot/chat        智能问答（支持 stream、session_id）
- POST /api/bot/analyze     代码分析（支持 stream）
- POST /api/bot/fix         代码修复
- POST /api/bot/generate    代码生成（支持 stream）
- CRUD /api/session         会话管理
"""
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from llm_client import get_llm, build_llm
from guardrails import get_input_guard, get_output_guard
from observability import get_observability_status
from session_store import get_session_store, SessionStoreBase
from token_manager import get_token_manager, TokenManager
from cost_tracker import get_cost_tracker, CostTracker
from llm_guard import init_llm_guard, get_llm_guard

# ============ Lifespan 上下文管理器 (2026 最佳实践) ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan 上下文管理器
    
    2026 标准：替代旧的 startup/shutdown 事件
    - 启动时初始化资源（LLM、数据库连接等）
    - 关闭时清理资源
    """
    # === Startup ===
    logger.info("KOB AI Service 启动中...")
    
    # 预热 LLM 实例
    try:
        llm = get_llm()
        if llm:
            logger.info("✓ LLM 实例预热成功")
        else:
            logger.warning("⚠ LLM 未配置，将使用 mock 模式")
    except Exception as e:
        logger.error("LLM 初始化失败: %s", e)
    
    # 检查可观测性状态
    obs_status = get_observability_status()
    if obs_status.get("langsmith_enabled"):
        logger.info("✓ LangSmith 追踪已启用")
    if obs_status.get("langfuse_enabled"):
        logger.info("✓ LangFuse 监控已启用")
    
    # 初始化新模块
    try:
        session_store = get_session_store()
        logger.info("✓ 会话存储初始化成功: %s", type(session_store).__name__)
    except Exception as e:
        logger.error("会话存储初始化失败: %s", e)
    
    try:
        init_llm_guard()
        logger.info("✓ LLM Guard 初始化成功")
    except Exception as e:
        logger.warning("⚠ LLM Guard 初始化失败: %s", e)
    
    logger.info("KOB AI Service 启动完成 ✅")
    
    yield  # 应用运行中
    
    # === Shutdown ===
    logger.info("KOB AI Service 关闭中...")
    logger.info("KOB AI Service 已关闭 ✅")

app = FastAPI(
    title="KOB AI Service",
    version="2.1.0",
    description="Python AI microservice - 2026 标准",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 全局状态（已升级为 Redis 持久化）============
# 通过 get_session_store() 获取会话存储实例
# 支持 Redis 持久化或内存存储（降级）

# ============ Request/Response Models ============

class BotChatRequest(BaseModel):
    """智能问答请求
    
    支持 Java 端传入自定义 system_prompt（含 RAG 上下文）
    """
    question: str = Field(..., description="问题")
    session_id: Optional[str] = Field(None, alias="sessionId", description="会话 ID")
    stream: bool = Field(default=False, description="是否流式输出")
    user_id: Optional[int] = Field(None, alias="userId", description="用户 ID")
    system_prompt: Optional[str] = Field(None, alias="systemPrompt", description="系统提示词（含 RAG 上下文）")
    
    model_config = ConfigDict(populate_by_name=True)

class BotAnalyzeRequest(BaseModel):
    """代码分析请求"""
    code: str = Field(..., description="代码")
    stream: bool = Field(default=False, description="是否流式输出")
    
    model_config = ConfigDict(populate_by_name=True)

class BotFixRequest(BaseModel):
    """代码修复请求"""
    code: str = Field(..., description="代码")
    error_log: Optional[str] = Field(None, alias="errorLog", description="错误日志")
    
    model_config = ConfigDict(populate_by_name=True)

class BotGenerateRequest(BaseModel):
    """代码生成请求"""
    description: str = Field(..., description="策略描述")
    stream: bool = Field(default=False, description="是否流式输出")
    
    model_config = ConfigDict(populate_by_name=True)

class SessionRequest(BaseModel):
    """会话请求"""
    user_id: int = Field(..., alias="userId", description="用户 ID")
    title: str = Field(default="新对话", description="会话标题")
    
    model_config = ConfigDict(populate_by_name=True)


class BotChatHITLRequest(BaseModel):
    """支持 Human-in-the-Loop 的聊天请求
    
    2026 LangGraph 最佳实践：
    - 新对话: 提供 question
    - 恢复执行: 提供 resume_data（用户确认/拒绝/编辑）
    """
    question: Optional[str] = Field(None, description="用户问题（新对话时必填）")
    session_id: Optional[str] = Field(None, alias="sessionId", description="会话 ID")
    user_id: Optional[int] = Field(None, alias="userId", description="用户 ID")
    resume_data: Optional[Dict[str, Any]] = Field(None, alias="resumeData", 
        description="恢复执行数据，包含 type (approve/reject/edit) 和可选的 edited_code")
    
    model_config = ConfigDict(populate_by_name=True)


class LegacyChatMessage(BaseModel):
    """兼容 Java 旧协议的消息结构"""
    role: str = Field(default="user")
    content: str = Field(default="")


class LegacyChatRequest(BaseModel):
    """
    兼容旧版 /api/chat 与 /api/chat/stream 请求结构。

    Java 侧历史字段：
    - trace_id
    - session_id
    - userId
    - message/question/messages/history
    """
    version: str = Field(default="v1")
    trace_id: Optional[str] = Field(None, alias="trace_id")
    message: Optional[str] = Field(None)
    question: Optional[str] = Field(None)
    system_prompt: Optional[str] = Field(None, alias="systemPrompt")
    session_id: Optional[str] = Field(None, alias="session_id")
    user_id: Optional[int] = Field(None, alias="userId")
    stream: bool = Field(default=False)
    messages: List[LegacyChatMessage] = Field(default_factory=list)
    history: List[LegacyChatMessage] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


class AgentExecuteRequest(BaseModel):
    """兼容 Java AgentRouter 的旧版执行请求"""
    task: str = Field(..., description="任务描述")
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文")
    trace_id: Optional[str] = Field(None, alias="trace_id")

    model_config = ConfigDict(populate_by_name=True)


# ============ KOB 游戏规则 System Prompt ============

KOB_SYSTEM_PROMPT = """你是 King of Bots (KOB) 贪吃蛇对战平台的 Bot 开发助手。

【必须遵守】只基于本项目规则回答，禁止使用其它游戏/通用设定：
- 禁止出现：生命值/HP、武器/攻击力、技能/装备、道具、关卡等机制
- 如果问题与 KOB 无关，直接说明无法回答

【项目规则（必须遵守）】
1. 双人贪吃蛇对战，地图 13x14
2. 本游戏没有食物，蛇长度自动增长：
   - 前 10 步每步增长 1 格
   - 之后每 3 步增长 1 格（step % 3 == 1）
3. 胜负：撞墙/撞到自己或对手身体判负；双方同时撞到对方头部为平局；回合数超限按长度判定
4. 方向：0=上, 1=右, 2=下, 3=左
5. 坐标从 (0,0) 开始，地图有固定障碍

【回答要求】
- 使用 Markdown；代码块使用 ```java
- 如果被问到“如何胜利/策略”，必须先明确本项目胜负判定，再给策略建议
"""

# ============ 辅助函数 ============

def _get_refusal_response(reason: str) -> str:
    """生成拒答响应"""
    responses = {
        "injection": "检测到潜在的安全风险，请重新表述您的问题。",
        "too_long": "输入内容过长，请精简后重试。",
        "sensitive": "您的问题涉及敏感内容，请换个话题。",
    }
    return responses.get(reason, f"输入不合法：{reason}")

def _get_session_context(session_id: str, max_messages: int = 10) -> List[Dict]:
    """获取会话上下文（使用 SessionStore + TokenManager）"""
    store = get_session_store()
    messages = store.get_messages(session_id)
    
    if not messages:
        return []
    
    # 使用 TokenManager 智能截断
    token_manager = get_token_manager()
    selected, total_tokens = token_manager.get_context_within_limit(
        messages,
        max_tokens=60000,  # DeepSeek 窗口 64K，留 4K 给输出
    )
    
    logger.debug("会话 %s: 选择 %d/%d 条消息, %d tokens",
                 session_id, len(selected), len(messages), total_tokens)
    
    return selected

def _add_message_to_session(session_id: str, role: str, content: str):
    """添加消息到会话（使用 SessionStore）
    
    自动创建会话：如果 session_id 对应的会话不存在，自动创建
    """
    store = get_session_store()
    
    # 检查会话是否存在，不存在则自动创建
    session = store.get_session(session_id)
    if not session:
        now = datetime.now().isoformat()
        session = {
            "id": session_id,
            "user_id": 0,  # 默认用户
            "title": "新对话",
            "created_at": now,
            "updated_at": now,
        }
        store.save_session(session_id, session)
        logger.info(f"自动创建会话: {session_id}")
    
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    store.add_message(session_id, message)
    
    # 更新会话时间
    session["updated_at"] = datetime.now().isoformat()
    store.save_session(session_id, session)

def _validate_input(text: str) -> tuple[bool, str]:
    """输入安全检查"""
    input_guard = get_input_guard()
    result = input_guard.validate(text)
    return result.is_safe, result.reason

def _filter_output(text: str) -> str:
    """输出过滤"""
    output_guard = get_output_guard()
    return output_guard.filter(text)

def _build_chat_history(session_id: Optional[str]) -> List[Any]:
    """构建对话历史（仅 user/assistant 消息）"""
    if not session_id:
        return []
    from langchain_core.messages import HumanMessage, AIMessage
    history = []
    for msg in _get_session_context(session_id):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))
    return history


def _build_messages(system_prompt: str, session_id: Optional[str], user_input: str):
    """构建 LangChain 消息列表"""
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    
    messages = [SystemMessage(content=system_prompt)]
    
    # 添加会话历史
    if session_id:
        messages.extend(_build_chat_history(session_id))
    
    messages.append(HumanMessage(content=user_input))
    return messages


def _extract_legacy_question(request: LegacyChatRequest) -> str:
    """从旧协议请求中提取用户问题，兼容 message/question/messages/history。"""
    if request.question and request.question.strip():
        return request.question.strip()
    if request.message and request.message.strip():
        return request.message.strip()

    for msg in reversed(request.messages):
        if (msg.role or "").lower() in ("user", "human") and (msg.content or "").strip():
            return msg.content.strip()
    for msg in reversed(request.history):
        if (msg.role or "").lower() in ("user", "human") and (msg.content or "").strip():
            return msg.content.strip()

    for msg in reversed(request.messages):
        if (msg.content or "").strip():
            return msg.content.strip()
    for msg in reversed(request.history):
        if (msg.content or "").strip():
            return msg.content.strip()

    return ""


async def _save_langgraph_checkpoint(session_id: Optional[str], messages: List[Any]) -> Optional[str]:
    """为 Time Travel 保存检查点（非 HITL 也支持）"""
    if not session_id:
        return None
    try:
        from memory.langgraph_memory import get_memory_manager
        manager = await get_memory_manager()
        checkpointer = await manager.get_checkpointer()

        config = {
            "configurable": {
                "thread_id": session_id,
                "checkpoint_ns": "",
            }
        }

        latest = await checkpointer.aget(config)
        current_version = None
        if latest:
            current_version = latest.get("channel_versions", {}).get("messages")
            if latest.get("id"):
                config["configurable"]["checkpoint_id"] = latest.get("id")

        new_version = checkpointer.get_next_version(current_version, None)
        checkpoint_id = uuid.uuid4().hex
        checkpoint = {
            "v": 1,
            "id": checkpoint_id,
            "ts": datetime.utcnow().isoformat() + "Z",
            "channel_values": {"messages": messages},
            "channel_versions": {"messages": new_version},
            "versions_seen": {},
            "updated_channels": ["messages"],
        }

        await checkpointer.aput(
            config=config,
            checkpoint=checkpoint,
            metadata={"source": "chat"},
            new_versions={"messages": new_version},
        )
        return checkpoint_id
    except Exception as e:
        logger.warning("保存 LangGraph checkpoint 失败: %s", e)
        return None

async def _stream_response(llm, messages, session_id: Optional[str], user_input: str):
    """生成流式响应 - 2026 异步最佳实践"""
    full_response = ""

    try:
        # 使用 astream 异步迭代，避免阻塞事件循环
        async for chunk in llm.astream(messages):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            full_response += content
            # SSE 格式：event + data
            yield f"event: chunk\ndata: {json.dumps({'delta': content}, ensure_ascii=False)}\n\n"
    except Exception as e:
        logger.error("流式生成失败: %s", e)
        payload = {
            "error": {
                "code": "LLM_STREAM_ERROR",
                "message": str(e),
            }
        }
        yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        return
    
    # 保存到会话
    filtered_response = _filter_output(full_response)
    if session_id:
        _add_message_to_session(session_id, "user", user_input)
        _add_message_to_session(session_id, "assistant", filtered_response)

    # 保存检查点供 Time Travel 使用
    checkpoint_id = None
    try:
        from langchain_core.messages import AIMessage
        checkpoint_id = await _save_langgraph_checkpoint(
            session_id,
            messages + [AIMessage(content=filtered_response)]
        )
    except Exception as e:
        logger.warning("构建检查点失败: %s", e)

    done_payload = {"status": "completed"}
    if checkpoint_id:
        done_payload["checkpointId"] = checkpoint_id
    yield f"event: done\ndata: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

# ============ Health Check ============

@app.get("/health")
def health():
    """健康检查"""
    llm = get_llm()
    return {
        "status": "healthy",
        "service": "kob-ai-service",
        "version": "2.0.0",
        "llm_available": llm is not None,
    }

# ============ Bot API ============

@app.post("/api/bot/chat")
async def bot_chat(request: BotChatRequest):
    """
    智能问答端点 - 支持流式输出和多轮对话
    
    - stream=false: 返回 JSON
    - stream=true: 返回 SSE 流
    """
    # 1. 安全检查
    is_safe, reason = _validate_input(request.question)
    if not is_safe:
        if request.stream:
            async def error_stream():
                yield f"event: error\ndata: {json.dumps({'error': _get_refusal_response(reason)})}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        return {"success": False, "error": _get_refusal_response(reason), "sessionId": request.session_id}
    
    try:
        # 2. 构建消息 - 优先使用 Java 传入的 system_prompt（含 RAG 上下文）
        effective_prompt = request.system_prompt if request.system_prompt else KOB_SYSTEM_PROMPT
        messages = _build_messages(effective_prompt, request.session_id, request.question)
        
        # 3. 流式输出
        if request.stream:
            llm = build_llm(streaming=True)
            if llm is None:
                async def error_stream():
                    yield f"event: error\ndata: {json.dumps({'error': 'LLM 服务不可用'})}\n\n"
                return StreamingResponse(error_stream(), media_type="text/event-stream")
            
            return StreamingResponse(
                _stream_response(llm, messages, request.session_id, request.question),
                media_type="text/event-stream"
            )
        
        # 4. 非流式输出 - 使用异步调用
        llm = get_llm()
        if llm is None:
            return {"success": False, "error": "LLM 服务不可用", "sessionId": request.session_id}
        
        response = await llm.ainvoke(messages)
        answer = _filter_output(response.content)
        
        # 保存到会话
        if request.session_id:
            _add_message_to_session(request.session_id, "user", request.question)
            _add_message_to_session(request.session_id, "assistant", answer)

        checkpoint_id = None
        try:
            from langchain_core.messages import AIMessage
            checkpoint_id = await _save_langgraph_checkpoint(
                request.session_id,
                messages + [AIMessage(content=answer)]
            )
        except Exception as e:
            logger.warning("保存检查点失败: %s", e)

        return {
            "success": True,
            "answer": answer,
            "checkpointId": checkpoint_id,
            "sessionId": request.session_id,
        }
        
    except Exception as e:
        logger.error(f"智能问答失败: {e}")
        if request.stream:
            async def error_stream():
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        return {"success": False, "error": str(e), "sessionId": request.session_id}


@app.post("/api/chat")
async def legacy_chat(request: LegacyChatRequest):
    """
    兼容旧版 Java 客户端的聊天端点。

    兼容输出字段：
    - success
    - response
    - session_id
    - error {code, message}
    """
    if request.stream:
        return await legacy_chat_stream(request)

    question = _extract_legacy_question(request)
    if not question:
        return {
            "success": False,
            "version": request.version,
            "trace_id": request.trace_id,
            "session_id": request.session_id,
            "error": {
                "code": "INVALID_REQUEST",
                "message": "message or question is required",
            },
        }

    normalized_request = BotChatRequest(
        question=question,
        sessionId=request.session_id,
        stream=False,
        userId=request.user_id,
        systemPrompt=request.system_prompt,
    )
    result = await bot_chat(normalized_request)
    if not isinstance(result, dict):
        return {
            "success": False,
            "version": request.version,
            "trace_id": request.trace_id,
            "session_id": request.session_id,
            "error": {
                "code": "PYTHON_ERROR",
                "message": "Unexpected response type from /api/bot/chat",
            },
        }

    if not result.get("success", False):
        return {
            "success": False,
            "version": request.version,
            "trace_id": request.trace_id,
            "session_id": result.get("sessionId") or request.session_id,
            "error": {
                "code": "PYTHON_ERROR",
                "message": str(result.get("error", "Python chat failed")),
            },
        }

    return {
        "success": True,
        "version": request.version,
        "trace_id": request.trace_id,
        "response": result.get("answer", ""),
        "session_id": result.get("sessionId") or request.session_id,
    }


@app.post("/api/chat/stream")
async def legacy_chat_stream(request: LegacyChatRequest):
    """兼容旧版 Java 客户端的流式聊天端点。"""
    question = _extract_legacy_question(request)
    if not question:
        async def error_stream():
            payload = {
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": "message or question is required",
                }
            }
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(error_stream(), media_type="text/event-stream")

    normalized_request = BotChatRequest(
        question=question,
        sessionId=request.session_id,
        stream=True,
        userId=request.user_id,
        systemPrompt=request.system_prompt,
    )
    return await bot_chat(normalized_request)


@app.post("/api/agent/execute")
async def legacy_agent_execute(request: AgentExecuteRequest):
    """兼容 Java AgentRouter 的旧版 Agent 执行端点。"""
    if not request.task or not request.task.strip():
        return {"success": False, "error": "task is required", "thought_chain": [], "steps": 0}

    try:
        from agent.executor import get_executor

        executor = get_executor()
        result = executor.execute(request.task, request.context or {})
        return {
            "success": result.success,
            "answer": result.answer,
            "thought_chain": result.thought_chain,
            "steps": result.steps,
            "error": result.error,
        }
    except Exception as e:
        logger.error("legacy_agent_execute failed: %s", e, exc_info=True)
        return {
            "success": False,
            "answer": "",
            "thought_chain": [],
            "steps": 0,
            "error": str(e),
        }


@app.get("/api/agent/tools")
async def legacy_agent_tools():
    """兼容 Java AgentRouter 的旧版工具列表端点。"""
    try:
        from agent.tools import get_tools

        tools = get_tools()
        serialized = []
        for tool_obj in tools:
            serialized.append({
                "name": getattr(tool_obj, "name", getattr(tool_obj, "__name__", "unknown_tool")),
                "description": getattr(tool_obj, "description", ""),
            })
        return {"success": True, "tools": serialized, "count": len(serialized)}
    except Exception as e:
        logger.error("legacy_agent_tools failed: %s", e, exc_info=True)
        return {"success": False, "tools": [], "count": 0, "error": str(e)}


@app.post("/api/bot/chat/hitl")
async def chat_with_hitl(request: BotChatHITLRequest):
    """
    支持 Human-in-the-Loop 的聊天端点 (LangGraph 1.0+)

    2026 LangGraph 最佳实践：
    - 使用 interrupt() 暂停执行，等待用户确认
    - 使用 Command(resume=...) 恢复执行

    流程：
    1. 用户发送问题 → Agent 执行
    2. 如果需要确认（如修改 Bot 代码）→ 返回 interrupted=True + interruptData
    3. 用户确认/拒绝/编辑 → 发送 resume_data
    4. Agent 继续执行 → 返回最终结果
    """
    thread_id = request.session_id or str(uuid.uuid4())

    try:
        from agent.executor import get_executor

        executor = get_executor()

        # 恢复执行 - 使用 Command(resume=...)
        if request.resume_data:
            logger.info("[HITL] 恢复执行，决策: %s", request.resume_data.get("type"))
            
            result = await executor.resume(thread_id, request.resume_data)
            
            if result.success:
                # 保存到会话
                if result.answer:
                    _add_message_to_session(thread_id, "assistant", _filter_output(result.answer))
                
                return {
                    "success": True,
                    "answer": _filter_output(result.answer),
                    "sessionId": thread_id,
                    "checkpointId": result.checkpoint_id,
                }
            else:
                return {
                    "success": False,
                    "error": result.error or "恢复执行失败",
                    "sessionId": thread_id,
                }

        # 新对话
        if not request.question:
            return {
                "success": False,
                "error": "请提供问题",
                "sessionId": thread_id
            }

        # 安全检查
        is_safe, reason = _validate_input(request.question)
        if not is_safe:
            return {
                "success": False,
                "error": _get_refusal_response(reason),
                "sessionId": thread_id
            }

        logger.info("[HITL] 新对话，问题: %s, user_id: %s", request.question[:50], request.user_id)

        # 使用 HITL 执行
        chat_history = _build_chat_history(thread_id)
        result = await executor.execute_with_hitl(
            task=request.question,
            session_id=thread_id,
            user_id=request.user_id or 0,
            chat_history=chat_history,
        )

        # 检查是否被 interrupt
        if result.interrupted:
            logger.info("[HITL] Agent 被 interrupt 暂停，等待用户确认")
            
            # 保存用户问题到会话
            _add_message_to_session(thread_id, "user", request.question)
            
            return {
                "success": True,
                "interrupted": True,
                "interruptData": result.interrupt_data,
                "checkpointId": result.checkpoint_id,
                "sessionId": thread_id,
            }

        # 正常完成
        if result.success:
            # 保存到会话
            _add_message_to_session(thread_id, "user", request.question)
            if result.answer:
                _add_message_to_session(thread_id, "assistant", _filter_output(result.answer))
            
            return {
                "success": True,
                "interrupted": False,
                "answer": _filter_output(result.answer),
                "checkpointId": result.checkpoint_id,
                "sessionId": thread_id,
            }
        else:
            return {
                "success": False,
                "error": result.error or "执行失败",
                "sessionId": thread_id,
            }

    except Exception as e:
        logger.error(f"HITL 聊天失败: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "sessionId": thread_id
        }


@app.post("/api/bot/analyze")
async def bot_analyze(request: BotAnalyzeRequest):
    """
    代码分析端点 - 支持流式输出
    """
    if not request.code or not request.code.strip():
        return {"success": False, "error": "请提供代码"}
    
    # 安全检查
    is_safe, reason = _validate_input(request.code)
    if not is_safe:
        if request.stream:
            async def error_stream():
                yield f"event: error\ndata: {json.dumps({'error': _get_refusal_response(reason)})}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        return {"success": False, "error": _get_refusal_response(reason)}
    
    analysis_prompt = f"""分析以下 KOB 贪吃蛇 Bot 代码：

```java
{request.code}
```

请从以下几个方面分析：
1. **策略概述**：代码的主要策略
2. **优点**：代码的优势
3. **问题**：潜在的 bug 或逻辑问题
4. **优化建议**：如何改进"""

    try:
        messages = _build_messages(KOB_SYSTEM_PROMPT, None, analysis_prompt)
        
        if request.stream:
            llm = build_llm(streaming=True)
            if llm is None:
                async def error_stream():
                    yield f"event: error\ndata: {json.dumps({'error': 'LLM 服务不可用'})}\n\n"
                return StreamingResponse(error_stream(), media_type="text/event-stream")
            
            return StreamingResponse(
                _stream_response(llm, messages, None, analysis_prompt),
                media_type="text/event-stream"
            )
        
        llm = get_llm()
        if llm is None:
            return {"success": False, "error": "LLM 服务不可用"}
        
        response = await llm.ainvoke(messages)
        return {"success": True, "analysis": _filter_output(response.content)}
        
    except Exception as e:
        logger.error(f"代码分析失败: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/bot/fix")
async def bot_fix(request: BotFixRequest):
    """代码修复端点"""
    if not request.code or not request.code.strip():
        return {"success": False, "error": "请提供代码"}
    
    # 安全检查
    is_safe, reason = _validate_input(request.code)
    if not is_safe:
        return {"success": False, "error": _get_refusal_response(reason)}
    
    error_info = f"\n\n错误信息：\n{request.error_log}" if request.error_log else ""
    
    fix_prompt = f"""修复以下 KOB 贪吃蛇 Bot 代码中的问题：

```java
{request.code}
```
{error_info}

请：
1. 找出代码中的问题
2. 提供修复后的完整代码
3. 简要说明修复了什么"""

    try:
        messages = _build_messages(KOB_SYSTEM_PROMPT, None, fix_prompt)
        
        llm = get_llm()
        if llm is None:
            return {"success": False, "error": "LLM 服务不可用"}
        
        response = await llm.ainvoke(messages)
        result = _filter_output(response.content)
        
        # 提取代码块
        import re
        code_match = re.search(r'```(?:java)?\s*([\s\S]*?)```', result)
        fixed_code = code_match.group(1).strip() if code_match else ""
        
        return {
            "success": True,
            "fixedCode": fixed_code,
            "explanation": result,
        }
        
    except Exception as e:
        logger.error(f"代码修复失败: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/bot/generate")
async def bot_generate(request: BotGenerateRequest):
    """
    代码生成端点 - 支持流式输出
    
    公平性检查：
    - 用户提供思路 → 可以给代码
    - 用户只说"给我厉害的AI" → 先引导思考
    """
    is_safe, reason = _validate_input(request.description)
    if not is_safe:
        if request.stream:
            async def error_stream():
                yield f"event: error\ndata: {json.dumps({'error': _get_refusal_response(reason)})}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        return {"success": False, "error": _get_refusal_response(reason)}
    
    # === 公平性检查 ===
    from agent.code_optimizer import FairCodeOptimizer
    optimizer = FairCodeOptimizer()
    
    if optimizer._is_lazy_request(request.description) and not optimizer._check_user_has_idea(request.description):
        # 伸手党，没有思路 → 引导思考
        guidance_response = {
            "success": True,
            "needsIdea": True,
            "message": """我很乐意帮助你！但为了公平和真正帮助你学习，我需要先了解你的想法：

1. **你想实现什么策略？**
   - 安全优先？追着敌人？还是占领空间？

2. **你了解哪些算法？**
   - BFS（广度优先）可以找最短路径
   - 连通区域计算可以评估空间大小

3. **你目前的思路是什么？**
   - 哪怕只是大概想法也可以

请告诉我你的思路，我会帮你实现！

💡 提示：说说你想用什么算法，或者描述一下你的策略思路。""",
            "suggestedDescriptions": [
                "我想用 BFS 找到离对手最远的安全位置",
                "我的思路是先计算四个方向的可用空间",
                "我想实现一个能预测对手移动的策略",
            ],
        }
        if request.stream:
            async def guidance_stream():
                yield f"event: guidance\ndata: {json.dumps(guidance_response, ensure_ascii=False)}\n\n"
                yield f"event: done\ndata: {{\"status\": \"needs_idea\"}}\n\n"
            return StreamingResponse(guidance_stream(), media_type="text/event-stream")
        return guidance_response
    
    # 用户有思路，可以生成代码
    generate_prompt = f"""根据以下策略描述生成 KOB 贪吃蛇 Bot 代码：

策略描述：{request.description}

要求：
1. 生成完整可运行的 Java 代码
2. 代码需要实现 nextMove() 方法，返回 0-3 的方向
3. 添加必要的中文注释说明策略逻辑"""

    try:
        messages = _build_messages(KOB_SYSTEM_PROMPT, None, generate_prompt)
        
        if request.stream:
            llm = build_llm(streaming=True)
            if llm is None:
                async def error_stream():
                    yield f"event: error\ndata: {json.dumps({'error': 'LLM 服务不可用'})}\n\n"
                return StreamingResponse(error_stream(), media_type="text/event-stream")
            
            return StreamingResponse(
                _stream_response(llm, messages, None, generate_prompt),
                media_type="text/event-stream"
            )
        
        llm = get_llm()
        if llm is None:
            return {"success": False, "error": "LLM 服务不可用"}
        
        response = await llm.ainvoke(messages)
        result = _filter_output(response.content)
        
        # 提取代码块
        import re
        code_match = re.search(r'```(?:java)?\s*([\s\S]*?)```', result)
        code = code_match.group(1).strip() if code_match else result
        
        return {
            "success": True,
            "code": code,
            "explanation": result,
        }
        
    except Exception as e:
        logger.error(f"代码生成失败: {e}")
        return {"success": False, "error": str(e)}

# ============ Session API (RESTful) ============

@app.post("/api/session")
async def create_session(request: SessionRequest):
    """创建会话（使用 SessionStore）"""
    store = get_session_store()
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    session_data = {
        "id": session_id,
        "user_id": request.user_id,
        "title": request.title,
        "created_at": now,
        "updated_at": now,
    }
    store.save_session(session_id, session_data)
    
    return {
        "success": True,
        "session": session_data,
    }

@app.get("/api/session")
async def list_sessions(userId: int = Query(..., description="用户 ID")):
    """列出用户会话（使用 SessionStore）"""
    store = get_session_store()
    sessions = store.list_sessions(userId)
    return {"success": True, "sessions": sessions}

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """获取会话详情和历史（使用 SessionStore）"""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    messages = store.get_messages(session_id)
    return {
        "success": True,
        "session": session,
        "messages": messages,
    }

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话（使用 SessionStore）"""
    store = get_session_store()
    store.delete_session(session_id)
    return {"success": True}

@app.put("/api/session/{session_id}/title")
async def update_session_title(session_id: str, title: str = Query(...)):
    """更新会话标题（使用 SessionStore）"""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session["title"] = title
    session["updated_at"] = datetime.now().isoformat()
    store.save_session(session_id, session)
    return {"success": True, "session": session}


# ============ Multi-Agent API（LangGraph Supervisor）============

class MultiAgentRequest(BaseModel):
    """多 Agent 请求"""
    question: str = Field(..., description="用户问题")
    session_id: Optional[str] = Field(None, alias="sessionId", description="会话 ID")
    user_id: Optional[int] = Field(None, alias="userId", description="用户 ID（用于 Bot 查询等工具）")
    
    model_config = ConfigDict(populate_by_name=True)


@app.post("/api/multi-agent")
async def multi_agent_chat(request: MultiAgentRequest):
    """
    多 Agent 协作端点 - LangGraph Supervisor 模式
    
    根据问题类型自动路由到专业 Agent：
    - RAGAgent: 知识检索与问答
    - CodeAgent: Bot 代码生成与分析
    - AnalysisAgent: 对战分析与策略优化
    """
    try:
        from agent.multi_agent import multi_agent_invoke
        
        result = await multi_agent_invoke(
            question=request.question,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        
        return {
            "success": result.get("success", False),
            "answer": result.get("answer", ""),
            "agentUsed": result.get("agent_used", "unknown"),
            "thoughtProcess": result.get("thought_process", []),
            "sessionId": request.session_id,
        }
        
    except ImportError as e:
        logger.warning("Multi-Agent 模块不可用: %s", e)
        return {
            "success": False,
            "error": "Multi-Agent 功能需要安装 langgraph>=0.2.0",
            "sessionId": request.session_id,
        }
    except Exception as e:
        logger.error("Multi-Agent 执行失败: %s", e)
        return {
            "success": False,
            "error": str(e),
            "sessionId": request.session_id,
        }


# ============ Vision API（多模态图片分析）============

class ImageAnalysisRequest(BaseModel):
    """图片分析请求"""
    image_data: str = Field(..., alias="imageData", description="Base64 编码的图片或 URL")
    image_source: str = Field(default="base64", alias="imageSource", description="图片来源: base64 | url")
    analysis_type: str = Field(default="battle", alias="analysisType", description="分析类型: battle | code | map")
    question: Optional[str] = Field(None, description="用户问题（可选）")
    
    model_config = ConfigDict(populate_by_name=True)


@app.post("/api/vision/analyze")
async def analyze_image_endpoint(request: ImageAnalysisRequest):
    """
    多模态图片分析端点
    
    支持分析类型：
    - battle: 对战截图分析（局面评估、策略建议）
    - code: 代码截图分析（识别代码、找出问题）
    - map: 地图分析（障碍物、关键位置）
    
    需要配置 OPENAI_API_KEY（GPT-4o）或 DASHSCOPE_API_KEY（Qwen-VL）
    """
    try:
        from agent.multimodal import analyze_image
        
        result = await analyze_image(
            image_data=request.image_data,
            analysis_type=request.analysis_type,
            image_source=request.image_source,
            question=request.question,
        )
        
        return {
            "success": result.get("success", False),
            "analysis": result.get("analysis", ""),
            "modelUsed": result.get("model_used", "unknown"),
            "error": result.get("error"),
        }
        
    except ImportError as e:
        logger.warning("Vision 模块不可用: %s", e)
        return {
            "success": False,
            "error": "Vision 功能需要配置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY",
        }
    except Exception as e:
        logger.error("图片分析失败: %s", e)
        return {
            "success": False,
            "error": str(e),
        }


@app.get("/api/vision/status")
async def vision_status():
    """检查 Vision 功能状态"""
    try:
        from agent.multimodal import get_vision_analyzer
        analyzer = get_vision_analyzer()
        return {
            "available": analyzer.is_available(),
            "model": analyzer._model_name if analyzer.is_available() else None,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


# ============ Fair Code Optimizer API（公平代码优化）============

class CodeOptimizeRequest(BaseModel):
    """代码优化请求"""
    code: str = Field(..., description="要分析的代码")
    focus_area: str = Field(default="general", alias="focusArea", 
                            description="关注领域: general | safety | strategy | performance")
    
    model_config = ConfigDict(populate_by_name=True)


@app.post("/api/code/analyze")
async def analyze_code_endpoint(request: CodeOptimizeRequest):
    """
    公平代码分析端点
    
    设计原则（公平性）：
    - 帮助用户学习和理解，而非直接给答案
    - 提供算法思路，不给完整实现
    - 指出问题，引导用户自己修复
    
    返回：
    - complexity_score: 复杂度评分 (1-10)
    - issues: 检测到的问题
    - patterns: 识别到的算法模式
    - suggestions: 优化建议
    - educational_notes: 学习资料
    """
    try:
        from agent.code_optimizer import analyze_code_fair
        
        result = await analyze_code_fair(request.code)
        
        return {
            "success": result.get("success", False),
            "complexityScore": result.get("complexity_score", 0),
            "issues": result.get("issues", []),
            "patterns": result.get("patterns", []),
            "suggestions": result.get("suggestions", []),
            "educationalNotes": result.get("educational_notes", []),
            "error": result.get("error"),
        }
        
    except Exception as e:
        logger.error("代码分析失败: %s", e)
        return {
            "success": False,
            "error": str(e),
        }


@app.post("/api/code/guidance")
async def get_code_guidance(request: CodeOptimizeRequest):
    """
    获取代码改进指导（使用 LLM）
    
    使用苏格拉底式提问引导用户思考，而非直接给答案
    """
    try:
        from agent.code_optimizer import get_code_optimizer
        
        optimizer = get_code_optimizer()
        guidance = await optimizer.get_improvement_guidance(
            code=request.code,
            focus_area=request.focus_area,
        )
        
        return {
            "success": True,
            "guidance": guidance,
            "focusArea": request.focus_area,
        }
        
    except Exception as e:
        logger.error("获取指导失败: %s", e)
        return {
            "success": False,
            "error": str(e),
        }


class AlgorithmSuggestionRequest(BaseModel):
    """算法建议请求"""
    problem: str = Field(..., description="问题描述")


class CodeHelpRequest(BaseModel):
    """代码帮助请求（公平性检查）"""
    request: str = Field(..., description="用户请求")
    code: Optional[str] = Field(None, description="用户现有代码")
    idea: Optional[str] = Field(None, description="用户的思路")
    
    model_config = ConfigDict(populate_by_name=True)


@app.post("/api/code/help")
async def help_with_code_endpoint(req: CodeHelpRequest):
    """
    公平代码帮助端点 - 2026 更新
    
    公平性规则：
    - 用户提供思路 → 可以给代码帮助实现
    - 用户无思路只说"给我厉害的AI" → 先引导思考
    
    返回：
    - mode: guidance | implementation | learning
    - canGiveCode: 是否可以给代码
    - message/response: 回复内容
    """
    try:
        from agent.code_optimizer import help_with_code
        
        result = await help_with_code(
            user_request=req.request,
            user_code=req.code,
            user_idea=req.idea,
        )
        
        return result
        
    except Exception as e:
        logger.error("代码帮助失败: %s", e)
        return {
            "success": False,
            "mode": "error",
            "error": str(e),
        }


@app.post("/api/code/algorithm-suggestion")
async def suggest_algorithm_endpoint(request: AlgorithmSuggestionRequest):
    """
    根据问题描述建议算法思路
    
    只提供学习指导和原理解释，不给具体实现代码
    """
    try:
        from agent.code_optimizer import suggest_algorithm
        
        suggestion = suggest_algorithm.invoke(request.problem)
        
        return {
            "success": True,
            "suggestion": suggestion,
        }
        
    except Exception as e:
        logger.error("算法建议失败: %s", e)
        return {
            "success": False,
            "error": str(e),
        }


# ============ Time Travel API（检查点/分叉/分支）============

class ForkRequest(BaseModel):
    """分叉请求"""
    session_id: str = Field(..., alias="sessionId", description="原会话 ID")
    checkpoint_id: str = Field(..., alias="checkpointId", description="检查点 ID")
    question: str = Field(..., description="新问题")
    user_id: Optional[int] = Field(None, alias="userId", description="用户 ID")
    
    model_config = ConfigDict(populate_by_name=True)


@app.get("/api/chat/checkpoints")
async def list_checkpoints(sessionId: str = Query(...), limit: int = Query(default=20)):
    """
    获取会话的检查点列表（Time Travel）
    
    返回每个检查点的 ID、消息预览、时间戳
    """
    try:
        from memory.langgraph_memory import get_memory_manager
        
        manager = await get_memory_manager()
        checkpoints = await manager.get_checkpoints_with_metadata(sessionId, limit)
        
        return {
            "success": True,
            "checkpoints": checkpoints,
            "sessionId": sessionId,
        }
        
    except Exception as e:
        logger.error("获取检查点列表失败: %s", e)
        return {
            "success": False,
            "error": str(e),
            "checkpoints": [],
        }


@app.post("/api/chat/fork")
async def fork_from_checkpoint(request: ForkRequest):
    """
    从检查点分叉对话（Time Travel 核心功能）
    
    流程：
    1. 复制指定检查点状态到新会话
    2. 在新会话上继续对话
    """
    try:
        from memory.langgraph_memory import get_memory_manager
        
        manager = await get_memory_manager()
        
        # 1. 生成新会话 ID
        new_session_id = f"{request.session_id}_fork_{uuid.uuid4().hex[:8]}"
        branch_id = f"branch_{uuid.uuid4().hex[:4]}"
        
        # 2. 分叉到新线程
        success = await manager.fork_to_new_thread(
            request.session_id,
            request.checkpoint_id,
            new_session_id
        )
        
        if not success:
            return {
                "success": False,
                "error": "分叉失败，检查点可能不存在",
            }
        
        # 3. 在新会话上继续对话
        # 构建消息
        effective_prompt = KOB_SYSTEM_PROMPT
        user_message = request.question
        if request.user_id:
            user_message = f"[当前用户ID: {request.user_id}]\n\n{request.question}"
        
        messages = _build_messages(effective_prompt, new_session_id, user_message)
        
        llm = get_llm()
        if llm is None:
            return {
                "success": False,
                "error": "LLM 服务不可用",
                "newSessionId": new_session_id,
            }
        
        response = await llm.ainvoke(messages)
        answer = _filter_output(response.content)
        
        # 保存到新会话
        _add_message_to_session(new_session_id, "user", request.question)
        _add_message_to_session(new_session_id, "assistant", answer)

        checkpoint_id = None
        try:
            from langchain_core.messages import AIMessage
            checkpoint_id = await _save_langgraph_checkpoint(
                new_session_id,
                messages + [AIMessage(content=answer)]
            )
        except Exception as e:
            logger.warning("分叉后保存检查点失败: %s", e)

        return {
            "success": True,
            "newSessionId": new_session_id,
            "branchId": branch_id,
            "answer": answer,
            "checkpointId": checkpoint_id,
            "parentSessionId": request.session_id,
            "forkedFromCheckpoint": request.checkpoint_id,
        }
        
    except Exception as e:
        logger.error("分叉对话失败: %s", e, exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


@app.get("/api/chat/branches")
async def get_branches(sessionId: str = Query(...)):
    """
    获取会话的分支信息
    """
    try:
        from memory.langgraph_memory import get_memory_manager
        
        manager = await get_memory_manager()
        branch_info = await manager.get_branch_info(sessionId)
        
        return {
            "success": True,
            "branchInfo": branch_info,
            "sessionId": sessionId,
        }
        
    except Exception as e:
        logger.error("获取分支信息失败: %s", e)
        return {
            "success": False,
            "error": str(e),
        }


# ============ Multimodal Chat API（多模态聊天）============

class MultimodalChatRequest(BaseModel):
    """多模态聊天请求"""
    question: str = Field(..., description="用户问题")
    image_data: Optional[str] = Field(None, alias="imageData", description="Base64 图片数据")
    image_source: str = Field(default="base64", alias="imageSource", description="图片来源: base64 | url")
    session_id: Optional[str] = Field(None, alias="sessionId", description="会话 ID")
    user_id: Optional[int] = Field(None, alias="userId", description="用户 ID")
    stream: bool = Field(default=False, description="是否流式输出（图片分析不支持）")
    
    model_config = ConfigDict(populate_by_name=True)


@app.post("/api/chat/multimodal")
async def chat_multimodal(request: MultimodalChatRequest):
    """
    多模态聊天端点（支持图片）
    
    - 有图片：调用 vision 分析
    - 无图片：普通文本聊天
    
    注意：图片分析目前不支持流式输出
    """
    # 安全检查
    is_safe, reason = _validate_input(request.question)
    if not is_safe:
        return {
            "success": False,
            "error": _get_refusal_response(reason),
            "sessionId": request.session_id,
        }
    
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        if request.image_data:
            # 有图片 - 使用 vision 分析
            from agent.multimodal import analyze_image
            
            # 构建包含上下文的问题
            context_question = request.question
            if request.user_id:
                context_question = f"[用户ID: {request.user_id}]\n\n{request.question}"
            
            result = await analyze_image(
                image_data=request.image_data,
                analysis_type="battle",  # 默认对战分析
                image_source=request.image_source,
                question=context_question,
            )
            
            if result.get("success"):
                answer = result.get("analysis", "")
                
                # 保存到会话（带图片标记）
                _add_message_to_session(session_id, "user", f"[图片] {request.question}")
                _add_message_to_session(session_id, "assistant", answer)
                
                return {
                    "success": True,
                    "answer": answer,
                    "sessionId": session_id,
                    "modelUsed": result.get("model_used", "vision"),
                    "hasImage": True,
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "图片分析失败"),
                    "sessionId": session_id,
                }
        
        else:
            # 无图片 - 普通文本聊天
            messages = _build_messages(KOB_SYSTEM_PROMPT, session_id, request.question)
            
            llm = get_llm()
            if llm is None:
                return {
                    "success": False,
                    "error": "LLM 服务不可用",
                    "sessionId": session_id,
                }
            
            response = await llm.ainvoke(messages)
            answer = _filter_output(response.content)
            
            # 保存到会话
            _add_message_to_session(session_id, "user", request.question)
            _add_message_to_session(session_id, "assistant", answer)
            
            return {
                "success": True,
                "answer": answer,
                "sessionId": session_id,
                "hasImage": False,
            }
            
    except ImportError as e:
        logger.warning("Multimodal 模块不可用: %s", e)
        return {
            "success": False,
            "error": "多模态功能需要配置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY",
            "sessionId": session_id,
        }
    except Exception as e:
        logger.error("多模态聊天失败: %s", e)
        return {
            "success": False,
            "error": str(e),
            "sessionId": session_id,
        }


# ============ Stats API（新增）============

@app.get("/api/stats")
async def get_stats():
    """获取系统统计信息"""
    from embedding_cache import get_embedding_cache_stats
    
    cost_tracker = get_cost_tracker()
    llm_guard = get_llm_guard()
    
    return {
        "success": True,
        "cost": cost_tracker.get_stats() if cost_tracker else {},
        "embedding_cache": get_embedding_cache_stats(),
        "llm_guard": llm_guard.get_stats() if llm_guard else {},
    }

# ============ 启动 ============

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3003))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
