"""
KOB AI Service - FastAPI 应用入口

功能：
- 基础对话 API（同步/流式）
- Agent 执行 API（支持 session_id）
- 会话管理 API（创建/列表/历史/删除）
- 安全检查 API（输入校验/注入检测）
- RAG 检索 API

2026 年 AI 架构标准：
- ✅ 输入安全检查（Guardrails）
- ✅ 多轮会话管理
- ✅ 结构化输出
- ✅ 可观测性（日志/追踪）
"""
import json
import logging
import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 导入模块
from llm_client import get_llm, get_llm_async
from guardrails import get_input_guard, get_output_guard, get_rate_limiter, ValidationResult
from agent.executor import get_executor, AgentResult

app = FastAPI(
    title="KOB AI Service",
    version="2.0.0",
    description="Python AI microservice with security guardrails and session management"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 全局状态（生产环境用 Redis）============

# 会话存储
_sessions: Dict[str, Dict] = {}
# 会话消息存储
_session_messages: Dict[str, List[Dict]] = {}
# 会话摘要存储
_session_summaries: Dict[str, str] = {}

# ============ Request/Response Models ============

class Message(BaseModel):
    role: str = Field(..., description="角色: system, user, assistant")
    content: str = Field(..., description="消息内容")

class ChatRequest(BaseModel):
    messages: Optional[List[Message]] = Field(None, description="消息历史")
    message: Optional[str] = Field(None, description="单条消息（兼容 Java 端）")
    session_id: Optional[str] = Field(None, description="会话 ID（用于多轮对话）")
    system_prompt: Optional[str] = Field(None, alias="systemPrompt", description="系统提示词")
    stream: bool = Field(default=False, description="是否流式输出")
    trace_id: Optional[str] = Field(None, alias="traceId", description="追踪 ID")
    version: Optional[str] = Field(None, description="版本")

    class Config:
        populate_by_name = True

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI 响应")
    model: str = Field(default="deepseek-chat", description="使用的模型")
    session_id: Optional[str] = Field(None, description="会话 ID")
    usage: Optional[Dict] = Field(None, description="Token 使用统计")

class AgentRequest(BaseModel):
    task: str = Field(..., description="任务描述")
    session_id: Optional[str] = Field(None, description="会话 ID")
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息")

class AgentResponse(BaseModel):
    success: bool
    answer: str
    thought_chain: List[Dict[str, Any]] = []
    steps: int = 0
    session_id: Optional[str] = None
    error: Optional[str] = None

class SecurityValidateRequest(BaseModel):
    text: str = Field(..., description="待校验文本")
    max_length: int = Field(default=10000, description="最大长度")

class SecurityValidateResponse(BaseModel):
    is_safe: bool
    reason: str = ""
    details: Dict = {}

class SessionCreateRequest(BaseModel):
    user_id: int = Field(..., description="用户 ID")
    title: str = Field(default="新对话", description="会话标题")

class SessionResponse(BaseModel):
    id: str
    user_id: int
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0

class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: List[Message]
    summary: Optional[str] = None

# ============ 健康检查 ============

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "kob-ai-service",
        "version": "2.0.0",
        "features": ["guardrails", "session", "agent", "rag"],
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

# ============ 安全检查 API ============

@app.post("/api/security/validate", response_model=SecurityValidateResponse)
async def validate_input(request: SecurityValidateRequest):
    """
    输入校验端点 - 供 Java 端调用
    
    检查：Prompt Injection、长度限制、敏感词
    """
    input_guard = get_input_guard()
    result = input_guard.validate(request.text)
    
    return SecurityValidateResponse(
        is_safe=result.is_safe,
        reason=result.reason,
        details=result.details,
    )

@app.post("/api/security/check-injection")
async def check_injection(request: SecurityValidateRequest):
    """
    Prompt Injection 检测端点
    """
    input_guard = get_input_guard()
    result = input_guard._detect_injection(request.text)
    
    return {
        "is_injection": not result.is_safe,
        "details": result.details,
    }

# ============ 会话管理 API ============

@app.post("/api/session/create", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """创建新会话"""
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    _sessions[session_id] = {
        "id": session_id,
        "user_id": request.user_id,
        "title": request.title,
        "created_at": now,
        "updated_at": now,
    }
    _session_messages[session_id] = []
    
    logger.info(f"创建会话: {session_id} for user {request.user_id}")
    
    return SessionResponse(
        id=session_id,
        user_id=request.user_id,
        title=request.title,
        created_at=now,
        updated_at=now,
        message_count=0,
    )

@app.get("/api/session/list")
async def list_sessions(user_id: int = Query(..., description="用户 ID")):
    """列出用户的所有会话"""
    user_sessions = [
        SessionResponse(
            id=s["id"],
            user_id=s["user_id"],
            title=s["title"],
            created_at=s["created_at"],
            updated_at=s["updated_at"],
            message_count=len(_session_messages.get(s["id"], [])),
        )
        for s in _sessions.values()
        if s["user_id"] == user_id
    ]
    
    # 按更新时间倒序
    user_sessions.sort(key=lambda x: x.updated_at, reverse=True)
    
    return {"sessions": user_sessions}

@app.get("/api/session/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str):
    """获取会话历史"""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    messages = _session_messages.get(session_id, [])
    summary = _session_summaries.get(session_id)
    
    return SessionHistoryResponse(
        session_id=session_id,
        messages=[Message(**m) for m in messages],
        summary=summary,
    )

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    del _sessions[session_id]
    _session_messages.pop(session_id, None)
    _session_summaries.pop(session_id, None)
    
    logger.info(f"删除会话: {session_id}")
    
    return {"success": True, "message": "会话已删除"}

@app.put("/api/session/{session_id}/title")
async def update_session_title(session_id: str, title: str = Query(...)):
    """更新会话标题"""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    _sessions[session_id]["title"] = title
    _sessions[session_id]["updated_at"] = datetime.now().isoformat()
    
    return {"success": True, "title": title}

# ============ 对话 API ============

def _get_refusal_response(reason: str) -> str:
    """获取拒答响应"""
    return f"抱歉，由于{reason}，我无法处理这个请求。请换一种方式提问。"

def _add_message_to_session(session_id: str, role: str, content: str):
    """添加消息到会话"""
    if session_id and session_id in _sessions:
        if session_id not in _session_messages:
            _session_messages[session_id] = []
        
        _session_messages[session_id].append({
            "role": role,
            "content": content,
        })
        _sessions[session_id]["updated_at"] = datetime.now().isoformat()
        
        # 检查是否需要压缩（超过 20 条消息）
        if len(_session_messages[session_id]) > 20:
            _compress_session_history(session_id)

def _compress_session_history(session_id: str):
    """压缩会话历史为摘要"""
    messages = _session_messages[session_id]
    old_messages = messages[:-5]  # 保留最近 5 条
    
    if not old_messages:
        return
    
    # 生成摘要（简化版：直接截取关键信息）
    history_text = "\n".join([f"{m['role']}: {m['content'][:100]}" for m in old_messages])
    
    try:
        llm = get_llm()
        if llm:
            from langchain_core.messages import HumanMessage
            summary_prompt = f"请用 2-3 句话总结以下对话的关键信息：\n{history_text}"
            result = llm.invoke([HumanMessage(content=summary_prompt)])
            _session_summaries[session_id] = result.content
    except Exception as e:
        logger.warning(f"摘要生成失败: {e}")
        _session_summaries[session_id] = f"[历史对话 {len(old_messages)} 条]"
    
    # 保留最近 5 条
    _session_messages[session_id] = messages[-5:]
    logger.info(f"会话 {session_id} 历史已压缩")

def _get_session_context(session_id: str) -> List[Dict]:
    """获取会话上下文（摘要 + 最近消息）"""
    context = []
    
    if session_id and session_id in _sessions:
        # 添加摘要
        summary = _session_summaries.get(session_id)
        if summary:
            context.append({
                "role": "system",
                "content": f"[历史摘要]: {summary}"
            })
        
        # 添加最近消息
        messages = _session_messages.get(session_id, [])
        context.extend(messages[-10:])  # 最多 10 条
    
    return context

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    基础对话端点 - 支持会话管理和安全检查
    """
    # 1. 安全检查
    input_guard = get_input_guard()
    user_input = request.messages[-1].content if request.messages else ""
    
    validation = input_guard.validate(user_input)
    if not validation.is_safe:
        logger.warning(f"输入被拒绝: {validation.reason}")
        return ChatResponse(
            response=_get_refusal_response(validation.reason),
            model="security-filter",
            session_id=request.session_id,
        )
    
    # 2. 获取会话上下文
    session_context = _get_session_context(request.session_id) if request.session_id else []
    
    # 3. 构建消息
    all_messages = session_context + [msg.dict() for msg in request.messages]
    
    try:
        llm = get_llm()
        if llm is None:
            raise HTTPException(status_code=503, detail="LLM 服务不可用")
        
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        lc_messages = []
        for msg in all_messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        
        response = llm.invoke(lc_messages)
        response_text = response.content
        
        # 4. 输出过滤
        output_guard = get_output_guard()
        response_text = output_guard.filter(response_text)
        
        # 5. 保存到会话
        if request.session_id:
            _add_message_to_session(request.session_id, "user", user_input)
            _add_message_to_session(request.session_id, "assistant", response_text)
        
        return ChatResponse(
            response=response_text,
            model="deepseek-chat",
            session_id=request.session_id,
        )
        
    except Exception as e:
        logger.error(f"对话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """流式对话端点 - 兼容 Java 端 PythonChatRequest 格式"""
    # 提取用户输入（兼容 message 和 messages 两种格式）
    user_input = ""
    if request.message:
        user_input = request.message
    elif request.messages:
        user_input = request.messages[-1].content
    
    if not user_input:
        async def error_stream():
            payload = {"event": "error", "error": {"code": "INVALID_REQUEST", "message": "message or messages is required"}}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    # 安全检查
    input_guard = get_input_guard()
    validation = input_guard.validate(user_input)
    if not validation.is_safe:
        async def error_stream():
            payload = {"delta": _get_refusal_response(validation.reason)}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: {\"event\":\"done\"}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    try:
        from llm_client import build_llm
        llm = build_llm(streaming=True)
        if llm is None:
            async def error_stream():
                payload = {"event": "error", "error": {"code": "LLM_UNAVAILABLE", "message": "LLM 服务不可用"}}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # 构建消息列表
        lc_messages = []
        if request.system_prompt:
            lc_messages.append(SystemMessage(content=request.system_prompt))
        
        # 添加会话上下文
        if request.session_id:
            session_context = _get_session_context(request.session_id)
            for msg in session_context:
                if msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    from langchain_core.messages import AIMessage
                    lc_messages.append(AIMessage(content=msg["content"]))
        
        # 添加当前消息
        lc_messages.append(HumanMessage(content=user_input))
        
        async def generate():
            full_response = ""
            for chunk in llm.stream(lc_messages):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                full_response += content
                payload = {"delta": content}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            
            # 保存到会话
            if request.session_id:
                _add_message_to_session(request.session_id, "user", user_input)
                _add_message_to_session(request.session_id, "assistant", full_response)
            
            yield "data: {\"event\":\"done\"}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"流式对话失败: {e}")
        async def error_stream():
            payload = {
                "event": "error",
                "error": {
                    "code": "PYTHON_STREAM_ERROR",
                    "message": str(e),
                },
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

# ============ Agent API ============

@app.post("/api/agent/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest):
    """
    Agent 执行端点 - 支持会话和安全检查
    """
    # 1. 安全检查
    input_guard = get_input_guard()
    validation = input_guard.validate(request.task)
    
    if not validation.is_safe:
        logger.warning(f"Agent 输入被拒绝: {validation.reason}")
        return AgentResponse(
            success=False,
            answer=_get_refusal_response(validation.reason),
            session_id=request.session_id,
            error="SECURITY_BLOCKED",
        )
    
    # 2. 获取会话上下文
    context = request.context or {}
    if request.session_id:
        session_context = _get_session_context(request.session_id)
        context["session_history"] = session_context
    
    try:
        # 3. 执行 Agent
        executor = get_executor()
        result: AgentResult = executor.execute(request.task, context)
        
        # 4. 输出过滤
        output_guard = get_output_guard()
        filtered_answer = output_guard.filter(result.answer)
        
        # 5. 保存到会话
        if request.session_id:
            _add_message_to_session(request.session_id, "user", request.task)
            _add_message_to_session(request.session_id, "assistant", filtered_answer)
        
        return AgentResponse(
            success=result.success,
            answer=filtered_answer,
            thought_chain=result.thought_chain,
            steps=result.steps,
            session_id=request.session_id,
            error=result.error,
        )
        
    except Exception as e:
        logger.error(f"Agent 执行失败: {e}")
        return AgentResponse(
            success=False,
            answer=f"执行失败: {e}",
            session_id=request.session_id,
            error=str(e),
        )

# ============ Bot API (兼容 Java 端 AiBotController) ============

class BotAskRequest(BaseModel):
    question: str = Field(..., description="问题")
    session_id: Optional[str] = Field(None, alias="sessionId", description="会话 ID")
    user_id: Optional[int] = Field(None, alias="userId", description="用户 ID")

    class Config:
        populate_by_name = True

class BotCodeRequest(BaseModel):
    code: str = Field(..., description="代码")
    error_log: Optional[str] = Field(None, alias="errorLog", description="错误日志")

    class Config:
        populate_by_name = True

# KOB 游戏规则系统提示词
KOB_SYSTEM_PROMPT = """你是 King of Bots (KOB) 贪吃蛇对战平台的 Bot 开发助手。

【重要】本项目的游戏规则（必须遵守，不要使用通用贪吃蛇知识）：
1. 这是一个双人贪吃蛇对战游戏，两条蛇在 13x14 的地图上对战
2. 本游戏【没有食物】！蛇的长度是自动增长的：
   - 前10步每步增长1格
   - 之后每3步增长1格（step % 3 == 1 时增长）
3. 获胜条件：让对手撞墙或撞到蛇身（自己或对手的）
4. 移动方向：0=上, 1=右, 2=下, 3=左
5. 地图坐标从(0,0)开始，有固定的障碍物墙壁

格式要求：使用 Markdown 格式，代码用 ```java 包裹。
"""

@app.post("/api/bot/ask")
async def bot_ask(request: BotAskRequest):
    """智能问答端点 - 支持多轮对话"""
    # 安全检查
    input_guard = get_input_guard()
    validation = input_guard.validate(request.question)
    if not validation.is_safe:
        return {
            "success": False,
            "error": f"输入不合法: {validation.reason}",
            "sessionId": request.session_id,
        }
    
    try:
        llm = get_llm()
        if llm is None:
            return {"success": False, "error": "AI 服务不可用", "sessionId": request.session_id}
        
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        # 构建消息
        lc_messages = [SystemMessage(content=KOB_SYSTEM_PROMPT)]
        
        # 添加会话上下文
        if request.session_id:
            session_context = _get_session_context(request.session_id)
            for msg in session_context:
                if msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    lc_messages.append(AIMessage(content=msg["content"]))
        
        lc_messages.append(HumanMessage(content=request.question))
        
        # 调用 LLM
        response = llm.invoke(lc_messages)
        answer = response.content
        
        # 输出过滤
        output_guard = get_output_guard()
        answer = output_guard.filter(answer)
        
        # 保存到会话
        if request.session_id:
            _add_message_to_session(request.session_id, "user", request.question)
            _add_message_to_session(request.session_id, "assistant", answer)
        
        return {
            "success": True,
            "answer": answer,
            "sessionId": request.session_id,
            "sources": [],
        }
        
    except Exception as e:
        logger.error(f"智能问答失败: {e}")
        return {"success": False, "error": f"服务暂时不可用: {e}", "sessionId": request.session_id}

@app.get("/api/bot/stream")
async def bot_stream(question: str = Query(...)):
    """流式智能问答端点 (SSE) - 兼容 Java 端"""
    # 安全检查
    input_guard = get_input_guard()
    validation = input_guard.validate(question)
    if not validation.is_safe:
        async def error_stream():
            yield f"event: error\ndata: {{\"error\": \"输入不合法\"}}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    try:
        from llm_client import build_llm
        llm = build_llm(streaming=True)
        if llm is None:
            async def error_stream():
                yield f"event: error\ndata: {{\"error\": \"AI 服务不可用\"}}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        lc_messages = [
            SystemMessage(content=KOB_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
        
        async def generate():
            # 发送开始事件
            yield f"event: start\ndata: {{\"status\": \"processing\"}}\n\n"
            
            for chunk in llm.stream(lc_messages):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                # 编码换行符
                encoded = content.replace("\n", "___NEWLINE___")
                yield f"event: chunk\ndata: {encoded}\n\n"
            
            yield f"event: done\ndata: {{\"status\": \"completed\"}}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"流式问答失败: {e}")
        async def error_stream():
            yield f"event: error\ndata: {{\"error\": \"{e}\"}}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

@app.post("/api/bot/analyze")
async def bot_analyze(request: BotCodeRequest):
    """代码分析端点"""
    if not request.code or not request.code.strip():
        return {"success": False, "error": "请提供代码"}
    
    try:
        llm = get_llm()
        if llm is None:
            return {"success": False, "error": "AI 服务不可用"}
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        analysis_prompt = f"""分析以下 KOB 贪吃蛇 Bot 代码，给出详细的分析报告：

```java
{request.code}
```

请从以下几个方面分析：
1. **代码逻辑**：代码的主要策略是什么？
2. **优点**：代码的优势和亮点
3. **问题**：潜在的 bug 或逻辑问题
4. **优化建议**：如何改进这个策略

注意：本游戏没有食物，蛇的长度是自动增长的。"""

        lc_messages = [
            SystemMessage(content=KOB_SYSTEM_PROMPT),
            HumanMessage(content=analysis_prompt),
        ]
        
        response = llm.invoke(lc_messages)
        analysis = response.content
        
        # 输出过滤
        output_guard = get_output_guard()
        analysis = output_guard.filter(analysis)
        
        return {"success": True, "analysis": analysis}
        
    except Exception as e:
        logger.error(f"代码分析失败: {e}")
        return {"success": False, "error": f"服务暂时不可用: {e}"}

@app.get("/api/bot/analyze/stream")
async def bot_analyze_stream(code: str = Query(...)):
    """流式代码分析端点 (SSE)"""
    if not code or not code.strip():
        async def error_stream():
            yield f"event: error\ndata: {{\"error\": \"请提供代码\"}}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    try:
        from llm_client import build_llm
        llm = build_llm(streaming=True)
        if llm is None:
            async def error_stream():
                yield f"event: error\ndata: {{\"error\": \"AI 服务不可用\"}}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        analysis_prompt = f"""分析以下 KOB 贪吃蛇 Bot 代码：

```java
{code}
```

请简洁地分析：1.策略概述 2.优点 3.问题 4.优化建议"""

        lc_messages = [
            SystemMessage(content=KOB_SYSTEM_PROMPT),
            HumanMessage(content=analysis_prompt),
        ]
        
        async def generate():
            yield f"event: start\ndata: 开始分析代码...\n\n"
            
            for chunk in llm.stream(lc_messages):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                encoded = content.replace("\n", "___NEWLINE___")
                yield f"event: chunk\ndata: {encoded}\n\n"
            
            yield f"event: done\ndata: 完成\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"流式代码分析失败: {e}")
        async def error_stream():
            yield f"event: error\ndata: {{\"error\": \"{e}\"}}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

@app.post("/api/bot/fix")
async def bot_fix(request: BotCodeRequest):
    """代码修复端点"""
    if not request.code or not request.code.strip():
        return {"success": False, "error": "请提供代码"}
    
    try:
        llm = get_llm()
        if llm is None:
            return {"success": False, "error": "AI 服务不可用"}
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        error_info = f"\n\n错误信息：\n{request.error_log}" if request.error_log else ""
        
        fix_prompt = f"""修复以下 KOB 贪吃蛇 Bot 代码中的问题：

```java
{request.code}
```
{error_info}

请：
1. 找出代码中的问题
2. 提供修复后的完整代码
3. 简要说明修复了什么

注意：本游戏没有食物，蛇的长度是自动增长的。移动方向：0=上, 1=右, 2=下, 3=左。"""

        lc_messages = [
            SystemMessage(content=KOB_SYSTEM_PROMPT),
            HumanMessage(content=fix_prompt),
        ]
        
        response = llm.invoke(lc_messages)
        result = response.content
        
        # 输出过滤
        output_guard = get_output_guard()
        result = output_guard.filter(result)
        
        # 尝试提取代码块
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
        return {"success": False, "error": f"服务暂时不可用: {e}"}

@app.get("/api/bot/status")
async def bot_status():
    """AI 服务状态"""
    llm = get_llm()
    return {
        "embeddingEnabled": False,  # Python 端暂不支持 embedding
        "chatEnabled": llm is not None,
        "pythonService": True,
    }

# ============ RAG API ============

@app.post("/api/rag/search")
async def rag_search(query: str = Query(...), limit: int = Query(default=5)):
    """RAG 检索端点"""
    try:
        from rag.hybrid_search import hybrid_search
        results = hybrid_search(query, limit=limit)
        return {"query": query, "results": results}
    except Exception as e:
        logger.error(f"RAG 检索失败: {e}")
        return {"query": query, "results": [], "error": str(e)}

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
