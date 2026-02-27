import $ from 'jquery'

const AI_STREAM_BACKEND = (process.env.VUE_APP_AI_STREAM_BACKEND || 'python').toLowerCase()
const PYTHON_AI_URL = process.env.VUE_APP_PYTHON_AI_URL || 'http://localhost:3003'

export function getAuthHeaders() {
    const token = localStorage.getItem('jwt_token')
    if (!token) return {}
    return { Authorization: `Bearer ${token}` }
}

function createSseParser(onEvent) {
    let buffer = ''
    let eventName = ''
    let dataBuffer = ''

    const dispatch = () => {
        if (!eventName && dataBuffer === '') return
        const data = dataBuffer.endsWith('\n') ? dataBuffer.slice(0, -1) : dataBuffer
        onEvent(eventName || 'message', data)
        eventName = ''
        dataBuffer = ''
    }

    const feedLine = (line) => {
        if (line === '') {
            dispatch()
            return
        }
        if (line.startsWith('event:')) {
            eventName = line.slice(6).trim()
            return
        }
        if (line.startsWith('data:')) {
            let value = line.slice(5)
            if (value.startsWith(' ')) value = value.slice(1)
            dataBuffer += value + '\n'
        }
    }

    return {
        feed(chunk) {
            buffer += chunk
            const lines = buffer.split('\n')
            buffer = lines.pop() || ''
            for (let line of lines) {
                if (line.endsWith('\r')) line = line.slice(0, -1)
                feedLine(line)
            }
        },
        flush() {
            if (!buffer) return
            let line = buffer
            if (line.endsWith('\r')) line = line.slice(0, -1)
            feedLine(line)
            buffer = ''
        },
    }
}

function safeJsonParse(raw) {
    if (!raw) return null
    try {
        return JSON.parse(raw)
    } catch (e) {
        return null
    }
}

function streamSsePost(endpoint, payload, onChunk, onComplete, onError) {
    const controller = new AbortController()
    let completed = false

    const finish = (payload = null) => {
        if (completed) return
        completed = true
        if (onComplete) onComplete(payload)
    }

    fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify(payload),
        signal: controller.signal
    }).then(response => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        if (!response.body) throw new Error('ReadableStream not supported')
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        const parser = createSseParser((event, raw) => {
            const json = safeJsonParse(raw)
            const rawText = raw || ''

            if (event === 'done' || (json && json.status === 'completed')) {
                finish(json || rawText)
                return
            }

            if (event === 'error') {
                const message = (json && (json.error || json.message)) || rawText || 'SSE error'
                if (onError) onError(new Error(message))
                return
            }

            if (json && json.error && onError) {
                onError(new Error(json.error))
                return
            }

            if (json && json.delta !== undefined) {
                if (onChunk) onChunk(json.delta)
                return
            }

            if (rawText && onChunk) {
                onChunk(rawText)
            }
        })

        const read = () => {
            reader.read().then(({ done, value }) => {
                if (done) {
                    parser.flush()
                    finish()
                    return
                }
                parser.feed(decoder.decode(value, { stream: true }))
                read()
            }).catch(err => {
                if (err.name !== 'AbortError' && onError) onError(err)
            })
        }

        read()
    }).catch(err => {
        if (err.name !== 'AbortError' && onError) onError(err)
    })

    return { close: () => controller.abort() }
}

export function fetchAiHint(data) {
    return $.ajax({
        url: "/ai/hint",
        type: "post",
        contentType: "application/json",
        data: JSON.stringify(data),
    });
}

// 流式 AI 提示（SSE + POST，2026 最佳实践）
export function streamAiHint(question, onChunk, onComplete, onError) {
    // 使用 Python 后端 POST 方式，与智能问答一致
    const usePython = AI_STREAM_BACKEND === 'python';
    const endpoint = usePython ? `${PYTHON_AI_URL}/api/bot/chat` : '/ai/hint/stream';
    
    console.log('[Hint API] Stream hint to:', endpoint);
    
    if (usePython) {
        // Python: POST /api/bot/chat (stream=true)
        return streamSsePost(endpoint, { question, stream: true }, onChunk, onComplete, onError)
    } else {
        // Java: GET /ai/hint/stream (SSE EventSource)
        const url = `/ai/hint/stream?question=${encodeURIComponent(question)}`;
        console.log('[Hint API] Creating EventSource for:', url);
        const eventSource = new EventSource(url);
        
        eventSource.onopen = () => {
            console.log('[Hint API] EventSource connection opened');
        };
        
        eventSource.addEventListener('chunk', (e) => {
            const decodedData = e.data.replace(/___NEWLINE___/g, '\n');
            onChunk(decodedData);
        });
        
        eventSource.addEventListener('done', () => {
            eventSource.close();
            onComplete();
        });
        
        eventSource.addEventListener('error', (e) => {
            console.error('[Hint API] error event:', e);
            eventSource.close();
            if (onError) onError(e);
        });
        
        return eventSource;
    }
}

// AI Bot 助手 API

// 智能问答（RAG 优化版，支持多轮对话）
export function askAiBot(question, sessionId = null, userId = null) {
    const data = { question };
    if (sessionId) data.sessionId = sessionId;
    if (userId) data.userId = userId;

    return $.ajax({
        url: "/ai/bot/ask",
        type: "post",
        contentType: "application/json",
        data: JSON.stringify(data),
    });
}

// ========== 会话管理 API ==========

// 创建新会话
export function createSession(userId = 0) {
    return $.ajax({
        url: "/ai/session/create",
        type: "post",
        contentType: "application/json",
        data: JSON.stringify({ userId }),
    });
}

// 获取会话列表
export function listSessions(userId = 0, limit = 20) {
    return $.ajax({
        url: `/ai/session/list?userId=${userId}&limit=${limit}`,
        type: "get",
    });
}

// 获取会话历史
export function getSessionHistory(sessionId) {
    return $.ajax({
        url: `/ai/session/history?sessionId=${sessionId}`,
        type: "get",
    });
}

// 删除会话
export function deleteSession(sessionId) {
    return $.ajax({
        url: `/ai/session/delete?sessionId=${sessionId}`,
        type: "delete",
    });
}

// 更新会话标题
export function updateSessionTitle(sessionId, title) {
    return $.ajax({
        url: "/ai/session/title",
        type: "put",
        contentType: "application/json",
        data: JSON.stringify({ sessionId, title }),
    });
}

// 流式智能问答（SSE + POST，支持多轮对话）
// 2026 最佳实践：使用 Python 后端 POST 方式，更稳定可靠
export function streamAskAiBot(question, onChunk, onComplete, onError, sessionId = null) {
    const endpoint = `${PYTHON_AI_URL}/api/bot/chat`;
    console.log('[API] Stream chat to Python:', endpoint, 'sessionId:', sessionId);
    return streamSsePost(endpoint, { question, sessionId, stream: true }, onChunk, onComplete, onError)
}

// 流式代码生成 (SSE)
// onGuidance: 公平性检查回调，当用户没有提供思路时触发
export function streamGenerateBotCode(description, onChunk, onComplete, onError, sessionId = null, onGuidance = null) {
    const usePython = AI_STREAM_BACKEND === 'python';

    if (usePython) {
        // Python: POST /api/bot/chat (stream=true)
        const endpoint = `${PYTHON_AI_URL}/api/bot/chat`;
        const prompt = `请帮我编写一个 KOB 贪吃蛇 Bot，策略如下：\n${description}\n\n请直接给出完整的 Java 代码，类名为 Bot，并实现 nextMove 方法。`;
        
        console.log('[API] Stream generate code from Python:', endpoint, 'sessionId:', sessionId);
        return streamSsePost(endpoint, { question: prompt, sessionId, stream: true }, onChunk, onComplete, onError)
    } else {
        // Java: GET /ai/bot/generate/stream (SSE)
        const url = `/ai/bot/generate/stream?description=${encodeURIComponent(description)}`;
        console.log('[API] Creating EventSource for code generation (Java):', url);
        const eventSource = new EventSource(url);
        
        eventSource.addEventListener('start', (e) => {
            console.log('[API] Code generation started:', e.data);
        });
        
        eventSource.addEventListener('chunk', (e) => {
            const decodedData = e.data.replace(/___NEWLINE___/g, '\n');
            onChunk(decodedData);
        });
        
        // 公平性检查：需要用户提供思路
        eventSource.addEventListener('guidance', (e) => {
            console.log('[API] Fairness check: user needs to provide idea');
            const decodedData = e.data.replace(/___NEWLINE___/g, '\n');
            if (onGuidance) onGuidance(decodedData);
        });
        
        eventSource.addEventListener('done', (e) => {
            console.log('[API] Code generation completed:', e.data);
            eventSource.close();
            onComplete(e.data); // 传递状态（可能是 'needs_idea'）
        });
        
        eventSource.addEventListener('error', (e) => {
            console.error('[API] Code generation error:', e);
            eventSource.close();
            if (onError) onError(e);
        });
        
        return eventSource;
    }
}

// 分析代码（支持 Python 后端）
export function analyzeCode(code) {
    const url = AI_STREAM_BACKEND === 'python' 
        ? `${PYTHON_AI_URL}/api/bot/analyze`
        : "/ai/bot/analyze";
    return $.ajax({
        url: url,
        type: "post",
        contentType: "application/json",
        data: JSON.stringify({ code }),
    });
}

// 流式代码分析 (POST + SSE，支持多轮对话)
export function streamAnalyzeCode(code, onChunk, onComplete, onError, sessionId = null) {
    const usePython = AI_STREAM_BACKEND === 'python';
    
    if (usePython) {
        // Python: 使用统一的 chat 端点，支持多轮对话上下文
        const endpoint = `${PYTHON_AI_URL}/api/bot/chat`;
        const prompt = `请分析以下 KOB 贪吃蛇 Bot 代码：\n\n\`\`\`java\n${code}\n\`\`\`\n\n请从策略概述、优点、问题、优化建议几个方面进行分析。`;
        
        console.log('[API] Stream analyze code from Python:', endpoint, 'sessionId:', sessionId);
        return streamSsePost(endpoint, { question: prompt, sessionId, stream: true }, onChunk, onComplete, onError)
    } else {
        // Java: POST /ai/bot/analyze
        const endpoint = '/ai/bot/analyze';
        
        fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code, stream: true })
        }).then(response => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            function processStream() {
                reader.read().then(({ done, value }) => {
                    if (done) { onComplete(); return; }
                    
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.delta) onChunk(data.delta);
                            } catch (e) { /* ignore */ }
                        } else if (line.startsWith('event: done')) {
                            onComplete(); return;
                        }
                    }
                    processStream();
                }).catch(err => { if (onError) onError(err); });
            }
            processStream();
        }).catch(err => { if (onError) onError(err); });
        
        return { close: () => {} };
    }
}

// 修复代码（支持 Python 后端）
export function fixCode(code, errorLog) {
    const url = AI_STREAM_BACKEND === 'python' 
        ? `${PYTHON_AI_URL}/api/bot/fix`
        : "/ai/bot/fix";
    return $.ajax({
        url: url,
        type: "post",
        contentType: "application/json",
        data: JSON.stringify({ code, errorLog }),
    });
}

// 流式修复代码 (POST + SSE，支持多轮对话)
export function streamFixCode(code, errorLog, onChunk, onComplete, onError, sessionId = null) {
    const usePython = AI_STREAM_BACKEND === 'python';
    
    if (usePython) {
        const endpoint = `${PYTHON_AI_URL}/api/bot/chat`;
        const errorInfo = errorLog ? `\n\n错误信息：\n${errorLog}` : '';
        const prompt = `请修复以下 KOB 贪吃蛇 Bot 代码中的问题：\n\n\`\`\`java\n${code}\n\`\`\`${errorInfo}\n\n请找出问题、提供修复后的完整代码，并说明修复了什么。`;
        
        console.log('[API] Stream fix code from Python:', endpoint, 'sessionId:', sessionId);
        return streamSsePost(endpoint, { question: prompt, sessionId, stream: true }, onChunk, onComplete, onError)
    } else {
        // Java: 降级为非流式
        fixCode(code, errorLog).then(resp => {
            if (resp.success) {
                onChunk(resp.explanation || resp.fixedCode);
                onComplete();
            } else {
                onError(new Error(resp.error));
            }
        }).catch(onError);
        
        return { close: () => {} };
    }
}

// 解释代码
export function explainCode(code) {
    return $.ajax({
        url: "/ai/bot/explain",
        type: "post",
        contentType: "application/json",
        data: JSON.stringify({ code }),
    });
}

// 对战分析
export function analyzeBattle(mapData, stepsA, stepsB, loser) {
    return $.ajax({
        url: "/ai/bot/battle/analyze",
        type: "post",
        contentType: "application/json",
        data: JSON.stringify({ mapData, stepsA, stepsB, loser }),
    });
}

// AI 服务状态
export function getAiStatus() {
    return $.ajax({
        url: "/ai/bot/status",
        type: "get",
    });
}

// ========== Human-in-the-Loop (HITL) API ==========

/**
 * 支持 Human-in-the-Loop 的聊天
 * 
 * 2026 LangGraph 最佳实践：
 * - 新对话: 提供 question
 * - 恢复执行: 提供 resumeData（用户确认/拒绝/编辑）
 * 
 * @param {Object} params - 请求参数
 * @param {string} params.question - 用户问题（新对话时必填）
 * @param {string} params.sessionId - 会话 ID
 * @param {Object} params.resumeData - 恢复执行数据
 * @returns {Promise} 返回响应
 */
export async function chatWithHITL(params) {
    const { question, sessionId, resumeData } = params
    
    try {
        const response = await fetch('/ai/bot/chat/hitl', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
            body: JSON.stringify({
                question,
                sessionId,
                resumeData
            })
        })
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`)
        }
        
        return await response.json()
    } catch (e) {
        console.error('[HITL API] Error:', e)
        throw e
    }
}

/**
 * 恢复 HITL 执行（用户确认/拒绝/编辑后）
 * 
 * @param {string} sessionId - 会话 ID
 * @param {Object} decision - 用户决策 { type: 'approve'|'reject'|'edit', edited_code?: string }
 * @returns {Promise} 返回响应
 */
export async function resumeHITL(sessionId, decision) {
    return chatWithHITL({
        sessionId,
        resumeData: decision
    })
}

// ========== Time Travel API ==========

/**
 * 获取会话的检查点列表
 * 
 * @param {string} sessionId - 会话 ID
 * @param {number} limit - 返回数量限制
 * @returns {Promise} 返回检查点列表
 */
export async function getCheckpoints(sessionId, limit = 20) {
    const response = await fetch(
        `${PYTHON_AI_URL}/api/chat/checkpoints?sessionId=${encodeURIComponent(sessionId)}&limit=${limit}`
    );
    return response.json();
}

/**
 * 从检查点分叉对话（Time Travel 核心功能）
 * 
 * @param {string} sessionId - 原会话 ID
 * @param {string} checkpointId - 检查点 ID
 * @param {string} question - 新问题
 * @param {number} userId - 用户 ID
 * @returns {Promise} 返回新会话信息和回答
 */
export async function forkFromCheckpoint(sessionId, checkpointId, question, userId = null) {
    const response = await fetch(`${PYTHON_AI_URL}/api/chat/fork`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            sessionId, 
            checkpointId, 
            question,
            userId
        })
    });
    return response.json();
}

/**
 * 获取会话的分支信息
 * 
 * @param {string} sessionId - 会话 ID
 * @returns {Promise} 返回分支信息
 */
export async function getBranches(sessionId) {
    const response = await fetch(
        `${PYTHON_AI_URL}/api/chat/branches?sessionId=${encodeURIComponent(sessionId)}`
    );
    return response.json();
}

// ========== Multimodal Chat API ==========

/**
 * 多模态聊天（支持图片）
 * 
 * @param {Object} params - 请求参数
 * @param {string} params.question - 用户问题
 * @param {string} params.imageData - Base64 图片数据
 * @param {string} params.sessionId - 会话 ID
 * @param {number} params.userId - 用户 ID
 * @param {boolean} params.stream - 是否流式输出（图片分析不支持）
 * @returns {Promise} 返回响应
 */
export async function chatWithImage(params) {
    const { question, imageData, sessionId, userId, stream = false } = params;
    
    const response = await fetch(`${PYTHON_AI_URL}/api/chat/multimodal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            question,
            imageData,
            imageSource: 'base64',
            sessionId,
            userId,
            stream
        })
    });
    return response.json();
}
