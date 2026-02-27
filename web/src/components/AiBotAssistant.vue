<template>
    <div class="ai-bot-assistant">
        <!-- 功能选择标签 -->
        <ul class="nav nav-tabs mb-3">
            <li class="nav-item">
                <a class="nav-link" :class="{ active: activeTab === 'ask' }" @click="activeTab = 'ask'" href="#">
                    智能问答
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" :class="{ active: activeTab === 'generate' }" @click="activeTab = 'generate'" href="#">
                    代码生成
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" :class="{ active: activeTab === 'analyze' }" @click="activeTab = 'analyze'" href="#">
                    代码分析
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" :class="{ active: activeTab === 'fix' }" @click="activeTab = 'fix'" href="#">
                    修复代码
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" :class="{ active: activeTab === 'vision' }" @click="activeTab = 'vision'" href="#">
                    📷 图片分析
                </a>
            </li>
        </ul>

        <!-- 智能问答 -->
        <div v-if="activeTab === 'ask'" class="tab-content">
            <!-- 聊天历史 -->
            <div class="chat-container mb-3" ref="chatContainer">
                <div v-if="askData.messages.length === 0" class="text-center text-muted mt-5">
                    <p>👋 你好！我是 KOB 智能助手。</p>
                    <p>你可以问我关于贪吃蛇策略、代码实现或比赛规则的问题。</p>
                </div>
                <div v-else class="chat-history">
                    <div v-for="(msg, index) in askData.messages" :key="index" 
                         class="message-item mb-3" :class="msg.role"
                         @mouseenter="hoveredMessage = index"
                         @mouseleave="hoveredMessage = null">
                        <div class="message-role small text-muted mb-1 d-flex align-items-center">
                            {{ msg.role === 'user' ? '你' : 'AI 助手' }}
                        </div>
                        <div class="message-body">
                            <div class="message-content p-2 rounded">
                                <!-- 显示图片（如有）-->
                                <img v-if="msg.image" :src="msg.image" class="msg-image mb-2 rounded" style="max-height: 150px;">
                                
                                <div v-if="msg.role === 'assistant'" v-html="formatMarkdown(msg.content)"></div>
                                <div v-else>{{ msg.content }}</div>
                                <div v-if="msg.loading" class="spinner-border spinner-border-sm ms-2"></div>
                            </div>
                            
                            <!-- Time Travel: 回退按钮 (仅 AI 消息且有 checkpointId) -->
                            <button v-if="msg.checkpointId && msg.role === 'assistant'"
                                    class="btn btn-sm btn-outline-primary fork-btn"
                                    @click.stop="handleForkFromMessage(msg.checkpointId, index)"
                                    title="从这里分叉对话">
                                回退
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- HITL: 修改建议确认面板 -->
            <BotModificationProposal 
                v-if="askData.pendingProposal"
                :proposal="askData.pendingProposal"
                :session-id="askData.sessionId"
                @resumed="handleProposalResumed"
                @error="handleProposalError"
                class="mb-3"
            />

            <!-- 输入框 -->
            <div class="input-group">
                <!-- 图片上传按钮 -->
                <input type="file" ref="chatImageInput" accept="image/*"
                       @change="handleChatImageSelect" style="display: none">
                <button class="btn btn-outline-secondary"
                        @click="$refs.chatImageInput.click()"
                        :disabled="loading || askData.pendingProposal"
                        title="添加图片">
                    <i class="bi bi-image"></i>
                </button>
                
                <textarea v-model="askData.question" class="form-control" rows="2" 
                    placeholder="请输入你的问题... (Ctrl+Enter 发送)" 
                    @keydown.enter.ctrl="handleAsk"
                    :disabled="askData.pendingProposal != null"></textarea>
                <button class="btn btn-primary" :disabled="loading || askData.pendingProposal" @click="handleAsk">
                    发送
                </button>
            </div>
            
            <!-- 图片预览 -->
            <div v-if="askData.pendingImage" class="mt-2 d-flex align-items-center gap-2">
                <img :src="askData.pendingImage.preview" class="img-thumbnail" style="max-height: 60px;">
                <button class="btn btn-sm btn-outline-danger" @click="askData.pendingImage = null">
                    移除
                </button>
            </div>
            <div class="d-flex justify-content-between align-items-center mt-2">
                <div>
                    <small class="text-muted me-2">支持多轮对话</small>
                    <span v-if="askData.useHITL" class="badge bg-info">Bot 修改模式</span>
                    
                    <!-- 分支指示器 -->
                    <span v-if="askData.branchId && askData.branchId !== 'main'"
                          class="badge bg-warning text-dark ms-2">
                        分支: {{ askData.branchId }}
                        <button class="btn-close btn-close-white ms-1"
                                @click="switchToMain"
                                style="font-size: 0.6em;"></button>
                    </span>
                </div>
                <button v-if="askData.messages.length > 0" class="btn btn-sm btn-outline-danger" @click="clearConversation">
                    清空对话
                </button>
            </div>
        </div>

        <!-- 代码生成 -->
        <div v-if="activeTab === 'generate'" class="tab-content">
            <div class="mb-3">
                <label class="form-label">策略描述</label>
                <textarea v-model="generateData.description" class="form-control" rows="3" 
                    placeholder="例如：我想用 BFS 计算四个方向的可达空间，选择空间最大的方向移动"></textarea>
            </div>
            <button class="btn btn-primary mb-3" :disabled="loading" @click="handleGenerate">
                <span v-if="loading" class="spinner-border spinner-border-sm me-1"></span>
                生成代码
            </button>
            
            <!-- 公平性提示：需要用户提供思路 -->
            <div v-if="generateData.needsIdea" class="alert alert-info">
                <div class="markdown-content" v-html="formatMarkdown(generateData.guidance)"></div>
                <hr>
                <p class="mb-2"><strong>💡 示例描述（点击使用）：</strong></p>
                <div class="d-flex flex-wrap gap-2">
                    <button class="btn btn-sm btn-outline-primary" @click="generateData.description = '我想用 BFS 找到离对手最远的安全位置'">
                        BFS 找安全位置
                    </button>
                    <button class="btn btn-sm btn-outline-primary" @click="generateData.description = '我的思路是先计算四个方向的可用空间，选择空间最大的方向'">
                        计算可用空间
                    </button>
                    <button class="btn btn-sm btn-outline-primary" @click="generateData.description = '我想实现一个能预测对手下一步移动的策略'">
                        预测对手移动
                    </button>
                </div>
            </div>
            
            <div v-if="generateData.code" class="result-box">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <h6 class="mb-0">生成的代码</h6>
                    <button class="btn btn-sm btn-outline-secondary" @click="copyCode(generateData.code)">
                        复制代码
                    </button>
                </div>
                <pre class="code-block"><code>{{ generateData.code }}</code></pre>
            </div>
        </div>

        <!-- 代码分析 -->
        <div v-if="activeTab === 'analyze'" class="tab-content">
            <div class="mb-3">
                <label class="form-label">Bot 代码</label>
                <textarea v-model="analyzeData.code" class="form-control code-input" rows="8" 
                    placeholder="粘贴你的 Bot 代码..."></textarea>
            </div>
            <button class="btn btn-primary mb-3" :disabled="loading" @click="handleAnalyze">
                <span v-if="loading" class="spinner-border spinner-border-sm me-1"></span>
                分析代码
            </button>
            <div v-if="analyzeData.analysis" class="result-box">
                <h6>分析结果</h6>
                <div class="markdown-content" v-html="formatMarkdown(analyzeData.analysis)"></div>
            </div>
        </div>

        <!-- 修复代码 -->
        <div v-if="activeTab === 'fix'" class="tab-content">
            <div class="mb-3">
                <label class="form-label">有问题的代码</label>
                <textarea v-model="fixData.code" class="form-control code-input" rows="6" 
                    placeholder="粘贴有问题的代码..."></textarea>
            </div>
            <div class="mb-3">
                <label class="form-label">错误信息（可选）</label>
                <textarea v-model="fixData.errorLog" class="form-control" rows="3" 
                    placeholder="粘贴错误日志..."></textarea>
            </div>
            <button class="btn btn-primary mb-3" :disabled="loading" @click="handleFix">
                <span v-if="loading" class="spinner-border spinner-border-sm me-1"></span>
                修复代码
            </button>
            <div v-if="fixData.explanation" class="result-box">
                <h6>修复建议</h6>
                <div class="markdown-content" v-html="formatMarkdown(fixData.explanation)"></div>
            </div>
        </div>

        <!-- 图片分析 -->
        <div v-if="activeTab === 'vision'" class="tab-content">
            <div class="mb-3">
                <label class="form-label">上传图片（对战截图、代码截图、地图）</label>
                <input type="file" class="form-control" accept="image/*" @change="handleImageUpload" ref="imageInput">
            </div>
            
            <div v-if="visionData.preview" class="mb-3">
                <img :src="visionData.preview" class="img-thumbnail" style="max-height: 200px;">
            </div>
            
            <div class="mb-3">
                <label class="form-label">分析类型</label>
                <select v-model="visionData.analysisType" class="form-select">
                    <option value="battle">对战截图分析</option>
                    <option value="code">代码截图分析</option>
                    <option value="map">地图分析</option>
                </select>
            </div>
            
            <div class="mb-3">
                <label class="form-label">问题（可选）</label>
                <input v-model="visionData.question" class="form-control" placeholder="例如：当前局面谁占优？我应该往哪个方向走？">
            </div>
            
            <button class="btn btn-primary mb-3" :disabled="loading || !visionData.imageData" @click="handleVisionAnalyze">
                <span v-if="loading" class="spinner-border spinner-border-sm me-1"></span>
                分析图片
            </button>
            
            <div v-if="visionData.analysis" class="result-box">
                <h6>分析结果</h6>
                <div class="markdown-content" v-html="formatMarkdown(visionData.analysis)"></div>
            </div>
            
            <div v-if="!visionData.available" class="alert alert-warning">
                ⚠️ 图片分析功能需要配置 Vision 模型（GPT-4o 或 Qwen-VL）
            </div>
        </div>

        <!-- 错误提示 -->
        <div v-if="error" class="alert alert-danger mt-3">
            {{ error }}
        </div>
    </div>
</template>

<script>
import { ref, reactive, nextTick, onMounted } from 'vue'
import { streamAskAiBot, streamGenerateBotCode, streamAnalyzeCode, streamFixCode, chatWithHITL, forkFromCheckpoint, chatWithImage, getCheckpoints } from '@/api/ai'
import { marked } from 'marked'
import BotModificationProposal from './BotModificationProposal.vue'

export default {
    name: 'AiBotAssistant',
    components: {
        BotModificationProposal
    },
    props: {
        userId: {
            type: [Number, String],
            default: null
        }
    },
    emits: ['bot-updated'],
    setup(props, { emit }) {
        const activeTab = ref('ask')
        const loading = ref(false)
        const error = ref('')
        const chatContainer = ref(null)

        const askData = reactive({
            question: '',
            messages: [],
            sessionId: '',
            pendingProposal: null,  // HITL: 待确认的修改建议
            useHITL: true,  // 是否使用 HITL 模式（支持 Bot 代码修改）
            // Time Travel 新增
            pendingImage: null,      // {data: base64, preview: dataUrl}
            branchId: 'main',        // 当前分支
            parentSessionId: null,   // 父会话（分叉来源）
        })
        
        const hoveredMessage = ref(null)
        const chatImageInput = ref(null)

        const generateData = reactive({
            description: '',
            code: '',
            explanation: '',
            needsIdea: false,
            guidance: ''
        })

        const analyzeData = reactive({
            code: '',
            analysis: ''
        })

        const fixData = reactive({
            code: '',
            errorLog: '',
            fixedCode: '',
            explanation: ''
        })

        const visionData = reactive({
            imageData: '',
            preview: '',
            analysisType: 'battle',
            question: '',
            analysis: '',
            available: true
        })
        
        const imageInput = ref(null)

        // 生成 UUID
        const generateUUID = () => {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        }

        onMounted(() => {
            // 初始化 sessionId
            if (!askData.sessionId) {
                askData.sessionId = generateUUID();
            }
            
            // 初始欢迎语
            if (askData.messages.length === 0) {
                askData.messages.push({
                    role: 'assistant',
                    content: '你好！我是 KOB 智能助手，有什么可以帮你的吗？'
                });
            }
        });

        const scrollToBottom = () => {
            nextTick(() => {
                if (chatContainer.value) {
                    const history = chatContainer.value.querySelector('.chat-history');
                    if (history) {
                        history.scrollTop = history.scrollHeight;
                    }
                }
            })
        }

        const handleAsk = async () => {
            if (!askData.question.trim() && !askData.pendingImage) return;
            if (loading.value) return;
            
            // 如果有图片，使用多模态端点
            if (askData.pendingImage) {
                await handleMultimodalAsk();
                return;
            }

            const userQuestion = askData.question;
            askData.messages.push({
                role: 'user',
                content: userQuestion
            });
            askData.question = '';
            askData.pendingProposal = null;  // 清除之前的 proposal
            scrollToBottom();

            loading.value = true;
            error.value = '';

            // HITL 模式：使用 Agent 执行，支持 Bot 代码修改
            if (askData.useHITL) {
                console.log('[HITL] Starting HITL chat, sessionId:', askData.sessionId);
                
                try {
                    const response = await chatWithHITL({
                        question: userQuestion,
                        sessionId: askData.sessionId
                    });
                    
                    if (response.success) {
                        if (response.interrupted) {
                            // 需要用户确认
                            console.log('[HITL] Interrupted, showing proposal');
                            askData.pendingProposal = response.interruptData?.proposal || response.interruptData;
                            askData.messages.push({
                                role: 'assistant',
                                content: '我已生成修改建议，请在下方确认或拒绝修改。',
                                isProposal: true,
                                checkpointId: response.checkpointId || null
                            });
                        } else {
                            // 正常完成
                            askData.messages.push({
                                role: 'assistant',
                                content: response.answer || '已处理完成。',
                                checkpointId: response.checkpointId || null
                            });
                        }
                    } else {
                        error.value = response.error || 'HITL 请求失败';
                        askData.messages.push({
                            role: 'assistant',
                            content: `抱歉，发生错误：${response.error}`
                        });
                    }
                } catch (e) {
                    console.error('[HITL] Error:', e);
                    error.value = 'AI 服务暂时不可用';
                    askData.messages.push({
                        role: 'assistant',
                        content: '抱歉，AI 服务暂时不可用，请稍后重试。'
                    });
                } finally {
                    loading.value = false;
                    scrollToBottom();
                }
                return;
            }

            // 普通流式模式
            const aiMessage = reactive({
                role: 'assistant',
                content: '',
                loading: true
            });
            askData.messages.push(aiMessage);
            scrollToBottom();

            console.log('[SSE] Starting stream chat, sessionId:', askData.sessionId);

            streamAskAiBot(
                userQuestion,
                (chunk) => {
                    aiMessage.loading = false;
                    aiMessage.content += chunk;
                    scrollToBottom();
                },
                async (donePayload) => {
                    console.log('[SSE] Chat completed');
                    if (donePayload && typeof donePayload === 'object' && donePayload.checkpointId) {
                        aiMessage.checkpointId = donePayload.checkpointId;
                    } else {
                        try {
                            const resp = await getCheckpoints(askData.sessionId, 1);
                            const latest = resp?.checkpoints?.[0];
                            if (latest?.checkpointId) {
                                aiMessage.checkpointId = latest.checkpointId;
                            }
                        } catch (e) {
                            console.warn('[SSE] Fetch checkpoints failed:', e);
                        }
                    }
                    aiMessage.loading = false;
                    loading.value = false;
                },
                (e) => {
                    console.error('[SSE] Chat error:', e);
                    error.value = 'AI 服务暂时不可用';
                    aiMessage.loading = false;
                    aiMessage.content += '\n\n[出错: 连接中断]';
                    loading.value = false;
                },
                askData.sessionId
            );
        }
        
        const handleProposalResumed = (result) => {
            console.log('[HITL] Proposal resumed:', result);
            askData.pendingProposal = null;
            
            if (result.answer) {
                askData.messages.push({
                    role: 'assistant',
                    content: result.answer,
                    checkpointId: result.checkpointId || null
                });
            }
            emit('bot-updated');  // 通知父组件刷新 Bot 列表
            scrollToBottom();
        }
        
        const handleProposalError = (errorMsg) => {
            console.error('[HITL] Proposal error:', errorMsg);
            error.value = errorMsg;
            askData.pendingProposal = null;
        }
        
        // ========== Time Travel 函数 ==========
        
        const handleForkFromMessage = async (checkpointId, messageIndex) => {
            // 弹出输入框让用户输入新问题
            const newQuestion = prompt('请输入你想问的新问题：');
            if (!newQuestion?.trim()) return;
            
            loading.value = true;
            error.value = '';
            
            try {
                const response = await forkFromCheckpoint(
                    askData.sessionId,
                    checkpointId,
                    newQuestion,
                    props.userId
                );
                
                if (response.success) {
                    // 保存父会话信息
                    askData.parentSessionId = askData.sessionId;
                    askData.sessionId = response.newSessionId;
                    askData.branchId = response.branchId;
                    
                    // 截断消息到分叉点，添加新对话
                    askData.messages = askData.messages.slice(0, messageIndex + 1);
                    askData.messages.push({
                        role: 'user',
                        content: newQuestion
                    });
                    askData.messages.push({
                        role: 'assistant',
                        content: response.answer,
                        checkpointId: response.checkpointId
                    });
                    
                    scrollToBottom();
                } else {
                    error.value = response.error || '分叉失败';
                }
            } catch (e) {
                console.error('[Fork] Error:', e);
                error.value = '分叉对话失败';
            } finally {
                loading.value = false;
            }
        };
        
        const handleChatImageSelect = (event) => {
            const file = event.target.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = (e) => {
                askData.pendingImage = {
                    preview: e.target.result,
                    data: e.target.result.split(',')[1]  // Base64
                };
            };
            reader.readAsDataURL(file);
        };
        
        const handleMultimodalAsk = async () => {
            const question = askData.question;
            const imageData = askData.pendingImage.data;
            const imagePreview = askData.pendingImage.preview;
            
            askData.messages.push({
                role: 'user',
                content: question,
                image: imagePreview
            });
            askData.question = '';
            askData.pendingImage = null;
            
            loading.value = true;
            scrollToBottom();
            
            try {
                const response = await chatWithImage({
                    question,
                    imageData,
                    sessionId: askData.sessionId,
                    userId: props.userId
                });
                
                if (response.success) {
                    askData.messages.push({
                        role: 'assistant',
                        content: response.answer,
                        checkpointId: response.checkpointId
                    });
                    if (response.sessionId) {
                        askData.sessionId = response.sessionId;
                    }
                } else {
                    error.value = response.error || '图片分析失败';
                    askData.messages.push({
                        role: 'assistant',
                        content: `抱歉，图片分析失败：${response.error}`
                    });
                }
            } catch (e) {
                console.error('[Multimodal] Error:', e);
                error.value = '多模态聊天失败';
            } finally {
                loading.value = false;
                scrollToBottom();
            }
        };
        
        const switchToMain = () => {
            // 切换回主分支（简单实现：清空当前对话）
            if (askData.parentSessionId) {
                askData.sessionId = askData.parentSessionId;
                askData.parentSessionId = null;
            }
            askData.branchId = 'main';
            clearConversation();
        };

        const clearConversation = () => {
            askData.sessionId = generateUUID();
            askData.messages = [{
                role: 'assistant',
                content: '你好！我是 KOB 智能助手，有什么可以帮你的吗？'
            }];
            askData.pendingProposal = null;
            error.value = '';
        }

        const handleGenerate = () => {
            if (!generateData.description.trim()) {
                error.value = '请输入策略描述';
                return;
            }
            loading.value = true;
            error.value = '';
            generateData.code = '';
            generateData.needsIdea = false;
            generateData.guidance = '';
            
            console.log('[SSE] Starting code generation, sessionId:', askData.sessionId);
            
            streamGenerateBotCode(
                generateData.description,
                (chunk) => {
                    generateData.code += chunk;
                },
                (status) => {
                    console.log('[SSE] Generation completed, status:', status);
                    loading.value = false;
                },
                (e) => {
                    console.error('[SSE] Generation error:', e);
                    error.value = 'AI 服务暂时不可用';
                    loading.value = false;
                },
                askData.sessionId,
                (guidance) => {
                    // 公平性检查：需要用户提供思路
                    generateData.needsIdea = true;
                    generateData.guidance = guidance;
                    loading.value = false;
                }
            );
        }

        const handleAnalyze = () => {
            if (!analyzeData.code.trim()) {
                error.value = '请输入代码';
                return;
            }
            loading.value = true;
            error.value = '';
            analyzeData.analysis = '';
            
            console.log('[SSE] Starting code analysis, sessionId:', askData.sessionId);
            
            streamAnalyzeCode(
                analyzeData.code,
                (chunk) => {
                    analyzeData.analysis += chunk;
                },
                () => {
                    console.log('[SSE] Analysis completed');
                    loading.value = false;
                },
                (e) => {
                    console.error('[SSE] Analysis error:', e);
                    error.value = 'AI 服务暂时不可用';
                    loading.value = false;
                },
                askData.sessionId
            );
        }

        const handleFix = () => {
            if (!fixData.code.trim()) {
                error.value = '请输入代码';
                return;
            }
            loading.value = true;
            error.value = '';
            fixData.explanation = '';
            
            console.log('[SSE] Starting code fix, sessionId:', askData.sessionId);
            
            streamFixCode(
                fixData.code,
                fixData.errorLog,
                (chunk) => {
                    fixData.explanation += chunk;
                },
                () => {
                    console.log('[SSE] Fix completed');
                    loading.value = false;
                },
                (e) => {
                    console.error('[SSE] Fix error:', e);
                    error.value = 'AI 服务暂时不可用';
                    loading.value = false;
                },
                askData.sessionId
            );
        }

        const copyCode = (code) => {
            navigator.clipboard.writeText(code).then(() => {
                alert('代码已复制到剪贴板');
            });
        }

        const handleImageUpload = (event) => {
            const file = event.target.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = (e) => {
                visionData.preview = e.target.result;
                // 提取 base64 数据（去掉前缀）
                visionData.imageData = e.target.result.split(',')[1];
            };
            reader.readAsDataURL(file);
        }

        const handleVisionAnalyze = async () => {
            if (!visionData.imageData) {
                error.value = '请先上传图片';
                return;
            }
            
            loading.value = true;
            error.value = '';
            visionData.analysis = '';
            
            try {
                const PYTHON_AI_URL = process.env.VUE_APP_PYTHON_AI_URL || 'http://localhost:3003';
                const response = await fetch(`${PYTHON_AI_URL}/api/vision/analyze`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        imageData: visionData.imageData,
                        imageSource: 'base64',
                        analysisType: visionData.analysisType,
                        question: visionData.question || undefined
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    visionData.analysis = result.analysis;
                    visionData.available = true;
                } else {
                    error.value = result.error || '图片分析失败';
                    if (result.error && result.error.includes('Vision')) {
                        visionData.available = false;
                    }
                }
            } catch (e) {
                console.error('[Vision] Analysis error:', e);
                error.value = '图片分析服务不可用';
                visionData.available = false;
            } finally {
                loading.value = false;
            }
        }

        const formatMarkdown = (text) => {
            if (!text) return '';
            try {
                return marked(text);
            } catch (e) {
                return text;
            }
        }

        return {
            activeTab,
            loading,
            error,
            chatContainer,
            imageInput,
            chatImageInput,
            askData,
            generateData,
            analyzeData,
            fixData,
            visionData,
            hoveredMessage,
            handleAsk,
            clearConversation,
            handleGenerate,
            handleAnalyze,
            handleFix,
            handleImageUpload,
            handleVisionAnalyze,
            copyCode,
            formatMarkdown,
            // HITL handlers
            handleProposalResumed,
            handleProposalError,
            // Time Travel handlers
            handleForkFromMessage,
            handleChatImageSelect,
            handleMultimodalAsk,
            switchToMain
        }
    }
}
</script>

<style scoped>
.ai-bot-assistant {
    padding: 15px;
}

.nav-link {
    cursor: pointer;
}

.result-box {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
    margin-top: 15px;
}

.code-block {
    background: #2d2d2d;
    color: #f8f8f2;
    padding: 15px;
    border-radius: 5px;
    overflow-x: auto;
    font-size: 13px;
    max-height: 400px;
}

.code-input {
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 13px;
}

.chat-container {
    height: 500px;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    background: #fff;
}

.chat-history {
    flex: 1;
    overflow-y: auto;
    padding: 15px;
}

.message-body {
    display: flex;
    align-items: flex-start;
    gap: 6px;
}

.message-item.user .message-body {
    justify-content: flex-end;
}

.message-item.assistant .message-body {
    justify-content: flex-start;
}

.fork-btn {
    flex: 0 0 auto;
    line-height: 1;
    white-space: nowrap;
    padding: 4px 8px;
    margin-top: 2px;
}

.message-item.user {
    text-align: right;
}

.message-item.user .message-content {
    background: #e3f2fd;
    display: inline-block;
    text-align: left;
    max-width: 80%;
}

.message-item.assistant .message-content {
    background: #f8f9fa;
    display: inline-block;
    max-width: 90%;
}

.markdown-content :deep(pre) {
    background-color: #f6f8fa;
    padding: 16px;
    border-radius: 6px;
    overflow-x: auto;
}

.markdown-content :deep(code) {
    font-family: monospace;
}
</style>
