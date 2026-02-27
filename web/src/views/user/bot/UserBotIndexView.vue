<template>
    <div class="container">
        <div class="row">
            <div class="col-3">
                <div class="card" style="margin-top: 20px;">
                    <div class="card-body">
                        <img :src="$store.state.user.photo" alt="" style="width: 100%;">
                    </div>
                </div>
            </div>
            <div class="col-9">
                <div class="card" style="margin-top: 20px;">
                    <div class="card-header">
                        <span style="font-size: 130%">我的Bot</span>
                        <button type="button" class="btn btn-success float-end" style="margin-left: 10px;" data-bs-toggle="modal" data-bs-target="#ai-bot-assistant-modal">
                            AI 助手
                        </button>
                        <button type="button" class="btn btn-outline-secondary float-end" style="margin-left: 10px;" @click="openAiHelper">
                            AI 提示
                        </button>
                        <button type="button" class="btn btn-primary float-end" data-bs-toggle="modal" data-bs-target="#add-bot-btn">
                            创建Bot
                        </button>

                        <!-- Modal -->
                        <div class="modal fade" id="add-bot-btn" tabindex="-1">
                            <div class="modal-dialog modal-xl">
                                <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title">创建Bot</h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                                </div>
                                <div class="modal-body">
                                    <div class="mb-3">
                                        <label for="add-bot-title" class="form-label">名称</label>
                                        <input v-model="botadd.title" type="text" class="form-control" id="add-bot-title" placeholder="请输入Bot名称">
                                    </div>
                                    <div class="mb-3">
                                        <label for="add-bot-description" class="form-label">简介</label>
                                        <textarea v-model="botadd.description" class="form-control" id="add-bot-description" rows="3" placeholder="请输入Bot简介"></textarea>
                                    </div>
                                    <div class="mb-3">
                                        <label for="add-bot-code" class="form-label">代码</label>
                                        <VAceEditor
                                            v-model:value="botadd.content"
                                            @init="editorInit"
                                            lang="c_cpp"
                                            theme="textmate"
                                            style="height: 300px" />
                                    </div>
                                </div>
                                <div class="modal-footer">
                                    <div class="error-message">{{ botadd.error_message }}</div>
                                    <button type="button" class="btn btn-primary" @click="add_bot">创建</button>
                                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
                                </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <!-- AI Bot Assistant Modal -->
                    <div class="modal fade" id="ai-bot-assistant-modal" tabindex="-1">
                        <div class="modal-dialog modal-xl">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title">AI Bot 助手</h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                                </div>
                                <div class="modal-body">
                                    <AiBotAssistant :user-id="$store.state.user.id" @bot-updated="refresh_bots" />
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- AI Hint Modal -->
                    <div class="modal fade" id="ai-hint-modal" tabindex="-1">
                        <div class="modal-dialog">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title">AI 编写提示</h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                                </div>
                                <div class="modal-body">
                                    <div class="mb-3">
                                        <label for="ai-question" class="form-label">问题</label>
                                        <textarea id="ai-question" class="form-control" rows="3" v-model="aiHelper.question"
                                            placeholder="描述你遇到的问题，例如：我的Bot总是超时，应该如何优化？"></textarea>
                                    </div>
                                    <div class="mb-3 d-flex justify-content-between align-items-center">
                                        <span>建议</span>
                                        <button class="btn btn-sm btn-primary" :disabled="aiHelper.loading" @click="requestAiHint">
                                            <span v-if="aiHelper.loading" class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                                            <span v-if="!aiHelper.loading">获取建议</span>
                                        </button>
                                    </div>
                                    <div class="ai-answer-box">
                                        <div v-if="aiHelper.error" class="text-danger mb-2">{{ aiHelper.error }}</div>
                                        <div v-else-if="aiHelper.answer || aiHelper.loading">
                                            <div v-if="aiHelper.loading && !aiHelper.answer" class="text-muted">
                                                <div class="d-flex align-items-center">
                                                    <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                                                    <span>AI 正在思考中...</span>
                                                </div>
                                            </div>
                                            <div v-if="aiHelper.answer">
                                                <div v-html="renderMarkdown(aiHelper.answer)" class="markdown-content"></div>
                                                <span v-if="aiHelper.loading" class="typing-cursor">|</span>
                                            </div>
                                            <div v-if="aiHelper.sources.length">
                                                <small class="text-muted">参考：</small>
                                                <ul class="small">
                                                    <li v-for="source in aiHelper.sources" :key="source.id">
                                                        {{ source.title }}（{{ source.category }}）
                                                    </li>
                                                </ul>
                                            </div>
                                        </div>
                                        <div v-else class="text-muted small">点击"获取建议"后将展示AI生成的提示。</div>
                                    </div>
                                </div>
                                <div class="modal-footer">
                                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">关闭</button>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="card-body">
                        <table class="table table-striped table-hover">
                            <thead>
                                <tr>
                                    <th>名称</th>
                                    <th>创建时间</th>
                                    <th>操作</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr v-for="bot in bots" :key="bot.id">
                                    <td>{{ bot.title }}</td>
                                    <td>{{ bot.createtime }}</td>
                                    <td>
                                        <button type="button" class="btn btn-secondary" style="margin-right: 10px;" data-bs-toggle="modal" :data-bs-target="'#update-bot-modal-' + bot.id">修改</button>
                                        <button type="button" class="btn btn-danger" @click="remove_bot(bot)">删除</button>

                                        <div class="modal fade" :id="'update-bot-modal-' + bot.id" tabindex="-1">
                                            <div class="modal-dialog modal-xl">
                                                <div class="modal-content">
                                                <div class="modal-header">
                                                    <h5 class="modal-title">创建Bot</h5>
                                                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                                                </div>
                                                <div class="modal-body">
                                                    <div class="mb-3">
                                                        <label for="add-bot-title" class="form-label">名称</label>
                                                        <input v-model="bot.title" type="text" class="form-control" id="add-bot-title" placeholder="请输入Bot名称">
                                                    </div>
                                                    <div class="mb-3">
                                                        <label for="add-bot-description" class="form-label">简介</label>
                                                        <textarea v-model="bot.description" class="form-control" id="add-bot-description" rows="3" placeholder="请输入Bot简介"></textarea>
                                                    </div>
                                                    <div class="mb-3">
                                                        <label for="add-bot-code" class="form-label">代码</label>
                                                        <VAceEditor
                                                            v-model:value="bot.content"
                                                            @init="editorInit"
                                                            lang="c_cpp"
                                                            theme="textmate"
                                                            style="height: 300px" />
                                                    </div>
                                                </div>
                                                <div class="modal-footer">
                                                    <div class="error-message">{{ bot.error_message }}</div>
                                                    <button type="button" class="btn btn-primary" @click="update_bot(bot)">保存修改</button>
                                                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
                                                </div>
                                                </div>
                                            </div>
                                        </div>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import { ref, reactive, nextTick } from 'vue'
import $ from 'jquery'
import { useStore } from 'vuex'
import { Modal } from 'bootstrap/dist/js/bootstrap'
import { VAceEditor } from 'vue3-ace-editor';
import ace from 'ace-builds';
import { streamAiHint } from '@/api/ai'
import AiBotAssistant from '@/components/AiBotAssistant.vue'
import { marked } from 'marked'

// 配置 marked 以正确处理换行
marked.setOptions({
    breaks: true,      // 将单个换行符转换为 <br>
    gfm: true,         // 启用 GitHub Flavored Markdown
})

export default {
    components: {
        VAceEditor,
        AiBotAssistant
    },
    setup() {
        ace.config.set(
            "basePath", 
            "https://cdn.jsdelivr.net/npm/ace-builds@" + require('ace-builds').version + "/src-noconflict/")

        const store = useStore();
        let bots = ref([]);
        const aiHelper = reactive({
            question: "",
            answer: "",
            error: "",
            loading: false,
            sources: [],
        });

        const renderMarkdown = (text) => {
            if (!text) return '';
            
            let formattedText = text;

            // ========== 第一阶段：保护和格式化代码块 ==========
            const codeBlocks = [];
            formattedText = formattedText.replace(/```([\s\S]*?)```/g, (match) => {
                let fixedMatch = formatCodeBlock(match);
                codeBlocks.push(fixedMatch);
                return `__CODE_BLOCK_${codeBlocks.length - 1}__`;
            });

            // ========== 第二阶段：修复标题格式 ==========
            formattedText = formattedText.replace(/([^#\n])(#{1,6})([^#\s])/g, '$1\n\n$2 $3');
            formattedText = formattedText.replace(/([^#\n])(#{1,6}\s)/g, '$1\n\n$2');
            formattedText = formattedText.replace(/(#{1,6})([^#\s\n])/g, '$1 $2');

            // ========== 第三阶段：修复段落格式 ==========
            formattedText = formattedText.replace(/([。！？])([^"'」）\s\n\d])/g, '$1\n\n$2');
            
            // ========== 第四阶段：修复列表格式 ==========
            formattedText = formattedText.replace(/([^\n\d])(\d+\.)\s*([^\d\s])/g, '$1\n$2 $3');
            formattedText = formattedText.replace(/([^\n])(-\s+)/g, '$1\n$2');
            formattedText = formattedText.replace(/([^\n])(\*\s+)/g, '$1\n$2');

            // ========== 第五阶段：处理特定关键词 ==========
            const sectionKeywords = ['注意', '总结', '示例', '说明', '步骤', '方法', '原理', 
                                     '特点', '优点', '缺点', '实现', '代码示例', '算法', 
                                     '复杂度分析', '时间复杂度', '空间复杂度'];
            sectionKeywords.forEach(keyword => {
                const regex = new RegExp(`([^\\n])(${keyword}[：:])\\s*`, 'g');
                formattedText = formattedText.replace(regex, '$1\n\n$2 ');
            });

            // ========== 第六阶段：还原代码块 ==========
            formattedText = formattedText.replace(/__CODE_BLOCK_(\d+)__/g, (match, index) => {
                return '\n\n' + codeBlocks[index] + '\n\n';
            });
            
            // ========== 第七阶段：清理多余换行 ==========
            formattedText = formattedText.replace(/\n{3,}/g, '\n\n');
            
            return marked(formattedText);
        }
        
        // 格式化代码块内容 - 保留AI原始格式
        const formatCodeBlock = (codeBlock) => {
            const match = codeBlock.match(/```(\w*)([\s\S]*)```/);
            if (!match) return codeBlock;
            
            let lang = match[1];
            let code = match[2];
            
            // 关键：如果代码已有换行（>=3行），直接返回原样
            if (code.split('\n').length >= 3) return codeBlock;
            
            // 只有代码完全没换行时才修复
            const keywords = ['def ', 'class ', 'if ', 'for ', 'while ', 'return ', 'import ', 'from '];
            keywords.forEach(kw => {
                const regex = new RegExp(`([^\\n])(${kw})`, 'g');
                code = code.replace(regex, '$1\n$2');
            });
            
            code = code.replace(/([)\]}'">0-9])\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=/g, '$1\n$2 =');
            code = code.replace(/\):([^\n])/g, '):\n    $1');
            
            if (!code.startsWith('\n')) code = '\n' + code;
            if (!code.endsWith('\n')) code = code + '\n';
            
            return '```' + lang + code + '```';
        }

        const botadd = reactive({
            title: "",
            description: "",
            content: "",
            error_message: "",
        });

        const refresh_bots = () => {
            console.log("Refreshing bots...");
            $.ajax({
                url: "http://127.0.0.1:3000/api/user/bot/getlist/",
                type: "get",
                headers: {
                    Authorization: "Bearer " + store.state.user.token,
                },
                success(resp) {
                    console.log("Bots fetched successfully:", resp);
                    bots.value = resp;
                },
                error(resp) {
                    console.error("Failed to fetch bots:", resp);
                }
            })
        }

        const editorInit = () => {
            require('ace-builds/src-noconflict/ext-language_tools');
            require('ace-builds/src-noconflict/snippets/c_cpp');
            require('ace-builds/src-noconflict/mode-c_cpp');
            require('ace-builds/src-noconflict/theme-textmate');
            require('ace-builds/src-noconflict/theme-chrome');
            require('ace-builds/src-noconflict/theme-crimson_editor');
        }

        refresh_bots();

        const add_bot = () => {
            botadd.error_message = "";
            $.ajax({
                url: "http://127.0.0.1:3000/api/user/bot/add/",
                type: "post",
                data: {
                    title: botadd.title,
                    description: botadd.description,
                    content: botadd.content,
                },
                headers: {
                    Authorization: "Bearer " + store.state.user.token,
                },
                success(resp) {
                    if (resp.error_message === "success") {
                        botadd.title = "";
                        botadd.description = "";
                        botadd.content = "";
                        Modal.getInstance("#add-bot-btn").hide();
                        refresh_bots();
                    } else {
                        botadd.error_message = resp.error_message;
                    }
                }
            })
        }

        const update_bot = (bot) => {
            botadd.error_message = "";
            $.ajax({
                url: "http://127.0.0.1:3000/api/user/bot/update/",
                type: "post",
                data: {
                    bot_id: bot.id,
                    title: bot.title,
                    description: bot.description,
                    content: bot.content,
                },
                headers: {
                    Authorization: "Bearer " + store.state.user.token,
                },
                success(resp) {
                    if (resp.error_message === "success") {
                        Modal.getInstance('#update-bot-modal-' + bot.id).hide();
                        refresh_bots();
                    } else {
                        botadd.error_message = resp.error_message;
                    }
                }
            })
        }

        const remove_bot = (bot) => {
            $.ajax({
                url: "http://127.0.0.1:3000/api/user/bot/remove/",
                type: "post",
                data: {
                    bot_id: bot.id,
                },
                headers: {
                    Authorization: "Bearer " + store.state.user.token,
                },
                success(resp) {
                    if (resp.error_message === "success") {
                        refresh_bots();
                    }
                }
            })
        }

        const openAiHelper = () => {
            aiHelper.error = "";
            aiHelper.answer = "";
            aiHelper.sources = [];
            if (!aiHelper.question) {
                aiHelper.question = "我的Bot总是超时，应该如何优化？";
            }
            const modal = Modal.getOrCreateInstance(document.getElementById('ai-hint-modal'));
            modal.show();
        }

        const requestAiHint = () => {
            if (!aiHelper.question.trim()) {
                aiHelper.error = "请先输入问题";
                return;
            }
            aiHelper.loading = true;
            aiHelper.error = "";
            aiHelper.answer = "";  // 清空之前的回答
            aiHelper.sources = [];
            
            console.log('[AI Hint SSE] Starting stream for question:', aiHelper.question);
            
            // 用于跟踪 <think> 标签状态
            let insideThink = false;
            let thinkBuffer = '';
            
            // 过滤 <think> 标签内容
            const filterThinkContent = (text) => {
                let result = '';
                thinkBuffer += text;
                
                while (thinkBuffer.length > 0) {
                    if (!insideThink) {
                        const thinkStart = thinkBuffer.indexOf('<think>');
                        if (thinkStart === -1) {
                            // 没有 <think>，检查是否有不完整的标签
                            if (thinkBuffer.endsWith('<') || thinkBuffer.endsWith('<t') || 
                                thinkBuffer.endsWith('<th') || thinkBuffer.endsWith('<thi') || 
                                thinkBuffer.endsWith('<thin') || thinkBuffer.endsWith('<think')) {
                                // 保留可能的不完整标签
                                break;
                            }
                            result += thinkBuffer;
                            thinkBuffer = '';
                        } else {
                            // 输出 <think> 之前的内容
                            result += thinkBuffer.substring(0, thinkStart);
                            thinkBuffer = thinkBuffer.substring(thinkStart + 7);
                            insideThink = true;
                        }
                    } else {
                        const thinkEnd = thinkBuffer.indexOf('</think>');
                        if (thinkEnd === -1) {
                            // 还在 <think> 内部，丢弃内容
                            thinkBuffer = '';
                            break;
                        } else {
                            // 跳过 </think> 之前的内容
                            thinkBuffer = thinkBuffer.substring(thinkEnd + 8);
                            insideThink = false;
                        }
                    }
                }
                return result;
            };
            
            // 使用流式 API
            const eventSource = streamAiHint(
                aiHelper.question + " (请用中文回答，使用Markdown格式，注意分段)",
                // onChunk: 每收到一个 token 追加显示
                (chunk) => {
                    console.log('[AI Hint SSE] Received chunk:', chunk);
                    const filtered = filterThinkContent(chunk);
                    if (filtered) {
                        aiHelper.answer = aiHelper.answer + filtered;
                    }
                    nextTick(() => {});
                },
                // onComplete
                () => {
                    console.log('[AI Hint SSE] Stream completed');
                    aiHelper.loading = false;
                },
                // onError
                (e) => {
                    console.error('[AI Hint SSE] Error:', e);
                    aiHelper.error = "AI 服务暂时不可用，请稍后再试";
                    aiHelper.loading = false;
                }
            );
            
            console.log('[AI Hint SSE] EventSource created:', eventSource);
        }

        return {
            bots,
            botadd,
            add_bot,
            update_bot,
            remove_bot,
            aiHelper,
            openAiHelper,
            requestAiHint,
            renderMarkdown,
            refresh_bots,
            editorInit,
        }
    }
}
</script>

<style scoped>
div.error-message {
    color: red;
}

.typing-cursor {
    animation: blink 1s infinite;
    font-weight: bold;
}

@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
}

.ai-answer-box {
    min-height: 100px;
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
}

.markdown-content {
    line-height: 1.6;
}

.markdown-content :deep(pre) {
    background-color: #f6f8fa;
    padding: 16px;
    border-radius: 6px;
    overflow: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
}

.markdown-content :deep(code) {
    font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
    font-size: 85%;
    padding: 0.2em 0.4em;
    margin: 0;
    border-radius: 3px;
    background-color: rgba(175, 184, 193, 0.2);
}

.markdown-content :deep(pre code) {
    background-color: transparent;
    padding: 0;
}

.markdown-content :deep(p) {
    margin-bottom: 16px;
}

.markdown-content :deep(ul), .markdown-content :deep(ol) {
    padding-left: 2em;
    margin-bottom: 16px;
}
</style>
