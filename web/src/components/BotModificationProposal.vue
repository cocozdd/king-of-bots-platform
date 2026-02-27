<template>
    <div class="modification-proposal card">
        <div class="card-header d-flex justify-content-between align-items-center">
            <div>
                <h5 class="mb-0">Bot 修改建议</h5>
                <small class="text-muted">{{ proposal.bot_name || 'Bot' }} (ID: {{ proposal.bot_id }})</small>
            </div>
            <span class="badge bg-warning text-dark">待确认</span>
        </div>
        
        <div class="card-body">
            <div class="summary mb-3">
                <strong>修改说明：</strong>
                <p class="mb-0">{{ proposal.summary || proposal.modification_request }}</p>
            </div>

            <div class="code-diff">
                <ul class="nav nav-tabs mb-2" role="tablist">
                    <li class="nav-item">
                        <button class="nav-link" :class="{ active: viewMode === 'side-by-side' }" 
                            @click="viewMode = 'side-by-side'">并排对比</button>
                    </li>
                    <li class="nav-item">
                        <button class="nav-link" :class="{ active: viewMode === 'before' }" 
                            @click="viewMode = 'before'">修改前</button>
                    </li>
                    <li class="nav-item">
                        <button class="nav-link" :class="{ active: viewMode === 'after' }" 
                            @click="viewMode = 'after'">修改后</button>
                    </li>
                </ul>

                <div v-if="viewMode === 'side-by-side'" class="row">
                    <div class="col-6">
                        <h6 class="text-danger">修改前</h6>
                        <pre class="code-block before"><code>{{ proposal.original_code }}</code></pre>
                    </div>
                    <div class="col-6">
                        <h6 class="text-success">修改后</h6>
                        <pre class="code-block after"><code>{{ editMode ? editedCode : proposal.new_code }}</code></pre>
                    </div>
                </div>
                
                <div v-else-if="viewMode === 'before'">
                    <h6 class="text-danger">修改前</h6>
                    <pre class="code-block before"><code>{{ proposal.original_code }}</code></pre>
                </div>
                
                <div v-else>
                    <h6 class="text-success">修改后</h6>
                    <div v-if="editMode">
                        <textarea v-model="editedCode" class="form-control code-editor" rows="15"></textarea>
                    </div>
                    <pre v-else class="code-block after"><code>{{ proposal.new_code }}</code></pre>
                </div>
            </div>

            <div class="actions mt-4 d-flex gap-2 flex-wrap">
                <button class="btn btn-success" :disabled="loading" @click="handleApprove">
                    <span v-if="loading && actionType === 'approve'" class="spinner-border spinner-border-sm me-1"></span>
                    确认应用
                </button>
                <button v-if="!editMode" class="btn btn-outline-primary" :disabled="loading" @click="startEdit">
                    编辑后应用
                </button>
                <button v-else class="btn btn-primary" :disabled="loading" @click="handleEditApprove">
                    <span v-if="loading && actionType === 'edit'" class="spinner-border spinner-border-sm me-1"></span>
                    应用编辑后的代码
                </button>
                <button v-if="editMode" class="btn btn-outline-secondary" @click="cancelEdit">
                    取消编辑
                </button>
                <button class="btn btn-danger" :disabled="loading" @click="handleReject">
                    <span v-if="loading && actionType === 'reject'" class="spinner-border spinner-border-sm me-1"></span>
                    取消修改
                </button>
            </div>
        </div>
    </div>
</template>

<script>
import { ref } from 'vue'
import { getAuthHeaders } from '@/api/ai'

export default {
    name: 'BotModificationProposal',
    props: {
        proposal: {
            type: Object,
            required: true
        },
        sessionId: {
            type: String,
            required: true
        }
    },
    emits: ['resumed', 'error'],
    setup(props, { emit }) {
        const loading = ref(false)
        const actionType = ref('')
        const viewMode = ref('side-by-side')
        const editMode = ref(false)
        const editedCode = ref('')

        const startEdit = () => {
            editedCode.value = props.proposal.new_code
            editMode.value = true
            viewMode.value = 'after'
        }

        const cancelEdit = () => {
            editMode.value = false
            editedCode.value = ''
        }

        const resume = async (decision) => {
            loading.value = true
            actionType.value = decision.type

            try {
                const response = await fetch('/ai/bot/chat/hitl', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                    body: JSON.stringify({
                        sessionId: props.sessionId,
                        resumeData: decision
                    })
                })

                const result = await response.json()

                if (result.success) {
                    emit('resumed', result)
                } else {
                    emit('error', result.error || '操作失败')
                }
            } catch (e) {
                console.error('[HITL] Resume error:', e)
                emit('error', '网络错误，请重试')
            } finally {
                loading.value = false
                actionType.value = ''
            }
        }

        const handleApprove = () => {
            resume({ type: 'approve', proposal: props.proposal })
        }

        const handleReject = () => {
            resume({ type: 'reject', proposal: props.proposal })
        }

        const handleEditApprove = () => {
            resume({ 
                type: 'edit', 
                edited_code: editedCode.value,
                proposal: props.proposal
            })
        }

        return {
            loading,
            actionType,
            viewMode,
            editMode,
            editedCode,
            startEdit,
            cancelEdit,
            handleApprove,
            handleReject,
            handleEditApprove
        }
    }
}
</script>

<style scoped>
.modification-proposal {
    border: 2px solid #ffc107;
    background: #fffef0;
}

.code-diff {
    max-height: 500px;
    overflow-y: auto;
}

.code-block {
    background: #2d2d2d;
    color: #f8f8f2;
    padding: 15px;
    border-radius: 5px;
    overflow-x: auto;
    font-size: 12px;
    max-height: 300px;
    margin-bottom: 0;
}

.code-block.before {
    border-left: 4px solid #dc3545;
}

.code-block.after {
    border-left: 4px solid #28a745;
}

.code-editor {
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 12px;
    background: #1e1e1e;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
}

.nav-link {
    cursor: pointer;
}

.summary {
    background: #f8f9fa;
    padding: 10px 15px;
    border-radius: 5px;
    border-left: 4px solid #17a2b8;
}
</style>
