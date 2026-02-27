-- pgvector 初始化脚本
-- 启用向量扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- AI知识库表 (用于RAG)
CREATE TABLE IF NOT EXISTS ai_corpus (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(50) UNIQUE,
    title VARCHAR(200),
    content TEXT,
    category VARCHAR(50),
    embedding vector(1536),  -- DashScope text-embedding-v2 维度
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建向量索引 (HNSW算法，适合高维向量)
CREATE INDEX IF NOT EXISTS idx_ai_corpus_embedding
ON ai_corpus USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 创建分类索引
CREATE INDEX IF NOT EXISTS idx_ai_corpus_category ON ai_corpus(category);

-- 插入初始知识库数据
INSERT INTO ai_corpus (doc_id, title, content, category) VALUES
('snake_basic_001', '蛇类游戏基础规则',
 '蛇类对战游戏在13x14的网格地图上进行。每回合双方同时选择移动方向：0表示向上，1表示向右，2表示向下，3表示向左。蛇头碰到边界、自身或对手身体则判负。',
 'game_rules'),

('snake_basic_002', '游戏胜负判定',
 '胜负判定规则：1. 撞墙判负；2. 撞到自己身体判负；3. 撞到对手身体判负；4. 双方同时撞到对方头部则平局；5. 回合数超过限制则按长度判定。',
 'game_rules'),

('bot_strategy_001', 'BFS寻路算法',
 'BFS（广度优先搜索）是最基础的寻路算法。从当前位置开始，逐层扩展搜索，找到目标的最短路径。实现时需要用队列存储待探索节点，用visited数组避免重复访问。适合简单场景。',
 'algorithm'),

('bot_strategy_002', 'A*寻路算法',
 'A*算法是BFS的优化版本，引入启发函数估计到目标的距离。f(n) = g(n) + h(n)，g(n)是起点到当前的代价，h(n)是当前到终点的估计代价。使用曼哈顿距离作为启发函数效果较好。',
 'algorithm'),

('bot_strategy_003', '避障策略',
 '避障需要预判危险区域：1. 检测四个方向是否会撞墙；2. 检测是否会撞到自己身体；3. 预判对手下一步可能位置；4. 计算每个方向的安全评分，选择最安全的方向。',
 'strategy'),

('bot_strategy_004', '进攻策略',
 '进攻策略的核心是围堵对手：1. 计算对手可能的移动空间；2. 尝试切断对手的移动路线；3. 利用地图边界形成包围；4. 在安全的前提下主动靠近对手。',
 'strategy'),

('bot_advanced_001', 'MCTS蒙特卡洛树搜索',
 'MCTS通过模拟对局来评估每个动作的胜率。四个步骤：选择(Selection)、扩展(Expansion)、模拟(Simulation)、回传(Backpropagation)。UCB1公式平衡探索和利用。适合复杂决策场景。',
 'algorithm'),

('bot_advanced_002', '对手行为预测',
 '预测对手行为可以提前规避风险：1. 分析对手历史移动模式；2. 假设对手会选择对其最优的方向；3. 使用Minimax思想考虑最坏情况；4. 结合概率分布评估各种可能性。',
 'strategy'),

('code_template_001', 'Java Bot代码模板',
 'Bot代码需要实现nextMove方法，接收当前游戏状态，返回0-3的方向值。可以通过getMySnake()获取自己位置，getOpponentSnake()获取对手位置，getMap()获取地图信息。',
 'code'),

('code_template_002', '调试技巧',
 '调试Bot代码的技巧：1. 使用System.err.println输出调试信息；2. 在本地模拟对战测试；3. 检查边界条件处理；4. 确保不会出现死循环；5. 注意时间限制，避免超时。',
 'code')
ON CONFLICT (doc_id) DO NOTHING;

-- 更新时间触发器
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_ai_corpus_updated_at
    BEFORE UPDATE ON ai_corpus
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- =====================================================
-- 多轮对话与记忆系统表
-- =====================================================

-- 会话表：存储对话会话信息
CREATE TABLE IF NOT EXISTS ai_session (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) UNIQUE NOT NULL,
    user_id BIGINT NOT NULL,
    title VARCHAR(200),
    summary TEXT,
    status VARCHAR(20) DEFAULT 'active',
    message_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 消息表：存储对话消息历史
CREATE TABLE IF NOT EXISTS ai_message (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL REFERENCES ai_session(session_id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    tokens_used INT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 用户长期记忆表：存储用户偏好和画像
CREATE TABLE IF NOT EXISTS ai_user_memory (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL UNIQUE,
    preferences JSONB DEFAULT '[]',
    topics JSONB DEFAULT '{}',
    profile_summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_ai_session_user ON ai_session(user_id);
CREATE INDEX IF NOT EXISTS idx_ai_session_status ON ai_session(status);
CREATE INDEX IF NOT EXISTS idx_ai_session_last_active ON ai_session(last_active_at);
CREATE INDEX IF NOT EXISTS idx_ai_message_session ON ai_message(session_id);
CREATE INDEX IF NOT EXISTS idx_ai_message_created ON ai_message(created_at);
CREATE INDEX IF NOT EXISTS idx_ai_user_memory_user ON ai_user_memory(user_id);

-- 更新时间触发器（复用已有的 update_updated_at 函数）
CREATE TRIGGER trigger_ai_session_updated_at
    BEFORE UPDATE ON ai_session
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trigger_ai_user_memory_updated_at
    BEFORE UPDATE ON ai_user_memory
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
