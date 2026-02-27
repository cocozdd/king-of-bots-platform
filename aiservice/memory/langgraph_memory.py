"""
LangGraph 原生记忆管理 - 2026 最佳实践

功能：
- 使用 LangGraph AsyncPostgresSaver 替代自定义 ConversationMemoryService
- 自动管理 thread_id 和状态持久化
- 支持 cross-thread 长期记忆

面试要点：
- 为什么用 Checkpointer：LangGraph 原生支持，无需手动管理 _messages
- thread_id：每个会话一个 thread，LangGraph 自动加载历史
- PostgresSaver vs MemorySaver：生产用 Postgres 持久化，开发用内存
- 减少代码量 50%+，提升状态管理可靠性

升级路径：
- 旧：ConversationMemoryService._messages + _summaries
- 新：LangGraph Checkpointer + thread_id
"""
import logging
import os
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# Postgres 连接配置（复用 pgvector 的数据库，默认对齐 docker-compose）
DEFAULT_PGVECTOR_HOST = "127.0.0.1"
DEFAULT_PGVECTOR_PORT = "15432"
DEFAULT_PGVECTOR_DB = "kob_ai"
DEFAULT_PGVECTOR_USER = "cocodzh"
DEFAULT_PGVECTOR_PASSWORD = "123"


def _build_pgvector_uri_from_env() -> str:
    """从 PGVECTOR_* 环境变量构建 PostgreSQL URI。"""
    host = os.getenv("PGVECTOR_HOST", DEFAULT_PGVECTOR_HOST)
    port = os.getenv("PGVECTOR_PORT", DEFAULT_PGVECTOR_PORT)
    database = os.getenv("PGVECTOR_DATABASE", DEFAULT_PGVECTOR_DB)
    user = os.getenv("PGVECTOR_USER", DEFAULT_PGVECTOR_USER)
    password = os.getenv("PGVECTOR_PASSWORD", DEFAULT_PGVECTOR_PASSWORD)
    return (
        f"postgresql://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{database}"
    )


class LangGraphMemoryManager:
    """
    LangGraph 原生记忆管理器
    
    使用 LangGraph Checkpointer 自动管理对话状态
    替代手动维护 _messages 列表
    
    使用方式：
    1. 初始化时指定 checkpointer（PostgresSaver 或 MemorySaver）
    2. 调用 Agent 时传入 thread_id
    3. LangGraph 自动加载/保存历史状态
    """
    
    def __init__(self, postgres_uri: str = None):
        # 优先级：显式传入 > LANGGRAPH_POSTGRES_URI > PGVECTOR_URI > DATABASE_URL > PGVECTOR_* 拼接
        self.postgres_uri = (
            postgres_uri
            or os.getenv("LANGGRAPH_POSTGRES_URI")
            or os.getenv("PGVECTOR_URI")
            or os.getenv("DATABASE_URL")
            or _build_pgvector_uri_from_env()
        )
        self._checkpointer = None
        self._checkpointer_cm = None
        self._is_setup = False
        self._require_persistent = os.getenv(
            "HITL_REQUIRE_PERSISTENT_CHECKPOINTER", "0"
        ).lower() in ("1", "true", "yes", "on")
        
        logger.info("LangGraphMemoryManager 初始化: %s", 
                   self.postgres_uri.split("@")[-1] if self.postgres_uri else "memory")
    
    async def get_checkpointer(self):
        """
        获取 Checkpointer 实例
        
        优先使用 AsyncPostgresSaver，失败则降级为 MemorySaver
        """
        if self._checkpointer is not None:
            return self._checkpointer
        
        # 尝试使用 PostgresSaver
        if self.postgres_uri:
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                # from_conn_string 在当前版本返回 async context manager，
                # 需要先进入上下文才能拿到可用的 saver 实例。
                self._checkpointer_cm = AsyncPostgresSaver.from_conn_string(
                    self.postgres_uri
                )
                self._checkpointer = await self._checkpointer_cm.__aenter__()

                # 首次使用需要初始化表结构
                if not self._is_setup:
                    await self._checkpointer.setup()
                    self._is_setup = True
                    logger.info("✓ AsyncPostgresSaver 初始化成功")

                return self._checkpointer
                
            except ImportError:
                logger.warning("langgraph-checkpoint-postgres 未安装，使用 MemorySaver")
                if self._require_persistent:
                    raise RuntimeError("缺少 langgraph-checkpoint-postgres，无法启用持久化 checkpointer")
            except Exception as e:
                # 如果 context 已经进入，失败时主动退出，避免连接泄漏
                if self._checkpointer_cm is not None:
                    try:
                        await self._checkpointer_cm.__aexit__(None, None, None)
                    except Exception:
                        pass
                    self._checkpointer_cm = None
                self._checkpointer = None
                logger.warning("PostgresSaver 初始化失败，降级为 MemorySaver: %s", e)
                if self._require_persistent:
                    raise RuntimeError(f"PostgresSaver 初始化失败: {e}") from e
        
        # 降级为 MemorySaver
        from langgraph.checkpoint.memory import MemorySaver
        self._checkpointer = MemorySaver()
        logger.info("使用 MemorySaver（内存存储，服务重启会丢失）")
        
        return self._checkpointer
    
    def get_thread_config(self, session_id: str) -> Dict[str, Any]:
        """
        生成 LangGraph 配置
        
        Args:
            session_id: 会话 ID（映射为 thread_id）
            
        Returns:
            LangGraph configurable 配置
        """
        return {
            "configurable": {
                "thread_id": session_id,
            }
        }
    
    async def delete_thread(self, session_id: str) -> bool:
        """
        删除会话的所有历史状态
        
        Args:
            session_id: 会话 ID
        """
        try:
            checkpointer = await self.get_checkpointer()
            
            # PostgresSaver 支持 delete_thread
            if hasattr(checkpointer, "adelete_thread"):
                await checkpointer.adelete_thread(session_id)
            elif hasattr(checkpointer, "delete_thread"):
                checkpointer.delete_thread(session_id)
            else:
                logger.warning("当前 Checkpointer 不支持删除线程")
                return False
            
            logger.info("已删除会话状态: %s", session_id)
            return True
            
        except Exception as e:
            logger.error("删除会话状态失败: %s", e)
            return False
    
    async def get_thread_state(self, session_id: str) -> Optional[Dict]:
        """
        获取会话的当前状态
        
        Args:
            session_id: 会话 ID
            
        Returns:
            状态字典或 None
        """
        try:
            checkpointer = await self.get_checkpointer()
            config = self.get_thread_config(session_id)
            
            # 获取最新的 checkpoint
            checkpoint = await checkpointer.aget(config)
            
            if checkpoint:
                return checkpoint.get("channel_values", {})
            return None
            
        except Exception as e:
            logger.error("获取会话状态失败: %s", e)
            return None
    
    async def list_thread_checkpoints(
        self, 
        session_id: str, 
        limit: int = 10
    ) -> list:
        """
        列出会话的历史状态（用于调试）
        
        Args:
            session_id: 会话 ID
            limit: 返回数量限制
        """
        try:
            checkpointer = await self.get_checkpointer()
            config = self.get_thread_config(session_id)
            
            checkpoints = []
            async for cp in checkpointer.alist(config, limit=limit):
                checkpoints.append({
                    "id": cp.config.get("configurable", {}).get("checkpoint_id"),
                    "ts": cp.checkpoint.get("ts"),
                    "metadata": cp.metadata,
                })
            
            return checkpoints
            
        except Exception as e:
            logger.error("列出会话历史失败: %s", e)
            return []
    
    async def get_checkpoints_with_metadata(
        self, 
        session_id: str, 
        limit: int = 20
    ) -> list:
        """
        获取检查点列表，包含消息预览（用于 Time Travel UI）
        
        Args:
            session_id: 会话 ID
            limit: 返回数量限制
            
        Returns:
            检查点列表，每个包含 checkpoint_id, message_preview, timestamp, message_index
        """
        try:
            checkpointer = await self.get_checkpointer()
            config = self.get_thread_config(session_id)
            
            checkpoints = []
            message_index = 0
            
            async for cp in checkpointer.alist(config, limit=limit):
                checkpoint_id = cp.config.get("configurable", {}).get("checkpoint_id")
                
                # 提取消息预览
                channel_values = cp.checkpoint.get("channel_values", {})
                messages = channel_values.get("messages", [])
                
                # 获取最后一条 AI 消息作为预览
                message_preview = ""
                for msg in reversed(messages):
                    if hasattr(msg, "content") and getattr(msg, "type", "") == "ai":
                        content = msg.content
                        message_preview = content[:100] + "..." if len(content) > 100 else content
                        break
                
                checkpoints.append({
                    "checkpointId": checkpoint_id,
                    "messagePreview": message_preview,
                    "timestamp": cp.checkpoint.get("ts"),
                    "messageIndex": len(messages) // 2,  # 近似消息轮次
                    "metadata": cp.metadata,
                })
                message_index += 1
            
            return checkpoints
            
        except Exception as e:
            logger.error("获取检查点元数据失败: %s", e)
            return []
    
    async def fork_to_new_thread(
        self, 
        original_session_id: str,
        checkpoint_id: str,
        new_session_id: str
    ) -> bool:
        """
        从指定检查点分叉到新线程（Time Travel 核心功能）
        
        实现原理：
        1. 获取原线程指定 checkpoint 的状态
        2. 将该状态保存到新 thread_id
        
        Args:
            original_session_id: 原始会话 ID
            checkpoint_id: 要分叉的检查点 ID
            new_session_id: 新会话 ID
            
        Returns:
            是否成功
        """
        try:
            checkpointer = await self.get_checkpointer()
            
            # 1. 获取原检查点
            original_config = {
                "configurable": {
                    "thread_id": original_session_id,
                    "checkpoint_id": checkpoint_id,
                }
            }
            
            checkpoint_tuple = await checkpointer.aget_tuple(original_config)
            
            if not checkpoint_tuple:
                logger.error("找不到检查点: %s/%s", original_session_id, checkpoint_id)
                return False
            
            # 2. 准备新线程配置
            new_config = {
                "configurable": {
                    "thread_id": new_session_id,
                }
            }
            
            # 3. 保存到新线程
            # 使用 aput 保存 checkpoint 到新 thread
            await checkpointer.aput(
                config=new_config,
                checkpoint=checkpoint_tuple.checkpoint,
                metadata={
                    **checkpoint_tuple.metadata,
                    "forked_from": original_session_id,
                    "forked_checkpoint": checkpoint_id,
                },
                new_versions=checkpoint_tuple.checkpoint.get("channel_versions", {}),
            )
            
            logger.info("成功分叉: %s/%s -> %s", 
                       original_session_id, checkpoint_id, new_session_id)
            return True
            
        except Exception as e:
            logger.error("分叉失败: %s", e, exc_info=True)
            return False
    
    async def get_branch_info(self, session_id: str) -> Optional[Dict]:
        """
        获取分支信息（是否为分叉、来源等）
        
        Args:
            session_id: 会话 ID
            
        Returns:
            分支信息字典
        """
        try:
            checkpointer = await self.get_checkpointer()
            config = self.get_thread_config(session_id)
            
            # 获取第一个 checkpoint 的 metadata
            async for cp in checkpointer.alist(config, limit=1):
                metadata = cp.metadata or {}
                
                if "forked_from" in metadata:
                    return {
                        "isFork": True,
                        "parentSessionId": metadata.get("forked_from"),
                        "forkedCheckpoint": metadata.get("forked_checkpoint"),
                    }
                
                return {"isFork": False}
            
            return None
            
        except Exception as e:
            logger.error("获取分支信息失败: %s", e)
            return None


# ============ 便捷函数 ============

_memory_manager: Optional[LangGraphMemoryManager] = None


async def get_memory_manager() -> LangGraphMemoryManager:
    """获取全局记忆管理器"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = LangGraphMemoryManager()
    return _memory_manager


async def get_langgraph_checkpointer():
    """
    获取 LangGraph Checkpointer
    
    用于创建 Agent 时传入
    
    示例：
        checkpointer = await get_langgraph_checkpointer()
        agent = create_react_agent(llm, tools, checkpointer=checkpointer)
        result = await agent.ainvoke(input, config={"configurable": {"thread_id": session_id}})
    """
    manager = await get_memory_manager()
    return await manager.get_checkpointer()


def get_thread_config(session_id: str) -> Dict[str, Any]:
    """
    生成 thread 配置
    
    用于 Agent 调用时传入
    """
    return {
        "configurable": {
            "thread_id": session_id,
        }
    }


# ============ Agent 集成示例 ============

async def create_persistent_agent(llm, tools: list):
    """
    创建带持久化的 Agent
    
    示例用法：
        agent = await create_persistent_agent(llm, tools)
        
        # 调用时传入 thread_id
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="你好")]},
            config=get_thread_config("session-123")
        )
        
        # LangGraph 自动：
        # 1. 从 Postgres 加载 session-123 的历史状态
        # 2. 执行 Agent
        # 3. 保存新状态到 Postgres
    """
    from langgraph.prebuilt import create_react_agent
    
    checkpointer = await get_langgraph_checkpointer()
    
    agent = create_react_agent(
        llm,
        tools,
        checkpointer=checkpointer,
    )
    
    return agent


# ============ 迁移指南 ============
"""
从 ConversationMemoryService 迁移到 LangGraph Checkpointer

## 旧代码
```python
from memory.conversation_memory import get_memory_service

memory = get_memory_service()
memory.add_message(session_id, "user", user_input)
context = memory.get_context(session_id)

# 手动构建 messages
messages = [SystemMessage(content=system_prompt)]
for msg in context:
    if msg["role"] == "user":
        messages.append(HumanMessage(content=msg["content"]))
    elif msg["role"] == "assistant":
        messages.append(AIMessage(content=msg["content"]))

response = await llm.ainvoke(messages)
memory.add_message(session_id, "assistant", response.content)
```

## 新代码
```python
from memory.langgraph_memory import create_persistent_agent, get_thread_config

# 创建一次，复用
agent = await create_persistent_agent(llm, tools)

# 调用时只需传入 thread_id，LangGraph 自动管理历史
result = await agent.ainvoke(
    {"messages": [HumanMessage(content=user_input)]},
    config=get_thread_config(session_id)
)

# 响应自动包含完整历史
response = result["messages"][-1].content
```

## 优势
1. 代码量减少 50%+
2. 无需手动管理 _messages 列表
3. 状态自动持久化到 Postgres
4. 支持 interrupt/resume、time-travel 等高级功能
"""
