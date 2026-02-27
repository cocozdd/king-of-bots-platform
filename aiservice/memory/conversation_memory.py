"""
对话记忆服务 - 对齐 Java ConversationMemoryService

功能：
- 短期记忆：最近 N 轮对话（滑动窗口）
- 长期记忆：摘要压缩（超过阈值时调用 LLM 生成摘要）
- 偏好抽取：从对话中提取用户偏好（可选）

2026 年标准：
- 使用 LangGraph memory 概念
- 支持 thread-scoped（短期）和 cross-thread（长期）记忆
"""
import logging
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """记忆条目"""
    role: str  # user, assistant, system
    content: str
    timestamp: str = ""
    metadata: Dict = field(default_factory=dict)


class ConversationMemoryService:
    """
    对话记忆服务 - 对齐 Java ConversationMemoryService
    
    特性：
    - 短期记忆（滑动窗口）
    - 摘要压缩（减少 token 消耗）
    - 内存存储（生产环境可替换为 Redis）
    """
    
    def __init__(
        self,
        max_messages: int = 20,
        summary_threshold: int = 15,
        keep_recent: int = 5,
    ):
        """
        Args:
            max_messages: 短期记忆最大消息数
            summary_threshold: 触发摘要的消息数阈值
            keep_recent: 压缩时保留的最近消息数
        """
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self.keep_recent = keep_recent
        
        # 会话消息存储
        self._messages: Dict[str, List[MemoryEntry]] = {}
        # 会话摘要存储
        self._summaries: Dict[str, str] = {}
        # 用户偏好存储
        self._preferences: Dict[str, Dict] = {}
        
        # LLM 实例（延迟初始化）
        self._llm = None
        
        logger.info(f"ConversationMemoryService 初始化: max={max_messages}, threshold={summary_threshold}")
    
    def _get_llm(self):
        """获取 LLM 实例（用于摘要生成）"""
        if self._llm is None:
            try:
                from llm_client import get_llm
                self._llm = get_llm()
            except Exception as e:
                logger.warning(f"LLM 初始化失败: {e}")
        return self._llm
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict = None,
    ):
        """
        添加消息到会话
        
        Args:
            session_id: 会话 ID
            role: 角色（user/assistant/system）
            content: 消息内容
            metadata: 额外元数据
        """
        if session_id not in self._messages:
            self._messages[session_id] = []
        
        from datetime import datetime
        entry = MemoryEntry(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        
        self._messages[session_id].append(entry)
        
        # 检查是否需要压缩
        if len(self._messages[session_id]) > self.summary_threshold:
            self._compress_history(session_id)
    
    def get_context(
        self,
        session_id: str,
        include_summary: bool = True,
    ) -> List[Dict]:
        """
        获取会话上下文
        
        Args:
            session_id: 会话 ID
            include_summary: 是否包含摘要
            
        Returns:
            消息列表（可直接用于 LLM 调用）
        """
        context = []
        
        # 添加摘要
        if include_summary:
            summary = self._summaries.get(session_id)
            if summary:
                context.append({
                    "role": "system",
                    "content": f"[历史对话摘要]: {summary}"
                })
        
        # 添加最近消息
        messages = self._messages.get(session_id, [])
        recent = messages[-self.max_messages:]
        
        for entry in recent:
            context.append({
                "role": entry.role,
                "content": entry.content,
            })
        
        return context
    
    def get_messages(self, session_id: str) -> List[MemoryEntry]:
        """获取会话的所有消息"""
        return self._messages.get(session_id, [])
    
    def get_summary(self, session_id: str) -> Optional[str]:
        """获取会话摘要"""
        return self._summaries.get(session_id)
    
    def set_summary(self, session_id: str, summary: str):
        """设置会话摘要"""
        self._summaries[session_id] = summary
    
    def _compress_history(self, session_id: str):
        """
        压缩历史记录为摘要
        
        策略：
        1. 保留最近 keep_recent 条消息
        2. 将较早的消息生成摘要
        3. 旧摘要与新摘要合并
        """
        messages = self._messages.get(session_id, [])
        if len(messages) <= self.keep_recent:
            return
        
        old_messages = messages[:-self.keep_recent]
        recent_messages = messages[-self.keep_recent:]
        
        # 生成摘要
        history_text = "\n".join([
            f"{m.role}: {m.content[:200]}"
            for m in old_messages
        ])
        
        # 合并旧摘要
        old_summary = self._summaries.get(session_id, "")
        if old_summary:
            history_text = f"[之前的摘要]: {old_summary}\n\n[新对话]:\n{history_text}"
        
        try:
            llm = self._get_llm()
            if llm:
                from langchain_core.messages import HumanMessage
                summary_prompt = f"""请用 2-3 句话总结以下对话的关键信息，保留重要的上下文：

{history_text}

摘要："""
                result = llm.invoke([HumanMessage(content=summary_prompt)])
                new_summary = result.content.strip()
            else:
                # 无 LLM 时的降级策略
                new_summary = f"[{len(old_messages)} 条历史消息]"
        except Exception as e:
            logger.warning(f"摘要生成失败: {e}")
            new_summary = f"[{len(old_messages)} 条历史消息]"
        
        # 更新存储
        self._summaries[session_id] = new_summary
        self._messages[session_id] = recent_messages
        
        logger.info(f"会话 {session_id} 历史已压缩: {len(old_messages)} -> 摘要")
    
    def clear_session(self, session_id: str):
        """清除会话的所有记忆"""
        self._messages.pop(session_id, None)
        self._summaries.pop(session_id, None)
        self._preferences.pop(session_id, None)
    
    def get_message_count(self, session_id: str) -> int:
        """获取会话的消息数量"""
        return len(self._messages.get(session_id, []))
    
    # ============ 用户偏好（可选功能）============
    
    def set_preference(self, session_id: str, key: str, value):
        """设置用户偏好"""
        if session_id not in self._preferences:
            self._preferences[session_id] = {}
        self._preferences[session_id][key] = value
    
    def get_preference(self, session_id: str, key: str, default=None):
        """获取用户偏好"""
        prefs = self._preferences.get(session_id, {})
        return prefs.get(key, default)
    
    def get_all_preferences(self, session_id: str) -> Dict:
        """获取所有用户偏好"""
        return self._preferences.get(session_id, {})


# 全局实例
_memory_service: Optional[ConversationMemoryService] = None


def get_memory_service() -> ConversationMemoryService:
    """获取全局记忆服务实例"""
    global _memory_service
    if _memory_service is None:
        _memory_service = ConversationMemoryService()
    return _memory_service
