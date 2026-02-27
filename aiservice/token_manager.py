"""
Token 窗口管理 - P1 改进

功能：
- 精确计算 Token 数量
- 智能截断上下文
- 防止超出模型窗口

面试要点：
- 为什么需要 Token 管理：模型有上下文窗口限制（如 GPT-4: 128K, DeepSeek: 64K）
- tiktoken：OpenAI 官方 tokenizer，准确计算 token 数
- 截断策略：保留最新消息，为输出预留空间
"""
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# 模型窗口大小配置
MODEL_CONTEXT_WINDOWS = {
    "gpt-4": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-3.5-turbo": 16385,
    "deepseek-chat": 64000,
    "deepseek-coder": 64000,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
}

# 默认配置
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MAX_TOKENS = 64000
DEFAULT_RESERVE_TOKENS = 4000  # 为输出预留


class TokenManager:
    """
    Token 窗口管理器
    
    功能：
    - 计算文本 token 数
    - 智能截断消息历史
    - 防止超出模型窗口
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = None,
        reserve_tokens: int = DEFAULT_RESERVE_TOKENS,
    ):
        self.model = model
        self.max_tokens = max_tokens or MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_MAX_TOKENS)
        self.reserve_tokens = reserve_tokens
        self._encoding = None
        
        logger.info(
            "TokenManager 初始化: model=%s, max_tokens=%d, reserve=%d",
            model, self.max_tokens, reserve_tokens,
        )
    
    def _get_encoding(self):
        """获取 tokenizer"""
        if self._encoding is not None:
            return self._encoding
        
        try:
            import tiktoken
            
            # 尝试获取模型专用编码
            try:
                self._encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # 回退到 cl100k_base（GPT-4 编码，通用性好）
                self._encoding = tiktoken.get_encoding("cl100k_base")
                logger.debug("使用 cl100k_base 编码（模型 %s 无专用编码）", self.model)
            
            return self._encoding
        except ImportError:
            logger.warning("tiktoken 未安装，使用估算方法")
            return None
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本 token 数
        
        Args:
            text: 文本内容
            
        Returns:
            token 数量
        """
        if not text:
            return 0
        
        encoding = self._get_encoding()
        
        if encoding:
            return len(encoding.encode(text))
        else:
            # 估算：平均每 4 个字符 1 个 token（中英文混合）
            return len(text) // 3
    
    def count_message_tokens(self, message: Dict) -> int:
        """
        计算单条消息的 token 数
        
        包含消息结构开销（role、分隔符等）
        """
        # 消息结构开销（约 4 tokens）
        overhead = 4
        content = message.get("content", "")
        return self.count_tokens(content) + overhead
    
    def count_messages_tokens(self, messages: List[Dict]) -> int:
        """计算消息列表总 token 数"""
        total = 0
        for msg in messages:
            total += self.count_message_tokens(msg)
        # 消息列表结构开销
        return total + 3
    
    def get_context_within_limit(
        self,
        messages: List[Dict],
        max_tokens: int = None,
        reserve_tokens: int = None,
        system_message: Optional[Dict] = None,
    ) -> Tuple[List[Dict], int]:
        """
        获取不超过 token 限制的上下文
        
        Args:
            messages: 消息列表（用户/助手消息，不含 system）
            max_tokens: 最大 token 数（默认使用 self.max_tokens）
            reserve_tokens: 为输出预留的 token 数
            system_message: 系统消息（如有，优先保留）
            
        Returns:
            (selected_messages, total_tokens)
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        if reserve_tokens is None:
            reserve_tokens = self.reserve_tokens
        
        available_tokens = max_tokens - reserve_tokens
        
        # 如果有系统消息，先计算其 token 数
        system_tokens = 0
        if system_message:
            system_tokens = self.count_message_tokens(system_message)
            available_tokens -= system_tokens
        
        if available_tokens <= 0:
            logger.warning("系统消息已超出 token 限制")
            return [], system_tokens
        
        # 从最新消息开始，倒序添加
        result = []
        total_tokens = 0
        
        for msg in reversed(messages):
            msg_tokens = self.count_message_tokens(msg)
            
            if total_tokens + msg_tokens > available_tokens:
                # 如果是第一条消息就超限，截断它
                if not result:
                    truncated = self._truncate_message(
                        msg,
                        available_tokens - 50,  # 留点余量
                    )
                    result.insert(0, truncated)
                    total_tokens = available_tokens
                break
            
            result.insert(0, msg)
            total_tokens += msg_tokens
        
        final_tokens = total_tokens + system_tokens
        
        logger.debug(
            "Token 管理: 原始 %d 条 -> 选择 %d 条, tokens=%d/%d",
            len(messages), len(result), final_tokens, max_tokens,
        )
        
        return result, final_tokens
    
    def _truncate_message(self, msg: Dict, max_tokens: int) -> Dict:
        """
        截断单条消息
        
        Args:
            msg: 消息
            max_tokens: 最大 token 数
            
        Returns:
            截断后的消息
        """
        content = msg.get("content", "")
        
        encoding = self._get_encoding()
        if encoding:
            tokens = encoding.encode(content)
            if len(tokens) > max_tokens:
                truncated_tokens = tokens[:max_tokens]
                truncated_content = encoding.decode(truncated_tokens)
            else:
                truncated_content = content
        else:
            # 估算截断
            char_limit = max_tokens * 3
            truncated_content = content[:char_limit]
        
        return {
            **msg,
            "content": truncated_content + "\n\n[消息已截断...]",
        }
    
    def estimate_output_tokens(self, prompt_tokens: int) -> int:
        """
        估算输出 token 数
        
        基于提示词长度估算
        """
        # 简单估算：输出约为输入的 0.5-2 倍
        return min(prompt_tokens, self.reserve_tokens)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "reserve_tokens": self.reserve_tokens,
            "available_tokens": self.max_tokens - self.reserve_tokens,
        }


# ============ 全局实例管理 ============

_token_manager: Optional[TokenManager] = None


def get_token_manager(model: str = None) -> TokenManager:
    """获取 Token 管理器"""
    global _token_manager
    
    if _token_manager is None:
        _token_manager = TokenManager(model=model or DEFAULT_MODEL)
    
    return _token_manager


def count_tokens(text: str) -> int:
    """便捷函数：计算 token 数"""
    return get_token_manager().count_tokens(text)


def get_context_within_limit(
    messages: List[Dict],
    max_tokens: int = None,
    system_message: Optional[Dict] = None,
) -> Tuple[List[Dict], int]:
    """便捷函数：获取限制内的上下文"""
    return get_token_manager().get_context_within_limit(
        messages,
        max_tokens=max_tokens,
        system_message=system_message,
    )
