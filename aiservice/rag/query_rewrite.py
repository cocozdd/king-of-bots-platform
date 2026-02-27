"""
查询改写服务 - 使用 LLM 优化用户查询

功能：
- 查询扩展：添加同义词和相关术语
- 查询纠错：修正拼写和语法错误
- 多查询生成：生成多个变体查询

面试要点：
- 为什么需要查询改写：用户查询往往不完整或有歧义
- 多查询策略：提升召回率，尤其对复杂问题
"""
import logging
import os
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RewriteResult:
    """查询改写结果"""
    original_query: str
    rewritten_query: str
    expanded_queries: List[str]
    intent: str  # search, question, command


class QueryRewriter:
    """查询改写器"""
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._llm = None
    
    def _get_llm(self):
        """获取 LLM 客户端"""
        if self._llm is not None:
            return self._llm
        
        try:
            from llm_client import build_llm, should_use_llm
            
            if not should_use_llm():
                return None
            
            self._llm = build_llm(streaming=False)
            return self._llm
        except Exception as e:
            logger.error("LLM 初始化失败: %s", e)
            return None
    
    def rewrite(self, query: str) -> RewriteResult:
        """
        改写查询
        
        Args:
            query: 原始查询
            
        Returns:
            改写结果
        """
        logger.info("查询改写: '%s'", query[:50])
        
        if self.use_llm:
            result = self._llm_rewrite(query)
        else:
            result = self._simple_rewrite(query)
        
        logger.info("改写完成: '%s' -> '%s', 扩展 %d 个",
                    query[:30], result.rewritten_query[:30], len(result.expanded_queries))
        return result
    
    def _llm_rewrite(self, query: str) -> RewriteResult:
        """使用 LLM 改写查询"""
        llm = self._get_llm()
        if llm is None:
            logger.warning("LLM 不可用，降级为简单改写")
            return self._simple_rewrite(query)
        
        prompt = f"""请分析并改写以下用户查询，使其更适合知识库检索。

原始查询: {query}

请按以下格式返回（每行一个）：
意图: [search|question|command]
改写查询: [优化后的查询]
扩展查询1: [第一个变体查询]
扩展查询2: [第二个变体查询]
扩展查询3: [第三个变体查询]

注意：
- 修正拼写和语法错误
- 添加必要的上下文
- 生成语义相似但表达不同的变体
"""
        
        try:
            from langchain_core.messages import HumanMessage
            
            response = llm.invoke([HumanMessage(content=prompt)])
            content = getattr(response, "content", str(response))
            
            return self._parse_llm_response(query, content)
        except Exception as e:
            logger.warning("LLM 改写失败: %s", e)
            return self._simple_rewrite(query)
    
    def _parse_llm_response(self, original_query: str, content: str) -> RewriteResult:
        """解析 LLM 响应"""
        lines = content.strip().split("\n")
        
        intent = "search"
        rewritten = original_query
        expanded = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("意图:") or line.startswith("意图："):
                intent = line.split(":", 1)[-1].strip().lower()
                if intent not in ["search", "question", "command"]:
                    intent = "search"
            elif line.startswith("改写查询:") or line.startswith("改写查询："):
                rewritten = line.split(":", 1)[-1].strip()
            elif line.startswith("扩展查询"):
                expanded_query = line.split(":", 1)[-1].strip() if ":" in line else ""
                if expanded_query and expanded_query != rewritten:
                    expanded.append(expanded_query)
        
        return RewriteResult(
            original_query=original_query,
            rewritten_query=rewritten or original_query,
            expanded_queries=expanded[:3],
            intent=intent,
        )
    
    def _simple_rewrite(self, query: str) -> RewriteResult:
        """简单改写（不使用 LLM）"""
        # 清理查询
        cleaned = query.strip()
        
        # 移除多余空格
        import re
        cleaned = re.sub(r"\s+", " ", cleaned)
        
        # 检测意图
        intent = self._detect_intent(cleaned)
        
        # 生成扩展查询（简单同义词替换）
        expanded = self._generate_expansions(cleaned)
        
        return RewriteResult(
            original_query=query,
            rewritten_query=cleaned,
            expanded_queries=expanded,
            intent=intent,
        )
    
    def _detect_intent(self, query: str) -> str:
        """检测查询意图"""
        query_lower = query.lower()
        
        # 问题类
        question_words = ["什么", "怎么", "为什么", "如何", "哪个", "哪些", "是否", "能不能", "?", "？"]
        for word in question_words:
            if word in query_lower:
                return "question"
        
        # 命令类
        command_words = ["帮我", "请", "生成", "创建", "分析", "计算"]
        for word in command_words:
            if word in query_lower:
                return "command"
        
        return "search"
    
    def _generate_expansions(self, query: str) -> List[str]:
        """生成扩展查询"""
        expansions = []
        
        # 简单的同义词替换
        synonyms = {
            "bot": ["机器人", "AI", "代码"],
            "机器人": ["bot", "AI"],
            "对战": ["比赛", "游戏", "PK"],
            "胜率": ["胜场", "赢", "获胜"],
            "策略": ["算法", "方法", "技巧"],
        }
        
        for original, replacements in synonyms.items():
            if original in query:
                for replacement in replacements[:2]:
                    expanded = query.replace(original, replacement)
                    if expanded not in expansions and expanded != query:
                        expansions.append(expanded)
        
        return expansions[:3]
    
    def multi_query(self, query: str, num_queries: int = 3) -> List[str]:
        """
        生成多个查询变体
        
        Args:
            query: 原始查询
            num_queries: 生成数量
            
        Returns:
            查询列表（包含原始查询）
        """
        result = self.rewrite(query)
        
        queries = [result.rewritten_query]
        queries.extend(result.expanded_queries)
        
        # 确保包含原始查询
        if query not in queries:
            queries.insert(0, query)
        
        return queries[:num_queries + 1]


# 全局实例
_rewriter: Optional[QueryRewriter] = None


def get_rewriter() -> QueryRewriter:
    """获取全局改写器实例"""
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter


def rewrite_query(query: str) -> dict:
    """便捷函数：改写查询"""
    rewriter = get_rewriter()
    result = rewriter.rewrite(query)
    return {
        "original_query": result.original_query,
        "rewritten_query": result.rewritten_query,
        "expanded_queries": result.expanded_queries,
        "intent": result.intent,
    }


def multi_query(query: str, num_queries: int = 3) -> List[str]:
    """便捷函数：生成多个查询"""
    rewriter = get_rewriter()
    return rewriter.multi_query(query, num_queries)
