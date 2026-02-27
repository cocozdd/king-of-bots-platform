"""
重排序服务 - 使用 Cohere, SiliconFlow 或 LLM 对检索结果重排序

功能：
- API Rerank (推荐): 调用 Cohere 或 SiliconFlow 的 Cross-Encoder 模型
- LLM Rerank (备选): 使用 LLM 对结果评分
- 简单重排序 (兜底): 基于关键词匹配

配置 (在 .env 中):
- RERANK_PROVIDER: auto | cohere | siliconflow | llm
- COHERE_API_KEY: Cohere API Key
- SILICONFLOW_API_KEY: SiliconFlow API Key

面试要点：
- 为什么使用 Cross-Encoder API：比 LLM 更快（~200ms vs ~1s），比向量检索更准（全注意力机制）
- 降级策略：API 失败时自动降级到 LLM 或 简单算法，保证服务高可用
"""
import logging
import os
import requests
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 默认相关性阈值
DEFAULT_RELEVANCE_THRESHOLD = 0.3


@dataclass
class RerankResult:
    """重排序结果"""
    id: int
    title: str
    content: str
    category: str
    original_score: float
    rerank_score: float
    relevance: str  # high, medium, low


class Reranker:
    """重排序器"""
    
    def __init__(
        self,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ):
        self.relevance_threshold = relevance_threshold
        
        # 加载配置
        self.provider = os.getenv("RERANK_PROVIDER", "auto").lower()
        self.cohere_key = os.getenv("COHERE_API_KEY")
        self.siliconflow_key = os.getenv("SILICONFLOW_API_KEY")
        self.rerank_model = os.getenv("RERANK_MODEL", "rerank-multilingual-v3.0")
        
        self._llm = None
        
        # 初始化日志
        logger.info(f"Reranker 初始化: Provider={self.provider}, Model={self.rerank_model}")
        if self.cohere_key:
            logger.info("Cohere API Key 已配置")
        if self.siliconflow_key:
            logger.info("SiliconFlow API Key 已配置")
    
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
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None,
    ) -> List[RerankResult]:
        """
        对检索结果重排序
        
        策略：
        1. 尝试 API Rerank (Cohere/SiliconFlow)
        2. 失败或未配置 -> 降级到 LLM Rerank
        3. 失败 -> 降级到 简单 Rerank
        """
        if not documents:
            return []
        
        logger.info("重排序开始: query='%s', docs=%d", query[:50], len(documents))
        
        results = None
        
        # 1. 尝试 API Rerank
        if self.provider in ["auto", "cohere", "siliconflow"]:
            results = self._try_api_rerank(query, documents, top_k)
            
        # 2. 尝试 LLM Rerank (如果 API 失败或配置为 LLM)
        if results is None and self.provider in ["auto", "llm"]:
            logger.info("尝试 LLM Rerank...")
            try:
                results = self._llm_rerank(query, documents)
            except Exception as e:
                logger.warning(f"LLM Rerank 失败: {e}")
        
        # 3. 兜底 Simple Rerank
        if results is None:
            logger.warning("降级到简单重排序")
            results = self._simple_rerank(query, documents)
        
        # 过滤低相关性结果
        results = [r for r in results if r.rerank_score >= self.relevance_threshold]
        
        # 限制返回数量
        if top_k:
            results = results[:top_k]
        
        logger.info("重排序完成: 返回 %d 结果", len(results))
        return results

    def _try_api_rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> Optional[List[RerankResult]]:
        """尝试调用 API 进行重排序"""
        
        # 优先尝试 Cohere
        if (self.provider == "cohere" or self.provider == "auto") and self.cohere_key:
            try:
                return self._cohere_rerank(query, documents, top_k)
            except Exception as e:
                logger.error(f"Cohere Rerank 失败: {e}")
        
        # 其次尝试 SiliconFlow
        if (self.provider == "siliconflow" or self.provider == "auto") and self.siliconflow_key:
            try:
                return self._siliconflow_rerank(query, documents, top_k)
            except Exception as e:
                logger.error(f"SiliconFlow Rerank 失败: {e}")
                
        return None

    def _cohere_rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[RerankResult]:
        """调用 Cohere Rerank API"""
        logger.info("调用 Cohere Rerank API...")
        
        url = "https://api.cohere.ai/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.cohere_key}",
            "Content-Type": "application/json",
            "X-Client-Name": "kob-ai-service"
        }
        
        # 提取文档文本
        docs_text = [
            f"{doc.get('title', '')} {doc.get('content', '')}"[:4000] # Cohere 限制
            for doc in documents
        ]
        
        payload = {
            "model": self.rerank_model,
            "query": query,
            "documents": docs_text,
            "top_n": top_k or len(documents)
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get("results", []):
            idx = item["index"]
            score = item["relevance_score"]
            doc = documents[idx]
            
            results.append(RerankResult(
                id=doc.get("id", 0),
                title=doc.get("title", ""),
                content=doc.get("content", ""),
                category=doc.get("category", ""),
                original_score=doc.get("score", 0),
                rerank_score=score,
                relevance=self._score_to_relevance(score)
            ))
            
        return results

    def _siliconflow_rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[RerankResult]:
        """调用 SiliconFlow Rerank API (兼容 BGE 模型)"""
        logger.info("调用 SiliconFlow Rerank API...")
        
        url = "https://api.siliconflow.cn/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.siliconflow_key}",
            "Content-Type": "application/json"
        }
        
        docs_text = [
            f"{doc.get('title', '')} {doc.get('content', '')}"[:4000]
            for doc in documents
        ]
        
        # SiliconFlow 模型通常是 BAAI/bge-reranker-v2-m3
        model = os.getenv("RERANK_MODEL_ALT", "BAAI/bge-reranker-v2-m3")
        
        payload = {
            "model": model,
            "query": query,
            "documents": docs_text,
            "top_n": top_k or len(documents),
            "return_documents": False
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get("results", []):
            idx = item["index"]
            score = item["relevance_score"]
            doc = documents[idx]
            
            results.append(RerankResult(
                id=doc.get("id", 0),
                title=doc.get("title", ""),
                content=doc.get("content", ""),
                category=doc.get("category", ""),
                original_score=doc.get("score", 0),
                rerank_score=score,
                relevance=self._score_to_relevance(score)
            ))
            
        return results
    
    def _llm_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> List[RerankResult]:
        """使用 LLM 批量重排序 - 2026 优化版：一次 API 调用评估所有文档"""
        llm = self._get_llm()
        if llm is None:
            logger.warning("LLM 不可用，降级为简单重排序")
            return self._simple_rerank(query, documents)
        
        # 批量评分：一次 LLM 调用评估所有文档
        scores = self._batch_score_documents(llm, query, documents)
        
        results = []
        for i, doc in enumerate(documents):
            score = scores[i] if i < len(scores) else doc.get("score", 0.5)
            relevance = self._score_to_relevance(score)
            
            results.append(RerankResult(
                id=doc.get("id", 0),
                title=doc.get("title", ""),
                content=doc.get("content", ""),
                category=doc.get("category", ""),
                original_score=doc.get("score", 0),
                rerank_score=score,
                relevance=relevance,
            ))
        
        # 按重排序分数排序
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results
    
    def _batch_score_documents(
        self,
        llm,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> List[float]:
        """批量评分所有文档 - 一次 API 调用"""
        if not documents:
            return []
        
        # 构建批量评分提示
        docs_text = "\n\n".join([
            f"[文档{i+1}]\n标题: {doc.get('title', '')}\n内容: {doc.get('content', '')[:300]}"
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""请评估以下文档与查询的相关性。

查询: {query}

{docs_text}

请为每个文档打分（0-1之间），格式如下（每行一个分数）：
文档1: 0.X
文档2: 0.X
...

评分标准：
- 0.9-1.0: 高度相关，直接回答查询
- 0.6-0.8: 中度相关，包含有用信息
- 0.3-0.5: 低度相关，仅部分相关
- 0-0.2: 不相关"""
        
        try:
            from langchain_core.messages import HumanMessage
            
            response = llm.invoke([HumanMessage(content=prompt)])
            content = getattr(response, "content", str(response))
            
            # 解析批量分数
            return self._parse_batch_scores(content, len(documents))
        except Exception as e:
            logger.warning("LLM 批量评分失败: %s", e)
            return [doc.get("score", 0.5) for doc in documents]
    
    def _parse_batch_scores(self, content: str, num_docs: int) -> List[float]:
        """解析批量评分结果"""
        import re
        
        scores = []
        lines = content.strip().split("\n")
        
        for line in lines:
            # 匹配 "文档X: 0.X" 或 "0.X" 格式
            match = re.search(r"(\d+\.?\d*)", line)
            if match:
                score = float(match.group(1))
                if score > 1:
                    score = score / 10 if score <= 10 else score / 100
                scores.append(min(max(score, 0), 1))
        
        # 如果解析的分数不够，用默认值补充
        while len(scores) < num_docs:
            scores.append(0.5)
        
        return scores[:num_docs]
    
    def _simple_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> List[RerankResult]:
        """简单重排序（基于原始分数和关键词匹配）"""
        query_terms = set(query.lower().split())
        
        results = []
        for doc in documents:
            original_score = doc.get("score", 0)
            
            # 计算关键词匹配度
            title = doc.get("title", "").lower()
            content = doc.get("content", "").lower()
            
            title_matches = sum(1 for term in query_terms if term in title)
            content_matches = sum(1 for term in query_terms if term in content)
            
            keyword_score = (title_matches * 2 + content_matches) / (len(query_terms) * 3) if query_terms else 0
            
            # 综合分数
            rerank_score = original_score * 0.7 + keyword_score * 0.3
            relevance = self._score_to_relevance(rerank_score)
            
            results.append(RerankResult(
                id=doc.get("id", 0),
                title=doc.get("title", ""),
                content=doc.get("content", ""),
                category=doc.get("category", ""),
                original_score=original_score,
                rerank_score=rerank_score,
                relevance=relevance,
            ))
        
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results
    
    def _score_to_relevance(self, score: float) -> str:
        """分数转相关性标签"""
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"


# 全局实例
_reranker: Optional[Reranker] = None


def get_reranker() -> Reranker:
    """获取全局重排序器实例"""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def rerank(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int = None,
) -> List[Dict[str, Any]]:
    """便捷函数：执行重排序"""
    reranker = get_reranker()
    results = reranker.rerank(query, documents, top_k)
    return [
        {
            "id": r.id,
            "title": r.title,
            "content": r.content,
            "category": r.category,
            "original_score": r.original_score,
            "rerank_score": r.rerank_score,
            "relevance": r.relevance,
        }
        for r in results
    ]
