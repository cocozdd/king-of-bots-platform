"""
成本追踪系统 - P1 改进

功能：
- 追踪 LLM API 调用成本
- 按用户/功能统计
- 预算控制和告警

面试要点：
- 为什么需要成本追踪：控制预算、优化成本分配、识别异常使用
- 价格模型：不同模型价格差异大（GPT-4 vs DeepSeek 差 300 倍）
- 预算控制：实时监控 + 告警 + 自动限流
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


# 价格表（每 1K tokens，USD）- 2026 年价格
PRICE_PER_1K_TOKENS = {
    # OpenAI
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    
    # DeepSeek（性价比之王）
    "deepseek-chat": {"input": 0.00014, "output": 0.00028},
    "deepseek-coder": {"input": 0.00014, "output": 0.00028},
    
    # Claude
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    
    # Embedding
    "text-embedding-v3": {"input": 0.00002, "output": 0},
    "text-embedding-ada-002": {"input": 0.0001, "output": 0},
}


@dataclass
class UsageRecord:
    """使用记录"""
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    user_id: Optional[int] = None
    feature: Optional[str] = None
    trace_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "user_id": self.user_id,
            "feature": self.feature,
            "trace_id": self.trace_id,
        }


class CostTracker:
    """
    成本追踪器
    
    功能：
    - 记录每次 LLM 调用的 token 使用量和成本
    - 按时间、用户、功能维度统计
    - 预算控制和告警
    """
    
    def __init__(
        self,
        daily_budget: float = 100.0,
        alert_threshold: float = 0.8,
    ):
        self.daily_budget = daily_budget
        self.alert_threshold = alert_threshold
        self.records: List[UsageRecord] = []
        self._daily_cost_cache: Dict[str, float] = {}
        
        logger.info(
            "CostTracker 初始化: daily_budget=$%.2f, alert_threshold=%.0f%%",
            daily_budget, alert_threshold * 100,
        )
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        计算单次调用成本
        
        Args:
            model: 模型名称
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数
            
        Returns:
            成本（美元）
        """
        prices = PRICE_PER_1K_TOKENS.get(model)
        
        if not prices:
            # 尝试模糊匹配
            for key in PRICE_PER_1K_TOKENS:
                if key in model.lower():
                    prices = PRICE_PER_1K_TOKENS[key]
                    break
        
        if not prices:
            logger.warning("未知模型 %s，使用 deepseek-chat 价格", model)
            prices = PRICE_PER_1K_TOKENS["deepseek-chat"]
        
        cost = (
            input_tokens * prices["input"] +
            output_tokens * prices["output"]
        ) / 1000
        
        return cost
    
    def track_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        user_id: Optional[int] = None,
        feature: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> float:
        """
        记录使用量并计算成本
        
        Args:
            model: 模型名称
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数
            user_id: 用户 ID（可选）
            feature: 功能名称（可选）
            trace_id: 追踪 ID（可选）
            
        Returns:
            本次调用成本（美元）
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        record = UsageRecord(
            timestamp=datetime.now(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            user_id=user_id,
            feature=feature,
            trace_id=trace_id,
        )
        
        self.records.append(record)
        
        # 更新日成本缓存
        today = datetime.now().strftime("%Y-%m-%d")
        self._daily_cost_cache[today] = self._daily_cost_cache.get(today, 0) + cost
        
        # 检查预算
        self._check_budget_alert(today)
        
        logger.info(
            "[Cost] model=%s, tokens=%d+%d, cost=$%.6f, user=%s, feature=%s",
            model, input_tokens, output_tokens, cost, user_id, feature,
        )
        
        return cost
    
    def _check_budget_alert(self, date: str):
        """检查预算告警"""
        current_cost = self._daily_cost_cache.get(date, 0)
        
        if current_cost >= self.daily_budget:
            logger.error(
                "[Budget] 每日预算已超出: $%.2f / $%.2f",
                current_cost, self.daily_budget,
            )
        elif current_cost >= self.daily_budget * self.alert_threshold:
            logger.warning(
                "[Budget] 接近每日预算: $%.2f / $%.2f (%.0f%%)",
                current_cost, self.daily_budget,
                current_cost / self.daily_budget * 100,
            )
    
    def get_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[int] = None,
        feature: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        获取统计数据
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            user_id: 用户 ID 过滤
            feature: 功能名称过滤
            
        Returns:
            统计数据
        """
        # 过滤记录
        filtered = self.records
        
        if start_time:
            filtered = [r for r in filtered if r.timestamp >= start_time]
        if end_time:
            filtered = [r for r in filtered if r.timestamp <= end_time]
        if user_id:
            filtered = [r for r in filtered if r.user_id == user_id]
        if feature:
            filtered = [r for r in filtered if r.feature == feature]
        
        if not filtered:
            return {
                "total_cost_usd": 0,
                "total_requests": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "average_cost_per_request": 0,
                "by_model": {},
                "by_feature": {},
            }
        
        # 统计
        total_cost = sum(r.cost_usd for r in filtered)
        total_input = sum(r.input_tokens for r in filtered)
        total_output = sum(r.output_tokens for r in filtered)
        total_requests = len(filtered)
        
        # 按模型分组
        by_model: Dict[str, Dict] = defaultdict(lambda: {
            "requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0
        })
        for r in filtered:
            by_model[r.model]["requests"] += 1
            by_model[r.model]["input_tokens"] += r.input_tokens
            by_model[r.model]["output_tokens"] += r.output_tokens
            by_model[r.model]["cost_usd"] += r.cost_usd
        
        # 按功能分组
        by_feature: Dict[str, Dict] = defaultdict(lambda: {
            "requests": 0, "cost_usd": 0
        })
        for r in filtered:
            feat = r.feature or "unknown"
            by_feature[feat]["requests"] += 1
            by_feature[feat]["cost_usd"] += r.cost_usd
        
        return {
            "total_cost_usd": round(total_cost, 6),
            "total_requests": total_requests,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "average_cost_per_request": round(total_cost / total_requests, 6),
            "by_model": dict(by_model),
            "by_feature": dict(by_feature),
        }
    
    def get_daily_cost(self, date: Optional[datetime] = None) -> float:
        """获取指定日期的成本"""
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime("%Y-%m-%d")
        
        # 先检查缓存
        if date_str in self._daily_cost_cache:
            return self._daily_cost_cache[date_str]
        
        # 计算
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        
        cost = sum(
            r.cost_usd for r in self.records
            if start <= r.timestamp < end
        )
        
        self._daily_cost_cache[date_str] = cost
        return cost
    
    def check_budget(self, daily_budget: float = None) -> bool:
        """
        检查是否超出预算
        
        Returns:
            True 如果在预算内，False 如果超出
        """
        if daily_budget is None:
            daily_budget = self.daily_budget
        
        current_cost = self.get_daily_cost()
        return current_cost < daily_budget
    
    def get_recent_records(self, limit: int = 100) -> List[dict]:
        """获取最近的使用记录"""
        recent = self.records[-limit:]
        return [r.to_dict() for r in reversed(recent)]
    
    def export_records(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[dict]:
        """导出记录（用于持久化）"""
        filtered = self.records
        
        if start_time:
            filtered = [r for r in filtered if r.timestamp >= start_time]
        if end_time:
            filtered = [r for r in filtered if r.timestamp <= end_time]
        
        return [r.to_dict() for r in filtered]


# ============ 全局实例管理 ============

_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """获取成本追踪器"""
    global _cost_tracker
    
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    
    return _cost_tracker


def track_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    user_id: Optional[int] = None,
    feature: Optional[str] = None,
) -> float:
    """便捷函数：记录使用量"""
    return get_cost_tracker().track_usage(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        user_id=user_id,
        feature=feature,
    )


def get_cost_stats(days: int = 7) -> dict:
    """便捷函数：获取最近 N 天的统计"""
    start = datetime.now() - timedelta(days=days)
    return get_cost_tracker().get_stats(start_time=start)
