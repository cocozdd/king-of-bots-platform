"""
多模态 AI 模块 - 2026 最佳实践

功能：
- 图片分析（对战截图、代码截图）
- 支持 GPT-4o（OpenAI）或 DeepSeek-VL2

技术要点：
- LangChain 多模态消息格式
- Base64 图片编码
- 自动选择可用的 Vision 模型

面试要点：
- 多模态消息结构：content 数组包含 text + image_url
- Vision 模型选择：GPT-4o 最强，DeepSeek-VL2 性价比高
"""
import base64
import logging
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


@dataclass
class VisionAnalysisResult:
    """Vision 分析结果"""
    success: bool
    analysis: str
    model_used: str
    error: Optional[str] = None


def _get_vision_model_config() -> Dict[str, Any]:
    """
    获取 Vision 模型配置
    
    优先级：
    1. VISION_PROVIDER 环境变量指定
    2. 有 OPENAI_API_KEY 且支持 GPT-4o → 使用 OpenAI
    3. 有 DEEPSEEK_API_KEY → 使用 DeepSeek-VL2（需要单独配置）
    4. 回退到 DashScope qwen-vl-max
    """
    vision_provider = os.getenv("VISION_PROVIDER", "auto").lower()
    
    # 1. 显式指定 OpenAI
    if vision_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return {
                "model": os.getenv("VISION_MODEL", "gpt-4o"),
                "api_key": api_key,
                "base_url": os.getenv("OPENAI_API_BASE"),
                "provider": "openai",
            }
    
    # 2. 显式指定 DeepSeek-VL
    if vision_provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            return {
                "model": os.getenv("VISION_MODEL", "deepseek-vl2"),
                "api_key": api_key,
                "base_url": os.getenv("DEEPSEEK_VL_API_BASE", "https://api.deepseek.com/v1"),
                "provider": "deepseek",
            }
    
    # 3. 显式指定 DashScope (Qwen-VL)
    if vision_provider == "dashscope":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            return {
                "model": os.getenv("VISION_MODEL", "qwen-vl-max"),
                "api_key": api_key,
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "provider": "dashscope",
            }
    
    # 4. Auto: 尝试按优先级选择
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return {
            "model": "gpt-4o",
            "api_key": openai_key,
            "base_url": os.getenv("OPENAI_API_BASE"),
            "provider": "openai",
        }
    
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    if dashscope_key:
        return {
            "model": "qwen-vl-max",
            "api_key": dashscope_key,
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "provider": "dashscope",
        }
    
    return {}


def build_vision_llm() -> Optional[ChatOpenAI]:
    """
    构建 Vision LLM 客户端
    
    Returns:
        支持多模态的 ChatOpenAI 实例
    """
    config = _get_vision_model_config()
    if not config:
        logger.warning("没有可用的 Vision 模型配置")
        return None
    
    provider = config.pop("provider", "unknown")
    logger.info("使用 Vision 模型: %s (%s)", config.get("model"), provider)
    
    return ChatOpenAI(
        model=config["model"],
        api_key=config["api_key"],
        base_url=config.get("base_url"),
        temperature=0.3,
    )


def encode_image_to_base64(image_path: str) -> str:
    """将图片文件编码为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """根据文件扩展名获取 MIME 类型"""
    ext = image_path.lower().split(".")[-1]
    mime_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    return mime_map.get(ext, "image/jpeg")


def build_multimodal_message(
    text: str,
    image_data: str,
    image_source: str = "base64",  # "base64" | "url"
    mime_type: str = "image/jpeg",
) -> HumanMessage:
    """
    构建多模态消息
    
    Args:
        text: 文本内容
        image_data: base64 编码的图片或 URL
        image_source: 图片来源类型
        mime_type: MIME 类型
    
    Returns:
        包含文本和图片的 HumanMessage
    """
    content = [
        {"type": "text", "text": text},
    ]
    
    if image_source == "url":
        content.append({
            "type": "image_url",
            "image_url": {"url": image_data},
        })
    else:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
        })
    
    return HumanMessage(content=content)


class VisionAnalyzer:
    """
    Vision 分析器
    
    支持功能：
    1. 对战截图分析 - 分析对战局面
    2. 代码截图分析 - 识别并分析代码
    3. 地图分析 - 分析地图布局
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or build_vision_llm()
        self._model_name = "unknown"
        if self.llm:
            self._model_name = getattr(self.llm, "model_name", "vision-model")
    
    def is_available(self) -> bool:
        """检查 Vision 功能是否可用"""
        return self.llm is not None
    
    async def analyze_battle_screenshot(
        self,
        image_data: str,
        image_source: str = "base64",
        question: Optional[str] = None,
    ) -> VisionAnalysisResult:
        """
        分析对战截图
        
        Args:
            image_data: 图片数据（base64 或 URL）
            image_source: "base64" | "url"
            question: 用户问题（可选）
        
        Returns:
            分析结果
        """
        if not self.is_available():
            return VisionAnalysisResult(
                success=False,
                analysis="",
                model_used="none",
                error="Vision 模型不可用，请配置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY",
            )
        
        system_prompt = """你是 KOB 贪吃蛇对战游戏的专家分析师。

游戏规则：
- 13x14 地图，双人对战（蓝色 vs 红色）
- 无食物，蛇自动增长（前10步每步+1，之后每3步+1）
- 方向：0=上, 1=右, 2=下, 3=左
- 撞墙或撞蛇身则输

分析要点：
1. 当前局面评估（蛇的位置、长度、可移动空间）
2. 关键风险点（可能被封死的位置）
3. 建议的下一步策略
4. 预测的胜负趋势

用中文回答，简洁专业。"""
        
        user_text = question or "请分析这个对战截图，说明当前局面和建议策略。"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                build_multimodal_message(user_text, image_data, image_source),
            ]
            
            response = await self.llm.ainvoke(messages)
            
            return VisionAnalysisResult(
                success=True,
                analysis=response.content,
                model_used=self._model_name,
            )
        except Exception as e:
            logger.error("对战截图分析失败: %s", e)
            return VisionAnalysisResult(
                success=False,
                analysis="",
                model_used=self._model_name,
                error=str(e),
            )
    
    async def analyze_code_screenshot(
        self,
        image_data: str,
        image_source: str = "base64",
        question: Optional[str] = None,
    ) -> VisionAnalysisResult:
        """
        分析代码截图
        
        从截图中识别代码并进行分析
        """
        if not self.is_available():
            return VisionAnalysisResult(
                success=False,
                analysis="",
                model_used="none",
                error="Vision 模型不可用",
            )
        
        system_prompt = """你是 KOB Bot 代码分析专家。

任务：
1. 识别截图中的代码
2. 分析代码逻辑
3. 找出潜在问题
4. 提供改进建议

KOB 游戏规则（重要）：
- 无食物！蛇自动增长
- 方向：0=上, 1=右, 2=下, 3=左
- nextMove() 返回方向值

公平性原则：
- 只提供学习指导，不直接给完整解决方案
- 帮助理解算法原理，不给现成答案

用中文回答。"""
        
        user_text = question or "请识别并分析这段代码，指出问题和改进方向。"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                build_multimodal_message(user_text, image_data, image_source),
            ]
            
            response = await self.llm.ainvoke(messages)
            
            return VisionAnalysisResult(
                success=True,
                analysis=response.content,
                model_used=self._model_name,
            )
        except Exception as e:
            logger.error("代码截图分析失败: %s", e)
            return VisionAnalysisResult(
                success=False,
                analysis="",
                model_used=self._model_name,
                error=str(e),
            )
    
    async def analyze_map(
        self,
        image_data: str,
        image_source: str = "base64",
        question: Optional[str] = None,
    ) -> VisionAnalysisResult:
        """
        分析地图布局
        """
        if not self.is_available():
            return VisionAnalysisResult(
                success=False,
                analysis="",
                model_used="none",
                error="Vision 模型不可用",
            )
        
        system_prompt = """你是 KOB 地图分析专家。

分析要点：
1. 障碍物分布
2. 关键位置（中心、角落、通道）
3. 初始位置优劣
4. 策略建议

用中文简洁回答。"""
        
        user_text = question or "请分析这个地图的布局和关键位置。"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                build_multimodal_message(user_text, image_data, image_source),
            ]
            
            response = await self.llm.ainvoke(messages)
            
            return VisionAnalysisResult(
                success=True,
                analysis=response.content,
                model_used=self._model_name,
            )
        except Exception as e:
            logger.error("地图分析失败: %s", e)
            return VisionAnalysisResult(
                success=False,
                analysis="",
                model_used=self._model_name,
                error=str(e),
            )


# 全局实例
_vision_analyzer: Optional[VisionAnalyzer] = None


def get_vision_analyzer() -> VisionAnalyzer:
    """获取全局 Vision 分析器"""
    global _vision_analyzer
    if _vision_analyzer is None:
        _vision_analyzer = VisionAnalyzer()
    return _vision_analyzer


async def analyze_image(
    image_data: str,
    analysis_type: str = "battle",  # "battle" | "code" | "map"
    image_source: str = "base64",
    question: Optional[str] = None,
) -> Dict[str, Any]:
    """
    便捷函数：分析图片
    
    Args:
        image_data: 图片数据
        analysis_type: 分析类型
        image_source: "base64" | "url"
        question: 用户问题
    
    Returns:
        分析结果字典
    """
    analyzer = get_vision_analyzer()
    
    if analysis_type == "battle":
        result = await analyzer.analyze_battle_screenshot(
            image_data, image_source, question
        )
    elif analysis_type == "code":
        result = await analyzer.analyze_code_screenshot(
            image_data, image_source, question
        )
    elif analysis_type == "map":
        result = await analyzer.analyze_map(
            image_data, image_source, question
        )
    else:
        result = await analyzer.analyze_battle_screenshot(
            image_data, image_source, question
        )
    
    return {
        "success": result.success,
        "analysis": result.analysis,
        "model_used": result.model_used,
        "error": result.error,
    }
