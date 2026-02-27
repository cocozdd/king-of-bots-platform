"""
工具定义 - 使用 LangChain @tool 装饰器

实现真正的工具（替代 Java 端的 Stub）：
- knowledge_search: 知识库检索
- code_analysis: 代码分析
- battle_query: 对战记录查询
- strategy_recommend: 策略推荐
- calculator: 数学计算

P1 改进：
- 添加 tenacity 重试机制，提升工具调用成功率
- 指数退避策略，避免雪崩
"""
import logging
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import requests
from langchain_core.tools import tool, ToolException
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# 重试配置
RETRY_MAX_ATTEMPTS = 3
RETRY_WAIT_MIN = 1  # 最小等待秒数
RETRY_WAIT_MAX = 10  # 最大等待秒数


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    output: str
    data: Optional[Dict[str, Any]] = None


# ==================== 知识库检索工具 ====================

def _knowledge_search_impl(query: str, top_k: int = 5) -> str:
    """知识库检索实现（带重试）"""
    from rag.hybrid_search import hybrid_search
    
    logger.info("[knowledge_search] query='%s', top_k=%d", query[:50], top_k)
    results = hybrid_search(query, top_k=top_k)
    
    if not results:
        return f"未找到与 '{query}' 相关的信息"
    
    output_parts = [f"找到 {len(results)} 条相关结果：\n"]
    for i, doc in enumerate(results, 1):
        output_parts.append(f"{i}. 【{doc.get('title', '无标题')}】")
        content = doc.get("content", "")[:300]
        output_parts.append(f"   {content}")
        output_parts.append(f"   [相关度: {doc.get('score', 0):.2f}]\n")
    
    return "\n".join(output_parts)


@tool
@retry(
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def knowledge_search(query: str, top_k: int = 5) -> str:
    """
    搜索知识库获取相关信息。用于回答关于游戏规则、Bot 开发、策略等问题。
    
    Args:
        query: 搜索查询词
        top_k: 返回结果数量，默认5条
        
    Returns:
        检索到的相关文档内容
        
    重试策略：最多3次，指数退避（1s, 2s, 4s）
    """
    try:
        return _knowledge_search_impl(query, top_k)
    except (ConnectionError, TimeoutError, OSError) as e:
        logger.warning("[knowledge_search] 可重试错误: %s", e)
        raise
    except Exception as e:
        logger.error("知识库检索失败: %s", e)
        raise ToolException(f"知识库检索失败: {e}")


# ==================== 代码分析工具 ====================

@tool(return_direct=True)
def code_analysis(code: str, analysis_type: str = "review") -> str:
    """
    分析 Bot 代码，提供改进建议或解释代码逻辑。
    
    Args:
        code: 要分析的代码片段
        analysis_type: 分析类型，可选值：review（代码审查）、explain（解释逻辑）、explain_with_code（附源码解释）、code_only（仅源码）、optimize（优化建议）
        
    Returns:
        代码分析结果
    """
    try:
        if analysis_type == "code_only":
            return f"""```java
{code}
```"""

        from llm_client import build_llm, should_use_llm
        
        if not should_use_llm():
            return _simple_code_analysis(code, analysis_type)
        
        llm = build_llm(streaming=False)
        if llm is None:
            return _simple_code_analysis(code, analysis_type)
        
        prompts = {
            "review": f"""请对以下 Bot 代码进行代码审查，指出潜在问题和改进建议：

```
{code}
```

请从以下方面分析：
1. 代码逻辑是否正确
2. 是否有潜在的 bug
3. 性能优化建议
4. 代码风格改进""",
            
            "explain": f"""请按严格格式解释以下 Bot 代码，避免重复：

```
{code}
```

只输出 4 行，格式必须完全一致（不要额外标题或段落）：
1) 结构: 说明 package/import、类声明与接口实现
2) 关键方法: 说明辅助方法的作用
3) 决策流程: 说明入口方法的步骤
4) 一句话策略: 用一句话概括策略

总字数 160-220 字以内，保持概括性；除非用户明确要求，不要扩展优缺点/改进建议。""",

            "explain_with_code": f"""请按以下格式输出，避免重复：

先输出完整源码（使用 ```java 代码块），然后输出 4 行说明，格式必须完全一致（不要额外标题或段落）：
1) 结构: 说明 package/import、类声明与接口实现
2) 关键方法: 说明辅助方法的作用
3) 决策流程: 说明入口方法的步骤
4) 一句话策略: 用一句话概括策略

说明部分需比 explain 更具体，每行至少包含 1 个具体标识符或条件（如方法名、变量名、常量、判断条件），并在 2) 或 3) 行包含 1 个简短代码片段（用反引号），避免与上一条“概括性解释”重复措辞。

源码如下：
```java
{code}
```

说明部分总字数 240-360 字以内；除非用户明确要求，不要扩展优缺点/改进建议。""",

            "code_only": f"""请仅输出完整源码，使用 ```java 代码块包裹，不要添加任何说明：

```java
{code}
```""",
            
            "optimize": f"""请对以下 Bot 代码提供优化建议：

```
{code}
```

请从以下方面提供建议：
1. 性能优化
2. 策略改进
3. 代码简化""",
        }
        
        prompt = prompts.get(analysis_type, prompts["review"])
        
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        return getattr(response, "content", str(response))
    except Exception as e:
        logger.error("代码分析失败: %s", e)
        raise ToolException(f"代码分析失败: {e}")


def _simple_code_analysis(code: str, analysis_type: str) -> str:
    """简单代码分析（不使用 LLM）"""
    if analysis_type in ("explain", "explain_with_code"):
        package_match = re.search(r"package\s+([\w\.]+);", code)
        package_name = package_match.group(1) if package_match else "未知包"
        class_match = re.search(r"class\s+(\w+)", code)
        class_name = class_match.group(1) if class_match else "Bot"
        implements_match = re.search(r"implements\s+([^{]+)", code)
        implements = implements_match.group(1).strip() if implements_match else ""
        imports = re.findall(r"import\s+([^;]+);", code)
        import_hint = ", ".join(imports[:2]) if imports else "未显式导入"

        method_names = re.findall(r"(?:public|private|protected)\s+[\w<>\[\]]+\s+(\w+)\s*\(", code)
        method_names = [name for name in method_names if name not in ("Bot", class_name)]
        method_hint = "、".join(method_names[:3]) if method_names else "主要入口方法"

        has_check_tail = "checkTailIncreasing" in code
        has_get_cells = "getCells" in code
        has_input = "System.getenv(\"INPUT\")" in code
        has_map_size = "new int[13][14]" in code
        has_safe_moves = "safeMoves" in code
        has_random = "random.nextInt" in code

        structure_line = (
            f"1) 结构: package 为 {package_name}，类 {class_name} "
            f"{('实现 ' + implements) if implements else '未显示接口'}，常用依赖如 {import_hint}。"
        )
        key_methods_parts = []
        if has_check_tail:
            key_methods_parts.append("`checkTailIncreasing(step)` 负责蛇身增长节奏")
        if has_get_cells:
            key_methods_parts.append("`getCells(sx, sy, steps)` 依据 steps 还原蛇身坐标")
        if not key_methods_parts:
            key_methods_parts.append(f"核心方法包含 {method_hint}")
        key_methods_line = f"2) 关键方法: {'；'.join(key_methods_parts)}。"

        flow_parts = []
        if has_input:
            flow_parts.append("从 `System.getenv(\"INPUT\")` 读取并 split('#') 解析")
        if has_map_size:
            flow_parts.append("构建 `gameMap[13][14]` 标记障碍物")
        if has_safe_moves:
            flow_parts.append("收集 `safeMoves` 作为安全方向集合")
        if not flow_parts:
            flow_parts.append("解析地图和蛇身后筛选可走方向")
        flow_line = f"3) 决策流程: {'，'.join(flow_parts)}，最后随机选取方向。"

        strategy_tail = "在 safeMoves 中随机挑选方向前进" if has_random else "随机在可行方向中选择"
        strategy_line = f"4) 一句话策略: {strategy_tail}，优先规避即时碰撞。"

        explain_block = "\n".join([structure_line, key_methods_line, flow_line, strategy_line])
        if analysis_type == "explain_with_code":
            return f"""```java
{code}
```

{explain_block}"""
        return explain_block
    if analysis_type == "code_only":
        return f"""```java
{code}
```"""

    lines = code.split("\n")
    line_count = len(lines)
    
    # 简单统计
    has_loop = any("for" in line or "while" in line for line in lines)
    has_condition = any("if" in line for line in lines)
    has_function = any("def " in line or "function" in line for line in lines)
    
    analysis = f"""代码分析结果（简化版）：
- 代码行数: {line_count}
- 包含循环: {'是' if has_loop else '否'}
- 包含条件判断: {'是' if has_condition else '否'}
- 包含函数定义: {'是' if has_function else '否'}

注意：完整的代码分析需要 LLM 支持，当前为简化分析。"""
    
    return analysis


# ==================== 对战记录查询工具 ====================

@tool
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def battle_query(user_id: int, limit: int = 10) -> str:
    """
    查询用户的对战记录和统计数据。
    
    Args:
        user_id: 用户ID
        limit: 返回记录数量，默认10条
        
    Returns:
        对战记录摘要
        
    重试策略：最多2次，指数退避
    """
    try:
        from db_client import get_battle_records, get_user_stats
        logger.info("[battle_query] user_id=%d, limit=%d", user_id, limit)
        
        # 获取统计数据
        stats = get_user_stats(user_id)
        
        # 获取最近对战记录
        records = get_battle_records(user_id, limit)
        
        if not records and stats["total_games"] == 0:
            return f"用户 {user_id} 暂无对战记录"
        
        output_parts = [
            f"用户 {user_id} 对战统计：",
            f"- 总场次: {stats['total_games']}",
            f"- 胜场: {stats['wins']}",
            f"- 负场: {stats['losses']}",
            f"- 胜率: {stats['win_rate']}%",
            "",
            f"最近 {len(records)} 场对战记录：",
        ]
        
        for i, record in enumerate(records, 1):
            is_player_a = record.get("a_id") == user_id
            opponent_id = record.get("b_id") if is_player_a else record.get("a_id")
            loser = record.get("loser", "")
            
            if loser == "A":
                result = "负" if is_player_a else "胜"
            elif loser == "B":
                result = "胜" if is_player_a else "负"
            else:
                result = "平局"
            
            output_parts.append(f"{i}. vs 玩家{opponent_id} - {result}")
        
        return "\n".join(output_parts)
    except Exception as e:
        logger.error("对战记录查询失败: %s", e)
        raise ToolException(f"对战记录查询失败: {e}")


# ==================== 对战分析工具 ====================

@tool
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, requests.RequestException)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def battle_analysis(record_id: int) -> str:
    """
    详细分析某场对战的回放，指出关键失误和优化策略。
    
    Args:
        record_id: 对战记录ID
        
    Returns:
        AI 分析报告，包含局面评估、关键帧和改进建议。
    """
    try:
        # 假设 Java 后端运行在本地 3000 端口
        # 在生产环境中应通过环境变量获取
        backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:3000")
        url = f"{backend_url}/ai/analysis/record/{record_id}"
        
        logger.info("[battle_analysis] calling %s", url)
        resp = requests.get(url, timeout=60)  # 分析可能较慢
        
        if resp.status_code == 404:
            return f"未找到对战记录 #{record_id}"
        if resp.status_code != 200:
            return f"分析服务异常: HTTP {resp.status_code}"
            
        data = resp.json()
        if "error" in data:
            return f"分析失败: {data['error']}"
            
        stats = data.get("stats", {})
        ai_analysis = data.get("ai_analysis", "（AI 未生成详细点评）")
        key_moments = data.get("key_moments", [])
        
        output_parts = [
            f"【对战分析报告 #{record_id}】",
            f"结果: {stats.get('winner', '未知')}",
            f"局面控制: A({stats.get('controlAreaA', 0)}) vs B({stats.get('controlAreaB', 0)})",
            f"移动风格: A[{stats.get('movementPatternA', '')}] vs B[{stats.get('movementPatternB', '')}]",
            "",
            "## AI 深度复盘",
            ai_analysis
        ]
        
        if key_moments:
            output_parts.append("\n## 关键时刻")
            for m in key_moments:
                output_parts.append(f"- 第{m.get('round')}回合 [{m.get('type')}]: {m.get('description')}")
                
        return "\n".join(output_parts)
        
    except Exception as e:
        logger.error("对战分析请求失败: %s", e)
        raise ToolException(f"无法连接分析服务: {e}")


# ==================== 失败原因分析工具 ====================

@tool
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, requests.RequestException)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def loss_reason_analyzer(record_id: int, user_id: int) -> str:
    """
    专门分析用户在某场对战中输掉的具体原因。

    Args:
        record_id: 对战记录ID
        user_id: 用户ID（用于确定分析视角）

    Returns:
        针对该用户的失败原因分析报告，包含关键失误时刻、策略对比、具体改进建议。
    """
    try:
        backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:3000")
        url = f"{backend_url}/ai/analysis/loss/{record_id}?userId={user_id}"

        logger.info("[loss_reason_analyzer] calling %s", url)
        resp = requests.get(url, timeout=60)

        if resp.status_code == 404:
            return f"未找到对战记录 #{record_id}"
        if resp.status_code == 400:
            data = resp.json()
            return f"分析失败: {data.get('error', '用户未参与此对战')}"
        if resp.status_code != 200:
            return f"分析服务异常: HTTP {resp.status_code}"

        data = resp.json()

        # 构建用户友好的失败分析报告
        is_loser = data.get("is_loser", False)
        player_role = data.get("player_role", "未知")

        output_parts = [
            f"【失败原因分析 #{record_id}】",
            "",
            "## 对战结果",
            f"你（玩家{player_role}）在这场对战中{'输了' if is_loser else '赢了'}",
            f"总回合数: {data.get('total_rounds', 0)}",
            f"比赛结果: {data.get('result', '未知')}",
            "",
            "## 控制区域对比",
            f"- 你的控制区域: {data.get('your_control_area', 0)} 格",
            f"- 对手控制区域: {data.get('opponent_control_area', 0)} 格",
            "",
            "## 策略分析",
            f"- 你的移动策略: {data.get('your_strategy', '未知')}",
            f"- 对手移动策略: {data.get('opponent_strategy', '未知')}",
        ]

        # 关键失误
        critical_mistakes = data.get("critical_mistakes", [])
        if critical_mistakes:
            output_parts.extend(["", "## 关键失误时刻"])
            for moment in critical_mistakes:
                output_parts.append(
                    f"- 第{moment.get('round', '?')}回合 [{moment.get('type', '')}]: {moment.get('description', '')}"
                )

        # 改进建议
        suggestions = data.get("suggestions", [])
        if suggestions:
            output_parts.extend(["", "## 改进建议"])
            for suggestion in suggestions:
                output_parts.append(f"- {suggestion}")

        # AI 深度分析
        ai_analysis = data.get("ai_analysis")
        if ai_analysis:
            output_parts.extend([
                "",
                "## AI 深度分析",
                ai_analysis
            ])

        return "\n".join(output_parts)

    except Exception as e:
        logger.error("失败原因分析请求失败: %s", e)
        raise ToolException(f"无法连接分析服务: {e}")


# ==================== 策略推荐工具 ====================

@tool
def strategy_recommend(scenario: str) -> str:
    """
    基于游戏场景推荐策略。
    
    Args:
        scenario: 游戏场景描述，如"开局"、"中期"、"被围堵"等
        
    Returns:
        策略推荐
    """
    try:
        # 预定义策略库
        strategies = {
            "开局": """开局策略推荐：
1. 优先扩展领地，占据更多空间
2. 保持蛇身紧凑，减少被截断的风险
3. 观察对手动向，准备应对

关键点：开局阶段要平衡扩张和安全""",
            
            "中期": """中期策略推荐：
1. 根据蛇长选择激进或保守策略
2. 利用地形优势，封堵对手路线
3. 保持逃生路线，避免死角

关键点：中期是决定胜负的关键阶段""",
            
            "被围堵": """被围堵时的策略：
1. 寻找最近的出口
2. 利用对手蛇身的空隙
3. 如果无法逃脱，尝试与对手同归于尽

关键点：保持冷静，计算所有可能的路径""",
            
            "领先": """领先时的策略：
1. 不要过于激进，保持优势
2. 逐步压缩对手空间
3. 避免冒险操作

关键点：稳中求胜""",
            
            "落后": """落后时的策略：
1. 寻找翻盘机会
2. 冒险追击可能带来转机
3. 利用对手失误

关键点：抓住机会，果断行动""",
        }
        
        # 匹配场景
        scenario_lower = scenario.lower()
        for key, value in strategies.items():
            if key in scenario_lower:
                return value
        
        # 默认策略
        return f"""针对场景 "{scenario}" 的通用策略：
1. 优先确保生存
2. 观察对手行为模式
3. 根据实际情况灵活调整

建议：可以尝试使用更具体的场景描述，如"开局"、"中期"、"被围堵"等"""
    except Exception as e:
        logger.error("策略推荐失败: %s", e)
        raise ToolException(f"策略推荐失败: {e}")


# ==================== 数学计算工具 ====================

@tool
def calculator(expression: str) -> str:
    """
    执行数学计算。支持基本运算和常用数学函数。
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4" 或 "sqrt(16)"
        
    Returns:
        计算结果
    """
    try:
        import math
        
        # 安全的数学函数白名单
        safe_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }
        
        # 清理表达式
        expr = expression.strip()
        
        # 验证表达式安全性
        if not _is_safe_expression(expr):
            raise ToolException("不安全的表达式")
        
        # 执行计算
        result = eval(expr, {"__builtins__": {}}, safe_names)
        
        return f"计算结果: {expression} = {result}"
    except ToolException:
        raise
    except ZeroDivisionError:
        return "错误: 除数不能为零"
    except Exception as e:
        logger.error("计算失败: %s", e)
        raise ToolException(f"计算失败: {e}")


def _is_safe_expression(expr: str) -> bool:
    """检查表达式是否安全"""
    # 只允许数字、运算符、括号和白名单函数名
    allowed_pattern = r'^[\d\s\+\-\*\/\(\)\.\,a-zA-Z_]+$'
    if not re.match(allowed_pattern, expr):
        return False
    
    # 禁止导入和执行
    forbidden = ["import", "exec", "eval", "__", "open", "file", "input", "print"]
    expr_lower = expr.lower()
    for word in forbidden:
        if word in expr_lower:
            return False
    
    return True


# ==================== 用户信息查询工具 ====================

@tool
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def user_info(user_id: int) -> str:
    """
    查询用户基本信息和 Bot 列表。
    
    Args:
        user_id: 用户ID
        
    Returns:
        用户信息摘要
        
    重试策略：最多2次，指数退避
    """
    try:
        from db_client import get_user_by_id, get_user_bots
        logger.info("[user_info] user_id=%d", user_id)
        
        user = get_user_by_id(user_id)
        if not user:
            return f"用户 {user_id} 不存在"
        
        bots = get_user_bots(user_id)
        
        output_parts = [
            f"用户信息：",
            f"- ID: {user.get('id')}",
            f"- 用户名: {user.get('username', '未知')}",
            f"- Rating: {user.get('rating', 0)}",
            "",
            f"Bot 列表 ({len(bots)} 个)：",
        ]
        
        for i, bot in enumerate(bots, 1):
            output_parts.append(f"{i}. {bot.get('title', '未命名')} (Rating: {bot.get('rating', 0)})")
        
        if not bots:
            output_parts.append("  暂无 Bot")
        
        return "\n".join(output_parts)
    except Exception as e:
        logger.error("用户信息查询失败: %s", e)
        raise ToolException(f"用户信息查询失败: {e}")


# ==================== Bot 管理工具 (HITL) ====================

# 尝试导入 LangGraph interrupt 支持
try:
    from langgraph.types import interrupt
    INTERRUPT_AVAILABLE = True
except ImportError:
    INTERRUPT_AVAILABLE = False
    logger.warning("langgraph.types.interrupt 不可用，HITL 功能受限")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:3000")


def _format_modification_proposal(bot_name: str, bot_id: int, summary: str, new_code: str) -> str:
    """格式化修改建议（当 HITL interrupt 不可用时）"""
    return f"""## 修改建议

**Bot**: {bot_name} (ID: {bot_id})
**修改说明**: {summary}

### 建议的新代码：
```java
{new_code}
```

⚠️ HITL 功能不可用，请手动确认并应用修改。"""


@tool
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, requests.RequestException)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def get_user_bots(user_id: Optional[int] = None, username: Optional[str] = None) -> str:
    """
    获取用户的 Bot 列表。支持通过用户ID或用户名查询。
    
    Args:
        user_id: 用户ID（可选，与 username 二选一）
        username: 用户名（可选，与 user_id 二选一）
        
    Returns:
        用户的 Bot 列表，包含 ID、名称和描述
        
    注意：如果 user_id 查询失败（用户不存在），可以尝试使用 username 参数。
    """
    try:
        resolved_user_id = user_id
        resolved_username = username
        
        # 如果提供了 username，先通过数据库查找对应的 user_id
        if username and not user_id:
            try:
                from db_client import get_user_by_username
                user_info = get_user_by_username(username)
                if user_info:
                    resolved_user_id = user_info.get("id")
                    resolved_username = user_info.get("username")
                    logger.info("[get_user_bots] 通过用户名 '%s' 找到 user_id=%d", username, resolved_user_id)
                else:
                    return f"未找到用户名为 '{username}' 的用户"
            except Exception as e:
                logger.warning("[get_user_bots] 通过用户名查找失败: %s", e)
                return f"通过用户名查找失败: {e}"
        
        # 如果提供了 user_id，尝试获取用户名（可选，失败不影响后续）
        if resolved_user_id and not username:
            try:
                from db_client import get_user_by_id
                user_info = get_user_by_id(resolved_user_id)
                if user_info:
                    resolved_username = user_info.get("username")
                # 注意：即使 user_info 为 None，也继续尝试从 Java 后端获取 bot 列表
                # 因为 Java 后端有自己的用户验证
            except Exception as e:
                logger.warning("[get_user_bots] 验证用户ID失败（继续尝试Java后端）: %s", e)
        
        if not resolved_user_id:
            return "请提供 user_id 或 username 参数"
        
        logger.info("[get_user_bots] user_id=%d, username=%s", resolved_user_id, resolved_username)
        response = requests.get(
            f"{BACKEND_URL}/ai/bot/manage/list",
            params={"userId": resolved_user_id},
            timeout=10
        )
        
        if response.status_code != 200:
            return f"获取 Bot 列表失败: HTTP {response.status_code}"
        
        data = response.json()
        bots = data.get("bots", [])
        
        display_name = resolved_username or str(resolved_user_id)
        if not bots:
            return f"用户 {display_name} (ID: {resolved_user_id}) 暂无 Bot"
        
        output_parts = [f"用户 {display_name} (ID: {resolved_user_id}) 的 Bot 列表 ({len(bots)} 个)：\n"]
        for bot in bots:
            output_parts.append(f"- **{bot.get('title', '未命名')}** (ID: {bot.get('id')})")
            if bot.get('description'):
                output_parts.append(f"  描述: {bot.get('description')}")
        
        return "\n".join(output_parts)
    except Exception as e:
        logger.error("获取 Bot 列表失败: %s", e)
        raise ToolException(f"获取 Bot 列表失败: {e}")


@tool
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, requests.RequestException)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def get_bot_code(bot_id: int, user_id: int) -> str:
    """
    获取指定 Bot 的代码内容。用于查看和分析 Bot 代码。
    
    Args:
        bot_id: Bot ID
        user_id: 用户ID（用于权限验证）
        
    Returns:
        Bot 的名称、描述和代码内容
    """
    try:
        logger.info("[get_bot_code] bot_id=%d, user_id=%d", bot_id, user_id)
        response = requests.get(
            f"{BACKEND_URL}/ai/bot/manage/code",
            params={"botId": bot_id, "userId": user_id},
            timeout=10
        )
        
        if response.status_code == 403:
            return "没有权限访问此 Bot"
        if response.status_code == 404:
            return f"Bot {bot_id} 不存在"
        if response.status_code != 200:
            return f"获取 Bot 代码失败: HTTP {response.status_code}"
        
        data = response.json()
        if data.get("error"):
            return f"错误: {data.get('error')}"
        
        output_parts = [
            f"**Bot 名称**: {data.get('title', '未命名')}",
            f"**Bot ID**: {data.get('id')}",
        ]
        if data.get('description'):
            output_parts.append(f"**描述**: {data.get('description')}")
        output_parts.append(f"\n**代码内容**:\n```java\n{data.get('content', '')}\n```")
        
        return "\n".join(output_parts)
    except Exception as e:
        logger.error("获取 Bot 代码失败: %s", e)
        raise ToolException(f"获取 Bot 代码失败: {e}")


@tool
def propose_and_apply_modification(
    bot_id: int,
    user_id: int,
    modification_request: str,
    current_code: str,
    bot_name: str = "Bot"
) -> str:
    """
    分析代码并提出修改建议，等待用户确认后应用。使用 LangGraph interrupt() 实现 Human-in-the-Loop。
    
    这是一个 HITL 工具：会暂停执行等待用户确认修改建议。
    
    Args:
        bot_id: 要修改的 Bot ID
        user_id: 用户ID（用于权限验证）
        modification_request: 用户的修改需求描述
        current_code: 当前的代码内容
        bot_name: Bot 名称
        
    Returns:
        修改结果或等待确认的状态
    """
    try:
        logger.info("[propose_and_apply_modification] bot_id=%d, request='%s'", 
                    bot_id, modification_request[:50])
        
        # 1. 使用 LLM 生成修改建议（幂等操作）
        from llm_client import build_llm
        from langchain_core.messages import HumanMessage, SystemMessage
        
        llm = build_llm(streaming=False)
        if llm is None:
            return "LLM 服务不可用，无法生成修改建议"
        
        prompt = f"""你是 KOB 贪吃蛇游戏的 Bot 代码专家。

用户希望对 Bot 代码进行以下修改：
{modification_request}

当前代码：
```java
{current_code}
```

请直接输出修改后的完整代码，用 ```java 包裹。不要解释，只输出代码。"""

        response = llm.invoke([
            SystemMessage(content="你是代码修改专家，只输出修改后的代码。"),
            HumanMessage(content=prompt)
        ])
        
        new_code_raw = response.content if hasattr(response, 'content') else str(response)
        
        # 提取代码块
        import re
        code_match = re.search(r'```(?:java)?\s*([\s\S]*?)```', new_code_raw)
        new_code = code_match.group(1).strip() if code_match else new_code_raw.strip()
        
        # 2. 生成修改摘要
        diff_prompt = f"""对比以下两段代码，简要说明修改了什么（用中文，50字以内）：

原代码：
{current_code[:500]}

新代码：
{new_code[:500]}"""
        
        summary_response = llm.invoke([HumanMessage(content=diff_prompt)])
        summary = summary_response.content if hasattr(summary_response, 'content') else "代码已修改"
        
        # 3. 构建修改建议
        proposal = {
            "bot_id": bot_id,
            "bot_name": bot_name,
            "user_id": user_id,
            "summary": summary[:100],
            "original_code": current_code,
            "new_code": new_code,
            "modification_request": modification_request,
        }
        
        # 4. 返回结构化的 proposal 数据，让 hitl_tool_node 调用 interrupt()
        # 工具内部不调用 interrupt()，因为 tool.invoke() 没有 LangGraph 上下文
        import json
        proposal_response = {
            "__hitl_proposal__": True,  # 标记这是一个需要 HITL 确认的响应
            "type": "bot_modification_proposal",
            "proposal": proposal,
            "message": f"请确认是否应用对 {bot_name} 的修改",
            "options": ["approve", "reject", "edit"]
        }
        logger.info("[HITL] 返回 proposal 数据，等待 hitl_tool_node 调用 interrupt()")
        return json.dumps(proposal_response, ensure_ascii=False)

    except Exception as e:
        # Re-raise GraphInterrupt so it propagates to graph runner
        try:
            from langgraph.errors import GraphInterrupt
        except ImportError:
            try:
                from langgraph.types import GraphInterrupt
            except ImportError:
                GraphInterrupt = None
        if GraphInterrupt and isinstance(e, GraphInterrupt):
            raise
        logger.error("代码修改失败: %s", e)
        raise ToolException(f"代码修改失败: {e}")


# ==================== 工具注册 ====================

def get_tools() -> List:
    """获取所有可用工具"""
    return [
        knowledge_search,
        code_analysis,
        battle_query,
        battle_analysis,
        loss_reason_analyzer,
        strategy_recommend,
        calculator,
        user_info,
        # HITL Bot 管理工具
        get_user_bots,
        get_bot_code,
        propose_and_apply_modification,
    ]


def get_tool_descriptions() -> List[Dict[str, str]]:
    """获取工具描述列表"""
    tools = get_tools()
    return [
        {
            "name": tool.name,
            "description": tool.description,
        }
        for tool in tools
    ]
