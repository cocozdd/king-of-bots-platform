"""
公平代码优化工具 - 2026 最佳实践

设计原则（公平性 - 2026 更新）：
- **可以给代码**，但要求用户先提供自己的思路
- 如果用户只说"给我一个厉害的AI"，先引导思考
- 用户提供思路后，可以帮助完善和实现
- 教育为主，鼓励独立思考

功能：
1. 代码复杂度分析
2. 算法模式识别
3. Bug 检测与提示
4. 性能优化建议
5. 代码风格检查

面试要点：
- 公平性设计：AI 辅助 vs 作弊的边界
- 静态分析技术：AST、复杂度计算
- 教育心理学：引导式学习

支持模型：
- DeepSeek V3（默认）
- GLM-4（智谱 AI）
- OpenAI GPT-4o
"""
import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """优化建议的详细程度"""
    HINT = "hint"          # 仅提示方向
    GUIDANCE = "guidance"  # 提供思路和伪代码
    EDUCATIONAL = "educational"  # 教育性解释（默认）


@dataclass
class CodeAnalysisResult:
    """代码分析结果"""
    success: bool
    complexity_score: int  # 1-10，越高越复杂
    issues: List[Dict[str, str]] = field(default_factory=list)
    patterns_detected: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    educational_notes: List[str] = field(default_factory=list)
    error: Optional[str] = None


def analyze_code_complexity(code: str) -> Dict[str, Any]:
    """
    静态分析代码复杂度（不需要 LLM）
    
    分析指标：
    - 行数
    - 嵌套深度
    - 条件分支数
    - 循环数
    """
    lines = code.split("\n")
    non_empty_lines = [l for l in lines if l.strip() and not l.strip().startswith("//")]
    
    # 计算嵌套深度
    max_depth = 0
    current_depth = 0
    for line in lines:
        current_depth += line.count("{") - line.count("}")
        max_depth = max(max_depth, current_depth)
    
    # 计算条件分支
    conditions = len(re.findall(r'\b(if|else|switch|case)\b', code))
    
    # 计算循环
    loops = len(re.findall(r'\b(for|while|do)\b', code))
    
    # 综合评分 (1-10)
    score = min(10, max(1, 
        1 + 
        (len(non_empty_lines) // 20) +
        (max_depth // 2) +
        (conditions // 3) +
        (loops // 2)
    ))
    
    return {
        "lines": len(non_empty_lines),
        "max_nesting_depth": max_depth,
        "conditions": conditions,
        "loops": loops,
        "complexity_score": score,
    }


def detect_algorithm_patterns(code: str) -> List[str]:
    """
    检测代码中的算法模式
    
    识别常见模式：
    - BFS/DFS
    - 动态规划
    - 贪心
    - 距离计算
    """
    patterns = []
    code_lower = code.lower()
    
    # BFS 模式
    if any(kw in code_lower for kw in ["queue", "bfs", "linkedlist", "poll()", "offer("]):
        patterns.append("BFS (广度优先搜索)")
    
    # DFS 模式
    if any(kw in code_lower for kw in ["stack", "dfs", "recursive", "backtrack"]):
        patterns.append("DFS (深度优先搜索)")
    
    # A* 或启发式搜索
    if any(kw in code_lower for kw in ["heuristic", "priority", "astar", "a*", "priorityqueue"]):
        patterns.append("A* 或启发式搜索")
    
    # 动态规划
    if any(kw in code_lower for kw in ["dp[", "memo", "cache", "memorize"]):
        patterns.append("动态规划/记忆化")
    
    # 距离计算
    if any(kw in code_lower for kw in ["math.abs", "distance", "manhattan"]):
        patterns.append("距离计算")
    
    # 方向数组
    if re.search(r'(dx|dy|dir|direction)\s*=\s*\[', code_lower) or \
       re.search(r'int\[\]\s*(dx|dy|dir)', code_lower):
        patterns.append("方向数组")
    
    # 随机策略
    if any(kw in code_lower for kw in ["random", "math.random"]):
        patterns.append("随机策略")
    
    return patterns


def detect_common_bugs(code: str) -> List[Dict[str, str]]:
    """
    检测常见 Bug 模式
    
    返回 Bug 提示（不直接给解决方案）
    """
    bugs = []
    
    # 边界检查缺失
    if "getMapCell" in code or "map[" in code:
        if not re.search(r'(>|<|>=|<=)\s*(0|rows|cols|13|14)', code):
            bugs.append({
                "type": "boundary",
                "hint": "可能缺少边界检查，蛇可能会撞墙",
                "learning": "在访问地图元素前，确保坐标在有效范围内",
            })
    
    # 自身碰撞检测
    if not re.search(r'(snake|body|self)', code.lower()):
        bugs.append({
            "type": "self_collision",
            "hint": "可能没有检测自身碰撞",
            "learning": "蛇不能撞到自己或对手的身体",
        })
    
    # 对手检测
    if not re.search(r'(opponent|enemy|other)', code.lower()):
        bugs.append({
            "type": "opponent",
            "hint": "可能没有考虑对手的位置",
            "learning": "对战游戏需要考虑对手的位置和可能行动",
        })
    
    # 硬编码方向
    if re.search(r'return\s+[0-3]\s*;', code):
        bugs.append({
            "type": "hardcoded",
            "hint": "检测到硬编码的返回值，可能在某些情况下不灵活",
            "learning": "策略应该根据当前局面动态决策",
        })
    
    # 无限循环风险
    if "while(true)" in code.replace(" ", "") or "for(;;)" in code.replace(" ", ""):
        if "break" not in code and "return" not in code:
            bugs.append({
                "type": "infinite_loop",
                "hint": "可能存在无限循环风险",
                "learning": "确保循环有明确的退出条件",
            })
    
    return bugs


def generate_fair_suggestions(
    code: str,
    complexity: Dict[str, Any],
    patterns: List[str],
    bugs: List[Dict[str, str]],
    level: OptimizationLevel = OptimizationLevel.EDUCATIONAL,
) -> List[str]:
    """
    生成公平的优化建议
    
    原则：引导思考，不给现成答案
    """
    suggestions = []
    
    # 复杂度建议
    if complexity["complexity_score"] >= 7:
        suggestions.append("代码复杂度较高，考虑拆分函数提高可读性")
    
    if complexity["max_nesting_depth"] >= 4:
        suggestions.append("嵌套层次较深，可以考虑提前返回或使用辅助函数")
    
    # 算法建议（教育性）
    if not patterns:
        suggestions.append("💡 学习建议：了解 BFS/DFS 等经典算法，它们在路径搜索中很有用")
    elif "随机策略" in patterns and len(patterns) == 1:
        suggestions.append("💡 随机策略简单但不够智能，考虑结合其他算法")
    
    # Bug 相关建议
    for bug in bugs:
        suggestions.append(f"⚠️ {bug['hint']}")
    
    # 通用建议
    if "BFS (广度优先搜索)" not in patterns and complexity["complexity_score"] <= 3:
        suggestions.append("💡 可以学习 BFS 算法来寻找安全路径")
    
    return suggestions


def generate_educational_notes(patterns: List[str]) -> List[str]:
    """
    生成教育性注释
    
    帮助用户理解算法原理，而非直接给代码
    """
    notes = []
    
    if "BFS (广度优先搜索)" in patterns:
        notes.append("""
📚 BFS 学习要点：
- 使用队列（Queue）按层遍历
- 适合找最短路径
- 时间复杂度 O(V+E)
- 思考：如何用 BFS 找到离墙壁最远的安全位置？
""".strip())
    
    if "DFS (深度优先搜索)" in patterns:
        notes.append("""
📚 DFS 学习要点：
- 使用栈或递归深入探索
- 适合探索所有可能路径
- 注意避免重复访问
- 思考：DFS 如何帮助判断是否被封死？
""".strip())
    
    if "A* 或启发式搜索" in patterns:
        notes.append("""
📚 A* 学习要点：
- 结合实际代价 g(n) 和启发式估计 h(n)
- f(n) = g(n) + h(n)
- 需要设计好的启发函数
- 思考：什么是好的启发函数？曼哈顿距离够吗？
""".strip())
    
    if not patterns:
        notes.append("""
📚 算法学习建议：
1. 先理解 BFS：用于寻找最短路径
2. 再学习评估函数：判断位置的安全性
3. 最后考虑对手预测：博弈论基础

推荐资源：
- 《算法导论》图论章节
- LeetCode 题目：200, 994, 542（BFS 相关）
""".strip())
    
    return notes


@tool
def analyze_bot_code_fair(code: str) -> str:
    """
    公平地分析 Bot 代码，提供学习指导而非直接答案。
    
    Args:
        code: 要分析的 Java Bot 代码
    
    Returns:
        分析报告，包含复杂度、模式、问题和学习建议
    """
    try:
        # 1. 复杂度分析
        complexity = analyze_code_complexity(code)
        
        # 2. 模式识别
        patterns = detect_algorithm_patterns(code)
        
        # 3. Bug 检测
        bugs = detect_common_bugs(code)
        
        # 4. 生成建议
        suggestions = generate_fair_suggestions(
            code, complexity, patterns, bugs
        )
        
        # 5. 教育笔记
        notes = generate_educational_notes(patterns)
        
        # 构建报告
        report = []
        report.append("## 代码分析报告\n")
        
        report.append(f"**复杂度评分**: {complexity['complexity_score']}/10")
        report.append(f"- 有效代码行: {complexity['lines']}")
        report.append(f"- 最大嵌套深度: {complexity['max_nesting_depth']}")
        report.append(f"- 条件分支: {complexity['conditions']}")
        report.append(f"- 循环数: {complexity['loops']}\n")
        
        if patterns:
            report.append("**检测到的算法模式**:")
            for p in patterns:
                report.append(f"- {p}")
            report.append("")
        
        if bugs:
            report.append("**潜在问题**:")
            for bug in bugs:
                report.append(f"- {bug['hint']}")
                report.append(f"  - 学习提示: {bug['learning']}")
            report.append("")
        
        if suggestions:
            report.append("**优化建议**:")
            for s in suggestions:
                report.append(f"- {s}")
            report.append("")
        
        if notes:
            report.append("**学习资料**:")
            for note in notes:
                report.append(note)
        
        return "\n".join(report)
        
    except Exception as e:
        logger.error("代码分析失败: %s", e)
        return f"分析失败: {e}"


@tool
def suggest_algorithm(problem_description: str) -> str:
    """
    根据问题描述建议适合的算法思路（不给具体实现）。
    
    Args:
        problem_description: 问题描述，如"如何避免被困死"
    
    Returns:
        算法建议和学习指导
    """
    desc_lower = problem_description.lower()
    
    suggestions = []
    suggestions.append("## 算法建议\n")
    
    # 路径寻找
    if any(kw in desc_lower for kw in ["路径", "最短", "到达", "path", "shortest"]):
        suggestions.append("""
**推荐算法**: BFS (广度优先搜索)

**思路**:
1. 从起点开始，逐层扩展
2. 记录已访问的位置
3. 第一次到达目标时的步数就是最短距离

**学习方向**:
- 理解队列的 FIFO 特性
- 学习如何记录路径（前驱数组）
- 思考：多目标 BFS 怎么做？
""")
    
    # 安全评估
    if any(kw in desc_lower for kw in ["安全", "困死", "空间", "safe", "trapped"]):
        suggestions.append("""
**推荐算法**: 连通区域计算 (Flood Fill / BFS)

**思路**:
1. 计算当前位置可达的空间大小
2. 比较各方向的可达空间
3. 选择空间更大的方向

**学习方向**:
- 如何用 BFS 计算连通区域面积
- 何时需要考虑对手的影响
- 思考：蛇的增长如何影响空间计算？
""")
    
    # 对手预测
    if any(kw in desc_lower for kw in ["对手", "预测", "博弈", "opponent", "predict"]):
        suggestions.append("""
**推荐算法**: Minimax 或简单博弈

**思路**:
1. 假设对手也会选择最优策略
2. 模拟几步后的局面
3. 选择对自己最有利的走法

**学习方向**:
- 博弈树的基本概念
- Alpha-Beta 剪枝优化
- 思考：搜索深度和计算时间的平衡
""")
    
    # 默认建议
    if len(suggestions) == 1:
        suggestions.append("""
**通用学习建议**:

1. **基础**: 先实现能安全移动的 Bot
   - 检测四个方向是否安全
   - 避免撞墙和蛇身

2. **进阶**: 添加空间评估
   - 用 BFS 计算可达空间
   - 选择空间更大的方向

3. **高级**: 考虑对手
   - 预测对手可能的移动
   - 避免进入可能被封死的区域

**推荐学习路径**:
随机 → 安全检测 → 空间评估 → 路径规划 → 博弈预测
""")
    
    suggestions.append("""
---
⚠️ 公平提示：以上是学习指导，具体实现需要自己完成。
这是为了帮助你真正理解和提升编程能力。
""")
    
    return "\n".join(suggestions)


@tool  
def review_code_style(code: str) -> str:
    """
    检查代码风格和最佳实践。
    
    Args:
        code: Java 代码
    
    Returns:
        风格检查报告
    """
    issues = []
    
    # 命名规范
    if re.search(r'\bint\s+[a-z]\b', code):  # 单字母变量
        issues.append("- 发现单字母变量名，建议使用有意义的名称（如 `row`, `col` 而非 `r`, `c`）")
    
    # 魔法数字
    magic_numbers = re.findall(r'(?<![a-zA-Z0-9_])[0-9]{2,}(?![a-zA-Z0-9_])', code)
    if magic_numbers:
        issues.append(f"- 发现魔法数字 {magic_numbers[:3]}，建议定义为常量")
    
    # 注释
    if code.count("//") < 3 and len(code.split("\n")) > 20:
        issues.append("- 代码较长但注释较少，建议添加关键逻辑的注释")
    
    # 函数长度
    methods = re.findall(r'(public|private|protected)\s+\w+\s+\w+\s*\([^)]*\)\s*\{', code)
    if len(methods) == 1 and len(code.split("\n")) > 50:
        issues.append("- 只有一个大函数，建议拆分为多个小函数")
    
    # 异常处理
    if "try" not in code and ("parseInt" in code or "get(" in code):
        issues.append("- 可能需要添加异常处理，防止运行时错误")
    
    report = ["## 代码风格检查\n"]
    
    if issues:
        report.append("**发现以下问题**:\n")
        report.extend(issues)
    else:
        report.append("✅ 代码风格良好！")
    
    report.append("\n**代码风格指南**:")
    report.append("- 变量名要有意义")
    report.append("- 常量使用 `static final`")
    report.append("- 函数职责单一")
    report.append("- 适当添加注释")
    
    return "\n".join(report)


class FairCodeOptimizer:
    """
    公平代码优化器 - 2026 更新
    
    设计理念：
    - **可以给代码**，但要求用户先提供思路
    - 如果用户只说"给我厉害的AI"，先引导思考
    - 用户提供思路后，帮助完善和实现
    - 教育为主，鼓励独立思考
    """
    
    def __init__(self, llm=None):
        self.llm = llm
    
    def _get_llm(self):
        if self.llm:
            return self.llm
        try:
            from llm_client import get_llm
            return get_llm()
        except:
            return None
    
    def _check_user_has_idea(self, user_input: str) -> bool:
        """
        检测用户是否提供了自己的思路
        
        有思路的表现：
        - 提到具体算法（BFS、DFS、A*）
        - 描述了具体策略
        - 给出了代码片段
        - 解释了自己的想法
        """
        idea_indicators = [
            # 算法关键词
            "bfs", "dfs", "广度", "深度", "a*", "dijkstra", "动态规划", "dp",
            "递归", "迭代", "队列", "栈", "优先队列",
            # 策略描述
            "我想", "我的思路", "我打算", "我觉得", "我认为",
            "先判断", "然后", "如果...就", "计算", "遍历",
            # 代码相关
            "int[]", "for", "while", "if", "queue", "return",
            # 具体问题
            "怎么实现", "如何优化", "这样写对吗", "帮我改进",
        ]
        
        lower_input = user_input.lower()
        return any(kw in lower_input for kw in idea_indicators)
    
    def _is_lazy_request(self, user_input: str) -> bool:
        """
        检测是否是"伸手党"请求
        """
        lazy_patterns = [
            "给我一个厉害", "给我一个强", "给我最强", "直接给我",
            "帮我写一个", "写一个能赢", "给我代码", "要必胜",
            "给我bot", "给我ai", "最强bot", "最强ai",
        ]
        lower_input = user_input.lower()
        return any(p in lower_input for p in lazy_patterns)
    
    async def analyze(self, code: str) -> CodeAnalysisResult:
        """
        全面分析代码
        """
        try:
            complexity = analyze_code_complexity(code)
            patterns = detect_algorithm_patterns(code)
            bugs = detect_common_bugs(code)
            suggestions = generate_fair_suggestions(
                code, complexity, patterns, bugs
            )
            notes = generate_educational_notes(patterns)
            
            return CodeAnalysisResult(
                success=True,
                complexity_score=complexity["complexity_score"],
                issues=[{"type": b["type"], "hint": b["hint"]} for b in bugs],
                patterns_detected=patterns,
                suggestions=suggestions,
                educational_notes=notes,
            )
        except Exception as e:
            return CodeAnalysisResult(
                success=False,
                complexity_score=0,
                error=str(e),
            )
    
    async def get_improvement_guidance(
        self,
        code: str,
        focus_area: str = "general",
    ) -> str:
        """
        获取改进指导（使用 LLM）
        
        Args:
            code: 代码
            focus_area: 关注领域（general, safety, strategy, performance）
        
        Returns:
            改进指导
        """
        llm = self._get_llm()
        if llm is None:
            return "LLM 服务不可用，请使用静态分析功能"
        
        # 公平性提示词 - 2026 更新
        system_prompt = """你是一个 KOB Bot 代码教练。

你的职责是帮助用户学习和提升，但要保持公平性。

规则：
1. 如果用户提供了自己的思路，可以帮助完善和给出代码
2. 如果用户只是"伸手党"，先引导他们思考
3. 解释代码背后的原理，帮助真正理解
4. 鼓励独立思考，而非直接复制

公平原则：
- 用户有思路 → 可以给代码帮助实现
- 用户无思路 → 先引导思考，问几个问题
- 禁止直接给"必胜"策略

用中文回答。"""
        
        focus_prompts = {
            "safety": "重点关注代码的安全性，如边界检查、碰撞检测等",
            "strategy": "重点关注策略的有效性，但只解释原理",
            "performance": "重点关注算法效率，解释时间/空间复杂度",
            "general": "全面分析代码，给出学习建议",
        }
        
        user_prompt = f"""请分析以下 KOB Bot 代码：

```java
{code[:2000]}  
```

{focus_prompts.get(focus_area, focus_prompts["general"])}

请用引导式问题帮助用户理解改进方向。"""
        
        try:
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            return response.content
        except Exception as e:
            logger.error("LLM 调用失败: %s", e)
            return f"分析失败: {e}"


# 全局实例
_optimizer: Optional[FairCodeOptimizer] = None


def get_code_optimizer() -> FairCodeOptimizer:
    """获取全局代码优化器"""
    global _optimizer
    if _optimizer is None:
        _optimizer = FairCodeOptimizer()
    return _optimizer


async def analyze_code_fair(code: str) -> Dict[str, Any]:
    """
    便捷函数：公平地分析代码
    """
    optimizer = get_code_optimizer()
    result = await optimizer.analyze(code)
    
    return {
        "success": result.success,
        "complexity_score": result.complexity_score,
        "issues": result.issues,
        "patterns": result.patterns_detected,
        "suggestions": result.suggestions,
        "educational_notes": result.educational_notes,
        "error": result.error,
    }


async def help_with_code(
    user_request: str,
    user_code: Optional[str] = None,
    user_idea: Optional[str] = None,
) -> Dict[str, Any]:
    """
    公平地帮助用户写代码
    
    规则：
    - 用户提供思路 → 可以给代码
    - 用户无思路 → 先引导思考
    """
    optimizer = get_code_optimizer()
    
    # 合并用户输入
    full_input = f"{user_request} {user_idea or ''}"
    
    # 检查是否有思路
    has_idea = optimizer._check_user_has_idea(full_input)
    is_lazy = optimizer._is_lazy_request(user_request)
    
    if is_lazy and not has_idea:
        # 伸手党，没有思路 → 引导思考
        return {
            "success": True,
            "mode": "guidance",
            "canGiveCode": False,
            "message": """我很乐意帮助你！但为了公平和真正帮助你学习，我需要先了解你的想法：

1. **你想实现什么策略？**
   - 安全优先？追着敌人？还是占领空间？

2. **你了解哪些算法？**
   - BFS（广度优先）可以找最短路径
   - 连通区域计算可以评估空间大小

3. **你目前的思路是什么？**
   - 哪怕只是大概想法也可以

请告诉我你的思路，我会帮你实现和完善！

💡 提示：说说你想用什么算法，或者描述一下你的策略思路。""",
            "suggestedQuestions": [
                "我想用 BFS 找到离对手最远的安全位置",
                "我的思路是先计算四个方向的可用空间",
                "我想实现一个能预测对手移动的策略",
            ],
        }
    
    elif has_idea:
        # 用户有思路 → 可以帮助实现
        llm = optimizer._get_llm()
        if llm is None:
            return {
                "success": False,
                "mode": "error",
                "error": "LLM 服务不可用",
            }
        
        system_prompt = """你是 KOB Bot 代码教练。用户已经提供了自己的思路，现在帮助他们实现。

规则：
1. 根据用户的思路给出代码实现
2. 解释代码的关键部分
3. 指出可能的改进方向
4. 代码要完整可运行

输出格式：
1. 简要确认用户的思路
2. 给出完整的 Java 代码
3. 解释关键逻辑
4. 建议下一步优化方向"""
        
        user_prompt = f"""用户的请求：{user_request}

用户的思路：{user_idea or '见上文'}

用户现有代码：
```java
{user_code or '无'}
```

请根据用户的思路帮助实现代码。"""
        
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            return {
                "success": True,
                "mode": "implementation",
                "canGiveCode": True,
                "response": response.content,
            }
        except Exception as e:
            logger.error("代码生成失败: %s", e)
            return {
                "success": False,
                "mode": "error",
                "error": str(e),
            }
    
    else:
        # 一般请求，给学习指导
        return {
            "success": True,
            "mode": "learning",
            "canGiveCode": False,
            "message": "请描述你的思路或想实现的策略，我会帮助你完善和实现。",
        }
