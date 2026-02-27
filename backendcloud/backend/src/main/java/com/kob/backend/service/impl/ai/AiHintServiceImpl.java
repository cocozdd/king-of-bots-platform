package com.kob.backend.service.impl.ai;

import com.kob.backend.controller.ai.dto.AiHintRequest;
import com.kob.backend.controller.ai.dto.AiHintResponse;
import com.kob.backend.controller.ai.dto.AiDoc;
import com.kob.backend.repository.AiCorpusRepository;
import com.kob.backend.service.ai.AiHintService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class AiHintServiceImpl implements AiHintService {

    @Autowired
    private AiCorpusRepository aiCorpusRepository;

    @Autowired
    private AiMetricsService metricsService;
    
    @Autowired
    private EmbeddingCacheService cacheService;

    @Value("${dashscope.api.key:}")
    private String dashscopeApiKey;

    private DashscopeEmbeddingClient dashscopeClient;
    private DeepseekClient deepseekClient;

    @PostConstruct
    public void initClients() {
        String key = dashscopeApiKey;
        if (key == null || key.isEmpty()) {
            key = System.getenv("DASHSCOPE_API_KEY");
        }
        dashscopeClient = new DashscopeEmbeddingClient(key, metricsService, cacheService);
        deepseekClient = new DeepseekClient(metricsService);
    }

    private String buildQuery(AiHintRequest req) {
        StringBuilder sb = new StringBuilder();
        if (req.getQuestion() != null) {
            sb.append(req.getQuestion());
        }
        if (req.getErrorLog() != null && !req.getErrorLog().isEmpty()) {
            sb.append("\n错误日志: ").append(trim(req.getErrorLog(), 500));
        }
        if (req.getCodeSnippet() != null && !req.getCodeSnippet().isEmpty()) {
            sb.append("\n代码片段: ").append(trim(req.getCodeSnippet(), 500));
        }
        return trim(sb.toString(), 800);
    }

    private String trim(String text, int maxLen) {
        if (text == null) return "";
        return text.length() > maxLen ? text.substring(0, maxLen) + "..." : text;
    }

    @Override
    public AiHintResponse hint(AiHintRequest request) {
        AiHintResponse resp = new AiHintResponse();
        // 需要 DashScope 和 DeepSeek 均可用
        if (dashscopeClient == null || !dashscopeClient.enabled() || !deepseekClient.enabled()) {
            List<AiHintResponse.AiSource> sources = aiCorpusRepository.listSafe(5);
            resp.setSources(sources);
            resp.setAnswer("占位提示：AI 服务未配置完整，请设置 DashScope 与 DeepSeek 的 API Key。");
            resp.setErrorMessage("DASHSCOPE_API_KEY or DEEPSEEK_API_KEY not set");
            return resp;
        }

        try {
            String query = buildQuery(request);
            double[] emb = dashscopeClient.embed(query);
            List<AiDoc> docs = aiCorpusRepository.searchSafe(emb, 5);
            List<String> ctx = docs.stream().map(AiDoc::getContent).collect(Collectors.toList());

            String system = """
                你是 KOB（King of Bots）贪吃蛇对战游戏的 Bot 开发助手。
                
                ## 项目背景
                - 这是一个双人贪吃蛇对战游戏，两条蛇在 13x14 的网格地图上对战
                - 用户编写 Java Bot 代码，系统会编译并执行用户的代码
                - **每回合 Bot 必须在 2 秒内返回移动方向（0-上, 1-右, 2-下, 3-左）**
                - 超时会导致 Bot 返回默认方向 0，可能导致撞墙失败
                
                ## Bot 代码结构
                用户的 Bot 代码需要实现 `Supplier<Integer>` 接口：
                ```java
                public class Bot implements java.util.function.Supplier<Integer> {
                    @Override
                    public Integer get() {
                        // 从 input.txt 读取游戏状态
                        // 返回方向 0-3
                    }
                }
                ```
                
                ## 常见问题及解决方案
                
                ### Bot 超时问题
                超时的真正原因通常是：
                1. **寻路算法效率低** - 使用 A* 或优化的 BFS 替代朴素 DFS
                2. **重复解析地图** - 预处理障碍物，避免每次重新解析 input
                3. **搜索深度过大** - 限制搜索深度，保证 2 秒内返回
                4. **未使用剪枝** - 提前排除死路
                
                ### 性能优化建议
                - 使用 BFS 而非 DFS（BFS 能找最短路径且不会栈溢出）
                - 位运算优化地图状态存储
                - 缓存计算结果，避免重复计算
                - 使用 `ArrayDeque` 而非 `LinkedList` 作为队列
                
                ## 回答要求
                - 结合项目实际和参考文档给出具体可操作的建议
                - 使用 Markdown 格式，代码块用 ```java 包裹
                - 回答简洁准确，不知道就说不知道
                """;
            String answer = deepseekClient.chat(system, request.getQuestion(), ctx);

            List<AiHintResponse.AiSource> sources = docs.stream()
                    .map(d -> new AiHintResponse.AiSource(d.getId(), d.getTitle(), d.getCategory()))
                    .collect(Collectors.toList());
            resp.setSources(sources);
            resp.setAnswer(answer);
            resp.setErrorMessage(null);
        } catch (Exception e) {
            List<AiHintResponse.AiSource> sources = aiCorpusRepository.listSafe(5);
            resp.setSources(sources);
            resp.setAnswer("占位提示：AI 服务暂时不可用，请稍后重试。");
            resp.setErrorMessage(e.getMessage());
        }
        return resp;
    }
}
