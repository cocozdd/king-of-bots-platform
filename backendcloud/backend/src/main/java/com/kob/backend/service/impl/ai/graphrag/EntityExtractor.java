package com.kob.backend.service.impl.ai.graphrag;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.kob.backend.service.impl.ai.DeepseekClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 实体关系提取器
 * 
 * 使用 LLM 从文本中提取实体和关系
 * 
 * 提取策略：
 * 1. 基于规则的快速提取（针对 Bot 领域）
 * 2. LLM 增强提取（更准确但成本高）
 */
public class EntityExtractor {
    
    private static final Logger log = LoggerFactory.getLogger(EntityExtractor.class);
    
    // 预定义的实体类型和关键词
    private static final Map<String, Set<String>> ENTITY_KEYWORDS = new LinkedHashMap<>();
    
    static {
        // 算法相关
        ENTITY_KEYWORDS.put("ALGORITHM", Set.of(
            "bfs", "dfs", "a*", "astar", "dijkstra", "贪心", "动态规划", "dp",
            "深度优先", "广度优先", "搜索算法", "寻路算法", "路径搜索"
        ));
        
        // 策略相关
        ENTITY_KEYWORDS.put("STRATEGY", Set.of(
            "策略", "战术", "进攻", "防守", "包围", "逃跑", "追击",
            "生存策略", "攻击策略", "移动策略", "博弈", "对战"
        ));
        
        // 游戏元素
        ENTITY_KEYWORDS.put("ELEMENT", Set.of(
            "蛇", "snake", "地图", "map", "障碍物", "墙", "边界",
            "身体", "头", "尾", "格子", "坐标", "方向"
        ));
        
        // 动作
        ENTITY_KEYWORDS.put("ACTION", Set.of(
            "移动", "move", "转向", "前进", "后退", "上", "下", "左", "右",
            "碰撞", "吃", "生长", "死亡"
        ));
        
        // 代码概念
        ENTITY_KEYWORDS.put("CODE", Set.of(
            "函数", "方法", "类", "接口", "变量", "数组", "队列",
            "nextStep", "getInput", "java", "代码", "实现"
        ));
        
        // 优化技术
        ENTITY_KEYWORDS.put("OPTIMIZATION", Set.of(
            "优化", "剪枝", "缓存", "预计算", "启发式", "评估函数",
            "性能", "效率"
        ));
    }
    
    // 关系模式
    private static final List<RelationPattern> RELATION_PATTERNS = List.of(
        new RelationPattern("(.+?)使用(.+)", "USES"),
        new RelationPattern("(.+?)可以(.+)", "CAN"),
        new RelationPattern("(.+?)比(.+?)更", "BETTER_THAN"),
        new RelationPattern("(.+?)是(.+?)的一部分", "PART_OF"),
        new RelationPattern("(.+?)与(.+?)相关", "RELATED_TO"),
        new RelationPattern("(.+?)实现(.+)", "IMPLEMENTS"),
        new RelationPattern("(.+?)依赖(.+)", "DEPENDS_ON"),
        new RelationPattern("(.+?)导致(.+)", "CAUSES"),
        new RelationPattern("(.+?)防止(.+)", "PREVENTS"),
        new RelationPattern("(.+?)通过(.+)", "USES"),
        new RelationPattern("(.+?)需要(.+)", "DEPENDS_ON"),
        new RelationPattern("(.+?)包含(.+)", "CONTAINS")
    );
    
    private final DeepseekClient llmClient;
    
    public EntityExtractor(DeepseekClient llmClient) {
        this.llmClient = llmClient;
    }
    
    /**
     * 从文本提取实体和关系（规则+LLM混合）
     */
    public ExtractionResult extract(String docId, String title, String content) {
        ExtractionResult result = new ExtractionResult(docId);
        
        // 1. 规则提取（快速）
        extractByRules(content, result);
        
        // 2. 从标题提取主题实体
        extractFromTitle(title, result);
        
        // 3. LLM 增强提取（可选）
        if (llmClient != null && llmClient.enabled() && result.getEntities().size() < 3) {
            extractByLLM(content, result);
        }
        
        // 4. 自动推断关系
        inferRelations(result);
        
        log.info("从文档 {} 提取: {} 实体, {} 关系", 
                docId, result.getEntities().size(), result.getRelations().size());
        
        return result;
    }
    
    /**
     * 规则提取
     */
    private void extractByRules(String content, ExtractionResult result) {
        String contentLower = content.toLowerCase();
        
        for (Map.Entry<String, Set<String>> entry : ENTITY_KEYWORDS.entrySet()) {
            String type = entry.getKey();
            for (String keyword : entry.getValue()) {
                if (contentLower.contains(keyword.toLowerCase())) {
                    // 提取包含关键词的上下文作为描述
                    String description = extractContext(content, keyword, 100);
                    result.addEntity(new ExtractedEntity(keyword, type, description));
                }
            }
        }
        
        // 提取关系
        for (RelationPattern pattern : RELATION_PATTERNS) {
            Matcher matcher = pattern.pattern.matcher(content);
            while (matcher.find()) {
                if (matcher.groupCount() >= 2) {
                    String source = matcher.group(1).trim();
                    String target = matcher.group(2).trim();
                    if (source.length() < 20 && target.length() < 20) {
                        result.addRelation(new ExtractedRelation(
                            source, target, pattern.relationType, matcher.group(0), 0.7
                        ));
                    }
                }
            }
        }
    }
    
    /**
     * 从标题提取实体
     */
    private void extractFromTitle(String title, ExtractionResult result) {
        if (title == null || title.isEmpty()) return;
        
        // 标题通常包含主要概念
        String[] parts = title.split("[：:,，、\\s]+");
        for (String part : parts) {
            part = part.trim();
            if (part.length() >= 2 && part.length() <= 20) {
                String type = inferEntityType(part);
                result.addEntity(new ExtractedEntity(part, type, "来自标题: " + title));
            }
        }
    }
    
    /**
     * LLM 增强提取
     */
    private void extractByLLM(String content, ExtractionResult result) {
        if (llmClient == null || !llmClient.enabled()) return;
        
        try {
            String prompt = """
                从以下文本中提取实体和关系，返回JSON格式：
                {
                    "entities": [{"name": "实体名", "type": "类型", "description": "描述"}],
                    "relations": [{"source": "源实体", "target": "目标实体", "type": "关系类型"}]
                }
                
                实体类型: ALGORITHM, STRATEGY, ELEMENT, ACTION, CODE, OPTIMIZATION, CONCEPT
                关系类型: USES, CAN, BETTER_THAN, PART_OF, RELATED_TO, IMPLEMENTS, DEPENDS_ON
                
                文本：
                """ + content.substring(0, Math.min(1500, content.length()));
            
            String response = llmClient.chat(
                "你是知识图谱构建专家，只返回JSON，不要其他解释。",
                prompt,
                List.of()
            );
            
            // 解析 JSON
            parseExtractedJson(response, result);
            
        } catch (Exception e) {
            log.warn("LLM 提取失败: {}", e.getMessage());
        }
    }
    
    /**
     * 自动推断关系
     */
    private void inferRelations(ExtractionResult result) {
        List<ExtractedEntity> entities = result.getEntities();
        
        // 算法与策略的关系
        for (ExtractedEntity algo : entities) {
            if ("ALGORITHM".equals(algo.type)) {
                for (ExtractedEntity strategy : entities) {
                    if ("STRATEGY".equals(strategy.type)) {
                        result.addRelation(new ExtractedRelation(
                            algo.name, strategy.name, "USES", 
                            algo.name + " 用于实现 " + strategy.name, 0.5
                        ));
                    }
                }
            }
        }
        
        // 动作与元素的关系
        for (ExtractedEntity action : entities) {
            if ("ACTION".equals(action.type)) {
                for (ExtractedEntity element : entities) {
                    if ("ELEMENT".equals(element.type)) {
                        result.addRelation(new ExtractedRelation(
                            element.name, action.name, "CAN",
                            element.name + " 可以 " + action.name, 0.5
                        ));
                    }
                }
            }
        }
    }
    
    private String extractContext(String content, String keyword, int contextLength) {
        int idx = content.toLowerCase().indexOf(keyword.toLowerCase());
        if (idx == -1) return "";
        
        int start = Math.max(0, idx - contextLength / 2);
        int end = Math.min(content.length(), idx + keyword.length() + contextLength / 2);
        return content.substring(start, end).trim();
    }
    
    private String inferEntityType(String name) {
        String nameLower = name.toLowerCase();
        for (Map.Entry<String, Set<String>> entry : ENTITY_KEYWORDS.entrySet()) {
            for (String keyword : entry.getValue()) {
                if (nameLower.contains(keyword.toLowerCase())) {
                    return entry.getKey();
                }
            }
        }
        return "CONCEPT";
    }
    
    private void parseExtractedJson(String response, ExtractionResult result) {
        try {
            // 提取 JSON 部分
            int start = response.indexOf("{");
            int end = response.lastIndexOf("}") + 1;
            if (start >= 0 && end > start) {
                String json = response.substring(start, end);
                JSONObject obj = JSON.parseObject(json);
                
                JSONArray entities = obj.getJSONArray("entities");
                if (entities != null) {
                    for (int i = 0; i < entities.size(); i++) {
                        JSONObject e = entities.getJSONObject(i);
                        result.addEntity(new ExtractedEntity(
                            e.getString("name"),
                            e.getString("type"),
                            e.getString("description")
                        ));
                    }
                }
                
                JSONArray relations = obj.getJSONArray("relations");
                if (relations != null) {
                    for (int i = 0; i < relations.size(); i++) {
                        JSONObject r = relations.getJSONObject(i);
                        result.addRelation(new ExtractedRelation(
                            r.getString("source"),
                            r.getString("target"),
                            r.getString("type"),
                            "",
                            0.8
                        ));
                    }
                }
            }
        } catch (Exception e) {
            log.warn("JSON 解析失败: {}", e.getMessage());
        }
    }
    
    // 内部类
    private record RelationPattern(Pattern pattern, String relationType) {
        RelationPattern(String regex, String relationType) {
            this(Pattern.compile(regex), relationType);
        }
    }
    
    public record ExtractedEntity(String name, String type, String description) {}
    
    public record ExtractedRelation(String source, String target, String type, 
                                     String description, double confidence) {}
    
    public static class ExtractionResult {
        private final String docId;
        private final List<ExtractedEntity> entities = new ArrayList<>();
        private final List<ExtractedRelation> relations = new ArrayList<>();
        
        public ExtractionResult(String docId) {
            this.docId = docId;
        }
        
        public void addEntity(ExtractedEntity entity) {
            // 去重
            if (entities.stream().noneMatch(e -> 
                    e.name.equalsIgnoreCase(entity.name) && e.type.equals(entity.type))) {
                entities.add(entity);
            }
        }
        
        public void addRelation(ExtractedRelation relation) {
            relations.add(relation);
        }
        
        public String getDocId() { return docId; }
        public List<ExtractedEntity> getEntities() { return entities; }
        public List<ExtractedRelation> getRelations() { return relations; }
    }
}
