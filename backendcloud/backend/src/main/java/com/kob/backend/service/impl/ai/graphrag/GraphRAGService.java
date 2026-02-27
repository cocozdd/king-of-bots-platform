package com.kob.backend.service.impl.ai.graphrag;

import com.kob.backend.controller.ai.dto.AiDoc;
import com.kob.backend.repository.AiCorpusRepository;
import com.kob.backend.service.impl.ai.*;
import com.kob.backend.service.impl.ai.search.HybridSearchService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.util.*;
import java.util.stream.Collectors;

/**
 * GraphRAG 检索服务
 * 
 * 核心功能：
 * 1. Local Search: 基于实体的局部图检索
 * 2. Global Search: 基于社区摘要的全局检索
 * 3. Hybrid: 结合传统 RAG 和图检索
 * 
 * 优势：
 * - 发现隐含关联（多跳推理）
 * - 全局视角回答（社区摘要）
 * - 更好的上下文理解
 */
@Service
public class GraphRAGService {
    
    private static final Logger log = LoggerFactory.getLogger(GraphRAGService.class);
    
    private final KnowledgeGraph knowledgeGraph = new KnowledgeGraph();
    private EntityExtractor entityExtractor;
    private volatile boolean initialized = false;
    
    @Autowired
    private AiCorpusRepository corpusRepository;
    
    @Autowired
    @Qualifier("pgvectorJdbcTemplate")
    private JdbcTemplate jdbcTemplate;
    
    @Autowired
    private AiMetricsService metricsService;
    
    @Autowired(required = false)
    private DashscopeEmbeddingClient embeddingClient;
    
    @Autowired(required = false)
    private DeepseekClient deepseekClient;
    
    @Autowired
    private HybridSearchService hybridSearchService;
    
    @Autowired
    private RerankService rerankService;
    
    @PostConstruct
    public void init() {
        this.entityExtractor = new EntityExtractor(deepseekClient);
        log.info("GraphRAGService 初始化完成");
    }
    
    /**
     * 构建知识图谱（从数据库加载文档并提取实体）
     */
    public synchronized void buildGraph() {
        if (initialized) {
            log.info("知识图谱已构建，跳过");
            return;
        }
        
        long startTime = System.currentTimeMillis();
        log.info("开始构建知识图谱...");
        
        try {
            // 1. 加载所有文档
            List<AiDoc> docs = loadAllDocuments();
            log.info("加载 {} 篇文档", docs.size());
            
            // 2. 提取实体和关系
            for (AiDoc doc : docs) {
                try {
                    EntityExtractor.ExtractionResult result = 
                        entityExtractor.extract(doc.getId(), doc.getTitle(), doc.getContent());
                    
                    // 添加实体到图谱
                    for (EntityExtractor.ExtractedEntity e : result.getEntities()) {
                        knowledgeGraph.addEntity(e.name(), e.type(), e.description(), doc.getId());
                    }
                    
                    // 添加关系到图谱
                    for (EntityExtractor.ExtractedRelation r : result.getRelations()) {
                        String sourceId = inferEntityId(r.source());
                        String targetId = inferEntityId(r.target());
                        if (sourceId != null && targetId != null) {
                            knowledgeGraph.addRelation(sourceId, targetId, r.type(), 
                                    r.description(), r.confidence());
                        }
                    }
                } catch (Exception e) {
                    log.warn("处理文档 {} 失败: {}", doc.getId(), e.getMessage());
                }
            }
            
            // 3. 构建社区
            buildCommunities();
            
            initialized = true;
            long elapsed = System.currentTimeMillis() - startTime;
            
            KnowledgeGraph.GraphStats stats = knowledgeGraph.getStats();
            log.info("知识图谱构建完成: {} 实体, {} 关系, {} 社区, 耗时 {}ms",
                    stats.entityCount(), stats.relationCount(), stats.communityCount(), elapsed);
                    
        } catch (Exception e) {
            log.error("构建知识图谱失败: {}", e.getMessage(), e);
        }
    }
    
    /**
     * Local Search: 基于实体的局部检索
     * 
     * 流程：
     * 1. 从查询中提取实体
     * 2. 在图中找到相关实体
     * 3. 遍历获取相关文档
     * 4. 结合传统向量检索
     */
    public GraphRAGResult localSearch(String query, double[] embedding, int topK) {
        long startTime = System.currentTimeMillis();
        ensureInitialized();
        
        GraphRAGResult result = new GraphRAGResult();
        result.setSearchType("LOCAL");
        
        // 1. 从查询提取实体
        List<String> queryEntities = extractQueryEntities(query);
        result.setExtractedEntities(queryEntities);
        
        // 2. 查找相关图实体
        Set<Entity> relevantEntities = new HashSet<>();
        for (String entityName : queryEntities) {
            List<Entity> found = knowledgeGraph.searchEntities(entityName, 5);
            relevantEntities.addAll(found);
            
            // 扩展：获取邻居实体（1跳）
            for (Entity e : new ArrayList<>(found)) {
                relevantEntities.addAll(knowledgeGraph.getNeighbors(e.getId()));
            }
        }
        
        // 3. 获取实体关联的文档
        Set<String> graphDocIds = new HashSet<>();
        for (Entity entity : relevantEntities) {
            graphDocIds.addAll(knowledgeGraph.getEntityDocuments(entity.getId()));
        }
        
        // 4. 传统向量检索
        List<AiDoc> vectorDocs = hybridSearchService.hybridSearch(query, embedding, topK * 2);
        
        // 5. 融合结果
        List<AiDoc> mergedDocs = mergeWithGraphContext(vectorDocs, graphDocIds, topK);
        
        // 6. Rerank
        List<AiDoc> finalDocs = rerankService.rerank(query, mergedDocs, topK);
        
        // 7. 构建图上下文
        String graphContext = buildGraphContext(relevantEntities, query);
        
        result.setDocuments(finalDocs);
        result.setGraphContext(graphContext);
        result.setRelevantEntities(relevantEntities.stream()
                .map(e -> new EntityInfo(e.getId(), e.getName(), e.getType(), e.getDegree()))
                .limit(10)
                .collect(Collectors.toList()));
        
        long elapsed = System.currentTimeMillis() - startTime;
        result.setLatencyMs(elapsed);
        
        log.info("GraphRAG Local Search: query={}, 提取{}实体, 图中找到{}实体, 融合{}文档, 耗时{}ms",
                query.substring(0, Math.min(20, query.length())),
                queryEntities.size(), relevantEntities.size(), finalDocs.size(), elapsed);
        
        return result;
    }
    
    /**
     * Global Search: 基于社区的全局检索
     * 
     * 适用于需要全局视角的问题，如：
     * - "有哪些策略可以提升胜率？"
     * - "总结一下所有的移动算法"
     */
    public GraphRAGResult globalSearch(String query, int topK) {
        long startTime = System.currentTimeMillis();
        ensureInitialized();
        
        GraphRAGResult result = new GraphRAGResult();
        result.setSearchType("GLOBAL");
        
        // 1. 搜索相关社区
        List<Community> relevantCommunities = searchCommunities(query, 5);
        
        // 2. 收集社区摘要
        List<String> communitySummaries = relevantCommunities.stream()
                .map(Community::getSummary)
                .filter(s -> s != null && !s.isEmpty())
                .collect(Collectors.toList());
        
        // 3. 收集社区内的实体
        Set<Entity> communityEntities = new HashSet<>();
        for (Community community : relevantCommunities) {
            for (String entityId : community.getEntityIds()) {
                Entity entity = knowledgeGraph.getEntity(entityId);
                if (entity != null) {
                    communityEntities.add(entity);
                }
            }
        }
        
        // 4. 获取相关文档
        Set<String> docIds = new HashSet<>();
        for (Entity entity : communityEntities) {
            docIds.addAll(knowledgeGraph.getEntityDocuments(entity.getId()));
        }
        
        // 5. 加载文档
        List<AiDoc> docs = loadDocumentsByIds(new ArrayList<>(docIds));
        
        // 6. Rerank
        List<AiDoc> finalDocs = rerankService.rerank(query, docs, topK);
        
        // 7. 构建全局上下文
        String globalContext = buildGlobalContext(relevantCommunities, communityEntities);
        
        result.setDocuments(finalDocs);
        result.setGraphContext(globalContext);
        result.setCommunitySummaries(communitySummaries);
        result.setRelevantEntities(communityEntities.stream()
                .map(e -> new EntityInfo(e.getId(), e.getName(), e.getType(), e.getDegree()))
                .limit(15)
                .collect(Collectors.toList()));
        
        long elapsed = System.currentTimeMillis() - startTime;
        result.setLatencyMs(elapsed);
        
        log.info("GraphRAG Global Search: query={}, 找到{}社区, {}实体, {}文档, 耗时{}ms",
                query.substring(0, Math.min(20, query.length())),
                relevantCommunities.size(), communityEntities.size(), finalDocs.size(), elapsed);
        
        return result;
    }
    
    /**
     * 混合检索：自动选择 Local 或 Global
     */
    public GraphRAGResult search(String query, double[] embedding, int topK) {
        // 判断查询类型
        boolean isGlobalQuery = isGlobalQuery(query);
        
        if (isGlobalQuery) {
            return globalSearch(query, topK);
        } else {
            return localSearch(query, embedding, topK);
        }
    }
    
    /**
     * 获取图谱统计信息
     */
    public Map<String, Object> getStats() {
        KnowledgeGraph.GraphStats stats = knowledgeGraph.getStats();
        Map<String, Object> result = new HashMap<>();
        result.put("initialized", initialized);
        result.put("entityCount", stats.entityCount());
        result.put("relationCount", stats.relationCount());
        result.put("communityCount", stats.communityCount());
        result.put("indexedDocs", stats.indexedDocs());
        return result;
    }
    
    // ==================== 私有方法 ====================
    
    private void ensureInitialized() {
        if (!initialized) {
            buildGraph();
        }
    }
    
    private List<AiDoc> loadAllDocuments() {
        try {
            String sql = "SELECT doc_id as id, title, category, content FROM ai_corpus";
            return jdbcTemplate.query(sql, (rs, rowNum) -> new AiDoc(
                    rs.getString("id"),
                    rs.getString("title"),
                    rs.getString("category"),
                    rs.getString("content")
            ));
        } catch (Exception e) {
            log.error("加载文档失败: {}", e.getMessage());
            return new ArrayList<>();
        }
    }
    
    private List<AiDoc> loadDocumentsByIds(List<String> docIds) {
        if (docIds.isEmpty()) return new ArrayList<>();
        
        try {
            String placeholders = docIds.stream().map(id -> "?").collect(Collectors.joining(","));
            String sql = "SELECT doc_id as id, title, category, content FROM ai_corpus WHERE doc_id IN (" + placeholders + ")";
            return jdbcTemplate.query(sql, ps -> {
                for (int i = 0; i < docIds.size(); i++) {
                    ps.setString(i + 1, docIds.get(i));
                }
            }, (rs, rowNum) -> new AiDoc(
                    rs.getString("id"),
                    rs.getString("title"),
                    rs.getString("category"),
                    rs.getString("content")
            ));
        } catch (Exception e) {
            log.error("加载文档失败: {}", e.getMessage());
            return new ArrayList<>();
        }
    }
    
    private List<String> extractQueryEntities(String query) {
        List<String> entities = new ArrayList<>();
        String queryLower = query.toLowerCase();
        
        // 基于关键词提取
        String[] keywords = {"bfs", "dfs", "a*", "策略", "移动", "蛇", "算法", "优化", 
                           "路径", "寻路", "攻击", "防守", "代码", "bot"};
        for (String keyword : keywords) {
            if (queryLower.contains(keyword)) {
                entities.add(keyword);
            }
        }
        
        // 提取名词短语（简化版）
        String[] words = query.split("[\\s，。？！,\\.\\?!]+");
        for (String word : words) {
            if (word.length() >= 2 && word.length() <= 10) {
                entities.add(word);
            }
        }
        
        return entities.stream().distinct().limit(10).collect(Collectors.toList());
    }
    
    private String inferEntityId(String entityName) {
        List<Entity> found = knowledgeGraph.findEntitiesByName(entityName);
        if (!found.isEmpty()) {
            return found.get(0).getId();
        }
        // 模糊匹配
        found = knowledgeGraph.searchEntities(entityName, 1);
        return found.isEmpty() ? null : found.get(0).getId();
    }
    
    private List<AiDoc> mergeWithGraphContext(List<AiDoc> vectorDocs, Set<String> graphDocIds, int topK) {
        Map<String, AiDoc> docMap = new LinkedHashMap<>();
        
        // 优先添加图检索的文档（有图关联的优先）
        for (AiDoc doc : vectorDocs) {
            if (graphDocIds.contains(doc.getId())) {
                docMap.put(doc.getId(), doc);
            }
        }
        
        // 再添加向量检索的文档
        for (AiDoc doc : vectorDocs) {
            if (!docMap.containsKey(doc.getId())) {
                docMap.put(doc.getId(), doc);
            }
            if (docMap.size() >= topK * 2) break;
        }
        
        return new ArrayList<>(docMap.values());
    }
    
    private String buildGraphContext(Set<Entity> entities, String query) {
        StringBuilder context = new StringBuilder();
        context.append("【图谱上下文】\n");
        
        // 实体信息
        context.append("相关概念: ");
        context.append(entities.stream()
                .sorted((a, b) -> b.getDegree() - a.getDegree())
                .limit(5)
                .map(e -> e.getName() + "(" + e.getType() + ")")
                .collect(Collectors.joining(", ")));
        context.append("\n");
        
        // 关系信息
        context.append("概念关系:\n");
        for (Entity entity : entities.stream().limit(3).collect(Collectors.toList())) {
            List<Relation> relations = knowledgeGraph.getEntityRelations(entity.getId());
            for (Relation rel : relations.stream().limit(3).collect(Collectors.toList())) {
                Entity source = knowledgeGraph.getEntity(rel.getSourceEntityId());
                Entity target = knowledgeGraph.getEntity(rel.getTargetEntityId());
                if (source != null && target != null) {
                    context.append("  - ").append(rel.toNaturalLanguage(source.getName(), target.getName())).append("\n");
                }
            }
        }
        
        return context.toString();
    }
    
    private String buildGlobalContext(List<Community> communities, Set<Entity> entities) {
        StringBuilder context = new StringBuilder();
        context.append("【全局视角】\n");
        
        for (Community community : communities) {
            context.append("主题: ").append(community.getTitle()).append("\n");
            if (community.getSummary() != null) {
                context.append("摘要: ").append(community.getSummary()).append("\n");
            }
            context.append("\n");
        }
        
        context.append("涉及概念: ");
        context.append(entities.stream()
                .limit(10)
                .map(Entity::getName)
                .collect(Collectors.joining(", ")));
        
        return context.toString();
    }
    
    private void buildCommunities() {
        // 简化版社区发现：基于实体类型分组
        Map<String, List<Entity>> typeGroups = new HashMap<>();
        
        for (int i = 0; i < 100; i++) {
            // 模拟迭代所有实体（实际应遍历所有实体）
        }
        
        // 基于类型创建社区
        String[] types = {"ALGORITHM", "STRATEGY", "ELEMENT", "ACTION", "CODE", "OPTIMIZATION"};
        String[] titles = {"算法与寻路", "策略与战术", "游戏元素", "动作与操作", "代码实现", "性能优化"};
        String[] summaries = {
            "包含BFS、DFS、A*等寻路算法，用于计算蛇的移动路径",
            "包含进攻、防守、生存等策略，用于提升Bot胜率",
            "包含蛇、地图、障碍物等游戏元素",
            "包含移动、转向、碰撞等操作动作",
            "包含代码实现相关的函数、类和接口",
            "包含性能优化、剪枝、缓存等技术"
        };
        
        for (int i = 0; i < types.length; i++) {
            Community community = new Community(
                "community-" + types[i].toLowerCase(),
                titles[i],
                summaries[i],
                0
            );
            
            // 添加该类型的所有实体
            List<Entity> found = knowledgeGraph.searchEntities(types[i], 100);
            for (Entity entity : found) {
                if (entity.getType().equals(types[i])) {
                    community.addEntity(entity.getId());
                }
            }
            
            if (community.getSize() > 0) {
                knowledgeGraph.addCommunity(community);
            }
        }
    }
    
    private List<Community> searchCommunities(String query, int limit) {
        String queryLower = query.toLowerCase();
        return knowledgeGraph.getCommunities().stream()
                .filter(c -> c.getTitle().toLowerCase().contains(queryLower) ||
                            c.getSummary().toLowerCase().contains(queryLower) ||
                            queryLower.contains(c.getTitle().substring(0, Math.min(4, c.getTitle().length()))))
                .limit(limit)
                .collect(Collectors.toList());
    }
    
    private boolean isGlobalQuery(String query) {
        String[] globalPatterns = {"有哪些", "总结", "所有", "列举", "概述", "整体", "全部"};
        String queryLower = query.toLowerCase();
        for (String pattern : globalPatterns) {
            if (queryLower.contains(pattern)) {
                return true;
            }
        }
        return false;
    }
    
    // ==================== 结果类 ====================
    
    public static class GraphRAGResult {
        private String searchType;
        private List<AiDoc> documents = new ArrayList<>();
        private String graphContext;
        private List<String> extractedEntities = new ArrayList<>();
        private List<EntityInfo> relevantEntities = new ArrayList<>();
        private List<String> communitySummaries = new ArrayList<>();
        private long latencyMs;
        
        // Getters and Setters
        public String getSearchType() { return searchType; }
        public void setSearchType(String searchType) { this.searchType = searchType; }
        
        public List<AiDoc> getDocuments() { return documents; }
        public void setDocuments(List<AiDoc> documents) { this.documents = documents; }
        
        public String getGraphContext() { return graphContext; }
        public void setGraphContext(String graphContext) { this.graphContext = graphContext; }
        
        public List<String> getExtractedEntities() { return extractedEntities; }
        public void setExtractedEntities(List<String> extractedEntities) { this.extractedEntities = extractedEntities; }
        
        public List<EntityInfo> getRelevantEntities() { return relevantEntities; }
        public void setRelevantEntities(List<EntityInfo> relevantEntities) { this.relevantEntities = relevantEntities; }
        
        public List<String> getCommunitySummaries() { return communitySummaries; }
        public void setCommunitySummaries(List<String> communitySummaries) { this.communitySummaries = communitySummaries; }
        
        public long getLatencyMs() { return latencyMs; }
        public void setLatencyMs(long latencyMs) { this.latencyMs = latencyMs; }
    }
    
    public record EntityInfo(String id, String name, String type, int degree) {}
}
