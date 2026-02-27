package com.kob.backend.service.impl.ai.graphrag;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 知识图谱数据结构
 * 
 * GraphRAG 核心组件，存储实体和关系
 * 
 * 架构：
 * - Entity: 实体节点（概念、技术、操作等）
 * - Relation: 实体之间的关系边
 * - Community: 实体社区（用于全局查询）
 */
public class KnowledgeGraph {
    
    // 实体存储: entityId -> Entity
    private final Map<String, Entity> entities = new ConcurrentHashMap<>();
    
    // 关系存储: relationId -> Relation
    private final Map<String, Relation> relations = new ConcurrentHashMap<>();
    
    // 实体索引: entityName -> entityIds (支持同名实体)
    private final Map<String, Set<String>> nameIndex = new ConcurrentHashMap<>();
    
    // 文档-实体映射: docId -> entityIds
    private final Map<String, Set<String>> docEntityIndex = new ConcurrentHashMap<>();
    
    // 实体-文档映射: entityId -> docIds
    private final Map<String, Set<String>> entityDocIndex = new ConcurrentHashMap<>();
    
    // 社区存储: communityId -> Community
    private final Map<String, Community> communities = new ConcurrentHashMap<>();
    
    /**
     * 添加实体
     */
    public Entity addEntity(String name, String type, String description, String sourceDocId) {
        String entityId = generateEntityId(name, type);
        
        Entity entity = entities.computeIfAbsent(entityId, id -> 
            new Entity(id, name, type, description));
        
        // 如果描述更长，更新描述
        if (description != null && description.length() > entity.getDescription().length()) {
            entity.setDescription(description);
        }
        
        // 更新索引
        nameIndex.computeIfAbsent(name.toLowerCase(), k -> ConcurrentHashMap.newKeySet()).add(entityId);
        
        if (sourceDocId != null) {
            docEntityIndex.computeIfAbsent(sourceDocId, k -> ConcurrentHashMap.newKeySet()).add(entityId);
            entityDocIndex.computeIfAbsent(entityId, k -> ConcurrentHashMap.newKeySet()).add(sourceDocId);
            entity.addSourceDoc(sourceDocId);
        }
        
        return entity;
    }
    
    /**
     * 添加关系
     */
    public Relation addRelation(String sourceEntityId, String targetEntityId, 
                                 String relationType, String description, double weight) {
        String relationId = sourceEntityId + "-" + relationType + "->" + targetEntityId;
        
        Relation relation = relations.computeIfAbsent(relationId, id ->
            new Relation(id, sourceEntityId, targetEntityId, relationType, description, weight));
        
        // 更新权重（取最大值）
        if (weight > relation.getWeight()) {
            relation.setWeight(weight);
        }
        
        // 更新实体的邻居关系
        Entity sourceEntity = entities.get(sourceEntityId);
        Entity targetEntity = entities.get(targetEntityId);
        
        if (sourceEntity != null) {
            sourceEntity.addOutRelation(relationId);
        }
        if (targetEntity != null) {
            targetEntity.addInRelation(relationId);
        }
        
        return relation;
    }
    
    /**
     * 根据名称查找实体
     */
    public List<Entity> findEntitiesByName(String name) {
        Set<String> entityIds = nameIndex.get(name.toLowerCase());
        if (entityIds == null) {
            return new ArrayList<>();
        }
        return entityIds.stream()
                .map(entities::get)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
    }
    
    /**
     * 模糊搜索实体
     */
    public List<Entity> searchEntities(String keyword, int limit) {
        String keywordLower = keyword.toLowerCase();
        return entities.values().stream()
                .filter(e -> e.getName().toLowerCase().contains(keywordLower) ||
                            e.getDescription().toLowerCase().contains(keywordLower))
                .sorted((a, b) -> {
                    // 精确匹配优先
                    boolean aExact = a.getName().toLowerCase().equals(keywordLower);
                    boolean bExact = b.getName().toLowerCase().equals(keywordLower);
                    if (aExact && !bExact) return -1;
                    if (!aExact && bExact) return 1;
                    // 名称包含优先
                    boolean aInName = a.getName().toLowerCase().contains(keywordLower);
                    boolean bInName = b.getName().toLowerCase().contains(keywordLower);
                    if (aInName && !bInName) return -1;
                    if (!aInName && bInName) return 1;
                    return 0;
                })
                .limit(limit)
                .collect(Collectors.toList());
    }
    
    /**
     * 获取实体的邻居（1跳）
     */
    public List<Entity> getNeighbors(String entityId) {
        Entity entity = entities.get(entityId);
        if (entity == null) {
            return new ArrayList<>();
        }
        
        Set<String> neighborIds = new HashSet<>();
        
        // 出边的目标
        for (String relationId : entity.getOutRelations()) {
            Relation rel = relations.get(relationId);
            if (rel != null) {
                neighborIds.add(rel.getTargetEntityId());
            }
        }
        
        // 入边的源
        for (String relationId : entity.getInRelations()) {
            Relation rel = relations.get(relationId);
            if (rel != null) {
                neighborIds.add(rel.getSourceEntityId());
            }
        }
        
        return neighborIds.stream()
                .map(entities::get)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
    }
    
    /**
     * 获取实体的关系
     */
    public List<Relation> getEntityRelations(String entityId) {
        Entity entity = entities.get(entityId);
        if (entity == null) {
            return new ArrayList<>();
        }
        
        List<Relation> result = new ArrayList<>();
        for (String relationId : entity.getOutRelations()) {
            Relation rel = relations.get(relationId);
            if (rel != null) result.add(rel);
        }
        for (String relationId : entity.getInRelations()) {
            Relation rel = relations.get(relationId);
            if (rel != null) result.add(rel);
        }
        return result;
    }
    
    /**
     * 图遍历：获取 N 跳内的所有实体
     */
    public Set<Entity> traverseWithinHops(String startEntityId, int maxHops) {
        Set<String> visited = new HashSet<>();
        Set<Entity> result = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        Map<String, Integer> depths = new HashMap<>();
        
        queue.offer(startEntityId);
        depths.put(startEntityId, 0);
        visited.add(startEntityId);
        
        while (!queue.isEmpty()) {
            String currentId = queue.poll();
            int currentDepth = depths.get(currentId);
            
            Entity current = entities.get(currentId);
            if (current != null) {
                result.add(current);
            }
            
            if (currentDepth >= maxHops) {
                continue;
            }
            
            // 遍历邻居
            for (Entity neighbor : getNeighbors(currentId)) {
                if (!visited.contains(neighbor.getId())) {
                    visited.add(neighbor.getId());
                    queue.offer(neighbor.getId());
                    depths.put(neighbor.getId(), currentDepth + 1);
                }
            }
        }
        
        return result;
    }
    
    /**
     * 获取实体关联的文档ID
     */
    public Set<String> getEntityDocuments(String entityId) {
        return entityDocIndex.getOrDefault(entityId, Collections.emptySet());
    }
    
    /**
     * 获取文档关联的实体
     */
    public Set<Entity> getDocumentEntities(String docId) {
        Set<String> entityIds = docEntityIndex.get(docId);
        if (entityIds == null) {
            return Collections.emptySet();
        }
        return entityIds.stream()
                .map(entities::get)
                .filter(Objects::nonNull)
                .collect(Collectors.toSet());
    }
    
    /**
     * 添加社区
     */
    public void addCommunity(Community community) {
        communities.put(community.getId(), community);
    }
    
    /**
     * 获取所有社区
     */
    public Collection<Community> getCommunities() {
        return communities.values();
    }
    
    /**
     * 获取实体所属的社区
     */
    public List<Community> getEntityCommunities(String entityId) {
        return communities.values().stream()
                .filter(c -> c.getEntityIds().contains(entityId))
                .collect(Collectors.toList());
    }
    
    public Entity getEntity(String entityId) {
        return entities.get(entityId);
    }
    
    public Relation getRelation(String relationId) {
        return relations.get(relationId);
    }
    
    public int getEntityCount() {
        return entities.size();
    }
    
    public int getRelationCount() {
        return relations.size();
    }
    
    public int getCommunityCount() {
        return communities.size();
    }
    
    private String generateEntityId(String name, String type) {
        return type.toLowerCase() + ":" + name.toLowerCase().replaceAll("\\s+", "_");
    }
    
    /**
     * 获取图谱统计信息
     */
    public GraphStats getStats() {
        return new GraphStats(
            entities.size(),
            relations.size(),
            communities.size(),
            docEntityIndex.size()
        );
    }
    
    public record GraphStats(int entityCount, int relationCount, int communityCount, int indexedDocs) {}
}
