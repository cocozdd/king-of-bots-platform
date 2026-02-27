package com.kob.backend.service.impl.ai.graphrag;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 知识图谱实体
 * 
 * 表示图谱中的一个节点，如：
 * - 技术概念（BFS、A*算法）
 * - 操作动作（移动、攻击）
 * - 游戏元素（蛇、地图、障碍物）
 */
public class Entity {
    
    private final String id;
    private final String name;
    private final String type;  // CONCEPT, ACTION, ELEMENT, STRATEGY, CODE
    private volatile String description;
    
    // 关联的关系ID
    private final Set<String> outRelations = ConcurrentHashMap.newKeySet();
    private final Set<String> inRelations = ConcurrentHashMap.newKeySet();
    
    // 来源文档
    private final Set<String> sourceDocs = ConcurrentHashMap.newKeySet();
    
    // 实体重要性权重（PageRank 或度中心性）
    private volatile double importance = 1.0;
    
    // 实体向量（可选，用于语义搜索）
    private volatile double[] embedding;
    
    public Entity(String id, String name, String type, String description) {
        this.id = id;
        this.name = name;
        this.type = type;
        this.description = description != null ? description : "";
    }
    
    public String getId() {
        return id;
    }
    
    public String getName() {
        return name;
    }
    
    public String getType() {
        return type;
    }
    
    public String getDescription() {
        return description;
    }
    
    public void setDescription(String description) {
        this.description = description;
    }
    
    public Set<String> getOutRelations() {
        return new HashSet<>(outRelations);
    }
    
    public Set<String> getInRelations() {
        return new HashSet<>(inRelations);
    }
    
    public void addOutRelation(String relationId) {
        outRelations.add(relationId);
    }
    
    public void addInRelation(String relationId) {
        inRelations.add(relationId);
    }
    
    public Set<String> getSourceDocs() {
        return new HashSet<>(sourceDocs);
    }
    
    public void addSourceDoc(String docId) {
        sourceDocs.add(docId);
    }
    
    public double getImportance() {
        return importance;
    }
    
    public void setImportance(double importance) {
        this.importance = importance;
    }
    
    public double[] getEmbedding() {
        return embedding;
    }
    
    public void setEmbedding(double[] embedding) {
        this.embedding = embedding;
    }
    
    /**
     * 获取实体的度（入度+出度）
     */
    public int getDegree() {
        return outRelations.size() + inRelations.size();
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Entity entity = (Entity) o;
        return Objects.equals(id, entity.id);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
    
    @Override
    public String toString() {
        return String.format("Entity{id='%s', name='%s', type='%s', degree=%d}", 
                id, name, type, getDegree());
    }
}
