package com.kob.backend.service.impl.ai.graphrag;

import java.util.Objects;

/**
 * 知识图谱关系
 * 
 * 表示两个实体之间的有向边，如：
 * - BFS -[用于]-> 寻路
 * - 蛇 -[可以]-> 移动
 * - A*算法 -[优于]-> BFS
 */
public class Relation {
    
    private final String id;
    private final String sourceEntityId;
    private final String targetEntityId;
    private final String relationType;  // USES, CAN, BETTER_THAN, PART_OF, RELATED_TO
    private final String description;
    private volatile double weight;
    
    public Relation(String id, String sourceEntityId, String targetEntityId, 
                   String relationType, String description, double weight) {
        this.id = id;
        this.sourceEntityId = sourceEntityId;
        this.targetEntityId = targetEntityId;
        this.relationType = relationType;
        this.description = description != null ? description : "";
        this.weight = weight;
    }
    
    public String getId() {
        return id;
    }
    
    public String getSourceEntityId() {
        return sourceEntityId;
    }
    
    public String getTargetEntityId() {
        return targetEntityId;
    }
    
    public String getRelationType() {
        return relationType;
    }
    
    public String getDescription() {
        return description;
    }
    
    public double getWeight() {
        return weight;
    }
    
    public void setWeight(double weight) {
        this.weight = weight;
    }
    
    /**
     * 获取关系的自然语言表示
     */
    public String toNaturalLanguage(String sourceName, String targetName) {
        return switch (relationType) {
            case "USES" -> sourceName + " 使用 " + targetName;
            case "CAN" -> sourceName + " 可以 " + targetName;
            case "BETTER_THAN" -> sourceName + " 优于 " + targetName;
            case "PART_OF" -> sourceName + " 是 " + targetName + " 的一部分";
            case "RELATED_TO" -> sourceName + " 与 " + targetName + " 相关";
            case "IMPLEMENTS" -> sourceName + " 实现 " + targetName;
            case "DEPENDS_ON" -> sourceName + " 依赖 " + targetName;
            case "CAUSES" -> sourceName + " 导致 " + targetName;
            case "PREVENTS" -> sourceName + " 防止 " + targetName;
            default -> sourceName + " [" + relationType + "] " + targetName;
        };
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Relation relation = (Relation) o;
        return Objects.equals(id, relation.id);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
    
    @Override
    public String toString() {
        return String.format("Relation{%s -[%s]-> %s, weight=%.2f}", 
                sourceEntityId, relationType, targetEntityId, weight);
    }
}
