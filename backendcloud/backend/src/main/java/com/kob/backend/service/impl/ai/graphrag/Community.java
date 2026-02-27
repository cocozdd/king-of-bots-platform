package com.kob.backend.service.impl.ai.graphrag;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/**
 * 知识图谱社区
 * 
 * 一组紧密相关的实体集合，用于全局查询
 * 
 * 特点：
 * - 社区内实体高度相关
 * - 每个社区有一个摘要描述
 * - 用于回答需要全局视角的问题
 */
public class Community {
    
    private final String id;
    private final String title;
    private volatile String summary;
    private final Set<String> entityIds;
    private final int level;  // 社区层级（0=最细粒度，数字越大越抽象）
    
    public Community(String id, String title, String summary, int level) {
        this.id = id;
        this.title = title;
        this.summary = summary;
        this.level = level;
        this.entityIds = new HashSet<>();
    }
    
    public String getId() {
        return id;
    }
    
    public String getTitle() {
        return title;
    }
    
    public String getSummary() {
        return summary;
    }
    
    public void setSummary(String summary) {
        this.summary = summary;
    }
    
    public Set<String> getEntityIds() {
        return new HashSet<>(entityIds);
    }
    
    public void addEntity(String entityId) {
        entityIds.add(entityId);
    }
    
    public void addEntities(Set<String> ids) {
        entityIds.addAll(ids);
    }
    
    public int getLevel() {
        return level;
    }
    
    public int getSize() {
        return entityIds.size();
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Community community = (Community) o;
        return Objects.equals(id, community.id);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
    
    @Override
    public String toString() {
        return String.format("Community{id='%s', title='%s', size=%d, level=%d}", 
                id, title, entityIds.size(), level);
    }
}
