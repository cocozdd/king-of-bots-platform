package com.kob.backend.service.impl.ai.graph;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 知识图谱服务 - GraphRAG 核心组件
 * 
 * 功能：
 * - 实体识别与关系抽取
 * - 图结构存储与遍历
 * - 基于图的上下文增强
 * 
 * 面试要点：
 * - GraphRAG vs 传统 RAG：图结构捕获实体关系，提升多跳推理能力
 * - 实体链接：将文本中的实体链接到知识图谱节点
 * - 社区检测：发现知识聚类，生成社区摘要
 */
@Service
public class KnowledgeGraphService {
    
    private static final Logger log = LoggerFactory.getLogger(KnowledgeGraphService.class);
    
    // 图存储（生产环境可用 Neo4j）
    private final Map<String, GraphNode> nodes = new ConcurrentHashMap<>();
    private final Map<String, List<GraphEdge>> edges = new ConcurrentHashMap<>();
    
    /**
     * 初始化 Bot 对战领域知识图谱
     */
    public void initializeBotKnowledgeGraph() {
        // 核心概念节点
        addNode("Bot", "concept", "自动化对战程序");
        addNode("Strategy", "concept", "Bot决策策略");
        addNode("Algorithm", "concept", "算法实现");
        addNode("Movement", "concept", "移动控制");
        addNode("GameMap", "concept", "游戏地图");
        
        // 策略节点
        addNode("BFS", "algorithm", "广度优先搜索，用于最短路径");
        addNode("DFS", "algorithm", "深度优先搜索，用于空间探索");
        addNode("AStar", "algorithm", "A*算法，启发式最优路径");
        addNode("Greedy", "strategy", "贪心策略，每步最优");
        addNode("Defensive", "strategy", "防守策略，优先生存");
        addNode("Aggressive", "strategy", "进攻策略，追击对手");
        
        // 游戏元素
        addNode("Snake", "entity", "蛇身，需要避开");
        addNode("Wall", "entity", "墙壁，不可通过");
        addNode("Food", "entity", "食物，增加长度");
        addNode("Head", "entity", "蛇头，控制移动方向");
        
        // 建立关系
        addEdge("Bot", "uses", "Strategy");
        addEdge("Strategy", "implements", "Algorithm");
        addEdge("BFS", "solves", "Movement");
        addEdge("AStar", "optimizes", "Movement");
        addEdge("Bot", "avoids", "Snake");
        addEdge("Bot", "avoids", "Wall");
        addEdge("Bot", "seeks", "Food");
        addEdge("Greedy", "subtype_of", "Strategy");
        addEdge("Defensive", "subtype_of", "Strategy");
        addEdge("Aggressive", "subtype_of", "Strategy");
        addEdge("BFS", "subtype_of", "Algorithm");
        addEdge("DFS", "subtype_of", "Algorithm");
        addEdge("AStar", "subtype_of", "Algorithm");
        
        log.info("知识图谱初始化完成: {} 节点, {} 关系", nodes.size(), 
                edges.values().stream().mapToInt(List::size).sum());
    }
    
    /**
     * 基于查询扩展上下文（GraphRAG 核心）
     * 
     * @param query 用户查询
     * @param hops 图遍历跳数
     * @return 图增强的上下文
     */
    public String expandContextWithGraph(String query, int hops) {
        // 1. 从查询中识别实体
        Set<String> entities = extractEntities(query);
        log.info("识别到实体: {}", entities);
        
        if (entities.isEmpty()) {
            return "";
        }
        
        // 2. 图遍历获取相关节点
        Set<String> relatedNodes = new HashSet<>();
        for (String entity : entities) {
            relatedNodes.addAll(traverseGraph(entity, hops));
        }
        
        // 3. 构建图上下文
        StringBuilder context = new StringBuilder();
        context.append("=== 知识图谱上下文 ===\n");
        
        // 添加节点信息
        for (String nodeId : relatedNodes) {
            GraphNode node = nodes.get(nodeId);
            if (node != null) {
                context.append(String.format("- %s (%s): %s\n", 
                        node.name, node.type, node.description));
            }
        }
        
        // 添加关系信息
        context.append("\n关系:\n");
        for (String nodeId : entities) {
            List<GraphEdge> nodeEdges = edges.getOrDefault(nodeId, Collections.emptyList());
            for (GraphEdge edge : nodeEdges) {
                if (relatedNodes.contains(edge.target)) {
                    context.append(String.format("- %s -[%s]-> %s\n", 
                            edge.source, edge.relation, edge.target));
                }
            }
        }
        
        log.info("GraphRAG 扩展: 查询涉及 {} 实体, 扩展到 {} 相关概念", 
                entities.size(), relatedNodes.size());
        
        return context.toString();
    }
    
    /**
     * 从查询中提取实体（简化版，生产可用 NER 模型）
     */
    private Set<String> extractEntities(String query) {
        Set<String> found = new HashSet<>();
        String queryLower = query.toLowerCase();
        
        // 关键词到实体的映射
        Map<String, String> keywordMap = Map.ofEntries(
                Map.entry("bot", "Bot"),
                Map.entry("策略", "Strategy"),
                Map.entry("strategy", "Strategy"),
                Map.entry("算法", "Algorithm"),
                Map.entry("algorithm", "Algorithm"),
                Map.entry("bfs", "BFS"),
                Map.entry("广度", "BFS"),
                Map.entry("dfs", "DFS"),
                Map.entry("深度", "DFS"),
                Map.entry("a*", "AStar"),
                Map.entry("astar", "AStar"),
                Map.entry("移动", "Movement"),
                Map.entry("move", "Movement"),
                Map.entry("贪心", "Greedy"),
                Map.entry("greedy", "Greedy"),
                Map.entry("防守", "Defensive"),
                Map.entry("进攻", "Aggressive"),
                Map.entry("蛇", "Snake"),
                Map.entry("墙", "Wall"),
                Map.entry("食物", "Food"),
                Map.entry("地图", "GameMap")
        );
        
        for (Map.Entry<String, String> entry : keywordMap.entrySet()) {
            if (queryLower.contains(entry.getKey())) {
                found.add(entry.getValue());
            }
        }
        
        return found;
    }
    
    /**
     * 图遍历 - BFS 获取 N 跳内的相关节点
     */
    private Set<String> traverseGraph(String startNode, int maxHops) {
        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        Map<String, Integer> depth = new HashMap<>();
        
        queue.offer(startNode);
        depth.put(startNode, 0);
        visited.add(startNode);
        
        while (!queue.isEmpty()) {
            String current = queue.poll();
            int currentDepth = depth.get(current);
            
            if (currentDepth >= maxHops) continue;
            
            // 遍历出边
            List<GraphEdge> outEdges = edges.getOrDefault(current, Collections.emptyList());
            for (GraphEdge edge : outEdges) {
                if (!visited.contains(edge.target)) {
                    visited.add(edge.target);
                    queue.offer(edge.target);
                    depth.put(edge.target, currentDepth + 1);
                }
            }
            
            // 遍历入边（双向图）
            for (Map.Entry<String, List<GraphEdge>> entry : edges.entrySet()) {
                for (GraphEdge edge : entry.getValue()) {
                    if (edge.target.equals(current) && !visited.contains(edge.source)) {
                        visited.add(edge.source);
                        queue.offer(edge.source);
                        depth.put(edge.source, currentDepth + 1);
                    }
                }
            }
        }
        
        return visited;
    }
    
    /**
     * 添加节点
     */
    public void addNode(String name, String type, String description) {
        nodes.put(name, new GraphNode(name, type, description));
    }
    
    /**
     * 添加边
     */
    public void addEdge(String source, String relation, String target) {
        edges.computeIfAbsent(source, k -> new ArrayList<>())
                .add(new GraphEdge(source, relation, target));
    }
    
    /**
     * 获取图统计信息
     */
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("nodeCount", nodes.size());
        stats.put("edgeCount", edges.values().stream().mapToInt(List::size).sum());
        stats.put("nodeTypes", nodes.values().stream()
                .collect(Collectors.groupingBy(n -> n.type, Collectors.counting())));
        return stats;
    }
    
    // 内部类
    private record GraphNode(String name, String type, String description) {}
    private record GraphEdge(String source, String relation, String target) {}
}
