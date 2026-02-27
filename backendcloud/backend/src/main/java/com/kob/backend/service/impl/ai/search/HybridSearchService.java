package com.kob.backend.service.impl.ai.search;

import com.kob.backend.controller.ai.dto.AiDoc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 混合检索服务 - 向量检索 + 关键词检索融合
 * 
 * 功能：
 * - 稠密检索：Embedding 向量相似度
 * - 稀疏检索：BM25 关键词匹配
 * - 融合排序：RRF (Reciprocal Rank Fusion)
 * 
 * 面试要点：
 * - 为什么混合检索：向量擅长语义，关键词擅长精确匹配
 * - RRF 融合算法：score = Σ 1/(k + rank_i)，k 通常取 60
 * - 优于单一检索 10-20%
 * 
 * @deprecated 自 Phase 3 起，推荐使用 Python RAG 实现。
 *             此类将在下个版本移除。请通过 RAGRouter 路由到 Python 后端。
 *             配置: ai.rag.backend=python
 * @see com.kob.backend.service.impl.ai.RAGRouter
 */
@Deprecated
@Service
public class HybridSearchService {
    
    private static final Logger log = LoggerFactory.getLogger(HybridSearchService.class);
    
    // RRF 常数
    private static final int RRF_K = 60;
    
    // 默认权重
    private static final double DENSE_WEIGHT = 0.6;
    private static final double SPARSE_WEIGHT = 0.4;
    
    private final JdbcTemplate jdbcTemplate;
    
    public HybridSearchService(@Qualifier("pgvectorJdbcTemplate") JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }
    
    /**
     * 混合检索
     * 
     * @param query 查询文本
     * @param queryEmbedding 查询向量（1536维）
     * @param topK 返回数量
     * @return 融合排序后的文档列表
     */
    public List<HybridSearchResult> search(String query, float[] queryEmbedding, int topK) {
        log.info("混合检索: query='{}', topK={}", truncate(query, 50), topK);
        
        // 1. 稠密检索（向量相似度）
        List<SearchHit> denseResults = denseSearch(queryEmbedding, topK * 2);
        log.debug("稠密检索返回 {} 结果", denseResults.size());
        
        // 2. 稀疏检索（BM25）
        List<SearchHit> sparseResults = sparseSearch(query, topK * 2);
        log.debug("稀疏检索返回 {} 结果", sparseResults.size());
        
        // 3. RRF 融合
        List<HybridSearchResult> fusedResults = fusionRRF(denseResults, sparseResults, topK);
        
        log.info("混合检索完成: 返回 {} 结果, 最高分 {:.4f}", 
                fusedResults.size(), 
                fusedResults.isEmpty() ? 0 : fusedResults.get(0).fusedScore);
        
        return fusedResults;
    }
    
    /**
     * 兼容方法 - 接受 double[] 并返回 AiDoc 列表
     */
    public List<AiDoc> hybridSearch(String query, double[] queryEmbedding, int topK) {
        // 转换 double[] 到 float[]
        float[] floatEmbedding = new float[queryEmbedding.length];
        for (int i = 0; i < queryEmbedding.length; i++) {
            floatEmbedding[i] = (float) queryEmbedding[i];
        }
        
        List<HybridSearchResult> results = search(query, floatEmbedding, topK);
        
        // 转换为 AiDoc 列表
        return results.stream()
                .map(r -> new AiDoc(
                        String.valueOf(r.id),
                        r.title,
                        r.content,
                        r.category
                ))
                .collect(Collectors.toList());
    }
    
    /**
     * 带权重的混合检索
     */
    public List<HybridSearchResult> searchWithWeights(
            String query, 
            float[] queryEmbedding, 
            int topK,
            double denseWeight,
            double sparseWeight) {
        
        List<SearchHit> denseResults = denseSearch(queryEmbedding, topK * 2);
        List<SearchHit> sparseResults = sparseSearch(query, topK * 2);
        
        return fusionWeighted(denseResults, sparseResults, topK, denseWeight, sparseWeight);
    }
    
    /**
     * 稠密检索 - pgvector 向量相似度
     */
    private List<SearchHit> denseSearch(float[] queryEmbedding, int limit) {
        String embeddingStr = Arrays.toString(queryEmbedding);
        
        String sql = """
            SELECT id, title, content, category,
                   1 - (embedding <=> ?::vector) as score
            FROM ai_corpus
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> ?::vector
            LIMIT ?
            """;
        
        try {
            return jdbcTemplate.query(sql, (rs, rowNum) -> {
                SearchHit hit = new SearchHit();
                hit.id = rs.getLong("id");
                hit.title = rs.getString("title");
                hit.content = rs.getString("content");
                hit.category = rs.getString("category");
                hit.score = rs.getDouble("score");
                hit.source = "dense";
                return hit;
            }, embeddingStr, embeddingStr, limit);
        } catch (Exception e) {
            log.warn("稠密检索失败: {}", e.getMessage());
            return Collections.emptyList();
        }
    }
    
    /**
     * 稀疏检索 - PostgreSQL 全文搜索（BM25 近似）
     */
    private List<SearchHit> sparseSearch(String query, int limit) {
        // 处理查询词
        String[] terms = query.split("\\s+");
        String tsQuery = Arrays.stream(terms)
                .filter(t -> t.length() > 1)
                .map(t -> t + ":*")
                .collect(Collectors.joining(" | "));
        
        if (tsQuery.isEmpty()) {
            return Collections.emptyList();
        }
        
        String sql = """
            SELECT id, title, content, category,
                   ts_rank_cd(
                       setweight(to_tsvector('simple', coalesce(title, '')), 'A') ||
                       setweight(to_tsvector('simple', coalesce(content, '')), 'B'),
                       to_tsquery('simple', ?)
                   ) as score
            FROM ai_corpus
            WHERE to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(content, ''))
                  @@ to_tsquery('simple', ?)
            ORDER BY score DESC
            LIMIT ?
            """;
        
        try {
            return jdbcTemplate.query(sql, (rs, rowNum) -> {
                SearchHit hit = new SearchHit();
                hit.id = rs.getLong("id");
                hit.title = rs.getString("title");
                hit.content = rs.getString("content");
                hit.category = rs.getString("category");
                hit.score = rs.getDouble("score");
                hit.source = "sparse";
                return hit;
            }, tsQuery, tsQuery, limit);
        } catch (Exception e) {
            log.warn("稀疏检索失败: {}", e.getMessage());
            return Collections.emptyList();
        }
    }
    
    /**
     * RRF 融合算法
     * score = Σ 1/(k + rank_i)
     */
    private List<HybridSearchResult> fusionRRF(
            List<SearchHit> denseResults,
            List<SearchHit> sparseResults,
            int topK) {
        
        Map<Long, HybridSearchResult> fusedMap = new HashMap<>();
        
        // 处理稠密检索结果
        for (int i = 0; i < denseResults.size(); i++) {
            SearchHit hit = denseResults.get(i);
            HybridSearchResult result = fusedMap.computeIfAbsent(hit.id, 
                    id -> createResult(hit));
            result.denseRank = i + 1;
            result.denseScore = hit.score;
            result.fusedScore += 1.0 / (RRF_K + i + 1);
        }
        
        // 处理稀疏检索结果
        for (int i = 0; i < sparseResults.size(); i++) {
            SearchHit hit = sparseResults.get(i);
            HybridSearchResult result = fusedMap.computeIfAbsent(hit.id,
                    id -> createResult(hit));
            result.sparseRank = i + 1;
            result.sparseScore = hit.score;
            result.fusedScore += 1.0 / (RRF_K + i + 1);
        }
        
        // 按融合分数排序
        return fusedMap.values().stream()
                .sorted(Comparator.comparingDouble(r -> -r.fusedScore))
                .limit(topK)
                .collect(Collectors.toList());
    }
    
    /**
     * 加权融合
     */
    private List<HybridSearchResult> fusionWeighted(
            List<SearchHit> denseResults,
            List<SearchHit> sparseResults,
            int topK,
            double denseWeight,
            double sparseWeight) {
        
        // 归一化分数
        normalizeScores(denseResults);
        normalizeScores(sparseResults);
        
        Map<Long, HybridSearchResult> fusedMap = new HashMap<>();
        
        // 处理稠密检索结果
        for (int i = 0; i < denseResults.size(); i++) {
            SearchHit hit = denseResults.get(i);
            HybridSearchResult result = fusedMap.computeIfAbsent(hit.id,
                    id -> createResult(hit));
            result.denseRank = i + 1;
            result.denseScore = hit.score;
            result.fusedScore += hit.score * denseWeight;
        }
        
        // 处理稀疏检索结果
        for (int i = 0; i < sparseResults.size(); i++) {
            SearchHit hit = sparseResults.get(i);
            HybridSearchResult result = fusedMap.computeIfAbsent(hit.id,
                    id -> createResult(hit));
            result.sparseRank = i + 1;
            result.sparseScore = hit.score;
            result.fusedScore += hit.score * sparseWeight;
        }
        
        return fusedMap.values().stream()
                .sorted(Comparator.comparingDouble(r -> -r.fusedScore))
                .limit(topK)
                .collect(Collectors.toList());
    }
    
    /**
     * 分数归一化（Min-Max）
     */
    private void normalizeScores(List<SearchHit> hits) {
        if (hits.isEmpty()) return;
        
        double min = hits.stream().mapToDouble(h -> h.score).min().orElse(0);
        double max = hits.stream().mapToDouble(h -> h.score).max().orElse(1);
        double range = max - min;
        
        if (range > 0) {
            for (SearchHit hit : hits) {
                hit.score = (hit.score - min) / range;
            }
        }
    }
    
    /**
     * 创建结果对象
     */
    private HybridSearchResult createResult(SearchHit hit) {
        HybridSearchResult result = new HybridSearchResult();
        result.id = hit.id;
        result.title = hit.title;
        result.content = hit.content;
        result.category = hit.category;
        return result;
    }
    
    /**
     * 截断文本
     */
    private String truncate(String text, int maxLen) {
        if (text == null) return "";
        if (text.length() <= maxLen) return text;
        return text.substring(0, maxLen) + "...";
    }
    
    // 内部类
    private static class SearchHit {
        Long id;
        String title;
        String content;
        String category;
        double score;
        String source;
    }
    
    public static class HybridSearchResult {
        public Long id;
        public String title;
        public String content;
        public String category;
        public double fusedScore;
        public double denseScore;
        public double sparseScore;
        public int denseRank;   // 0 表示未出现在该检索结果中
        public int sparseRank;
        
        public String getMatchSource() {
            if (denseRank > 0 && sparseRank > 0) return "both";
            if (denseRank > 0) return "dense";
            if (sparseRank > 0) return "sparse";
            return "none";
        }
    }
}
