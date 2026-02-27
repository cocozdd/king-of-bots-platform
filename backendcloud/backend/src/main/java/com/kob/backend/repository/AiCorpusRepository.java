package com.kob.backend.repository;

import com.kob.backend.controller.ai.dto.AiDoc;
import com.kob.backend.controller.ai.dto.AiHintResponse;
import org.postgresql.util.PGobject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.dao.DataAccessException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Repository
public class AiCorpusRepository {

    @Autowired
    @Qualifier("pgvectorJdbcTemplate")
    private JdbcTemplate jdbcTemplate;

    public List<AiHintResponse.AiSource> listTop(int limit) {
        String sql = "SELECT id, title, category FROM ai_corpus ORDER BY id LIMIT ?";
        return jdbcTemplate.query(sql, ps -> ps.setInt(1, limit), (rs, rowNum) ->
                new AiHintResponse.AiSource(
                        rs.getString("id"),
                        rs.getString("title"),
                        rs.getString("category")
                )
        );
    }

    public List<AiHintResponse.AiSource> listSafe(int limit) {
        try {
            return listTop(limit);
        } catch (DataAccessException e) {
            return new ArrayList<>();
        }
    }

    public List<AiDoc> searchByEmbedding(double[] embedding, int limit) {
        // 使用余弦距离 <=> 替代 L2 距离 <->，更适合文本语义相似度
        String sql = "SELECT id, title, category, content, embedding <=> ?::vector AS distance " +
                "FROM ai_corpus ORDER BY distance LIMIT ?";
        PGobject vectorObject = new PGobject();
        vectorObject.setType("vector");
        try {
            vectorObject.setValue(Arrays.stream(embedding)
                    .mapToObj(Double::toString)
                    .collect(Collectors.joining(",", "[", "]")));
        } catch (SQLException e) {
            throw new RuntimeException("Failed to set vector value", e);
        }
        return jdbcTemplate.query(sql, ps -> {
            ps.setObject(1, vectorObject);
            ps.setInt(2, limit);
        }, (rs, rowNum) -> new AiDoc(
                rs.getString("id"),
                rs.getString("title"),
                rs.getString("category"),
                rs.getString("content")
        ));
    }

    public List<AiDoc> searchSafe(double[] embedding, int limit) {
        try {
            return searchByEmbedding(embedding, limit);
        } catch (DataAccessException e) {
            return new ArrayList<>();
        }
    }
}
