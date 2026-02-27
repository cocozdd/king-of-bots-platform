package com.kob.backend.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.kob.backend.pojo.AiMessage;
import org.apache.ibatis.annotations.Delete;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface AiMessageMapper extends BaseMapper<AiMessage> {

    @Select("SELECT * FROM ai_message WHERE session_id = #{sessionId} ORDER BY created_at ASC")
    List<AiMessage> selectBySessionId(@Param("sessionId") String sessionId);

    @Select("SELECT * FROM ai_message WHERE session_id = #{sessionId} ORDER BY created_at ASC LIMIT #{limit}")
    List<AiMessage> selectBySessionIdWithLimit(@Param("sessionId") String sessionId, @Param("limit") int limit);

    @Select("SELECT * FROM ai_message WHERE session_id = #{sessionId} ORDER BY created_at DESC LIMIT #{limit}")
    List<AiMessage> selectRecentBySessionId(@Param("sessionId") String sessionId, @Param("limit") int limit);

    @Select("SELECT COUNT(*) FROM ai_message WHERE session_id = #{sessionId}")
    int countBySessionId(@Param("sessionId") String sessionId);

    @Delete("DELETE FROM ai_message WHERE session_id = #{sessionId}")
    int deleteBySessionId(@Param("sessionId") String sessionId);

    @Select("SELECT SUM(tokens_used) FROM ai_message WHERE session_id = #{sessionId}")
    Integer sumTokensBySessionId(@Param("sessionId") String sessionId);
}
