package com.kob.backend.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.kob.backend.pojo.AiSession;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.List;

@Mapper
public interface AiSessionMapper extends BaseMapper<AiSession> {

    @Select("SELECT * FROM ai_session WHERE session_id = #{sessionId}")
    AiSession selectBySessionId(@Param("sessionId") String sessionId);

    @Select("SELECT * FROM ai_session WHERE user_id = #{userId} AND status = 'active' ORDER BY last_active_at DESC")
    List<AiSession> selectActiveByUserId(@Param("userId") Long userId);

    @Select("SELECT * FROM ai_session WHERE user_id = #{userId} ORDER BY last_active_at DESC LIMIT #{limit}")
    List<AiSession> selectRecentByUserId(@Param("userId") Long userId, @Param("limit") int limit);

    @Update("UPDATE ai_session SET message_count = message_count + 1, last_active_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE session_id = #{sessionId}")
    int incrementMessageCount(@Param("sessionId") String sessionId);

    @Update("UPDATE ai_session SET status = #{status}, updated_at = CURRENT_TIMESTAMP WHERE session_id = #{sessionId}")
    int updateStatus(@Param("sessionId") String sessionId, @Param("status") String status);

    @Update("UPDATE ai_session SET summary = #{summary}, updated_at = CURRENT_TIMESTAMP WHERE session_id = #{sessionId}")
    int updateSummary(@Param("sessionId") String sessionId, @Param("summary") String summary);

    @Update("UPDATE ai_session SET title = #{title}, updated_at = CURRENT_TIMESTAMP WHERE session_id = #{sessionId}")
    int updateTitle(@Param("sessionId") String sessionId, @Param("title") String title);
}
