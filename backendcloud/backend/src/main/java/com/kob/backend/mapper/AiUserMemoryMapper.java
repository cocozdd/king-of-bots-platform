package com.kob.backend.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.kob.backend.pojo.AiUserMemory;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

@Mapper
public interface AiUserMemoryMapper extends BaseMapper<AiUserMemory> {

    @Select("SELECT * FROM ai_user_memory WHERE user_id = #{userId}")
    AiUserMemory selectByUserId(@Param("userId") Long userId);

    @Update("UPDATE ai_user_memory SET preferences = #{preferences}, updated_at = CURRENT_TIMESTAMP WHERE user_id = #{userId}")
    int updatePreferences(@Param("userId") Long userId, @Param("preferences") String preferences);

    @Update("UPDATE ai_user_memory SET topics = #{topics}, updated_at = CURRENT_TIMESTAMP WHERE user_id = #{userId}")
    int updateTopics(@Param("userId") Long userId, @Param("topics") String topics);

    @Update("UPDATE ai_user_memory SET profile_summary = #{profileSummary}, updated_at = CURRENT_TIMESTAMP WHERE user_id = #{userId}")
    int updateProfileSummary(@Param("userId") Long userId, @Param("profileSummary") String profileSummary);
}
