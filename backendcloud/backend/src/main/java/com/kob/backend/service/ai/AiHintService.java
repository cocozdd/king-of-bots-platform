package com.kob.backend.service.ai;

import com.kob.backend.controller.ai.dto.AiHintRequest;
import com.kob.backend.controller.ai.dto.AiHintResponse;

public interface AiHintService {
    AiHintResponse hint(AiHintRequest request);
    
    /**
     * 同步获取 AI 提示（简化版）
     */
    default String getHintSync(String question) {
        AiHintRequest request = new AiHintRequest();
        request.setQuestion(question);
        AiHintResponse response = hint(request);
        return response != null ? response.getAnswer() : "";
    }
}
