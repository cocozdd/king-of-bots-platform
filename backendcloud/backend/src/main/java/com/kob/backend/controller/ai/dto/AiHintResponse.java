package com.kob.backend.controller.ai.dto;

import java.util.List;

public class AiHintResponse {
    private String answer;
    private List<AiSource> sources;
    private String errorMessage;

    public String getErrorMessage() {
        return errorMessage;
    }

    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }

    public String getAnswer() {
        return answer;
    }

    public void setAnswer(String answer) {
        this.answer = answer;
    }

    public List<AiSource> getSources() {
        return sources;
    }

    public void setSources(List<AiSource> sources) {
        this.sources = sources;
    }

    /**
     * 创建错误响应
     */
    public static AiHintResponse error(String errorMessage) {
        AiHintResponse response = new AiHintResponse();
        response.setErrorMessage(errorMessage);
        return response;
    }

    public static class AiSource {
        private String id;
        private String title;
        private String category;

        public AiSource() {}

        public AiSource(String id, String title, String category) {
            this.id = id;
            this.title = title;
            this.category = category;
        }

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public String getTitle() {
            return title;
        }

        public void setTitle(String title) {
            this.title = title;
        }

        public String getCategory() {
            return category;
        }

        public void setCategory(String category) {
            this.category = category;
        }
    }
}
