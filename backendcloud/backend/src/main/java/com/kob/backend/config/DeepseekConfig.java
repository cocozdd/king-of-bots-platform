package com.kob.backend.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;

import javax.annotation.PostConstruct;

/**
 * DeepSeek 配置类
 * 将 application.properties 中的配置设置为系统属性，供 DeepseekClient 使用
 */
@Configuration
public class DeepseekConfig {

    @Value("${deepseek.api.key:}")
    private String apiKey;

    @PostConstruct
    public void init() {
        if (apiKey != null && !apiKey.isEmpty()) {
            System.setProperty("deepseek.api.key", apiKey);
        }
    }
}
