package com.kob.backend.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.web.servlet.config.annotation.AsyncSupportConfigurer;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * 异步配置 - 设置异步请求超时时间
 */
@Configuration
@EnableAsync
public class AsyncConfig implements WebMvcConfigurer {
    
    @Override
    public void configureAsyncSupport(AsyncSupportConfigurer configurer) {
        // 设置异步请求超时时间为 120 秒 (AI 调用可能较慢)
        configurer.setDefaultTimeout(120000);
    }
}
