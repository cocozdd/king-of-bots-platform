package com.kob.botrunningsystem.service.impl.utils;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

import com.kob.botrunningsystem.executor.ExecutionResult;
import com.kob.botrunningsystem.executor.JavaRunner;
import com.kob.botrunningsystem.executor.Runner;

@Component
public class Consumer extends Thread {
    private Bot bot;
    private static RestTemplate restTemplate;
    private final static String receiveBotMoveUrl = "http://127.0.0.1:3000/pk/receive/bot/move/";
    private final Runner runner = new JavaRunner(2000);

    @Autowired
    public void setRestTemplate(RestTemplate restTemplate) {
        Consumer.restTemplate = restTemplate;
    }

    public void startTimeout(long timeout, Bot bot) {
        this.bot = bot;
        this.start();

        try {
            this.join(timeout);  // 最多等待timeout秒
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            this.interrupt();  // 终端当前线程
        }
    }

    @Override
    public void run() {
        ExecutionResult result = runner.run(bot);
        Integer direction = result.getDirection();
        if (direction == null) {
            direction = 0;  // 兜底，避免阻塞对局
        }
        System.out.println("move-direction: " + bot.getUserId() + " " + direction + " status=" + result.getStatus() + " errorType=" + result.getErrorType());

        MultiValueMap<String, String> data = new LinkedMultiValueMap<>();
        data.add("user_id", bot.getUserId().toString());
        data.add("direction", direction.toString());

        restTemplate.postForObject(receiveBotMoveUrl, data, String.class);
    }
}
