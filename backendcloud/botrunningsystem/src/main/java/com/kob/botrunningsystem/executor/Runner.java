package com.kob.botrunningsystem.executor;

import com.kob.botrunningsystem.service.impl.utils.Bot;

public interface Runner {
    ExecutionResult run(Bot bot);
}
