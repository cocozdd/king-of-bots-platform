#!/bin/bash

# KOB 项目停止脚本

PROJECT_DIR="/Users/cocodzh/Downloads/kob"

echo "=== 停止 KOB 项目 ==="
echo ""

# 读取并停止进程
if [ -f "$PROJECT_DIR/pids/backend.pid" ]; then
    PID=$(cat "$PROJECT_DIR/pids/backend.pid")
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "✅ Backend 已停止 (PID: $PID)"
    fi
    rm "$PROJECT_DIR/pids/backend.pid"
fi

if [ -f "$PROJECT_DIR/pids/matchingsystem.pid" ]; then
    PID=$(cat "$PROJECT_DIR/pids/matchingsystem.pid")
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "✅ MatchingSystem 已停止 (PID: $PID)"
    fi
    rm "$PROJECT_DIR/pids/matchingsystem.pid"
fi

if [ -f "$PROJECT_DIR/pids/botrunningsystem.pid" ]; then
    PID=$(cat "$PROJECT_DIR/pids/botrunningsystem.pid")
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "✅ BotRunningSystem 已停止 (PID: $PID)"
    fi
    rm "$PROJECT_DIR/pids/botrunningsystem.pid"
fi

if [ -f "$PROJECT_DIR/pids/aiservice.pid" ]; then
    PID=$(cat "$PROJECT_DIR/pids/aiservice.pid")
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "✅ AI Service 已停止 (PID: $PID)"
    fi
    rm "$PROJECT_DIR/pids/aiservice.pid"
fi

if [ -f "$PROJECT_DIR/pids/web.pid" ]; then
    PID=$(cat "$PROJECT_DIR/pids/web.pid")
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "✅ Web 前端已停止 (PID: $PID)"
    fi
    rm "$PROJECT_DIR/pids/web.pid"
fi

# 兜底清理
pkill -f "spring-boot:run" 2>/dev/null || true
pkill -f "vue-cli-service serve" 2>/dev/null || true
# 停止 Docker 容器
if docker ps -q --filter "name=kob-postgres" | grep -q .; then
    echo "停止 Postgres 容器..."
    docker stop kob-postgres
fi

echo ""
echo "=== 所有服务已停止 ==="
echo "注意：MySQL 服务 (brew) 仍在运行，如需停止请运行: brew services stop mysql"
