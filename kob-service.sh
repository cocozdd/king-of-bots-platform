#!/bin/bash

# ===========================================
# KOB 服务管理脚本 (Phase 4.1)
# 支持: 启动、停止、定时关闭、状态查看
# 
# 2026-01 更新:
# - 添加 Redis 检查（会话持久化 + 缓存）
# - 支持 LangGraph Checkpointer（Postgres 状态管理）
# ===========================================

set -e

PROJECT_DIR="/Users/cocodzh/Downloads/kob"
PID_DIR="$PROJECT_DIR/pids"
LOG_DIR="$PROJECT_DIR/logs"
AI_ENV_FILE="${AI_ENV_FILE:-$HOME/.kob/ai.env}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 确保目录存在
mkdir -p "$PID_DIR" "$LOG_DIR"

# ===========================================
# 辅助函数
# ===========================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

get_service_port() {
    case "$1" in
        backend) echo "3000" ;;
        matchingsystem) echo "3001" ;;
        botrunningsystem) echo "3002" ;;
        aiservice) echo "3003" ;;
        web) echo "8080" ;;
        *) echo "" ;;
    esac
}

get_port_pid() {
    local port="$1"
    lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -n 1
}

# 检查服务是否运行
is_running() {
    local service="$1"
    local pid_file="$PID_DIR/$service.pid"

    # 优先检查 pid 文件
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi

    # pid 文件失效时回退到端口检查，并自动修复 pid 文件
    local port
    port=$(get_service_port "$service")
    if [ -n "$port" ]; then
        local port_pid
        port_pid=$(get_port_pid "$port")
        if [ -n "$port_pid" ]; then
            echo "$port_pid" > "$pid_file"
            return 0
        fi
    fi

    return 1
}

# 获取服务 PID
get_pid() {
    local service="$1"
    local pid_file="$PID_DIR/$service.pid"

    if is_running "$service" && [ -f "$pid_file" ]; then
        cat "$pid_file"
    fi
}

# 加载 AI 环境变量（用于后端/aiservice）
load_ai_env() {
    local env_file="${AI_ENV_FILE:-$HOME/.kob/ai.env}"
    local local_env="$PROJECT_DIR/aiservice/.env"

    if [ -f "$env_file" ]; then
        set -a
        # shellcheck disable=SC1090
        source "$env_file"
        set +a
        export AI_ENV_LOADED=1
        log_info "已加载 AI_ENV_FILE: $env_file"
        return 0
    fi

    if [ -f "$local_env" ]; then
        set -a
        # shellcheck disable=SC1090
        source "$local_env"
        set +a
        export AI_ENV_LOADED=1
        log_info "已加载 aiservice/.env"
        return 0
    fi

    log_warn "未找到 AI 环境文件: $env_file 或 $local_env"
    return 0
}

# 配置 Java 环境
ensure_java() {
    log_info "配置 Java 环境..."

    local java_home_candidate=""
    if [ -x "/usr/libexec/java_home" ]; then
        set +e
        java_home_candidate=$(/usr/libexec/java_home -v 17 2>/dev/null)
        set -e
        if [ -n "$java_home_candidate" ]; then
            export JAVA_HOME="$java_home_candidate"
            log_info "自动检测到 Java 17: $JAVA_HOME"
        fi
    fi

    if [ -z "$JAVA_HOME" ] || [ ! -d "$JAVA_HOME" ]; then
        if [ -d "/opt/homebrew/Cellar/openjdk@17/17.0.17/libexec/openjdk.jdk/Contents/Home" ]; then
            export JAVA_HOME="/opt/homebrew/Cellar/openjdk@17/17.0.17/libexec/openjdk.jdk/Contents/Home"
            log_info "检测到 Homebrew OpenJDK 17 (Direct): $JAVA_HOME"
        else
            local brew_java_candidate=""
            brew_java_candidate=$(find /opt/homebrew/Cellar/openjdk@17 -name "Home" -type d 2>/dev/null | head -n 1)
            if [ -n "$brew_java_candidate" ]; then
                export JAVA_HOME="$brew_java_candidate"
                log_info "检测到 Homebrew OpenJDK 17 (Find): $JAVA_HOME"
            fi
        fi
    fi

    if [ -z "$JAVA_HOME" ] || [ ! -d "$JAVA_HOME" ]; then
        if [ -d "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home" ]; then
            export JAVA_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
            log_info "使用 Homebrew Java 17 (Symlink): $JAVA_HOME"
        elif [ -d "/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home" ]; then
            export JAVA_HOME="/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
            log_info "使用 Homebrew Java 17 (Intel): $JAVA_HOME"
        fi
    fi

    if [ -n "$JAVA_HOME" ] && [ -x "$JAVA_HOME/bin/java" ]; then
        export PATH="$JAVA_HOME/bin:$PATH"
    else
        log_warn "未找到明确的 Java 17 路径，尝试使用系统默认 java..."
    fi

    local java_bin=""
    if [ -x "$JAVA_HOME/bin/java" ]; then
        java_bin="$JAVA_HOME/bin/java"
    elif command -v java > /dev/null 2>&1; then
        java_bin="$(command -v java)"
    fi

    if [ -z "$java_bin" ]; then
        log_error "错误: 无法执行 'java' 命令！"
        log_error "请确保已安装 Java 17 (推荐: brew install openjdk@17)"
        exit 1
    fi

    local java_ver=""
    java_ver=$("$java_bin" -version 2>&1 | head -n 1)
    log_info "Java 环境就绪: $java_ver"
}

# 配置 Python 虚拟环境
ensure_python_env() {
    if [ ! -d "$PROJECT_DIR/aiservice/venv" ]; then
        log_info "创建 Python 虚拟环境..."
        cd "$PROJECT_DIR/aiservice"
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt -q
        deactivate
        cd "$PROJECT_DIR"
    else
        log_info "Python 虚拟环境已存在"
    fi
}

# 检查 MySQL
ensure_mysql() {
    log_info "检查 MySQL..."
    if ! pgrep -x mysqld > /dev/null; then
        log_warn "MySQL 未运行，正在启动..."
        if command -v mysql.server > /dev/null; then
            mysql.server start
        else
            brew services start mysql
        fi
        sleep 5
    fi
    log_info "MySQL 运行中"
}

# 检查 Postgres (Docker)
ensure_postgres() {
    log_info "[Postgres] 检查 Postgres (AI 向量数据库)..."
    cd "$PROJECT_DIR"

    if ! docker info > /dev/null 2>&1; then
        log_error "Docker 未运行！本项目依赖 Docker 运行 Postgres。"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            log_warn "正在尝试启动 Docker Desktop..."
            open -a Docker
        fi
        exit 1
    fi

    if ! docker compose ps --services --filter "status=running" | grep -q "postgres"; then
        log_info "[Postgres] 正在启动 Postgres (via Docker)..."
        docker compose up -d postgres

        log_info "[Postgres] 等待数据库就绪..."
        local count=0
        while ! nc -z localhost 15432 && [ $count -lt 30 ]; do
            sleep 1
            count=$((count+1))
        done

        if [ $count -ge 30 ]; then
            log_warn "Postgres 启动超时 (30s)，请检查 docker logs kob-postgres"
        else
            log_info "Postgres 已就绪 (端口 15432)"
        fi
    else
        log_info "Postgres 运行中 (Docker)"
    fi
}

# 检查 MySQL 数据库
ensure_mysql_db() {
    log_info "[MySQL] 检查数据库..."
    if ! mysql -u root -p123456 -e "USE kob" 2>/dev/null; then
        log_warn "数据库未初始化，正在初始化..."
        mysql -u root -p123456 < "$PROJECT_DIR/backendcloud/db/init.sql"
        log_info "数据库初始化完成"
    else
        log_info "数据库已存在"
    fi
}

# 检查 Redis (Phase 4.1 新增)
ensure_redis() {
    log_info "[Redis] 检查 Redis（会话持久化 + 缓存）..."
    
    # 1. 尝试直接连接 (适用于本地安装或Docker端口映射)
    if command -v redis-cli > /dev/null 2>&1; then
        if redis-cli ping > /dev/null 2>&1; then
            log_info "Redis 运行中 (via redis-cli)"
            return 0
        fi
    fi

    # 2. 检查 Docker 容器
    if command -v docker > /dev/null 2>&1; then
        # 检查是否有正在运行的 redis 容器 (kob-redis 或 compose 服务名 redis)
        if docker compose ps --services --filter "status=running" | grep -q "redis"; then
            log_info "Redis 运行中 (Docker service: redis)"
            return 0
        fi
        
        # 检查是否有 kob-redis 容器运行中
        if docker ps --format '{{.Names}}' | grep -q "^kob-redis$"; then
            log_info "Redis 运行中 (Container: kob-redis)"
            return 0
        fi
        
        log_warn "Redis 未运行，尝试通过 Docker 启动..."
        docker compose up -d redis
        
        # 等待启动
        local count=0
        while [ $count -lt 10 ]; do
            if docker compose ps --services --filter "status=running" | grep -q "redis"; then
                log_info "Redis 已启动 (Docker)"
                return 0
            fi
            sleep 1
            count=$((count+1))
        done
    fi
    
    # 3. 尝试本地启动 (如果 Docker 失败或不可用)
    log_warn "尝试本地启动 Redis..."
    
    if command -v brew > /dev/null 2>&1; then
        brew services start redis 2>/dev/null || true
        sleep 2
    elif command -v redis-server > /dev/null 2>&1; then
        redis-server --daemonize yes 2>/dev/null || true
        sleep 2
    fi
    
    # 4. 最终检查
    if command -v redis-cli > /dev/null 2>&1 && redis-cli ping > /dev/null 2>&1; then
        log_info "Redis 已启动 (Local)"
    else
        log_warn "⚠ Redis 未安装或无法启动，AI 服务将使用内存存储（降级模式）"
        log_warn "建议: docker compose up -d redis 或 brew install redis"
    fi
}

# ===========================================
# 启动 AI 服务 (仅 Python)
# ===========================================

start_aiservice() {
    log_info "启动 AI Service (端口 3003)..."

    if is_running "aiservice"; then
        log_warn "AI Service 已在运行 (PID: $(get_pid aiservice), 端口 3003)，跳过启动"
        return 0
    fi

    if [ -z "${AI_ENV_LOADED:-}" ]; then
        load_ai_env
    fi

    ensure_python_env
    cd "$PROJECT_DIR/aiservice"
    source venv/bin/activate
    
    # 启动服务
    nohup python app.py > "$LOG_DIR/aiservice.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_DIR/aiservice.pid"
    
    deactivate
    cd "$PROJECT_DIR"
    
    log_info "AI Service 已启动 (PID: $pid)"
}

# ===========================================
# 启动后端与前端服务
# ===========================================

start_backend() {
    log_info "启动 backend (端口 3000)..."

    if is_running "backend"; then
        log_warn "backend 已在运行 (PID: $(get_pid backend), 端口 3000)，跳过启动"
        return 0
    fi

    cd "$PROJECT_DIR/backendcloud"
    export JAVA_HOME
    nohup ./mvnw spring-boot:run -pl backend -Dmaven.test.skip=true > "$LOG_DIR/backend.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_DIR/backend.pid"
    cd "$PROJECT_DIR"
    log_info "backend 已启动 (PID: $pid)"
}

start_matchingsystem() {
    log_info "启动 matchingsystem (端口 3001)..."

    if is_running "matchingsystem"; then
        log_warn "matchingsystem 已在运行 (PID: $(get_pid matchingsystem), 端口 3001)，跳过启动"
        return 0
    fi

    cd "$PROJECT_DIR/backendcloud"
    export JAVA_HOME
    nohup ./mvnw spring-boot:run -pl matchingsystem -Dmaven.test.skip=true > "$LOG_DIR/matchingsystem.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_DIR/matchingsystem.pid"
    cd "$PROJECT_DIR"
    log_info "matchingsystem 已启动 (PID: $pid)"
}

start_botrunningsystem() {
    log_info "启动 botrunningsystem (端口 3002)..."

    if is_running "botrunningsystem"; then
        log_warn "botrunningsystem 已在运行 (PID: $(get_pid botrunningsystem), 端口 3002)，跳过启动"
        return 0
    fi

    cd "$PROJECT_DIR/backendcloud"
    export JAVA_HOME
    nohup ./mvnw spring-boot:run -pl botrunningsystem -Dmaven.test.skip=true > "$LOG_DIR/botrunningsystem.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_DIR/botrunningsystem.pid"
    cd "$PROJECT_DIR"
    log_info "botrunningsystem 已启动 (PID: $pid)"
}

start_web() {
    log_info "启动前端服务 (端口 8080)..."

    if is_running "web"; then
        log_warn "web 已在运行 (PID: $(get_pid web), 端口 8080)，跳过启动"
        return 0
    fi

    cd "$PROJECT_DIR/web"
    if [ ! -d "node_modules" ]; then
        log_info "安装前端依赖..."
        npm install
    fi
    nohup npm run serve > "$LOG_DIR/web.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_DIR/web.pid"
    cd "$PROJECT_DIR"
    log_info "前端已启动 (PID: $pid)"
}

# ===========================================
# 启动所有服务
# ===========================================

start_all() {
    local timeout=${1:-15}

    echo ""
    echo -e "${BLUE}=== KOB 项目启动 ===${NC}"
    echo ""

    if [ -f "$PID_DIR/timer.pid" ]; then
        cancel_timer
    fi

    ensure_java
    load_ai_env
    ensure_python_env

    echo ""
    log_info "检查数据库和缓存..."
    ensure_mysql
    ensure_postgres
    ensure_mysql_db
    ensure_redis

    echo ""
    log_info "启动后端服务..."
    start_backend
    sleep 10
    start_matchingsystem
    start_botrunningsystem
    start_aiservice

    echo ""
    log_info "启动前端服务..."
    start_web

    echo ""
    log_info "启动完成"
    echo ""
    echo "服务状态："
    echo "  - Backend:          http://localhost:3000 (PID: $(get_pid backend))"
    echo "  - MatchingSystem:   http://localhost:3001 (PID: $(get_pid matchingsystem))"
    echo "  - BotRunningSystem: http://localhost:3002 (PID: $(get_pid botrunningsystem))"
    echo "  - AI Service:       http://localhost:3003 (PID: $(get_pid aiservice))"
    echo "  - Web:              http://localhost:8080 (PID: $(get_pid web))"
    echo ""
    echo "日志文件："
    echo "  - Backend:          $LOG_DIR/backend.log"
    echo "  - MatchingSystem:   $LOG_DIR/matchingsystem.log"
    echo "  - BotRunningSystem: $LOG_DIR/botrunningsystem.log"
    echo "  - AI Service:       $LOG_DIR/aiservice.log"
    echo "  - Web:              $LOG_DIR/web.log"

    if [ "$timeout" -gt 0 ]; then
        echo ""
        log_info "⏳ 将在 $timeout 分钟后自动停止所有服务"
        start_timer "$timeout"
        log_info "立即停止: ./kob-service.sh stop"
        log_info "取消定时: ./kob-service.sh cancel-timer"
    fi
}

# ===========================================
# 停止单个服务
# ===========================================

stop_service() {
    local service=$1
    local pid_file="$PID_DIR/$service.pid"

    local stopped=0
    local pid=""
    local port
    port=$(get_service_port "$service")

    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
            kill "$pid" 2>/dev/null || true
            stopped=1
            log_info "已停止 $service (PID: $pid)"
        fi
    fi

    # pid 文件失效时，按端口兜底停止（mvnw 子进程场景）
    if [ -n "$port" ]; then
        local port_pid
        port_pid=$(get_port_pid "$port")
        if [ -n "$port_pid" ] && [ "$port_pid" != "$pid" ]; then
            kill "$port_pid" 2>/dev/null || true
            stopped=1
            log_info "已停止 $service (端口 $port, PID: $port_pid)"
        fi
    fi

    rm -f "$pid_file"

    if [ "$stopped" -eq 0 ]; then
        log_warn "$service 未运行"
    fi
}

# ===========================================
# 停止所有服务
# ===========================================

stop_all() {
    log_info "停止所有服务..."
    
    # 停止所有已知服务
    for service in backend matchingsystem botrunningsystem aiservice web; do
        stop_service "$service"
    done
    
    # 杀死定时器进程
    if [ -f "$PID_DIR/timer.pid" ]; then
        local timer_pid=$(cat "$PID_DIR/timer.pid")
        kill "$timer_pid" 2>/dev/null || true
        rm -f "$PID_DIR/timer.pid"
        log_info "已取消定时关闭"
    fi
    
    # 额外清理：查找并杀死相关进程
    pkill -f "spring-boot:run -pl backend" 2>/dev/null || true
    pkill -f "spring-boot:run -pl matchingsystem" 2>/dev/null || true
    pkill -f "spring-boot:run -pl botrunningsystem" 2>/dev/null || true
    pkill -f "aiservice/app.py" 2>/dev/null || true
    pkill -f "npm run serve" 2>/dev/null || true
    
    log_info "所有服务已停止"
}

# ===========================================
# 查看状态
# ===========================================

show_status() {
    echo ""
    echo -e "${BLUE}=== KOB 服务状态 ===${NC}"
    echo ""
    
    local services=("backend:3000" "matchingsystem:3001" "botrunningsystem:3002" "aiservice:3003" "web:8080")
    
    for item in "${services[@]}"; do
        local service="${item%%:*}"
        local port="${item##*:}"
        
        if is_running "$service"; then
            local pid=$(get_pid "$service")
            echo -e "  ${GREEN}●${NC} $service (端口 $port) - 运行中 (PID: $pid)"
        else
            echo -e "  ${RED}○${NC} $service (端口 $port) - 已停止"
        fi
    done
    
    # 显示定时器状态
    echo ""
    if [ -f "$PID_DIR/timer.pid" ]; then
        local timer_pid=$(cat "$PID_DIR/timer.pid")
        if ps -p "$timer_pid" > /dev/null 2>&1; then
            echo -e "  ${YELLOW}⏰${NC} 定时关闭已设置"
        fi
    fi
    
    echo ""
}

# ===========================================
# 定时关闭
# ===========================================

start_timer() {
    local minutes=${1:-15}
    local seconds=$((minutes * 60))
    
    log_info "设置 $minutes 分钟后自动关闭..."
    
    # 后台启动定时器
    (
        sleep "$seconds"
        log_warn "⏰ 定时器触发，正在停止所有服务..."
        "$0" stop
    ) &
    
    local timer_pid=$!
    echo "$timer_pid" > "$PID_DIR/timer.pid"
    
    log_info "定时器已启动 (PID: $timer_pid)"
}

# ===========================================
# 取消定时器
# ===========================================

cancel_timer() {
    if [ -f "$PID_DIR/timer.pid" ]; then
        local timer_pid=$(cat "$PID_DIR/timer.pid")
        kill "$timer_pid" 2>/dev/null || true
        rm -f "$PID_DIR/timer.pid"
        log_info "定时关闭已取消"
    else
        log_warn "没有活动的定时器"
    fi
}

# ===========================================
# 仅启动 AI 服务并测试
# ===========================================

start_ai_only() {
    local timeout=${1:-15}
    
    echo ""
    echo -e "${BLUE}=== 启动 AI 服务测试模式 ===${NC}"
    echo ""

    if [ -f "$PID_DIR/timer.pid" ]; then
        cancel_timer
    fi

    # 停止已有的 AI 服务
    stop_service "aiservice"
    
    # 检查依赖服务 (Phase 4.1)
    ensure_postgres  # LangGraph Checkpointer 需要
    ensure_redis     # SessionStore + EmbeddingCache 需要
    
    # 启动 AI 服务
    if [ -z "${AI_ENV_LOADED:-}" ]; then
        load_ai_env
    fi
    start_aiservice
    
    # 等待启动
    sleep 3
    
    # 测试健康检查
    log_info "测试健康检查..."
    if curl -s http://localhost:3003/health > /dev/null 2>&1; then
        log_info "✅ AI Service 健康检查通过"
        curl -s http://localhost:3003/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:3003/health
    else
        log_error "❌ AI Service 健康检查失败"
        log_info "查看日志: tail -f $LOG_DIR/aiservice.log"
        return 1
    fi
    
    echo ""
    
    # 设置定时关闭
    if [ "$timeout" -gt 0 ]; then
        start_timer "$timeout"
        echo ""
        log_info "服务将在 $timeout 分钟后自动关闭"
        log_info "立即停止: ./kob-service.sh stop"
        log_info "取消定时: ./kob-service.sh cancel-timer"
    fi
    
    echo ""
    log_info "AI Service 地址: http://localhost:3003"
    log_info "查看日志: tail -f $LOG_DIR/aiservice.log"
    echo ""
}

# ===========================================
# 显示帮助
# ===========================================

show_help() {
    echo ""
    echo -e "${BLUE}KOB 服务管理脚本${NC}"
    echo ""
    echo "用法: $0 <命令> [参数]"
    echo ""
    echo "命令:"
    echo "  start [分钟]      启动所有服务，可选定时关闭（默认15分钟）"
    echo "  start-ai [分钟]   仅启动 AI 服务（含 Postgres/Redis）"
    echo "  stop              立即停止所有服务"
    echo "  restart           重启所有服务"
    echo "  status            查看服务状态"
    echo "  timer <分钟>      设置定时关闭"
    echo "  cancel-timer      取消定时关闭"
    echo "  logs [服务名]     查看日志"
    echo "  test              运行 Phase 4 测试"
    echo ""
    echo "依赖服务 (Phase 4.1):"
    echo "  - Redis:    会话持久化 + Embedding 缓存"
    echo "  - Postgres: LangGraph Checkpointer 状态管理"
    echo ""
    echo "示例:"
    echo "  $0 start-ai 30    启动 AI 服务，30分钟后自动关闭"
    echo "  $0 stop           立即停止所有服务"
    echo "  $0 timer 10       10分钟后关闭"
    echo "  $0 logs aiservice 查看 AI 服务日志"
    echo ""
}

# ===========================================
# 查看日志
# ===========================================

show_logs() {
    local service=${1:-aiservice}
    local log_file="$LOG_DIR/$service.log"
    
    if [ -f "$log_file" ]; then
        tail -f "$log_file"
    else
        log_error "日志文件不存在: $log_file"
    fi
}

# ===========================================
# 运行测试
# ===========================================

run_tests() {
    log_info "运行 Phase 4 测试..."
    cd "$PROJECT_DIR"
    python3 -m pytest test/test_phase4.py -v --tb=short
}

# ===========================================
# 主入口
# ===========================================

case "${1:-}" in
    start)
        start_all "${2:-15}"
        ;;
    start-ai)
        start_ai_only "${2:-15}"
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        sleep 2
        start_all "${2:-15}"
        ;;
    status)
        show_status
        ;;
    timer)
        if [ -z "$2" ]; then
            log_error "请指定分钟数，如: $0 timer 10"
            exit 1
        fi
        start_timer "$2"
        ;;
    cancel-timer)
        cancel_timer
        ;;
    logs)
        show_logs "${2:-aiservice}"
        ;;
    test)
        run_tests
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        show_help
        exit 1
        ;;
esac
