#!/bin/bash

# KOB 项目一键启动脚本（兼容入口，转交 kob-service.sh）

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$SCRIPT_DIR/kob-service.sh" start "$@"
