# king-of-bots-platform

Public showcase of an AI-augmented King of Bots platform.

This project combines a Java microservice game backend with a Python AI sidecar
for code-assist and strategy workflows.

## Tech stack

- Java: Spring Boot / Spring Cloud
- Python: FastAPI + Agent workflows
- Realtime: WebSocket, SSE
- Data: MySQL / PostgreSQL (pgvector), Redis
- Frontend: Vue

## Project structure

- `backendcloud/`: Java backend services
- `aiservice/`: Python AI service and agent logic
- `web/`: web frontend
- `acapp/`: game app module

## Quick start

```bash
# Start all services
./start-all.sh

# Or use service manager script
./kob-service.sh start
```

Stop services:

```bash
./stop-all.sh
# or
./kob-service.sh stop
```

## Notes

- This repository is a cleaned showcase snapshot for portfolio/demo usage.
- Internal notes, logs, and personal documents are intentionally excluded.
