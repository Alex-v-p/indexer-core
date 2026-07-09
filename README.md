# indexer-core

Agent-ready RAG foundation. The current implementation intentionally focuses on the core API baseline only: FastAPI setup, typed settings, logging, error handling, PostgreSQL connectivity, health checks, Docker Compose deployment, and Alembic migration wiring.

Core RAG features such as ingestion, parsing, chunking, embeddings, retrieval, answer generation, graph execution, and UI are not implemented yet.

## Run with Docker Compose

```bash
cp .env.example .env
# optional: edit .env
docker compose up --build
```

API docs: http://localhost:8000/docs

Health checks:

```bash
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/health/ready
```

## Local API development

```bash
cd apps/api
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

For local development outside Docker, make sure `DATABASE_URL` points to a reachable PostgreSQL database.

## Database migrations

Alembic is wired, but there are no domain models yet.

```bash
alembic revision --autogenerate -m "create initial tables"
alembic upgrade head
```

## Current API surface

- `GET /` — root metadata
- `GET /api/v1/health` — liveness probe
- `GET /api/v1/health/ready` — readiness probe with PostgreSQL check
