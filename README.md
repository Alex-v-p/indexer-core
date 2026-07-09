# indexer-core

Agent-ready RAG foundation. The current implementation focuses on the core API baseline and persistence model: FastAPI setup, typed settings, logging, error handling, PostgreSQL connectivity, health checks, Docker Compose deployment, Alembic migration wiring, and clear SQLAlchemy domain models.

Core RAG features such as ingestion, parsing, embeddings, retrieval, answer generation, graph execution, and UI are not implemented yet.

## Run with Docker Compose

```bash
cp .env.example .env
# optional: edit .env
docker compose up --build
```

On startup, Compose runs a short-lived `bootstrap` service before the API starts. The bootstrap service waits for PostgreSQL, runs `alembic upgrade head`, and exits successfully. On the first startup this creates the schema. On later startups it checks the Alembic version table and only applies migrations that are still pending.

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

## Database bootstrap and migrations

Docker Compose uses the `bootstrap` service for database initialization and schema upgrades:

```bash
docker compose up bootstrap
```

You normally do not need to run this manually because the API service depends on the bootstrap service completing successfully. For local development outside Docker, run Alembic directly from the repository root after setting `DATABASE_URL`:

```bash
alembic revision --autogenerate -m "describe schema change"
alembic upgrade head
```

## Current persistence model

PostgreSQL is the source of truth for application state. Qdrant will be used later as the vector index only. The current database model includes:

- `documents`
- `document_versions`
- `qdrant_chunk_indexes` — lightweight Postgres registry for Qdrant points, not chunk text/vector storage
- `query_runs`
- `evidence`
- `citations`
- `trace_steps`

## Current API surface

- `GET /` — root metadata
- `GET /api/v1/health` — liveness probe
- `GET /api/v1/health/ready` — readiness probe with PostgreSQL check
