# indexer-core

Agent-ready RAG foundation. The current implementation focuses on the core API baseline, persistence model, and the first query graph boundary: `retrieve → generate_answer`.

Implemented so far:

- FastAPI setup, typed settings, logging, error handling, PostgreSQL connectivity, health checks, Docker Compose deployment, and Alembic migration wiring.
- SQLAlchemy domain models for documents, document versions, Qdrant chunk indexes, query runs, evidence, citations, and trace steps.
- A minimal graph runner built around a shared `QueryState`.
- A baseline query graph that routes API questions through graph nodes instead of calling retrieval or generation directly.
- A local/container LLM provider abstraction backed by Ollama.

Still intentionally pending for later Phase 1 steps: ingestion, parsing, chunking, embeddings, vector retrieval, real evidence retrieval, and UI.

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

Run a query through the graph runner:

```bash
curl -X POST http://localhost:8000/api/v1/queries \
  -H "Content-Type: application/json" \
  -d '{"question":"What documents are available?","top_k":5}'
```

Because retrieval/ingestion is not implemented yet, the current retriever returns no evidence. The graph still executes both nodes and returns a safe no-evidence answer plus trace output.

## Ollama

Docker Compose includes an `ollama` service and configures the API container with:

```env
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TIMEOUT_SECONDS=120
```

You still need to pull the configured model into the Ollama volume before evidence-backed generation can use it, for example:

```bash
docker compose exec ollama ollama pull llama3.2
```

For local API development outside Docker, point `OLLAMA_BASE_URL` at a local Ollama process, usually `http://localhost:11434`.

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

## Current query architecture

The query API now uses the graph runner path:

```text
POST /api/v1/queries
        ↓
create query_run
        ↓
GraphRunner(QueryState)
        ↓
retrieve → generate_answer
        ↓
persist answer, evidence, citations, trace_steps
```

Key files:

- `packages/rag_core/query/state.py` — shared `QueryState`, `EvidenceItem`, `CitationItem`, and `TraceEvent`.
- `packages/rag_core/graph/runner.py` — minimal sequential graph runner with trace emission.
- `packages/rag_core/pipelines/baseline.py` — first graph definition: `retrieve → generate_answer`.
- `packages/rag_core/agents/nodes/retrieve.py` — retrieval node using a retriever interface.
- `packages/rag_core/agents/nodes/generate_answer.py` — answer node using an LLM provider interface.
- `packages/rag_core/providers/ollama.py` — Ollama HTTP provider.
- `apps/api/app/services/query_runs.py` — API-side persistence around graph execution.
- `apps/api/app/api/routes/queries.py` — query endpoints.

## Current API surface

- `GET /` — root metadata
- `GET /api/v1/health` — liveness probe
- `GET /api/v1/health/ready` — readiness probe with PostgreSQL check
- `POST /api/v1/queries` — create and execute a query run through the graph runner
- `GET /api/v1/queries/{query_run_id}` — fetch a persisted query run with evidence, citations, and trace
