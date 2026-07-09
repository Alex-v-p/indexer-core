# indexer-core

Agent-ready RAG foundation. The current implementation focuses on the core API baseline, persistence model, first query graph boundary, and basic document ingestion into Qdrant-backed chunk indexes.

Implemented so far:

- FastAPI setup, typed settings, logging, error handling, PostgreSQL connectivity, health checks, Docker Compose deployment, and Alembic migration wiring.
- SQLAlchemy domain models for documents, document versions, Qdrant chunk indexes, query runs, evidence, citations, and trace steps.
- A minimal graph runner built around a shared `QueryState`.
- A baseline query graph that routes API questions through graph nodes instead of calling retrieval or generation directly.
- A local/container LLM provider abstraction backed by Ollama.
- Basic ingestion for PDF, text, and markdown uploads: local file storage, parsing, character-window chunking, deterministic Phase 1 embeddings, Qdrant point upserts, and Postgres chunk-index metadata.

Still intentionally pending for later Phase 1 steps: vector retrieval from Qdrant, answer generation grounded in retrieved chunks, and UI.

## Run with Docker Compose

```bash
cp .env.example .env
# optional: edit .env
docker compose up --build
```

On startup, Compose runs a short-lived `bootstrap` service before the API starts. The bootstrap service waits for PostgreSQL, runs `alembic upgrade head`, and exits successfully. On the first startup this creates the schema. On later startups it checks the Alembic version table and only applies migrations that are still pending.

Docker Compose starts:

- `api` — FastAPI application
- `bootstrap` — one-shot database migration service
- `db` — PostgreSQL
- `qdrant` — vector store used by ingestion
- `ollama` — local LLM runtime for answer generation

API docs: http://localhost:8000/docs

Health checks:

```bash
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/health/ready
```

## Ingest a document

Upload a PDF, text file, or markdown file:

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "title=Example document" \
  -F "file=@./datasets/sample_docs/example.md"
```

List documents:

```bash
curl http://localhost:8000/api/v1/documents
```

Read a document with versions and chunk index metadata:

```bash
curl http://localhost:8000/api/v1/documents/<document_id>
```

Ingestion currently stores original files in the configured local storage directory, parses source text, chunks it, creates deterministic local embeddings, upserts vectors and chunk text into Qdrant payloads, and stores lightweight Qdrant point references in PostgreSQL.

## Run a query through the graph runner

```bash
curl -X POST http://localhost:8000/api/v1/queries \
  -H "Content-Type: application/json" \
  -d '{"question":"What documents are available?","top_k":5}'
```

Vector retrieval is intentionally still pending, so the current retriever returns no evidence. The graph still executes both nodes and returns a safe no-evidence answer plus trace output. The next Phase 1 step should replace the `EmptyRetriever` with a Qdrant-backed vector retriever.

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

For local development outside Docker, make sure `DATABASE_URL` points to a reachable PostgreSQL database and `QDRANT_URL` points to a reachable Qdrant instance.

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

PostgreSQL is the source of truth for application state. Qdrant is the vector index and stores chunk text in point payloads for retrieval. The current database model includes:

- `documents`
- `document_versions`
- `qdrant_chunk_indexes` — lightweight Postgres registry for Qdrant points, not chunk text/vector storage
- `query_runs`
- `evidence`
- `citations`
- `trace_steps`

## Current ingestion architecture

```text
POST /api/v1/documents
        ↓
LocalDocumentStorage
        ↓
parse_document(PDF/text/markdown)
        ↓
chunk_document
        ↓
HashingEmbeddingProvider
        ↓
QdrantVectorStore.upsert_points
        ↓
persist Document, DocumentVersion, QdrantChunkIndex metadata
```

Key files:

- `apps/api/app/api/routes/documents.py` — document upload/list/detail endpoints.
- `apps/api/app/services/document_storage.py` — local source-file storage.
- `apps/api/app/services/document_ingestion.py` — API-side ingestion orchestration and persistence.
- `packages/rag_core/documents/parsers.py` — PDF, text, and markdown parsers.
- `packages/rag_core/documents/chunking.py` — basic chunking and metadata generation.
- `packages/rag_core/documents/models.py` — parser/chunking domain models.
- `packages/rag_core/providers/embeddings.py` — Phase 1 deterministic embedding provider interface.
- `packages/rag_core/providers/vector_store.py` — Qdrant REST adapter.

## Current query architecture

The query API uses the graph runner path:

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

- `packages/rag_core/agents/state.py` — shared `QueryState`, `EvidenceItem`, `CitationItem`, and `TraceEvent`.
- `packages/rag_core/agents/graph.py` — minimal sequential graph runner with trace emission.
- `packages/rag_core/pipelines/baseline.py` — first graph definition: `retrieve → generate_answer`.
- `packages/rag_core/agents/nodes/retrieve.py` — retrieval node using a retriever interface.
- `packages/rag_core/agents/nodes/generate_answer.py` — answer node using an LLM provider interface.
- `packages/rag_core/providers/llm.py` — Ollama HTTP provider.
- `apps/api/app/services/query_runs.py` — API-side persistence around graph execution.
- `apps/api/app/api/routes/queries.py` — query endpoints.

## Current API surface

- `GET /` — root metadata
- `GET /api/v1/health` — liveness probe
- `GET /api/v1/health/ready` — readiness probe with PostgreSQL check
- `POST /api/v1/documents` — upload, parse, chunk, embed, and index a document
- `GET /api/v1/documents` — list ingested documents
- `GET /api/v1/documents/{document_id}` — fetch a document with version and chunk metadata
- `POST /api/v1/queries` — create and execute a query run through the graph runner
- `GET /api/v1/queries/{query_run_id}` — fetch a persisted query run with evidence, citations, and trace
