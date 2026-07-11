# indexer-core

Indexer Core is an agent-ready RAG application for uploading source documents, indexing them into a retrievable knowledge base, and asking evidence-backed questions through a graph-runner API and Angular UI.

Implemented so far:

- FastAPI setup, typed settings, logging, error handling, PostgreSQL connectivity, health checks, Docker Compose deployment, and Alembic migration wiring.
- SQLAlchemy domain models for documents, document versions, Qdrant chunk indexes, query runs, evidence, citations, and trace steps.
- A minimal graph runner built around a shared `QueryState`.
- A baseline query graph that routes API questions through graph nodes instead of calling retrieval or generation directly.
- Local/container provider abstractions backed by Ollama for embeddings, reranking, and answer generation.
- Basic ingestion for PDF, text, and markdown uploads: MinIO object storage, parser staging, character-window chunking, Ollama-backed embeddings, Qdrant point upserts, and Postgres chunk-index metadata.
- Baseline dense-vector retrieval from Qdrant using the same embedding-provider boundary as ingestion.
- Evidence-grounded answer generation with citation metadata, persisted evidence snapshots, and graph trace output.
- Angular UI for uploading documents, viewing indexed documents, asking questions, and inspecting answers, citations/evidence, and graph trace steps.
- A graph-level evaluation harness with versioned JSON datasets, recall@k, MRR, citation hit rate, an answer-faithfulness extension point, and JSON reports.
- Named pipeline and tool registries with a configured default, per-query selection, discovery API, and explicit pipeline-selection trace output.
- A selectable `hybrid_rag` pipeline that combines dense-vector retrieval with BM25 keyword retrieval through weighted reciprocal-rank fusion.
- A selectable `hybrid_rerank_rag` pipeline that expands hybrid candidates, reranks them with an Ollama relevance scorer, and only sends the final top-k evidence to answer generation.

## Run with Docker Compose

```bash
cp .env.example .env
# optional: edit .env
docker compose up --build
```

On startup, Compose runs a short-lived `bootstrap` service before the API starts. The bootstrap service waits for PostgreSQL, runs `alembic upgrade head`, and exits successfully. On the first startup this creates the schema. On later startups it checks the Alembic version table and only applies migrations that are still pending.

Docker Compose starts:

- `web` — Angular UI served by Nginx and proxying `/api/*` to the API container
- `api` — FastAPI application
- `bootstrap` — one-shot database migration service
- `db` — PostgreSQL
- `qdrant` — vector store and chunk-payload corpus used by dense and keyword retrieval
- `minio` — object storage for uploaded source documents
- `ollama` — local runtime for answer generation, embeddings, and reranking

Web UI: http://localhost:4200

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

## Web UI

The Angular app lives in `apps/web` and mirrors the main API surface:

- upload a PDF, text, or markdown document;
- view indexed documents and selected document chunk metadata;
- discover and select a registered retrieval pipeline before asking a question through `POST /api/v1/queries`;
- show the returned answer, citations, evidence snapshots, and execution trace.

Run it with the full stack:

```bash
docker compose up --build
```

Then open http://localhost:4200. In Docker, Nginx serves the compiled Angular app and proxies `/api/*` to the API container.

### Upload size limits

The UI is served through Nginx, so document uploads pass through two limits:

- `WEB_MAX_UPLOAD_SIZE` controls the Nginx proxy limit for browser uploads. The default is `250m`.
- `MAX_UPLOAD_SIZE_MB` controls the API/object-storage safety limit. The default is `250`.

Keep these values aligned when you want to allow larger PDFs. For example, to allow uploads up to 500 MB, set `WEB_MAX_UPLOAD_SIZE=500m` and `MAX_UPLOAD_SIZE_MB=500` in `.env`, then rebuild/restart the stack.

For local Angular development:

```bash
cd apps/web
npm install
npm start
```

The dev server uses `proxy.conf.json`, so browser calls to `/api/v1/*` are forwarded to `http://localhost:8000` without requiring extra CORS settings.

Ingestion stores original source files in MinIO, stages them briefly for parsing, chunks the extracted text, creates embeddings through Ollama by default, upserts vectors and chunk text into Qdrant payloads, and stores lightweight Qdrant point references in PostgreSQL. Set `DOCUMENT_STORAGE_BACKEND=local` or `EMBEDDING_PROVIDER=hashing` only for tests/offline development.

## Run a query through the graph runner

```bash
curl -X POST http://localhost:8000/api/v1/queries \
  -H "Content-Type: application/json" \
  -d '{"question":"What documents are available?","top_k":5,"pipeline_name":"baseline_rag"}'
```

List the currently registered pipelines and their logical tools with:

```bash
curl http://localhost:8000/api/v1/pipelines
```

`pipeline_name` is optional. When it is omitted, `DEFAULT_QUERY_PIPELINE` selects the configured default. The registry exposes:

- `baseline_rag` — embeds the question and performs dense-vector search in Qdrant.
- `hybrid_rag` — retrieves an expanded candidate set from dense-vector search and BM25 lexical search, deduplicates matching chunks, and combines both rankings with weighted reciprocal-rank fusion before answer generation.
- `hybrid_rerank_rag` — retrieves a larger hybrid candidate set, scores candidates for query relevance through the configured Ollama reranker, truncates back to the requested top-k, and then generates the answer.

The keyword provider builds a bounded in-process BM25 index from the chunk text already stored in Qdrant payloads, so existing indexed documents remain usable without a schema migration. Its corpus cache is invalidated after API ingestion and bounded by `KEYWORD_CACHE_TTL_SECONDS`. Every run begins with a `select_pipeline` trace step and persists the selected name/version alongside the answer, evidence, citations, and remaining graph trace. Hybrid evidence metadata includes the contributing vector/keyword ranks, original scores, fusion weights, and final fusion score. Reranked evidence keeps that retrieval metadata and adds the reranker model, original rank/score, and final relevance score. The reranked graph emits a separate `rerank` trace step. When no chunks are retrieved, the graph returns a safe no-evidence answer without calling the LLM.

## Evaluation harness

Evaluation datasets live in `datasets/eval_sets` and use the versioned JSON format documented in `datasets/eval_sets/README.md`. Every case is executed through the complete configured graph (`retrieve → generate_answer` for the current baseline), rather than scoring the retriever in isolation. The generated report keeps the expected answer/evidence beside the actual answer, retrieved chunks, citations, and graph trace.

The current metrics are:

- **Recall@k** — the fraction of separately annotated expected evidence items matched within the configured top-k results.
- **MRR** — the mean reciprocal rank of the first retrieved item matching expected evidence.
- **Citation hit rate** — the fraction of emitted citations whose linked retrieved evidence matches an expected evidence annotation.
- **Answer faithfulness** — an explicit `not_implemented` placeholder behind a replaceable evaluator interface.

A portable demo document and dataset are included. First upload and index the sample document:

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "title=Evaluation demo" \
  -F "file=@./datasets/sample_docs/evaluation_demo.md"
```

Then run the dataset through the API container, which uses the same Ollama and Qdrant configuration as normal queries. Compose mounts `datasets` read-only and writes reports back to the host `reports` directory:

```bash
docker compose exec api \
  python -m scripts.run_evaluation datasets/eval_sets/baseline_demo.json
```

For local API development outside Docker, the same module command works after configuring the local service URLs in `.env`:

```bash
python -m scripts.run_evaluation datasets/eval_sets/baseline_demo.json
```

Useful options:

```bash
python -m scripts.run_evaluation datasets/eval_sets/baseline_demo.json \
  --pipeline baseline_rag \
  --top-k 10 \
  --output reports/evaluations/baseline-top-10.json

python -m scripts.run_evaluation datasets/eval_sets/baseline_demo.json \
  --pipeline hybrid_rag \
  --top-k 10 \
  --output reports/evaluations/hybrid-top-10.json

python -m scripts.run_evaluation datasets/eval_sets/baseline_demo.json \
  --pipeline hybrid_rerank_rag \
  --top-k 10 \
  --output reports/evaluations/hybrid-rerank-top-10.json
```

The command exits non-zero when a case fails to execute, but low metric values remain valid evaluation results. Generated JSON reports are written under `reports/evaluations` by default and are ignored by Git.

## MinIO

Docker Compose includes a `minio` service for durable uploaded source files. The API creates the configured bucket on first upload when it does not already exist. Default Compose values are:

```env
DOCUMENT_STORAGE_BACKEND=minio
MINIO_ENDPOINT=minio:9000
MINIO_BUCKET_NAME=indexer-documents
MINIO_OBJECT_PREFIX=documents
```

MinIO console: http://localhost:9001

For local API development outside Docker, point `MINIO_ENDPOINT` at your local MinIO API endpoint, usually `localhost:9000`.

## Ollama

Docker Compose includes an `ollama` service and configures the API container with:

```env
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_RERANK_MODEL=llama3.2:3b
OLLAMA_TIMEOUT_SECONDS=120
EMBEDDING_PROVIDER=ollama
EMBEDDING_VECTOR_SIZE=768
```

The default configuration uses `llama3.2:3b` for answer generation and pointwise reranking, and `nomic-embed-text` for embeddings. Pull both models into the Ollama volume before ingestion, reranking, and evidence-backed generation:

```bash
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama pull nomic-embed-text
```

For local API development outside Docker, install and start Ollama on your machine, pull the same two models with `ollama pull llama3.2:3b` and `ollama pull nomic-embed-text`, then point `OLLAMA_BASE_URL` at the local Ollama process, usually `http://localhost:11434`.

## Local API development

```bash
cd apps/api
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

For local development outside Docker, make sure `DATABASE_URL` points to PostgreSQL, `QDRANT_URL` points to Qdrant, `MINIO_ENDPOINT` points to MinIO, and `OLLAMA_BASE_URL` points to Ollama. For tests or fully offline API development, set `DOCUMENT_STORAGE_BACKEND=local` and `EMBEDDING_PROVIDER=hashing`.

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
MinioDocumentStorage
        ↓
parse_document(PDF/text/markdown from temporary staging file)
        ↓
chunk_document
        ↓
OllamaEmbeddingProvider
        ↓
QdrantVectorStore.upsert_points
        ↓
persist Document, DocumentVersion, QdrantChunkIndex metadata
```

Key files:

- `apps/api/app/api/routes/documents.py` — document upload/list/detail endpoints.
- `apps/api/app/adapters/object_storage/` — MinIO source-file storage with local test fallback and temporary parser staging.
- `apps/api/app/services/document_ingestion.py` — API-side ingestion orchestration and persistence.
- `packages/rag_core/documents/parsers.py` — PDF, text, and markdown parsers.
- `packages/rag_core/documents/chunking.py` — basic chunking and metadata generation.
- `packages/rag_core/documents/models.py` — parser/chunking domain models.
- `packages/rag_core/providers/embeddings/` — embedding provider interface with Ollama and deterministic hashing implementations.
- `packages/rag_core/providers/vector_stores/` — vector-store interface and Qdrant REST adapter.
- `packages/rag_core/providers/keyword_stores/` — keyword-store contracts, dependency-free BM25 scoring, and Qdrant corpus scrolling.

## Current query architecture

The query API uses a registry-backed graph runner path:

```text
GET /api/v1/pipelines
        ↓
discover PipelineConfig + ToolConfig entries

POST /api/v1/queries (optional pipeline_name)
        ↓
PipelineRegistry selects configured/default pipeline
        ↓
ToolRegistry resolves retriever + optional reranker + generator
        ↓
GraphRunner(QueryState)
        ↓
select_pipeline → retrieve → [rerank] → generate_answer
        ↓
persist answer, evidence, citations, trace_steps
```

Key files:

- `packages/rag_core/agents/state.py` — shared `QueryState`, `EvidenceItem`, `CitationItem`, and `TraceEvent`.
- `packages/rag_core/agents/graph.py` — minimal sequential graph runner with pipeline and node trace emission.
- `packages/rag_core/agents/tools/` — named tool metadata/lookup registry for retrievers, generators, and future rerankers or graders.
- `packages/rag_core/pipelines/base.py` — common retrieval-pipeline protocol and `PipelineConfig` metadata.
- `packages/rag_core/pipelines/registry.py` — default/explicit pipeline selection and factory validation.
- `packages/rag_core/pipelines/baseline.py` — registered dense-vector graph definition: `retrieve → generate_answer`.
- `packages/rag_core/pipelines/hybrid.py` — registered hybrid graph and tool dependencies.
- `packages/rag_core/pipelines/hybrid_rerank.py` — registered hybrid candidate retrieval, reranking, and generation graph.
- `packages/rag_core/agents/nodes/retrieve.py` — retrieval node using a retriever interface.
- `packages/rag_core/agents/nodes/rerank.py` — reranking node that reduces candidate evidence back to the requested top-k.
- `packages/rag_core/retrieval/retrievers/vector.py` — dense-vector retriever that embeds the question and searches Qdrant.
- `packages/rag_core/retrieval/retrievers/keyword.py` — lexical retriever that normalizes keyword-store hits into evidence.
- `packages/rag_core/retrieval/retrievers/hybrid.py` — concurrent candidate retrieval, chunk deduplication, and weighted reciprocal-rank fusion.
- `packages/rag_core/retrieval/rerankers/` — provider-neutral reranker protocol and errors.
- `packages/indexer_infrastructure/ollama/reranker.py` — Ollama structured-output relevance scorer and evidence reordering.
- `packages/rag_core/agents/nodes/generate_answer.py` — answer node using an LLM provider interface and citation-oriented prompt.
- `packages/rag_core/providers/llms/` — LLM provider interface and Ollama HTTP provider.
- `packages/rag_core/providers/vector_stores/` — Qdrant upsert/search adapter used by ingestion and dense retrieval.
- `packages/rag_core/providers/keyword_stores/` — BM25 keyword index and Qdrant payload corpus source.
- `apps/api/app/adapters/keyword_store/` — configured, process-cached keyword provider factory.
- `apps/api/app/services/query_graph.py` — API-side tool registration, pipeline registration, and selected pipeline construction.
- `apps/api/app/services/query_runs.py` — API-side persistence around graph execution.
- `apps/api/app/api/routes/queries.py` — query endpoints with optional `pipeline_name` selection.
- `apps/api/app/api/routes/pipelines.py` — pipeline/tool discovery endpoint used by the UI.

## Current API surface

- `GET /` — root metadata
- `GET /api/v1/health` — liveness probe
- `GET /api/v1/health/ready` — readiness probe with PostgreSQL check
- `POST /api/v1/documents` — upload, parse, chunk, embed, and index a document
- `GET /api/v1/documents` — list ingested documents
- `GET /api/v1/documents/{document_id}` — fetch a document with version and chunk metadata
- `GET /api/v1/pipelines` — list registered pipelines, default selection, and logical tools
- `POST /api/v1/queries` — create and execute a query run through the selected/default pipeline
- `GET /api/v1/queries/{query_run_id}` — fetch a persisted query run with evidence, citations, and trace

## Web app structure

The UI keeps feature code under `apps/web/src/app/features` and shared concerns under `core`/`shared`:

- `core/config` — API base URL configuration.
- `core/http` — API URL interceptor and error normalization.
- `core/layout` — application shell.
- `features/documents` — document API models and data-access service.
- `features/queries` — query API models, data-access service, and the main RAG workbench page.
- `shared/ui` — small reusable UI pieces such as status badges.
