# indexer-core

Indexer Core is an agent-ready RAG application for uploading source documents, indexing them into a retrievable knowledge base, and asking evidence-backed questions through a graph-runner API and Angular UI.

Implemented so far:

- FastAPI setup, typed settings, logging, error handling, PostgreSQL connectivity, health checks, Docker Compose deployment, and Alembic migration wiring.
- SQLAlchemy domain models for documents, document versions, Qdrant chunk indexes, query runs, evidence, citations, and trace steps.
- A minimal graph runner built around a shared `QueryState`.
- A baseline query graph that routes API questions through graph nodes instead of calling retrieval or generation directly.
- Local/container provider abstractions backed by Ollama for embeddings, query expansion, reranking, and answer generation.
- Basic ingestion for PDF, text, and markdown uploads: MinIO object storage, parser staging, character-window chunking, Ollama-backed embeddings, Qdrant point upserts, and Postgres chunk-index metadata.
- Baseline dense-vector retrieval from Qdrant using the same embedding-provider boundary as ingestion.
- Evidence-grounded answer generation with citation metadata, persisted evidence snapshots, and graph trace output.
- Angular UI for uploading documents, viewing indexed documents, asking questions, and inspecting answers, citations/evidence, and graph trace steps.
- A graph-level evaluation harness with versioned JSON datasets, recall@k, MRR, citation hit rate, an answer-faithfulness extension point, and JSON reports.
- Named pipeline and tool registries with a configured default, per-query selection, discovery API, and explicit pipeline-selection trace output.
- A selectable `hybrid_rag` pipeline that combines dense-vector retrieval with BM25 keyword retrieval through weighted reciprocal-rank fusion.
- A selectable `hybrid_llm_rerank_rag` pipeline that expands hybrid candidates, reranks them with a resilient Ollama relevance scorer, and only sends the final top-k evidence to answer generation.
- A selectable `hybrid_cross_encoder_rerank_rag` pipeline that uses a dedicated local Sentence Transformers cross-encoder for deterministic query/passage scoring.
- Opt-in neighborhood-aware chunk contextualization that stores original and contextual named vectors on the same Qdrant point and exposes a selectable `contextual_rag` comparison pipeline.
- A selectable `multi_query_rag` pipeline that generates intent-preserving query variants with the configured Ollama model, runs hybrid retrieval for each query concurrently, deduplicates chunks, and fuses the rankings with weighted reciprocal-rank fusion.
- Query classification as the first graph node in every pipeline, covering factual lookups, broad explanations, comparisons, and version-specific questions while detecting likely metadata-filter dimensions.

## Run with Docker Compose

```bash
cp .env.example .env
# optional: edit .env
docker compose up --build
```

On startup, Compose runs two short-lived bootstrap services before the API starts. `bootstrap` waits for PostgreSQL, runs `alembic upgrade head`, and exits successfully. `cross-encoder-bootstrap` ensures the configured cross-encoder snapshot exists in the persistent `cross_encoder_cache` Docker volume. The model service contacts Hugging Face only when the configured model/revision is missing or a forced refresh is requested; later startups validate the local manifest and exit without a network request.

Docker Compose starts:

- `web` — Angular UI served by Nginx and proxying `/api/*` to the API container
- `api` — FastAPI application
- `bootstrap` — one-shot database migration service
- `cross-encoder-bootstrap` — one-shot model downloader that populates the named cross-encoder volume
- `db` — PostgreSQL
- `qdrant` — vector store and chunk-payload corpus used by dense and keyword retrieval
- `minio` — object storage for uploaded source documents
- `ollama` — local runtime for answer generation, embeddings, query expansion, and optional LLM-based reranking

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

Ingestion stores original source files in MinIO, stages them briefly for parsing, chunks the extracted text, creates embeddings through Ollama by default, upserts vectors and chunk text into Qdrant payloads, and stores lightweight Qdrant point references in PostgreSQL. When contextualization is enabled, each chunk also receives a short neighborhood-aware description and a second named embedding on the same Qdrant point. Set `DOCUMENT_STORAGE_BACKEND=local` or `EMBEDDING_PROVIDER=hashing` only for tests/offline development.

## Run a query through the graph runner

```bash
curl -X POST http://localhost:8000/api/v1/queries \
  -H "Content-Type: application/json" \
  -d '{"question":"What documents are available?","top_k":5}'
```

List the currently registered pipelines and their logical tools with:

```bash
curl http://localhost:8000/api/v1/pipelines
```

`pipeline_name` is optional. When it is omitted, `DEFAULT_QUERY_PIPELINE` selects the configured default. The default is now `agentic_rag`, which uses a readable top-level query graph plus a reusable cyclic information-need subgraph. The top-level graph classifies and decomposes the request, initializes one bounded work item per information need, invokes the subgraph until the queue is empty, aggregates grader-approved evidence, and generates a complete or explicitly partial answer. Explicit pipeline selection remains available for controlled evaluation. The registry exposes:

- `agentic_rag` — independently classifies, plans, retrieves, grades, and retries each decomposed information need. Every item has its own pipeline choice, query, top-k, evidence references, classification/plan history, and retry budget; a query-level attempt cap prevents compound requests from growing without bound.
- `baseline_rag` — embeds the question and performs dense-vector search in Qdrant.
- `hybrid_rag` — retrieves an expanded candidate set from dense-vector search and BM25 lexical search, deduplicates matching chunks, and combines both rankings with weighted reciprocal-rank fusion before answer generation.
- `hybrid_llm_rerank_rag` — retrieves a larger hybrid candidate set, scores candidates through the configured Ollama LLM reranker, retries incomplete structured responses as single-candidate requests, falls back to original hybrid order only for candidates that remain unscored, and then generates the answer.
- `hybrid_cross_encoder_rerank_rag` — retrieves the same expanded hybrid candidate set and reranks it with a dedicated local query/passage cross-encoder loaded from the persistent Docker volume before answer generation.
- `contextual_rag` — searches the `contextual` named vector and `contextualized_text` BM25 field in the same `indexer_chunks` collection, fuses the rankings with weighted RRF, and still returns the untouched original chunk text for answers and citations.
- `multi_query_rag` — asks the configured Ollama model for alternative search formulations, includes the original question by default, runs the normal hybrid retriever for every query concurrently, deduplicates chunks, and fuses cross-query rankings with weighted RRF before answer generation.

The keyword provider builds separate bounded in-process BM25 indexes from the `text` and `contextualized_text` payload fields, while evidence always uses the original `text` field. Points without a contextual representation remain available to normal pipelines but are intentionally excluded from contextual BM25 retrieval. Its corpus caches are invalidated after API ingestion and bounded by `KEYWORD_CACHE_TTL_SECONDS`. Every run begins with `select_pipeline`. Agentic runs then emit top-level trace steps for `classify_query`, `decompose_information_needs`, `initialize_information_need_work`, `resolve_information_needs`, `aggregate_information_needs`, and `generate_answer`. Inside `resolve_information_needs`, the trace records the named subgraph cycle `select_information_need → classify_information_need → plan_information_need → execute_information_need_plan → grade_information_need → decide_information_need → complete_information_need`. Weak evidence routes only the active item back to planning; a low-confidence missing result may route that item back through classification; supported or exhausted items return control to the queue. Every subgraph trace event includes its graph name, information-need id, attempt number, and graph depth. Hybrid, multi-query, and reranked evidence retain their detailed retrieval metadata. Grader-rejected chunks remain inspectable in the attempt trace but are removed from the item evidence index and are not persisted or passed into answer generation. If at least one required information need is supported, the LLM answers that supported subset and the runtime appends an explicit unresolved-information notice. Generation is blocked without calling the LLM only when no required information need is fully supported.

## Query classification

Query classification is implemented as a query-understanding capability and invoked by a shared agent node, rather than being tied to one retrieval pipeline. The `query_understanding` package owns classification contracts, models, LLM parsing, and deterministic fallback rules; the agent node only updates `QueryState` and records trace metadata. It classifies each question as one of:

- `factual_lookup` — focused facts, values, definitions, or direct details;
- `broad_explanation` — overviews, summaries, processes, reasoning, or implications;
- `comparison` — differences, similarities, or contrasts between multiple subjects;
- `version_specific` — latest, current, previous, dated, revision-specific, or explicitly numbered versions.

The classifier also emits `needs_metadata_filters` and zero or more stable hints: `document`, `document_version`, `date_range`, `section`, `file_type`, and `author`. Retrieval planning now uses these hints when choosing between dense and hybrid-style strategies. They remain advisory at the vector-store boundary until the later version-aware retrieval task translates them into concrete filters.

The configured Ollama model returns strict JSON through the provider-neutral `LLMProvider` boundary. Invalid output or a temporary model failure can fall back to deterministic rules, preserving query availability while recording `fallback_used=true` in both `QueryState.metadata["query_classification"]` and the classification trace metadata. Configure this behavior with:

```env
QUERY_CLASSIFICATION_FAIL_OPEN=true
QUERY_CLASSIFICATION_MAX_RATIONALE_CHARS=500
```

The query API also exposes the persisted result as the optional top-level `classification` field. Older query runs without classification metadata remain readable and return `classification: null`.

## Information-need decomposition

Information-need decomposition is an independent query-understanding capability under `packages/rag_core/query_understanding/decomposition`. It owns the decomposer protocol, typed models, LLM parsing, and deterministic fallback. `DecomposeInformationNeedsNode` only invokes that capability, stores the result in `QueryState.information_need_decomposition`, persists it under `QueryState.metadata["information_need_decomposition"]`, and emits its own trace step.

The configured Ollama model decomposes the question into one or more atomic `information_needs` without receiving or choosing a query classification or retrieval strategy. Each need contains a stable id, a description of what the evidence must establish, and a focused retrieval query that the retry/fallback controller can reuse. For example, “What are the pipeline flows and how do they function?” becomes separate needs for identifying the flows and explaining their functionality. Invalid output can fall back to conservative deterministic splitting.

## Retrieval planning

Retrieval planning is implemented independently under `packages/rag_core/query_understanding/planning`. The agentic pipeline uses one planner invocation per active information need rather than producing one global pipeline decision for the complete question. `PlanInformationNeedNode` receives the item description and focused retrieval query, its own classification, previous grader feedback, previous queries and plans, available pipelines, and the item attempt budget. It returns a fully executable `InformationNeedRetrievalPlan` containing the pipeline, strategy, query, top-k, metadata hints, rationale, and attempt number.

The default first-attempt policy selects:

- `baseline_rag` for focused factual items without special constraints;
- `hybrid_rag` for version-specific items or items with document, section, author, file-type, or date hints;
- `contextual_rag` for broad explanatory items that benefit from document-aware context;
- `multi_query_rag` for comparative items;
- `hybrid_cross_encoder_rerank_rag` for discriminative wording or low-confidence classifications.

When the preferred pipeline is unavailable, the planner produces an executable bounded fallback instead of an invalid plan. After weak or missing evidence, the same planner is called again for only that information need. It can expand the focused query with grader feedback, increase top-k, and choose an untried available pipeline. A repeated `(pipeline, query, top_k)` signature terminates the item as `no_effective_fallback`. The deterministic retry controller does not choose retrieval content; it only decides whether another item attempt, reclassification, or terminal completion is allowed.

The original query classification remains available for overall request understanding and trace feedback. Each decomposed item also has an independent classification and classification history, allowing one compound question to use different retrieval strategies for factual, broad, comparative, or version-specific requirements.

Configure decomposition and planning with:

```env
DEFAULT_QUERY_PIPELINE=agentic_rag
RETRIEVAL_PLANNING_LOW_CONFIDENCE_THRESHOLD=0.55
INFORMATION_NEED_DECOMPOSITION_FAIL_OPEN=true
INFORMATION_NEED_MAX_COUNT=6
INFORMATION_NEED_MAX_CHARS=240
INFORMATION_NEED_DECOMPOSITION_MAX_RATIONALE_CHARS=500
```

Manual pipeline selection is intentionally preserved. This allows the evaluation harness to compare fixed phase-2 pipelines against the hierarchical planner's end-to-end choices without changing the API contract.

## Evidence grading

Evidence grading is implemented under `packages/rag_core/retrieval/graders`; `GradeEvidenceNode` only orchestrates it. The agentic pipeline retains the existing per-chunk relevance score, but sufficiency is now derived from per-information-need support rather than one question-level verdict.

For every retrieved chunk, the grader records:

- a normalized relevance score;
- whether the chunk is relevant;
- the information-need ids the chunk materially supports;
- a short rationale.

For every decomposed information need, the grader records:

- `missing`, `partial`, or `supported`;
- a coverage score;
- the supporting evidence ranks;
- a short rationale.

The application derives overall status from these grades: no relevant evidence is `missing`, some relevant evidence with unresolved required claims is `weak`, and only complete support for every required claim is `sufficient`. This means evidence that merely lists pipeline names cannot satisfy a separate requirement asking how those pipelines function. A weak report is still `answerable` when at least one required claim is fully supported; this produces an explicitly partial answer rather than discarding the supported result. `supported_information`, `unresolved_information`, `partial_answer_available`, and the grader-approved evidence ranks are persisted in query metadata and exposed by the API. The retry flow converts unresolved grades into independent claim-retrieval tasks rather than broadening and rerunning the complete question as one lookup.

Invalid model output can fall back to deterministic lexical grading. Configure the thresholds with:

```env
EVIDENCE_GRADING_FAIL_OPEN=true
EVIDENCE_GRADING_RELEVANCE_THRESHOLD=0.60
EVIDENCE_GRADING_INFORMATION_NEED_SUPPORT_THRESHOLD=0.75
EVIDENCE_GRADING_MAX_CHARS_PER_EVIDENCE=2000
EVIDENCE_GRADING_MAX_RATIONALE_CHARS=500
```

## Retry and fallback logic

The agentic pipeline uses two graph levels:

```text
Top-level graph
classify_query
  → decompose_information_needs
  → initialize_information_need_work
  → resolve_information_needs (subgraph)
  → aggregate_information_needs
  → generate_answer

Information-need subgraph
select_information_need
  → classify_information_need
  → plan_information_need
  → execute_information_need_plan
  → grade_information_need
  → decide_information_need
      ├─ supported → complete_information_need
      ├─ retry → plan_information_need
      ├─ reclassify → classify_information_need
      └─ exhausted → complete_information_need
  → select_information_need
```

`ConditionalGraphRunner` owns the named routes and bounded cycle. The queue is used instead of Python recursion, making every transition traceable and preventing stack growth. Each `InformationNeedExecution` stores independent classification and plan histories, attempts, grader-approved evidence keys, final grade, stop reason, retry limit, parent id, and depth. The current implementation initializes top-level information needs at depth zero; the parent/depth fields leave a stable boundary for future discovered child needs without requiring another state redesign.

Weak or missing evidence is evaluated per active item. The retry policy can route it back to planning, route a low-confidence missing item through classification once, or finish it as exhausted. A supported item completes immediately and cannot consume another item's budget. The planner then uses the item's own grade and attempt history to decide whether to broaden its query, increase top-k, or switch pipeline.

The execution is bounded by both per-item and query-level limits:

```env
RETRIEVAL_RETRY_MAX_RETRIES=2
RETRIEVAL_RETRY_TOP_K_MULTIPLIER=2.0
RETRIEVAL_RETRY_MAX_TOP_K=20
RETRIEVAL_RETRY_EXPAND_QUERY=true
RETRIEVAL_RETRY_MAX_QUERY_CHARS=1200
RETRIEVAL_RETRY_MAX_TOTAL_ATTEMPTS=20
RETRIEVAL_RETRY_MAX_RECLASSIFICATIONS=1
RETRIEVAL_RETRY_MAX_ACCUMULATED_EVIDENCE=40
```

`RETRIEVAL_RETRY_MAX_RETRIES=2` means at most three retrieval attempts for each information need. `RETRIEVAL_RETRY_MAX_TOTAL_ATTEMPTS` caps the full compound query, and the graph runner also has a derived maximum step count. When the global budget is reached, pending items are completed as unresolved without retrieval so aggregation and partial-answer generation can still finish deterministically.

Evidence is indexed once at query level and referenced by the information needs it supports. After each attempt, the active grader removes irrelevant item references while retaining rejected candidate grades in the trace. Final aggregation takes the union of grader-approved evidence for supported items, deduplicates it, derives overall complete/weak/missing status, and exposes `information_need_resolution` through the API. This report includes every item lifecycle, all plans and attempts, supported and unresolved ids, per-item and global budgets, and final stop reasons.

## Multi-query retrieval

`multi_query_rag` isolates query expansion as a query-time retrieval experiment. It wraps the hybrid retriever rather than contextual retrieval, so evaluation can compare single-query hybrid retrieval with multi-query hybrid retrieval without also changing the indexed representation.

The retrieval flow is:

1. Generate up to `MULTI_QUERY_VARIANT_COUNT` alternatives that preserve the original intent.
2. Include the original question when `MULTI_QUERY_INCLUDE_ORIGINAL=true`.
3. Retrieve a bounded candidate set for every query through the existing hybrid retriever.
4. Run those independent lookups concurrently.
5. Deduplicate chunks by chunk ID, Qdrant point ID, document-version ordinal, or normalized text hash.
6. Fuse all query rankings with weighted reciprocal-rank fusion and return the final requested top-k.
7. Fall back to the original question when expansion fails and `MULTI_QUERY_FAIL_OPEN=true`.

The main tuning options are:

```env
MULTI_QUERY_VARIANT_COUNT=3
MULTI_QUERY_INCLUDE_ORIGINAL=true
MULTI_QUERY_CANDIDATE_MULTIPLIER=2
MULTI_QUERY_MAX_CANDIDATES_PER_QUERY=20
MULTI_QUERY_RRF_K=60
MULTI_QUERY_ORIGINAL_QUERY_WEIGHT=1.2
MULTI_QUERY_VARIANT_QUERY_WEIGHT=1.0
MULTI_QUERY_MAX_VARIANT_CHARS=300
MULTI_QUERY_FAIL_OPEN=true
```

The original question has a slightly higher default fusion weight, which reduces query drift while still rewarding evidence found by multiple paraphrases. Increasing the variant count or per-query candidate count can improve recall, but it also increases embedding, vector-search, BM25, and latency costs. Use the evaluation harness to compare `hybrid_rag` and `multi_query_rag` on the same dataset before changing the defaults.

## Contextual retrieval

Contextualization runs automatically during normal document ingestion and uses a bounded two-level document context hierarchy. This is inspired by RAPTOR's bottom-up summarization, but it does **not** add summary nodes to retrieval and does not change query-time lookup. The hierarchy exists only during preprocessing so a local model can approximate whole-document awareness without receiving the complete document for every chunk.

The ingestion flow is:

1. Parse and chunk the source document.
2. Embed every original chunk once.
3. Group semantically similar chunk embeddings into bounded clusters.
4. Generate one compact summary for each semantic cluster.
5. Generate one document summary from the cluster summaries.
6. Contextualize each target chunk using the document summary, its semantic-cluster summary, and a bounded adjacent-chunk window.
7. Reuse the already-computed original embeddings and embed only the contextualized representations.
8. Store both `original` and `contextual` named vectors on the same Qdrant point.

Configure the hierarchy and per-chunk contextualization with:

```env
CONTEXTUALIZATION_ENABLED=true
CONTEXTUALIZATION_FAIL_OPEN=false
CONTEXTUALIZATION_MODEL=llama3.2:3b
CONTEXTUALIZATION_NEIGHBOR_CHUNK_COUNT=2
CONTEXTUALIZATION_MAX_NEIGHBOR_CHARS=6000
CONTEXTUALIZATION_MAX_CONTEXT_CHARS=400
CONTEXTUALIZATION_MAX_CONCURRENCY=2
CONTEXTUALIZATION_CLUSTER_TARGET_SIZE=8
CONTEXTUALIZATION_MAX_CLUSTERS=24
CONTEXTUALIZATION_MAX_CLUSTER_SOURCE_CHARS=8000
CONTEXTUALIZATION_MAX_DOCUMENT_SOURCE_CHARS=12000
CONTEXTUALIZATION_MAX_CLUSTER_SUMMARY_CHARS=600
CONTEXTUALIZATION_MAX_DOCUMENT_SUMMARY_CHARS=900
QDRANT_ORIGINAL_VECTOR_NAME=original
QDRANT_CONTEXTUAL_VECTOR_NAME=contextual
```

The semantic grouping is deterministic and dependency-free: cosine similarity is calculated from the original chunk embeddings, semantic outliers become group seeds, and each seed is filled with its nearest remaining chunks. `CONTEXTUALIZATION_CLUSTER_TARGET_SIZE` controls the normal group size, while `CONTEXTUALIZATION_MAX_CLUSTERS` prevents very large documents from producing an excessive number of summary calls. The document summary is synthesized from the bounded cluster summaries rather than raw full-document text.

For every target chunk, the final prompt receives:

- the document title and generated document summary;
- the summary of the semantic group containing that chunk;
- chunk location metadata;
- a configurable window of preceding and following chunks;
- the untouched target chunk.

The prompt explicitly asks for only missing retrieval context as a topic-first line, normally 15-45 words. It rejects container-first boilerplate such as “The document” or “This chunk,” as well as a rehash of information already explicit in the target. Output normalization also removes generic container leads and model meta-commentary such as compliance notes. `CONTEXTUALIZATION_MAX_CONTEXT_CHARS` provides a hard output bound and truncation prefers a complete sentence boundary. The generated description is prepended only to `contextualized_text`; the original chunk remains in `text`, so answer generation, evidence snapshots, and citation quotes never present generated context as source material.

The hierarchy is recorded in document/version contextualization metadata for inspection: strategy, document summary, cluster count, cluster summaries, and chunk assignments. Each indexed chunk stores its `context_cluster_id` alongside the final chunk-specific context. The hierarchy itself is not indexed and therefore cannot change query-time retrieval behavior independently of the contextualized chunk representation.

This preprocessing adds one LLM call per semantic cluster, one document-summary call, and one call per chunk. For example, 80 chunks with a target cluster size of 8 normally require about 91 generation calls. `CONTEXTUALIZATION_MAX_CONCURRENCY` bounds simultaneous local Ollama requests.

When `CONTEXTUALIZATION_FAIL_OPEN=true` is explicitly configured, an unavailable contextualization model does not block normal ingestion: each point is written with only its `original` named vector and metadata records `failed_open`. Such points remain available to baseline/hybrid pipelines but do not appear in `contextual_rag` until successfully re-ingested.

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
  --pipeline hybrid_llm_rerank_rag \
  --top-k 10 \
  --output reports/evaluations/hybrid-ollama-rerank-top-10.json

python -m scripts.run_evaluation datasets/eval_sets/baseline_demo.json \
  --pipeline hybrid_cross_encoder_rerank_rag \
  --top-k 10 \
  --output reports/evaluations/hybrid-cross-encoder-rerank-top-10.json

python -m scripts.run_evaluation datasets/eval_sets/baseline_demo.json \
  --pipeline contextual_rag \
  --top-k 10 \
  --output reports/evaluations/contextual-top-10.json

python -m scripts.run_evaluation datasets/eval_sets/baseline_demo.json \
  --pipeline multi_query_rag \
  --top-k 10 \
  --output reports/evaluations/multi-query-top-10.json

python -m scripts.run_evaluation datasets/eval_sets/baseline_demo.json \
  --pipeline agentic_rag \
  --top-k 10 \
  --output reports/evaluations/agentic-top-10.json
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

### Reranker choices

The LLM-based `hybrid_llm_rerank_rag` option remains available. Small generative models can occasionally omit an item even when JSON-schema output is requested, so this implementation now uses request-local candidate ids, an exact-length schema, the schema in the prompt, bounded retries, single-candidate recovery requests, and an optional original-rank fallback. Control that behavior with:

```env
RERANK_BATCH_SIZE=8
RERANK_MAX_CHARS_PER_CANDIDATE=4000
OLLAMA_RERANK_MAX_ATTEMPTS=2
OLLAMA_RERANK_FALLBACK_TO_ORIGINAL_RANK=true
```

The dedicated `hybrid_cross_encoder_rerank_rag` option does not generate JSON. It jointly scores each `(question, chunk)` pair with `cross-encoder/ms-marco-MiniLM-L6-v2` by default. Compose downloads the snapshot through `cross-encoder-bootstrap` into the named `cross_encoder_cache` volume before the API starts:

```env
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
CROSS_ENCODER_MODEL_REVISION=c5ee24cb16019beea0893ab7796b1df96625c6b8
CROSS_ENCODER_MODEL_PATH=/root/.cache/huggingface/indexer/cross-encoder
CROSS_ENCODER_LOCAL_FILES_ONLY=true
CROSS_ENCODER_DOWNLOAD_FORCE=false
CROSS_ENCODER_BATCH_SIZE=16
CROSS_ENCODER_MAX_LENGTH=512
CROSS_ENCODER_DEVICE=cpu
```

The API loads only `CROSS_ENCODER_MODEL_PATH`, passes `local_files_only=True` to Sentence Transformers, and runs with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. It therefore cannot perform Hugging Face update checks during queries. The model object is retained in the application-scoped tool registry, so the first cross-encoder query loads the weights from the local volume into memory and later queries reuse that same in-process model.

`cross-encoder-bootstrap` stores a small manifest beside the model. When the model ID and revision still match and the local files are complete, the service exits without invoking the Hugging Face downloader. During the first migration to this layout it also tries to materialize the pinned snapshot from the existing volume cache in local-only mode before using the network. A new download fetches only the configuration, tokenizer, and PyTorch/Safetensors runtime files instead of unrelated ONNX, OpenVINO, or Flax exports. The default revision is pinned to a specific model commit. Change `CROSS_ENCODER_MODEL` or `CROSS_ENCODER_MODEL_REVISION` to download another snapshot, or explicitly force a refresh with:

```bash
docker compose run --rm \
  -e CROSS_ENCODER_DOWNLOAD_FORCE=true \
  cross-encoder-bootstrap
```

The default model is compact and English-focused. Replace `CROSS_ENCODER_MODEL` with another Sentence Transformers-compatible reranker when your documents require another language or domain. `CROSS_ENCODER_DEVICE=auto` lets Sentence Transformers choose an available device; using `cuda` also requires exposing a compatible GPU to the API container.

## Local API development

```bash
cd apps/api
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

For local development outside Docker, make sure `DATABASE_URL` points to PostgreSQL, `QDRANT_URL` points to Qdrant, `MINIO_ENDPOINT` points to MinIO, and `OLLAMA_BASE_URL` points to Ollama. To use cross-encoder reranking outside Compose, set `CROSS_ENCODER_MODEL_PATH` to a writable local directory, run `python -m scripts.download_cross_encoder` once, and then enable `HF_HUB_OFFLINE=1` plus `TRANSFORMERS_OFFLINE=1` for the API process. For tests or fully offline API development, set `DOCUMENT_STORAGE_BACKEND=local` and `EMBEDDING_PROVIDER=hashing`.

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
DocumentObjectStore (MinIO by default)
        ↓
parse_document → chunk_document
        ↓
[optional] LLMChunkContextualizer per chunk
        ↓
embed original text + optional contextualized text
        ↓
upsert one Qdrant point with `original` + optional `contextual` named vectors
        ↓
persist Document, DocumentVersion, QdrantChunkIndex metadata
```

Key files:

- `apps/api/app/api/routes/documents.py` — document upload/list/detail endpoints.
- `packages/indexer_infrastructure/minio/` and `packages/indexer_infrastructure/object_storage/` — MinIO storage plus the local test fallback.
- `packages/indexer_application/services/document_ingestion.py` — use-case orchestration, contextualization policy, and persistence flow.
- `packages/rag_core/documents/parsers.py` — PDF, text, and markdown parsers.
- `packages/rag_core/documents/chunking.py` — basic chunking and metadata generation.
- `packages/rag_core/documents/models.py` — parser/chunking domain models.
- `packages/rag_core/ports/embeddings.py` and `packages/indexer_infrastructure/embeddings/` / `ollama/` — embedding port and implementations.
- `packages/rag_core/ports/vector_indexes.py` and `packages/indexer_infrastructure/qdrant/vector_store.py` — vector-index port and Qdrant adapter.
- `packages/rag_core/ingestion/contextualizer.py` and `packages/rag_core/prompts/contextualize_chunk.md` — contextualization algorithm and prompt.
- `packages/indexer_application/services/chunk_indexing.py` — writes original and contextual representations without duplicating source evidence.
- `packages/indexer_infrastructure/bm25/` and `packages/indexer_infrastructure/qdrant/keyword_corpus.py` — BM25 scoring and payload-backed corpora.

## Current query architecture

The query API uses registry-backed fixed pipelines plus a hierarchical agentic graph:

```text
GET /api/v1/pipelines
        ↓
discover PipelineConfig + ToolConfig entries

POST /api/v1/queries (optional pipeline_name)
        ↓
PipelineRegistry selects configured/default pipeline
        ↓
ToolRegistry resolves query-understanding, retrieval, grading, retry-policy, and generation tools
        ↓
Top-level GraphRunner(QueryState)
        ↓
classify → decompose → initialize work queue → run information-need subgraph → aggregate → generate
        ↓
ConditionalGraphRunner loops each item through classify → plan → retrieve → grade → decide
        ↓
persist answer, grader-approved evidence, citations, nested trace events, and information-need resolution
```

Key files:

- `packages/rag_core/agents/runtime/` — generic sequential and conditional graph runners, node contracts, conditional edges, bounded execution, and graph-aware trace emission.
- `packages/rag_core/agents/query_graph/` — top-level query state, graph factory, trace summaries, and nodes for classification, decomposition, work initialization, subgraph invocation, aggregation, and generation.
- `packages/rag_core/agents/information_need_graph/` — per-information-need execution models, routes, reporting, evidence references, trace summaries, graph factory, and cyclic lifecycle nodes.
- `packages/rag_core/agents/shared/retrieval/` — reusable retrieval-plan execution models, executor, fixed-pipeline retrieval nodes, and retrieval trace summaries.
- `packages/rag_core/generation/` — answer-generation request/result models, evidence selection, prompt construction, citation conversion, and generation service.
- `packages/rag_core/query_understanding/classification/` — query and per-item classification contracts, models, LLM parsing, and deterministic fallback rules.
- `packages/rag_core/query_understanding/decomposition/` — independent atomic information-need extraction.
- `packages/rag_core/query_understanding/planning/` — per-item executable planning and retry re-planning from grader history.
- `packages/rag_core/retrieval/retry/` — bounded route decisions and per-item/query-level execution limits.
- `packages/rag_core/pipelines/agentic.py` — thin dependency wiring between the query-graph and information-need-graph factories.
- `packages/rag_core/pipelines/registry.py` and `packages/rag_core/agents/tools/` — pipeline and tool discovery.
- `packages/rag_core/pipelines/baseline.py`, `hybrid.py`, `contextual.py`, `multi_query.py`, and rerank pipeline modules — fixed phase-2 retrieval implementations reused by per-item plans.
- `packages/rag_core/agents/query_graph/nodes/generate_answer.py` — thin orchestration node applying the citation-aware generation service result.
- `apps/api/app/composition/pipelines.py` — tool registration, fixed pipeline registration, and hierarchical agentic graph construction.
- `apps/api/app/api/routes/queries.py` — query API including `information_need_resolution`.
- `packages/indexer_application/services/query_runs.py` — persistence around graph execution.

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
