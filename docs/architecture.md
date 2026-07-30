# Architecture contract

This document defines where code belongs in `indexer-core` and which dependency directions are allowed. It describes the current layered modular monolith; it does not introduce new runtime services, database entities, or processing behavior.

## Layer ownership

| Area | Owns | May depend on | Must not depend on |
| --- | --- | --- | --- |
| `packages/rag_core` | Pure RAG algorithms, graph definitions and state, retrieval and generation behavior, evaluation calculations, domain-level models, and provider protocols | Standard library, third-party libraries, and other `rag_core` modules | `indexer_application`, `indexer_infrastructure`, deployable apps, or `scripts` |
| `packages/indexer_application` | Reusable use cases, commands, queries, orchestration, application DTOs, ports, and transaction/side-effect decisions | `rag_core` | Concrete infrastructure, deployable apps, or `scripts` |
| `packages/indexer_infrastructure` | PostgreSQL, MinIO, Qdrant, Ollama, BM25, cross-encoder, and other concrete adapters; persistence mappings | `indexer_application` contracts and `rag_core` protocols/models | Deployable apps or `scripts` |
| `apps/api` | FastAPI HTTP contracts, request validation, response presentation, configuration, dependency injection, and composition | All three package layers | `scripts` or another deployable app |
| `apps/web` | Angular feature slices, transport models, UI view models, and presentation behavior | Backend HTTP contracts | Python package responsibilities or direct imports from backend layers |

The intended dependency direction is:

```text
apps/api ───────────────┐
  │                     │
  ├──> indexer_infrastructure ──┐
  ├──> indexer_application ─────┼──> rag_core
  └─────────────────────────────┘

apps/web ──HTTP──> apps/api
```

`apps/api` is the current composition root. Operational scripts may invoke the API composition root where necessary, but production packages and deployable apps must never import from `scripts`. A future independently deployable process belongs under `apps/<name>` and must compose shared package behavior rather than import another deployable app.

## Placement rules

Use these rules before creating a new module:

- Put provider-independent retrieval, generation, graph, grading, or evaluation behavior in `rag_core`.
- Put a user-visible or operational use case in `indexer_application`, with an application port when an external capability is required.
- Put a database, object-store, vector-store, model-runtime, or vendor implementation in `indexer_infrastructure`.
- Put HTTP validation, schemas, presenters, configuration, dependency wiring, and process-level caches in `apps/api`.
- Put Angular transport models, feature state, view models, components, and pages in `apps/web`.
- Keep `scripts` as thin operational entry points. Shared behavior used by a script belongs in one of the package layers.

Do not add a new architectural layer merely because a feature is large. Split it into cohesive modules inside the layer that owns the behavior.

## State and external-system ownership

### PostgreSQL: application truth

PostgreSQL is the authoritative store for application state, including logical documents, source revisions, query runs, evidence/citation records, traces, and references to external artifacts or indexes. Application use cases decide transaction boundaries and when a unit of work commits or rolls back; routes and repositories do not make that decision independently.

### MinIO: source and artifact storage

MinIO stores uploaded source bytes and durable generated artifacts. PostgreSQL stores the stable application references and metadata that identify those objects. The presence of an object alone is not application truth.

### Qdrant: rebuildable retrieval index

Qdrant stores vector and retrieval payloads derived from source content and application metadata. It is an index, not the authoritative owner of document identity or workflow state. Its contents must be treated as rebuildable from authoritative source bytes and PostgreSQL records.

When a use case spans PostgreSQL and an external system, the application layer owns ordering, failure handling, and cleanup policy. Infrastructure adapters own only the concrete operation against their system.

For synchronous document ingestion, the application coordinator owns this sequence:

1. persist the uploaded source and retain its stable storage reference;
2. create the PostgreSQL processing document/version identity;
3. materialize the source temporarily for parsing and always release that materialization;
4. derive and write Qdrant/chunk-index state;
5. activate the indexed version, mark PostgreSQL ready, and commit.

Failures after a processing version exists are recorded and committed as failed. MinIO source objects remain durable, temporary parser files are cleaned by the coordinator, and partially written Qdrant data remains rebuildable rather than being treated as application truth. No distributed transaction is implied across PostgreSQL, MinIO, and Qdrant.

## Document and revision identity

A `document` is the logical source family. A row in `document_versions` is the canonical identity of one source revision and remains the existing source-revision boundary.

“Source revision” is architectural terminology for that existing identity. It does **not** introduce a replacement table, parallel identifier, or schema migration.

## Reserved future terminology

The following terms provide a shared vocabulary for future work only. They do not create packages, tables, APIs, workers, or behavior in the current architecture.

- **Job** — a durable request to perform work that may later be queued, retried, resumed, or executed outside the request lifecycle.
- **Query execution** — one invocation of query understanding, retrieval, evidence processing, and answer generation, including its trace and result.
- **Source revision** — one immutable source state, represented today by `document_versions`.
- **Processing generation** — a future lineage concept for one processing attempt or derived-output generation associated with a source revision.
- **Evaluation run** — one execution of an evaluation dataset and configuration against a pipeline or query-execution entry point, producing metrics and artifacts.

Until a dedicated change explicitly introduces one of these concepts, use the current models and synchronous behavior.

## Enforcement

`apps/api/tests/test_architecture_boundaries.py` enforces the dependency direction across package layers, scripts, and all Python deployable apps discovered under `apps/`. It also prevents non-Python trees and operational directories from becoming accidental Python packages.

Architecture changes must update this document and its boundary tests in the same change. Feature changes must preserve these rules unless an explicitly approved architecture change replaces them.
