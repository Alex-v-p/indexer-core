# Evaluation datasets

Evaluation datasets are JSON files using schema version `1.0`. Each case is run through the complete configured query graph, including retrieval and answer generation.

```json
{
  "schema_version": "1.0",
  "name": "my-dataset",
  "version": "1.0.0",
  "default_top_k": 5,
  "cases": [
    {
      "id": "question-1",
      "question": "What does the source say?",
      "expected_answer": "A reference answer for inspection and future answer metrics.",
      "expected_evidence": [
        {
          "qdrant_chunk_index_id": "optional-stable-chunk-id",
          "document_id": "optional-document-id",
          "document_version_id": "optional-version-id",
          "metadata": {
            "original_filename": "source.pdf",
            "section_title": "Optional section"
          },
          "text_contains": ["required case-insensitive text fragment"]
        }
      ],
      "top_k": 5,
      "tags": ["baseline"]
    }
  ]
}
```

Within one expected-evidence object, every populated matcher must match. Separate expected-evidence objects represent separate relevant chunks. ID matchers are suited to a stable index; metadata and `text_contains` are more portable when documents are re-ingested and receive new IDs.

Cases may omit `expected_evidence` for answer-only or unanswerable examples. Retrieval and citation metrics are then marked `not_applicable`. `expected_answer` is stored in reports but is not scored yet. Answer faithfulness is intentionally reported as `not_implemented` until a groundedness evaluator is added.

## Subject-scoping behavior

`subject_scoping_v1.json` is a versioned synthetic contract for subject resolution and retrieval boundaries. It references `subject_scoping_fixture_v1.json`, whose stable names and filenames are resolved against IDs created by normal ingestion. Cases may add `requested_subject_ids` or portable `requested_subject_names`, `coverage_mode`, and `behavioral_expectations` for product outcome, matched scope subjects, forbidden sources, maximum leakage, relevant-document diversity, per-subject comparison lanes, and citation-scope validity. Existing schema-1.0 datasets such as `baseline_demo.json` remain valid without these additive fields and run with global scope.

Metric computation is deterministic for a captured run. Observed service-backed values are not stable CI results unless the fixture, catalog, index, retrievers, prompts, models, and model outputs are fixed. Recall, rank, evidence selection, and generated wording can vary with runtime configuration; `expected_answer` remains informational and answer faithfulness remains `not_implemented`.

Seed or reconcile the bounded fixture through the supported local API workflow, then run its read-only preflight:

```powershell
python -m scripts.seed_subject_scoping_fixture
python -m scripts.seed_subject_scoping_fixture --preflight-only
```

From a configured host environment:

```powershell
python -m scripts.run_evaluation datasets/eval_sets/subject_scoping_v1.json
```

With the local Compose stack and the synthetic fixture seeded:

```powershell
docker compose exec api python -m scripts.run_evaluation datasets/eval_sets/subject_scoping_v1.json
```

The evaluation runner preflights the manifest revision, active catalog entries, READY documents with non-empty chunk indexes, and required memberships before executing a scoped case. Filename-only, failed, processing, or unindexed matches are rejected and the seed command will ingest them again. Missing providers or fixtures fail with an actionable configuration error. A live subject-scoping run must be skipped when PostgreSQL, Qdrant, or the configured model service is unavailable.
