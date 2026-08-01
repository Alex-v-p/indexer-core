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

Subject-aware evaluation fields remain additive and portable: cases may provide `requested_subject_ids` or `requested_subject_names`, `coverage_mode`, and `behavioral_expectations`. Set dataset metadata `subject_scope_evaluation` to `true` when every case should run through automatic subject resolution. Metric computation is deterministic for a captured run, but service-backed retrieval and model outputs are not stable CI guarantees.
