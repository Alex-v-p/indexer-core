from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.rag_core.evaluation import EvaluationDatasetError, load_evaluation_dataset


def test_load_evaluation_dataset_parses_expected_answer_and_evidence(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "test-set",
                "version": "1.0.0",
                "default_top_k": 7,
                "cases": [
                    {
                        "id": "case-1",
                        "question": "What is evaluated?",
                        "expected_answer": "The complete graph run.",
                        "expected_evidence": [
                            {
                                "metadata": {"original_filename": "source.md"},
                                "text_contains": ["complete graph"],
                            },
                        ],
                        "tags": ["baseline"],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    dataset = load_evaluation_dataset(dataset_path)

    assert dataset.name == "test-set"
    assert dataset.default_top_k == 7
    assert dataset.cases[0].expected_answer == "The complete graph run."
    assert dataset.cases[0].expected_evidence[0].metadata == {"original_filename": "source.md"}
    assert dataset.cases[0].expected_evidence[0].text_contains == ("complete graph",)


def test_load_evaluation_dataset_rejects_evidence_without_matchers(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "invalid",
                "version": "1.0.0",
                "cases": [
                    {
                        "id": "case-1",
                        "question": "Question",
                        "expected_evidence": [{"description": "No matcher configured"}],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationDatasetError, match="at least one matching field"):
        load_evaluation_dataset(dataset_path)


def test_repository_demo_dataset_is_valid() -> None:
    dataset = load_evaluation_dataset(Path(__file__).resolve().parents[3] / "datasets/eval_sets/baseline_demo.json")

    assert dataset.name == "baseline-demo"
    assert len(dataset.cases) == 3
