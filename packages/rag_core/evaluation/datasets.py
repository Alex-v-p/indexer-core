from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from packages.rag_core.evaluation.models import EvaluationCase, EvaluationDataset, EvidenceExpectation


class EvaluationDatasetError(ValueError):
    """Raised when an evaluation dataset cannot be loaded or validated."""


def load_evaluation_dataset(path: str | Path) -> EvaluationDataset:
    """Load and validate a schema-versioned JSON evaluation dataset."""

    dataset_path = Path(path)
    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationDatasetError(f"Evaluation dataset not found: {dataset_path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationDatasetError(
            f"Evaluation dataset {dataset_path} is not valid JSON: line {exc.lineno}, column {exc.colno}.",
        ) from exc

    try:
        return _parse_dataset(_mapping(raw, "dataset"))
    except (TypeError, ValueError) as exc:
        raise EvaluationDatasetError(f"Invalid evaluation dataset {dataset_path}: {exc}") from exc


def _parse_dataset(raw: dict[str, Any]) -> EvaluationDataset:
    raw_cases = _list(raw.get("cases"), "cases")
    cases = tuple(_parse_case(_mapping(item, f"cases[{index}]")) for index, item in enumerate(raw_cases))
    return EvaluationDataset(
        schema_version=_required_string(raw, "schema_version"),
        name=_required_string(raw, "name"),
        version=_required_string(raw, "version"),
        description=_optional_string(raw.get("description"), "description"),
        default_top_k=_positive_int(raw.get("default_top_k", 5), "default_top_k"),
        metadata=_optional_mapping(raw.get("metadata"), "metadata"),
        cases=cases,
    )


def _parse_case(raw: dict[str, Any]) -> EvaluationCase:
    raw_evidence = _list(raw.get("expected_evidence", []), "expected_evidence")
    expected_evidence = tuple(
        _parse_evidence(_mapping(item, f"expected_evidence[{index}]")) for index, item in enumerate(raw_evidence)
    )
    top_k_value = raw.get("top_k")
    return EvaluationCase(
        id=_required_string(raw, "id"),
        question=_required_string(raw, "question"),
        expected_answer=_optional_string(raw.get("expected_answer"), "expected_answer"),
        expected_evidence=expected_evidence,
        top_k=None if top_k_value is None else _positive_int(top_k_value, "top_k"),
        tags=tuple(_string_list(raw.get("tags", []), "tags")),
        metadata=_optional_mapping(raw.get("metadata"), "metadata"),
    )


def _parse_evidence(raw: dict[str, Any]) -> EvidenceExpectation:
    return EvidenceExpectation(
        description=_optional_string(raw.get("description"), "description"),
        qdrant_chunk_index_id=_optional_string(raw.get("qdrant_chunk_index_id"), "qdrant_chunk_index_id"),
        document_id=_optional_string(raw.get("document_id"), "document_id"),
        document_version_id=_optional_string(raw.get("document_version_id"), "document_version_id"),
        metadata=_optional_mapping(raw.get("metadata"), "metadata"),
        text_contains=tuple(_string_list(raw.get("text_contains", []), "text_contains")),
    )


def _required_string(raw: dict[str, Any], key: str) -> str:
    if key not in raw:
        raise ValueError(f"Missing required field {key!r}.")
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"Field {key!r} must be a non-empty string.")
    return value.strip()


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Field {field_name!r} must be a string or null.")
    return value.strip() or None


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"Field {field_name!r} must be a positive integer.")
    return value


def _mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Field {field_name!r} must be an object.")
    return value


def _optional_mapping(value: object, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(_mapping(value, field_name))


def _list(value: object, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"Field {field_name!r} must be an array.")
    return value


def _string_list(value: object, field_name: str) -> list[str]:
    raw_values = _list(value, field_name)
    values: list[str] = []
    for index, item in enumerate(raw_values):
        if not isinstance(item, str) or not item.strip():
            raise TypeError(f"Field {field_name!r}[{index}] must be a non-empty string.")
        values.append(item.strip())
    return values
