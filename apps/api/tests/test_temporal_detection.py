from __future__ import annotations

from datetime import UTC, datetime

from packages.rag_core.query_understanding.classification import classify_query_heuristically
from packages.rag_core.query_understanding.temporal import (
    DocumentDateField,
    detect_document_date_constraints,
)


def test_detects_uploaded_month_range() -> None:
    constraints = detect_document_date_constraints(
        "Find documents uploaded in May 2026.",
        timezone_name="UTC",
    )

    assert len(constraints) == 1
    constraint = constraints[0]
    assert constraint.field is DocumentDateField.UPLOADED_AT
    assert constraint.date_range.start == datetime(2026, 5, 1, tzinfo=UTC)
    assert constraint.date_range.end == datetime(2026, 6, 1, tzinfo=UTC)


def test_detects_published_specific_date() -> None:
    constraint = detect_document_date_constraints(
        "Use policies published on March 15, 2025.",
        timezone_name="UTC",
    )[0]

    assert constraint.field is DocumentDateField.PUBLISHED_AT
    assert constraint.date_range.start == datetime(2025, 3, 15, tzinfo=UTC)
    assert constraint.date_range.end == datetime(2025, 3, 16, tzinfo=UTC)


def test_detects_relative_upload_window_with_absolute_trace_range() -> None:
    constraint = detect_document_date_constraints(
        "Show documents uploaded in the last 3 months.",
        reference_datetime=datetime(2026, 7, 18, 12, 0, tzinfo=UTC),
        timezone_name="UTC",
    )[0]

    assert constraint.date_range.start == datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
    assert constraint.date_range.end == datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    assert constraint.original_expression == "last 3 months"


def test_generic_year_without_date_field_is_not_silently_mapped() -> None:
    assert detect_document_date_constraints("Find the 2025 documents.") == ()


def test_heuristic_classification_carries_date_constraint() -> None:
    classification = classify_query_heuristically(
        "What policies were published in 2024?",
        timezone_name="UTC",
    )

    assert classification.needs_metadata_filters is True
    assert [hint.value for hint in classification.metadata_filter_hints].count("date_range") == 1
    assert classification.date_constraints[0].field is DocumentDateField.PUBLISHED_AT


def test_detects_generic_data_month_as_any_recorded_date() -> None:
    constraint = detect_document_date_constraints(
        "Only use data from May 2026.",
        timezone_name="UTC",
    )[0]

    assert constraint.field is DocumentDateField.ANY_RECORDED_AT
    assert constraint.date_range.start == datetime(2026, 5, 1, tzinfo=UTC)
    assert constraint.date_range.end == datetime(2026, 6, 1, tzinfo=UTC)
    assert "does not ignore the date" in constraint.rationale


def test_standalone_month_resolves_to_most_recent_occurrence() -> None:
    constraint = detect_document_date_constraints(
        "Only use documents from May.",
        reference_datetime=datetime(2026, 7, 18, 12, 0, tzinfo=UTC),
        timezone_name="UTC",
    )[0]

    assert constraint.field is DocumentDateField.ANY_RECORDED_AT
    assert constraint.date_range.start == datetime(2026, 5, 1, tzinfo=UTC)
    assert constraint.date_range.end == datetime(2026, 6, 1, tzinfo=UTC)
    assert constraint.original_expression == "May 2026"


def test_only_use_date_phrase_does_not_require_a_document_noun() -> None:
    constraint = detect_document_date_constraints(
        "Only use May 2026 for the answer.",
        timezone_name="UTC",
    )[0]

    assert constraint.field is DocumentDateField.ANY_RECORDED_AT
    assert constraint.date_range.start == datetime(2026, 5, 1, tzinfo=UTC)
    assert constraint.date_range.end == datetime(2026, 6, 1, tzinfo=UTC)


def test_since_month_remains_an_open_ended_range() -> None:
    constraint = detect_document_date_constraints(
        "Use documents uploaded since May 2026.",
        timezone_name="UTC",
    )[0]

    assert constraint.field is DocumentDateField.UPLOADED_AT
    assert constraint.date_range.start == datetime(2026, 5, 1, tzinfo=UTC)
    assert constraint.date_range.end is None
