from __future__ import annotations

import uuid
import math

import pytest

from packages.rag_core.subjects import (
    ConfidenceBand,
    DecisionControlSource,
    DecisionState,
    DocumentSubjectDecision,
    SubjectKind,
    SubjectName,
    normalize_subject_name,
)


def test_subject_kinds_and_decision_values_are_stable_checked_strings() -> None:
    assert {item.value for item in SubjectKind} == {
        "project",
        "topic",
        "organization",
        "custom",
    }
    assert {item.value for item in DecisionState} == {
        "suggested",
        "assigned",
        "rejected",
    }
    assert {item.value for item in DecisionControlSource} == {
        "manual",
        "automatic",
    }


def test_subject_names_share_unicode_whitespace_and_punctuation_normalization() -> None:
    name = SubjectName.from_value("  Project\u3000Orion—Phase_2  ")

    assert name.value == "Project Orion—Phase_2"
    assert name.normalized == "project orion phase 2"
    assert normalize_subject_name("ＰＲＯＪＥＣＴ orion / phase-2") == name.normalized


@pytest.mark.parametrize("value", ["", "   ", "---", "___"])
def test_subject_name_rejects_empty_normalized_values(value: str) -> None:
    with pytest.raises(ValueError):
        SubjectName.from_value(value)


def test_manual_rejection_is_a_non_membership_tombstone() -> None:
    decision = DocumentSubjectDecision(
        document_id=uuid.uuid4(),
        subject_id=uuid.uuid4(),
        state=DecisionState.REJECTED,
        control_source=DecisionControlSource.MANUAL,
        rationale="Removed by an operator",
    )

    assert decision.is_membership is False
    assert decision.state is DecisionState.REJECTED


def test_only_assigned_decisions_count_as_membership() -> None:
    decision = DocumentSubjectDecision(
        document_id=uuid.uuid4(),
        subject_id=uuid.uuid4(),
        state=DecisionState.ASSIGNED,
        control_source=DecisionControlSource.MANUAL,
    )

    assert decision.is_membership is True


def test_automatic_decision_requires_auditable_provenance() -> None:
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    decision = DocumentSubjectDecision(
        document_id=document_id,
        subject_id=uuid.uuid4(),
        state=DecisionState.ASSIGNED,
        control_source=DecisionControlSource.AUTOMATIC,
        confidence=0.63,
        confidence_band=ConfidenceBand.MEDIUM,
        rationale="The title and summary both identify Orion.",
        classifier_version="subject-classifier-v1",
        policy_version="auto-assignment-v1",
        signals={"title_match": True, "summary_score": 0.63},
        classified_document_version_id=version_id,
    )

    assert decision.document_id == document_id
    assert decision.classified_document_version_id == version_id
    assert decision.confidence_band is ConfidenceBand.MEDIUM


def test_automatic_decision_rejects_missing_provenance() -> None:
    with pytest.raises(ValueError, match="confidence"):
        DocumentSubjectDecision(
            document_id=uuid.uuid4(),
            subject_id=uuid.uuid4(),
            state=DecisionState.SUGGESTED,
            control_source=DecisionControlSource.AUTOMATIC,
        )


def test_manual_decision_rejects_automatic_provenance() -> None:
    with pytest.raises(ValueError, match="manual decisions must not carry confidence"):
        DocumentSubjectDecision(
            document_id=uuid.uuid4(),
            subject_id=uuid.uuid4(),
            state=DecisionState.ASSIGNED,
            control_source=DecisionControlSource.MANUAL,
            confidence=0.9,
            confidence_band=ConfidenceBand.HIGH,
        )


def test_classification_signals_are_bounded_and_json_serializable() -> None:
    common = {
        "document_id": uuid.uuid4(),
        "subject_id": uuid.uuid4(),
        "state": DecisionState.SUGGESTED,
        "control_source": DecisionControlSource.AUTOMATIC,
        "confidence": 0.4,
        "confidence_band": ConfidenceBand.LOW,
        "rationale": "Weak title evidence.",
        "classifier_version": "v1",
        "policy_version": "v1",
        "classified_document_version_id": uuid.uuid4(),
    }
    with pytest.raises(ValueError, match="32 entries"):
        DocumentSubjectDecision(
            **common,
            signals={f"signal-{index}": index for index in range(33)},
        )
    with pytest.raises(ValueError, match="JSON serializable"):
        DocumentSubjectDecision(**common, signals={"unsupported": object()})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_classification_signals_reject_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="JSON serializable"):
        DocumentSubjectDecision(
            document_id=uuid.uuid4(),
            subject_id=uuid.uuid4(),
            state=DecisionState.SUGGESTED,
            control_source=DecisionControlSource.AUTOMATIC,
            confidence=0.4,
            confidence_band=ConfidenceBand.LOW,
            rationale="Weak title evidence.",
            classifier_version="v1",
            policy_version="v1",
            classified_document_version_id=uuid.uuid4(),
            signals={"score": value},
        )
