from __future__ import annotations

import uuid

import pytest

from packages.rag_core.document_organization import (
    ClassificationConfidenceBand,
    ClassificationSource,
    ContentGroupAssignmentState,
    ContentGroupName,
    DocumentContentGroupAssignment,
    DocumentTypeDecision,
    DocumentTypeDecisionState,
    DocumentTypeKey,
)


def test_document_type_keys_are_extensible_stable_identifiers() -> None:
    assert DocumentTypeKey.from_value("field_research").value == "field_research"
    with pytest.raises(ValueError):
        DocumentTypeKey.from_value("Field Research")


def test_automatic_content_group_proposals_require_two_to_six_meaningful_words() -> None:
    assert ContentGroupName.from_automatic_proposal("Factory Monitoring Dashboard").normalized == "factory monitoring dashboard"
    with pytest.raises(ValueError, match="2 to 6"):
        ContentGroupName.from_automatic_proposal("Dashboard")
    with pytest.raises(ValueError, match="80"):
        ContentGroupName.from_value("x" * 81)


def test_document_type_decisions_enforce_manual_and_automatic_provenance() -> None:
    DocumentTypeDecision(
        document_id=uuid.uuid4(),
        document_type_id=uuid.uuid4(),
        state=DocumentTypeDecisionState.ASSIGNED,
        source=ClassificationSource.MANUAL,
    )
    with pytest.raises(ValueError, match="automatic confidence"):
        DocumentTypeDecision(
            document_id=uuid.uuid4(),
            document_type_id=uuid.uuid4(),
            state=DocumentTypeDecisionState.SUGGESTED,
            source=ClassificationSource.AUTOMATIC,
        )


def test_content_group_assignment_state_and_provenance_matrix() -> None:
    document_id = uuid.uuid4()
    group_id = uuid.uuid4()
    version_id = uuid.uuid4()
    assignment = DocumentContentGroupAssignment(
        document_id=document_id,
        content_group_id=group_id,
        state=ContentGroupAssignmentState.SUGGESTED,
        source=ClassificationSource.AUTOMATIC,
        confidence=0.72,
        confidence_band=ClassificationConfidenceBand.MEDIUM,
        rationale="Representative content matches.",
        classifier_version="group-v1",
        policy_version="group-policy-v1",
        signals={"content_match": True},
        summary_hash="a" * 64,
        classified_document_version_id=version_id,
    )
    assert assignment.is_resolved

    with pytest.raises(ValueError, match="require a group"):
        DocumentContentGroupAssignment(
            document_id=document_id,
            content_group_id=None,
            state=ContentGroupAssignmentState.ASSIGNED,
            source=ClassificationSource.MANUAL,
        )
    with pytest.raises(ValueError, match="must be assigned"):
        DocumentContentGroupAssignment(
            document_id=document_id,
            content_group_id=None,
            state=ContentGroupAssignmentState.PENDING,
            source=ClassificationSource.MANUAL,
        )
    with pytest.raises(ValueError, match="unresolved_reason"):
        DocumentContentGroupAssignment(
            document_id=document_id,
            content_group_id=None,
            state=ContentGroupAssignmentState.UNRESOLVED,
            source=ClassificationSource.AUTOMATIC,
        )
