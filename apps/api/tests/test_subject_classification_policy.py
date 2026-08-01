from __future__ import annotations

import uuid

from packages.rag_core.subjects import (
    ConfidenceBand,
    DecisionState,
    SignalFamily,
    SubjectClassificationCandidate,
    SubjectClassificationInput,
    SubjectKind,
    classify_document_subjects,
)


def _candidate(name: str, *, kind: SubjectKind = SubjectKind.PROJECT, aliases=()):
    return SubjectClassificationCandidate(
        subject_id=uuid.uuid5(uuid.NAMESPACE_URL, f"{kind.value}:{name}"),
        kind=kind,
        canonical_name=name,
        aliases=tuple(aliases),
    )


def test_exact_normalized_name_is_a_high_confidence_assignment() -> None:
    candidate = _candidate("Apollo")

    outcomes = classify_document_subjects(
        SubjectClassificationInput(title="Apollo"),
        (candidate,),
    )

    assert outcomes[0].state is DecisionState.ASSIGNED
    assert outcomes[0].confidence_band is ConfidenceBand.HIGH
    assert outcomes[0].signals[0].family is SignalFamily.TITLE


def test_medium_confidence_requires_two_independent_families() -> None:
    candidate = _candidate("Apollo")

    one_family = classify_document_subjects(
        SubjectClassificationInput(
            title="Architecture notes",
            document_summary="Apollo deployment work",
            model_scores={candidate.subject_id: 0.70},
        ),
        (candidate,),
    )
    two_families = classify_document_subjects(
        SubjectClassificationInput(
            title="Apollo architecture notes",
            filename="apollo-release.md",
        ),
        (candidate,),
    )

    assert one_family[0].confidence_band is ConfidenceBand.MEDIUM
    assert one_family[0].state is DecisionState.SUGGESTED
    assert two_families[0].confidence_band is ConfidenceBand.MEDIUM
    assert two_families[0].independent_family_count == 2
    assert two_families[0].state is DecisionState.ASSIGNED


def test_ambiguous_same_kind_matches_are_suggestions_but_kinds_are_independent() -> None:
    project_a = _candidate("Alpha", aliases=("Apollo",))
    project_b = _candidate("Beta", aliases=("Apollo",))
    topic = _candidate("Apollo", kind=SubjectKind.TOPIC)

    outcomes = classify_document_subjects(
        SubjectClassificationInput(title="Apollo"),
        (project_a, project_b, topic),
    )
    by_subject = {outcome.subject_id: outcome for outcome in outcomes}

    assert by_subject[project_a.subject_id].state is DecisionState.SUGGESTED
    assert by_subject[project_b.subject_id].state is DecisionState.SUGGESTED
    assert by_subject[topic.subject_id].state is DecisionState.ASSIGNED


def test_no_match_remains_unclassified() -> None:
    outcomes = classify_document_subjects(
        SubjectClassificationInput(title="Unrelated notes"),
        (_candidate("Apollo"),),
    )

    assert outcomes == ()
