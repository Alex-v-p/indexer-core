from __future__ import annotations

import re
import unicodedata
from itertools import combinations
from typing import Callable, Sequence

from packages.rag_core.evaluation.models import MetricValue
from packages.rag_core.evaluation.stability.models import (
    StabilityAttemptSnapshot,
    StabilityMetrics,
    StructuredDiagnosticSnapshot,
    StructuredStageRates,
)

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def normalize_answer(value: str | None) -> str:
    """NFKC-normalize, case-fold, and collapse Unicode whitespace."""

    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def answer_tokens(value: str | None) -> frozenset[str]:
    """Return deterministic Unicode word tokens from the normalized answer."""

    return frozenset(_TOKEN_PATTERN.findall(normalize_answer(value)))


def calculate_stability_metrics(
    attempt_groups: Sequence[Sequence[StabilityAttemptSnapshot]],
) -> StabilityMetrics:
    all_attempts = [attempt for group in attempt_groups for attempt in group]
    successful_groups = [
        [attempt for attempt in group if attempt.status == "succeeded"]
        for group in attempt_groups
    ]
    successful_attempts = [attempt for group in successful_groups for attempt in group]
    technical = _rate(
        sum(attempt.status == "succeeded" for attempt in all_attempts),
        len(all_attempts),
        empty_details="No attempts were supplied.",
    )
    return StabilityMetrics(
        technical_success_rate=technical,
        outcome_consistency=_pairwise_exact_metric(
            [[attempt.outcome for attempt in group] for group in successful_groups],
            label="successful attempt outcomes",
        ),
        route_signature_consistency=_pairwise_exact_metric(
            [[attempt.route_signature for attempt in group] for group in successful_groups],
            label="successful route signatures",
        ),
        evidence_exact_set_agreement=_pairwise_exact_metric(
            [[attempt.evidence_identities for attempt in group] for group in successful_groups],
            label="successful evidence identity sets",
        ),
        evidence_mean_pairwise_jaccard=_pairwise_similarity_metric(
            [[frozenset(attempt.evidence_identities) for attempt in group] for group in successful_groups],
            _jaccard,
            label="successful evidence identity sets",
        ),
        normalized_answer_exact_match_rate=_pairwise_exact_metric(
            [[normalize_answer(attempt.answer) for attempt in group] for group in successful_groups],
            label="successful normalized answers",
        ),
        normalized_answer_mean_pairwise_token_jaccard=_pairwise_similarity_metric(
            [[answer_tokens(attempt.answer) for attempt in group] for group in successful_groups],
            _jaccard,
            label="successful normalized answer token sets",
        ),
        presentation_signature_consistency=_pairwise_exact_metric(
            [
                [_presentation_signature_from_snapshot(attempt) for attempt in group]
                for group in successful_groups
            ],
            label="successful presentation signatures",
        ),
        structured_stage_rates=_structured_stage_rates(successful_attempts),
    )


def _presentation_signature_from_snapshot(attempt: StabilityAttemptSnapshot) -> tuple[str, ...]:
    presentation = attempt.answer_presentation
    if presentation is None:
        return ("absent", str(attempt.outcome), f"body:{bool(attempt.answer)}")
    supported = presentation.get("supported_information")
    unresolved = presentation.get("unresolved_information")
    return (
        f"schema:{presentation.get('schema_version')}",
        f"outcome:{presentation.get('outcome')}",
        f"body:{bool(presentation.get('body'))}",
        f"supported_section:{bool(supported)}",
        f"supported_count:{len(supported) if isinstance(supported, list) else 0}",
        f"unresolved_section:{bool(unresolved)}",
        f"unresolved_count:{len(unresolved) if isinstance(unresolved, list) else 0}",
        f"citation_count:{presentation.get('citation_count', 0)}",
    )


def _pairwise_exact_metric(groups: Sequence[Sequence[object]], *, label: str) -> MetricValue:
    return _pairwise_similarity_metric(
        groups,
        lambda left, right: 1.0 if left == right else 0.0,
        label=label,
    )


def _pairwise_similarity_metric(
    groups: Sequence[Sequence[object]],
    similarity: Callable[[object, object], float],
    *,
    label: str,
) -> MetricValue:
    values: list[float] = []
    singletons = 0
    for group in groups:
        if not group:
            continue
        if len(group) == 1:
            values.append(1.0)
            singletons += 1
            continue
        values.extend(similarity(left, right) for left, right in combinations(group, 2))
    if not values:
        return MetricValue(
            value=None,
            status="not_applicable",
            details=f"No technically successful {label} were available.",
        )
    return MetricValue(
        value=sum(values) / len(values),
        status="computed",
        details=(
            f"Mean across {len(values)} within-case comparison(s); "
            f"{singletons} single-success case(s) contributed a defined value of 1.0."
        ),
    )


def _jaccard(left: object, right: object) -> float:
    left_set = frozenset(left)  # type: ignore[arg-type]
    right_set = frozenset(right)  # type: ignore[arg-type]
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def _rate(numerator: int, denominator: int, *, empty_details: str) -> MetricValue:
    if denominator == 0:
        return MetricValue(value=None, status="not_applicable", details=empty_details)
    return MetricValue(
        value=numerator / denominator,
        status="computed",
        details=f"{numerator}/{denominator}.",
    )


def _structured_stage_rates(
    attempts: Sequence[StabilityAttemptSnapshot],
) -> tuple[StructuredStageRates, ...]:
    by_stage: dict[str, list[StructuredDiagnosticSnapshot]] = {}
    for attempt in attempts:
        for diagnostic in attempt.structured_diagnostics:
            by_stage.setdefault(diagnostic.stage, []).append(diagnostic)
    return tuple(
        StructuredStageRates(
            stage=stage,
            observation_count=len(items),
            repair_rate=_rate(
                sum(item.repair_attempted for item in items),
                len(items),
                empty_details="No safe structured diagnostics were available.",
            ),
            fallback_rate=_rate(
                sum(item.outcome == "fallback" for item in items),
                len(items),
                empty_details="No safe structured diagnostics were available.",
            ),
        )
        for stage, items in sorted(by_stage.items())
    )
