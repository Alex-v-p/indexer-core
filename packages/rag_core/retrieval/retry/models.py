from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from packages.rag_core.query_understanding.planning import RetrievalPlan, RetrievalStrategy
from packages.rag_core.retrieval.graders import EvidenceGradingReport


class RetryAction(StrEnum):
    """One controlled adjustment applied before another retrieval attempt."""

    EXPAND_QUERY = "expand_query"
    INCREASE_TOP_K = "increase_top_k"
    SWITCH_PIPELINE = "switch_pipeline"


class RetryStopReason(StrEnum):
    """Why the retry controller stopped evaluating fallback actions."""

    EVIDENCE_SUFFICIENT = "evidence_sufficient"
    RETRY_LIMIT_REACHED = "retry_limit_reached"
    NO_EFFECTIVE_FALLBACK = "no_effective_fallback"


@dataclass(frozen=True, slots=True)
class RetrievalAttempt:
    """Snapshot of one retrieval and evidence-grading attempt."""

    attempt_number: int
    retry_number: int
    query: str
    top_k: int
    retrieval_plan: RetrievalPlan
    evidence_grading: EvidenceGradingReport
    evidence_count: int
    actions: tuple[RetryAction, ...] = ()
    decision_rationale: str | None = None

    def __post_init__(self) -> None:
        if self.attempt_number <= 0:
            raise ValueError("attempt_number must be positive.")
        if self.retry_number < 0:
            raise ValueError("retry_number must not be negative.")
        if self.attempt_number != self.retry_number + 1:
            raise ValueError("attempt_number must equal retry_number + 1.")
        if not self.query.strip():
            raise ValueError("query must not be empty.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.evidence_count < 0:
            raise ValueError("evidence_count must not be negative.")
        if self.retry_number == 0 and (self.actions or self.decision_rationale is not None):
            raise ValueError("The initial attempt cannot contain retry actions or a retry rationale.")
        if self.retry_number > 0:
            if not self.actions:
                raise ValueError("Retry attempts require at least one action.")
            if self.decision_rationale is None or not self.decision_rationale.strip():
                raise ValueError("Retry attempts require a decision rationale.")

    @property
    def strategy(self) -> RetrievalStrategy:
        return self.retrieval_plan.strategy

    def to_metadata(self) -> dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "retry_number": self.retry_number,
            "query": self.query,
            "top_k": self.top_k,
            "pipeline_name": self.retrieval_plan.selected_pipeline_name,
            "strategy": self.retrieval_plan.strategy.value,
            "evidence_count": self.evidence_count,
            "actions": [action.value for action in self.actions],
            "decision_rationale": self.decision_rationale,
            "retrieval_plan": self.retrieval_plan.to_metadata(),
            "evidence_grading": self.evidence_grading.to_metadata(),
        }


@dataclass(frozen=True, slots=True)
class RetrievalRetryContext:
    """Inputs used by a retry policy to choose the next controlled attempt."""

    original_question: str
    current_query: str
    current_top_k: int
    current_plan: RetrievalPlan
    evidence_grading: EvidenceGradingReport
    retries_used: int
    attempted_strategies: tuple[RetrievalStrategy, ...]
    unresolved_retrieval_queries: tuple[str, ...]
    available_pipeline_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.original_question.strip():
            raise ValueError("original_question must not be empty.")
        if not self.current_query.strip():
            raise ValueError("current_query must not be empty.")
        if self.current_top_k <= 0:
            raise ValueError("current_top_k must be positive.")
        if self.retries_used < 0:
            raise ValueError("retries_used must not be negative.")


@dataclass(frozen=True, slots=True)
class RetrievalRetryDecision:
    """Policy output describing either the next attempt or a stop condition."""

    should_retry: bool
    rationale: str
    actions: tuple[RetryAction, ...] = ()
    next_query: str | None = None
    next_top_k: int | None = None
    next_plan: RetrievalPlan | None = None
    stop_reason: RetryStopReason | None = None

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")
        if self.should_retry:
            if not self.actions:
                raise ValueError("Retry decisions require at least one action.")
            if self.next_query is None or not self.next_query.strip():
                raise ValueError("Retry decisions require next_query.")
            if self.next_top_k is None or self.next_top_k <= 0:
                raise ValueError("Retry decisions require a positive next_top_k.")
            if self.next_plan is None:
                raise ValueError("Retry decisions require next_plan.")
            if self.stop_reason is not None:
                raise ValueError("Retry decisions cannot also contain a stop reason.")
        else:
            if self.stop_reason is None:
                raise ValueError("Stop decisions require a stop_reason.")
            if self.actions or self.next_query is not None or self.next_top_k is not None or self.next_plan is not None:
                raise ValueError("Stop decisions cannot contain retry actions or next-attempt values.")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "should_retry": self.should_retry,
            "rationale": self.rationale,
            "actions": [action.value for action in self.actions],
            "next_query": self.next_query,
            "next_top_k": self.next_top_k,
            "next_plan": self.next_plan.to_metadata() if self.next_plan is not None else None,
            "stop_reason": self.stop_reason.value if self.stop_reason is not None else None,
        }


@dataclass(frozen=True, slots=True)
class RetrievalRetryReport:
    """Complete retry history stored in QueryState metadata and the agent trace."""

    policy_name: str
    max_retries: int
    attempts: tuple[RetrievalAttempt, ...]
    stop_reason: RetryStopReason
    stop_rationale: str

    def __post_init__(self) -> None:
        if not self.policy_name.strip():
            raise ValueError("policy_name must not be empty.")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative.")
        if not self.attempts:
            raise ValueError("At least one retrieval attempt is required.")
        if not self.stop_rationale.strip():
            raise ValueError("stop_rationale must not be empty.")
        if self.retries_used > self.max_retries:
            raise ValueError("Retry report exceeds max_retries.")
        expected_numbers = list(range(1, len(self.attempts) + 1))
        if [attempt.attempt_number for attempt in self.attempts] != expected_numbers:
            raise ValueError("Retrieval attempts must be sequentially numbered.")

    @property
    def retries_used(self) -> int:
        return len(self.attempts) - 1

    @property
    def final_attempt(self) -> RetrievalAttempt:
        return self.attempts[-1]

    @property
    def final_sufficient(self) -> bool:
        return self.final_attempt.evidence_grading.sufficient

    @property
    def query_changed(self) -> bool:
        return self.final_attempt.query != self.attempts[0].query

    def to_metadata(self) -> dict[str, Any]:
        final = self.final_attempt
        return {
            "policy_name": self.policy_name,
            "max_retries": self.max_retries,
            "retries_used": self.retries_used,
            "attempt_count": len(self.attempts),
            "stop_reason": self.stop_reason.value,
            "stop_rationale": self.stop_rationale,
            "final_sufficient": self.final_sufficient,
            "final_pipeline_name": final.retrieval_plan.selected_pipeline_name,
            "final_strategy": final.retrieval_plan.strategy.value,
            "final_top_k": final.top_k,
            "final_query": final.query,
            "query_changed": self.query_changed,
            "attempts": [attempt.to_metadata() for attempt in self.attempts],
        }
