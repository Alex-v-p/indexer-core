from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

from packages.rag_core.query_understanding.planning import RetrievalPlan, RetrievalStrategy
from packages.rag_core.retrieval.retry.models import (
    RetryAction,
    RetryStopReason,
    RetrievalRetryContext,
    RetrievalRetryDecision,
)

_FALLBACK_ORDER: Mapping[RetrievalStrategy, tuple[RetrievalStrategy, ...]] = MappingProxyType(
    {
        RetrievalStrategy.BASELINE: (
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.MULTI_QUERY,
            RetrievalStrategy.RERANK,
        ),
        RetrievalStrategy.HYBRID: (
            RetrievalStrategy.MULTI_QUERY,
            RetrievalStrategy.RERANK,
        ),
        RetrievalStrategy.CONTEXTUAL: (
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.MULTI_QUERY,
            RetrievalStrategy.RERANK,
        ),
        RetrievalStrategy.MULTI_QUERY: (
            RetrievalStrategy.RERANK,
            RetrievalStrategy.HYBRID,
        ),
        RetrievalStrategy.RERANK: (
            RetrievalStrategy.MULTI_QUERY,
            RetrievalStrategy.HYBRID,
        ),
    },
)


class RuleBasedRetrievalRetryPolicy:
    """Deterministic escalation policy for weak or missing evidence.

    The policy only decides what should change. Agent nodes remain responsible
    for executing retrieval and evidence grading, which keeps fallback rules
    independently testable and avoids embedding domain policy in orchestration.
    """

    name = "rule_based_retrieval_retry_policy"

    def __init__(
        self,
        *,
        pipeline_names: Mapping[RetrievalStrategy, str],
        max_retries: int = 2,
        top_k_multiplier: float = 2.0,
        max_top_k: int = 20,
        expand_query: bool = True,
        max_query_chars: int = 1_200,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative.")
        if top_k_multiplier < 1.0:
            raise ValueError("top_k_multiplier must be at least 1.0.")
        if max_top_k <= 0:
            raise ValueError("max_top_k must be positive.")
        if max_query_chars <= 0:
            raise ValueError("max_query_chars must be positive.")

        normalized: dict[RetrievalStrategy, str] = {}
        for strategy, pipeline_name in pipeline_names.items():
            value = pipeline_name.strip()
            if not value:
                raise ValueError("pipeline names must not be empty.")
            normalized[strategy] = value

        self._pipeline_names = MappingProxyType(normalized)
        self.max_retries = max_retries
        self._top_k_multiplier = top_k_multiplier
        self._max_top_k = max_top_k
        self._expand_query = expand_query
        self._max_query_chars = max_query_chars

    def decide(self, context: RetrievalRetryContext) -> RetrievalRetryDecision:
        if context.evidence_grading.sufficient:
            return RetrievalRetryDecision(
                should_retry=False,
                rationale="All required information needs are supported by the current evidence.",
                stop_reason=RetryStopReason.EVIDENCE_SUFFICIENT,
            )

        if context.retries_used >= self.max_retries:
            return RetrievalRetryDecision(
                should_retry=False,
                rationale=(
                    f"Evidence remains {context.evidence_grading.status.value}, but the controlled retry limit "
                    f"of {self.max_retries} has been reached."
                ),
                stop_reason=RetryStopReason.RETRY_LIMIT_REACHED,
            )

        next_strategy = self._next_strategy(context)
        next_query = self._next_query(context)
        next_top_k = self._next_top_k(context.current_top_k)

        actions: list[RetryAction] = []
        if next_query != context.current_query:
            actions.append(RetryAction.EXPAND_QUERY)
        if next_top_k != context.current_top_k:
            actions.append(RetryAction.INCREASE_TOP_K)
        if next_strategy is not None and next_strategy is not context.current_plan.strategy:
            actions.append(RetryAction.SWITCH_PIPELINE)

        if not actions:
            return RetrievalRetryDecision(
                should_retry=False,
                rationale=(
                    "Evidence is still insufficient, but no untried fallback pipeline, query expansion, or top-k "
                    "increase is available within the configured bounds."
                ),
                stop_reason=RetryStopReason.NO_EFFECTIVE_FALLBACK,
            )

        selected_strategy = next_strategy or context.current_plan.strategy
        selected_pipeline = self._pipeline_names.get(
            selected_strategy,
            context.current_plan.selected_pipeline_name,
        )
        next_plan = RetrievalPlan(
            strategy=selected_strategy,
            selected_pipeline_name=selected_pipeline,
            rationale=self._rationale(context, actions, selected_strategy, next_top_k),
            planner_name=self.name,
            based_on_query_type=context.current_plan.based_on_query_type,
            metadata_filter_hints=context.current_plan.metadata_filter_hints,
            requires_reranking=selected_strategy is RetrievalStrategy.RERANK,
            target_information_need_ids=context.current_plan.target_information_need_ids,
        )
        return RetrievalRetryDecision(
            should_retry=True,
            rationale=next_plan.rationale,
            actions=tuple(actions),
            next_query=next_query,
            next_top_k=next_top_k,
            next_plan=next_plan,
        )

    def _next_strategy(self, context: RetrievalRetryContext) -> RetrievalStrategy | None:
        attempted = set(context.attempted_strategies)
        available = set(context.available_pipeline_names)
        for strategy in _FALLBACK_ORDER[context.current_plan.strategy]:
            pipeline_name = self._pipeline_names.get(strategy)
            if pipeline_name is None or pipeline_name not in available or strategy in attempted:
                continue
            return strategy
        return None

    def _next_top_k(self, current_top_k: int) -> int:
        if current_top_k >= self._max_top_k or self._top_k_multiplier == 1.0:
            return current_top_k
        multiplied = math.ceil(current_top_k * self._top_k_multiplier)
        return min(self._max_top_k, max(current_top_k + 1, multiplied))

    def _next_query(self, context: RetrievalRetryContext) -> str:
        if not self._expand_query or not context.unresolved_retrieval_queries:
            return context.current_query

        parts = [context.original_question]
        seen = {" ".join(context.original_question.lower().split())}
        for query in context.unresolved_retrieval_queries:
            normalized = " ".join(query.strip().split())
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            parts.append(normalized)
            seen.add(key)

        expanded = " | ".join(parts)
        if len(expanded) > self._max_query_chars:
            expanded = expanded[: self._max_query_chars].rstrip(" |")
        return expanded or context.current_query

    @staticmethod
    def _rationale(
        context: RetrievalRetryContext,
        actions: list[RetryAction],
        selected_strategy: RetrievalStrategy,
        next_top_k: int,
    ) -> str:
        action_text = ", ".join(action.value for action in actions)
        unresolved_count = len(context.evidence_grading.unresolved_information)
        return (
            f"Evidence was graded {context.evidence_grading.status.value} with "
            f"{unresolved_count} unresolved required information need(s). Apply {action_text}; "
            f"retry with strategy {selected_strategy.value} and top_k={next_top_k}."
        )
