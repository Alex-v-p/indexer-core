from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

from packages.rag_core.query_understanding.planning import RetrievalPlan, RetrievalStrategy
from packages.rag_core.retrieval.retry.models import (
    InformationNeedRetryAction,
    InformationNeedRetryContext,
    InformationNeedRetryDecision,
    RetryAction,
    RetryStopReason,
    RetrievalRetryContext,
    RetrievalRetryDecision,
)

_FALLBACK_ORDER: Mapping[RetrievalStrategy, tuple[RetrievalStrategy, ...]] = MappingProxyType(
    {
        RetrievalStrategy.BASELINE: (
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.HIERARCHICAL,
            RetrievalStrategy.MULTI_QUERY,
            RetrievalStrategy.RERANK,
        ),
        RetrievalStrategy.HYBRID: (
            RetrievalStrategy.HIERARCHICAL,
            RetrievalStrategy.MULTI_QUERY,
            RetrievalStrategy.RERANK,
        ),
        RetrievalStrategy.CONTEXTUAL: (
            RetrievalStrategy.HIERARCHICAL,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.MULTI_QUERY,
            RetrievalStrategy.RERANK,
        ),
        RetrievalStrategy.HIERARCHICAL: (
            RetrievalStrategy.CONTEXTUAL,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.MULTI_QUERY,
            RetrievalStrategy.RERANK,
        ),
        RetrievalStrategy.MULTI_QUERY: (
            RetrievalStrategy.RERANK,
            RetrievalStrategy.HIERARCHICAL,
            RetrievalStrategy.HYBRID,
        ),
        RetrievalStrategy.RERANK: (
            RetrievalStrategy.MULTI_QUERY,
            RetrievalStrategy.HIERARCHICAL,
            RetrievalStrategy.HYBRID,
        ),
    },
)


class RuleBasedRetrievalRetryPolicy:
    """Deterministic escalation policy for claim-level weak or missing evidence.

    Claim re-planning decides what unresolved answer requirements to retrieve.
    This policy decides how aggressively those focused lookups should run by
    changing top-k and the retrieval pipeline within a bounded retry budget.
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

    def decide_information_need(
        self,
        context: InformationNeedRetryContext,
    ) -> InformationNeedRetryDecision:
        """Route one information need without selecting its next pipeline.

        The planner owns query, top-k, and strategy selection. This controller
        only enforces support, per-item/global budgets, and one conservative
        reclassification opportunity after a low-confidence missing result.
        """

        if context.evidence_grading.sufficient:
            return InformationNeedRetryDecision(
                action=InformationNeedRetryAction.COMPLETE_SUPPORTED,
                reason="evidence_sufficient",
                rationale="The latest grade supports this required information need.",
            )
        if context.total_attempts_used >= context.max_total_attempts:
            return InformationNeedRetryDecision(
                action=InformationNeedRetryAction.COMPLETE_EXHAUSTED,
                reason="global_attempt_limit_reached",
                rationale=(
                    f"The query-level retrieval budget of {context.max_total_attempts} attempts has been reached; "
                    "this information need remains unresolved."
                ),
            )
        if context.attempts_used >= context.max_attempts:
            return InformationNeedRetryDecision(
                action=InformationNeedRetryAction.COMPLETE_EXHAUSTED,
                reason="information_need_attempt_limit_reached",
                rationale=(
                    f"This information need used all {context.max_attempts} allowed retrieval attempts and remains "
                    f"{context.evidence_grading.status.value}."
                ),
            )
        if (
            context.evidence_grading.status.value == "missing"
            and context.classification_confidence < 0.55
            and context.reclassifications_used < context.max_reclassifications
        ):
            return InformationNeedRetryDecision(
                action=InformationNeedRetryAction.RECLASSIFY,
                reason="low_confidence_missing_evidence",
                rationale=(
                    "No relevant evidence was found and the information-need classification is low-confidence. "
                    "Reclassify this item before producing the next retrieval plan."
                ),
            )
        return InformationNeedRetryDecision(
            action=InformationNeedRetryAction.RETRY,
            reason="evidence_insufficient_retry_available",
            rationale=(
                f"Evidence remains {context.evidence_grading.status.value}; return this information need to its "
                "planner with the latest grader feedback and independent attempt history."
            ),
        )

    def decide(self, context: RetrievalRetryContext) -> RetrievalRetryDecision:
        if context.evidence_grading.sufficient:
            return RetrievalRetryDecision(
                should_retry=False,
                rationale="All required claims are supported by the accumulated evidence.",
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

        claim_plan = context.claim_retrieval_plan
        if claim_plan is None or not claim_plan.tasks:
            return RetrievalRetryDecision(
                should_retry=False,
                rationale=(
                    "Evidence is still insufficient, but claim-level re-planning produced no executable unresolved "
                    "claim lookups."
                ),
                stop_reason=RetryStopReason.NO_EFFECTIVE_FALLBACK,
            )

        next_strategy = self._next_strategy(context)
        next_query = self._next_query(context)
        next_top_k = self._next_top_k(context.current_top_k)

        query_changed = next_query != context.current_query
        top_k_changed = next_top_k != context.current_top_k
        strategy_changed = next_strategy is not None and next_strategy is not context.current_plan.strategy
        claim_targets_changed = (
            claim_plan.target_information_need_ids != context.current_plan.target_information_need_ids
        )
        if not (query_changed or top_k_changed or strategy_changed or claim_targets_changed):
            return RetrievalRetryDecision(
                should_retry=False,
                rationale=(
                    "Evidence is still insufficient, but the claim targets, focused queries, top-k, and retrieval "
                    "strategy are unchanged, so another lookup would repeat the previous attempt."
                ),
                stop_reason=RetryStopReason.NO_EFFECTIVE_FALLBACK,
            )

        actions: list[RetryAction] = [RetryAction.TARGET_UNRESOLVED_CLAIMS]
        if query_changed:
            actions.append(RetryAction.EXPAND_QUERY)
        if top_k_changed:
            actions.append(RetryAction.INCREASE_TOP_K)
        if strategy_changed:
            actions.append(RetryAction.SWITCH_PIPELINE)

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
            target_information_need_ids=claim_plan.target_information_need_ids,
            document_constraint=context.current_plan.document_constraint,
            version_constraint=context.current_plan.version_constraint,
            date_constraints=context.current_plan.date_constraints,
        )
        return RetrievalRetryDecision(
            should_retry=True,
            rationale=next_plan.rationale,
            actions=tuple(actions),
            next_query=next_query,
            next_top_k=next_top_k,
            next_plan=next_plan,
            claim_retrieval_plan=claim_plan,
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
        claim_plan = context.claim_retrieval_plan
        if claim_plan is not None:
            queries = tuple(task.retrieval_query for task in claim_plan.tasks)
        else:
            queries = context.unresolved_retrieval_queries
        if not self._expand_query or not queries:
            return context.current_query

        expanded = " || ".join(queries)
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
        claim_plan = context.claim_retrieval_plan
        target_ids = (
            ", ".join(claim_plan.target_information_need_ids)
            if claim_plan is not None
            else "unresolved claims"
        )
        return (
            f"Evidence was graded {context.evidence_grading.status.value}. Apply {action_text}; run independent "
            f"lookups for {target_ids} with strategy {selected_strategy.value} and top_k={next_top_k}, then merge "
            "the new chunks with evidence retained from earlier attempts before re-grading every claim."
        )
