from __future__ import annotations

import math
import re
from collections.abc import Mapping
from types import MappingProxyType

from packages.rag_core.query_understanding.classification import QueryClassification, QueryType
from packages.rag_core.query_understanding.decomposition import InformationNeedDecomposition
from packages.rag_core.query_understanding.planning.models import (
    ClaimPlanningInput,
    ClaimRetrievalPlan,
    ClaimRetrievalTask,
    ClaimSupportStatus,
    InformationNeedPlanningContext,
    InformationNeedPlanningStop,
    InformationNeedRetrievalPlan,
    RetrievalPlan,
    RetrievalStrategy,
)
from packages.rag_core.retrieval.graders import InformationNeedSupport

_RERANK_PATTERN = re.compile(
    r"\b(most relevant|best evidence|strongest evidence|most important|rank|prioriti[sz]e|which .* best)\b",
    re.IGNORECASE,
)

_RETRIEVAL_QUERY_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_RETRIEVAL_QUERY_NOISE_TERMS = frozenset(
    {
        "a",
        "an",
        "answer",
        "are",
        "describe",
        "determine",
        "do",
        "does",
        "explain",
        "find",
        "for",
        "give",
        "identify",
        "in",
        "is",
        "of",
        "on",
        "provide",
        "tell",
        "the",
        "to",
        "what",
        "which",
        "who",
        "why",
    },
)

_FALLBACK_ORDER: Mapping[RetrievalStrategy, tuple[RetrievalStrategy, ...]] = MappingProxyType(
    {
        RetrievalStrategy.BASELINE: (
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.HIERARCHICAL,
            RetrievalStrategy.MULTI_QUERY,
            RetrievalStrategy.RERANK,
            RetrievalStrategy.CONTEXTUAL,
        ),
        RetrievalStrategy.HYBRID: (
            RetrievalStrategy.HIERARCHICAL,
            RetrievalStrategy.MULTI_QUERY,
            RetrievalStrategy.RERANK,
            RetrievalStrategy.CONTEXTUAL,
            RetrievalStrategy.BASELINE,
        ),
        RetrievalStrategy.CONTEXTUAL: (
            RetrievalStrategy.HIERARCHICAL,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.MULTI_QUERY,
            RetrievalStrategy.RERANK,
            RetrievalStrategy.BASELINE,
        ),
        RetrievalStrategy.HIERARCHICAL: (
            RetrievalStrategy.CONTEXTUAL,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.MULTI_QUERY,
            RetrievalStrategy.RERANK,
            RetrievalStrategy.BASELINE,
        ),
        RetrievalStrategy.MULTI_QUERY: (
            RetrievalStrategy.RERANK,
            RetrievalStrategy.HIERARCHICAL,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.CONTEXTUAL,
            RetrievalStrategy.BASELINE,
        ),
        RetrievalStrategy.RERANK: (
            RetrievalStrategy.MULTI_QUERY,
            RetrievalStrategy.HIERARCHICAL,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.CONTEXTUAL,
            RetrievalStrategy.BASELINE,
        ),
    },
)


class RuleBasedRetrievalPlanner:
    """Explainable query and information-need retrieval planner.

    ``plan`` is retained for the selectable single-plan compatibility boundary.
    The agentic pipeline uses ``plan_information_need`` so every decomposed item
    receives an independent query, pipeline, top-k, and retry history.
    """

    name = "rule_based_retrieval_planner"

    def __init__(
        self,
        *,
        baseline_pipeline_name: str,
        hybrid_pipeline_name: str,
        contextual_pipeline_name: str,
        multi_query_pipeline_name: str,
        rerank_pipeline_name: str,
        hierarchical_pipeline_name: str = "hierarchical_rag",
        low_confidence_threshold: float = 0.55,
        contextual_available: bool = True,
        hierarchical_available: bool = False,
        top_k_multiplier: float = 2.0,
        max_top_k: int = 20,
        expand_query: bool = True,
        max_query_chars: int = 1_200,
    ) -> None:
        if not 0.0 <= low_confidence_threshold <= 1.0:
            raise ValueError("low_confidence_threshold must be between 0 and 1.")
        if top_k_multiplier < 1.0:
            raise ValueError("top_k_multiplier must be at least 1.0.")
        if max_top_k <= 0:
            raise ValueError("max_top_k must be positive.")
        if max_query_chars <= 0:
            raise ValueError("max_query_chars must be positive.")
        self._pipeline_names = {
            RetrievalStrategy.BASELINE: _require_pipeline_name(baseline_pipeline_name),
            RetrievalStrategy.HYBRID: _require_pipeline_name(hybrid_pipeline_name),
            RetrievalStrategy.CONTEXTUAL: _require_pipeline_name(contextual_pipeline_name),
            RetrievalStrategy.MULTI_QUERY: _require_pipeline_name(multi_query_pipeline_name),
            RetrievalStrategy.RERANK: _require_pipeline_name(rerank_pipeline_name),
            RetrievalStrategy.HIERARCHICAL: _require_pipeline_name(hierarchical_pipeline_name),
        }
        self._low_confidence_threshold = low_confidence_threshold
        self._contextual_available = contextual_available
        self._hierarchical_available = hierarchical_available
        self._top_k_multiplier = top_k_multiplier
        self._max_top_k = max_top_k
        self._expand_query = expand_query
        self._max_query_chars = max_query_chars

    async def plan(
        self,
        question: str,
        classification: QueryClassification,
        decomposition: InformationNeedDecomposition,
    ) -> RetrievalPlan:
        """Build the previous whole-question plan for non-agentic compatibility."""

        normalized = " ".join(question.strip().split())
        if not normalized:
            raise ValueError("question must not be empty.")

        strategy, rationale = self._select_strategy(
            normalized,
            classification,
            information_need_count=len(decomposition.information_needs),
        )
        return RetrievalPlan(
            strategy=strategy,
            selected_pipeline_name=self._pipeline_names[strategy],
            rationale=rationale,
            planner_name=self.name,
            based_on_query_type=classification.query_type,
            metadata_filter_hints=classification.metadata_filter_hints,
            requires_reranking=strategy is RetrievalStrategy.RERANK,
            target_information_need_ids=tuple(
                need.need_id for need in decomposition.information_needs if need.required
            ),
            document_constraint=classification.document_constraint,
            version_constraint=classification.version_constraint,
            date_constraints=classification.date_constraints,
        )

    async def plan_information_need(
        self,
        context: InformationNeedPlanningContext,
    ) -> InformationNeedRetrievalPlan | InformationNeedPlanningStop:
        """Plan one distinct attempt using only this item's history and grade."""

        attempt_number = context.attempts_used + 1
        previous_strategies = tuple(plan.strategy for plan in context.previous_plans)
        if not context.previous_plans:
            preferred_strategy, base_rationale = self._select_information_need_strategy(
                context.information_need.retrieval_query,
                context.classification,
            )
            strategy = self._available_initial_strategy(
                preferred_strategy,
                context.available_pipeline_names,
            )
            if strategy is not preferred_strategy:
                base_rationale += (
                    f" The preferred {preferred_strategy.value} pipeline is unavailable, so "
                    f"{strategy.value} retrieval is used as the bounded starting fallback."
                )
            adjustments_list: list[str] = []
            if context.preferred_document is not None:
                adjustments_list.append("prefer_primary_document")
                base_rationale += (
                    f" Prefer {context.preferred_document.document.display_name!r} as the primary source "
                    "without excluding directly relevant supporting documents."
                )
            adjustments = tuple(adjustments_list)
            top_k = context.current_top_k
            query = _normalized_query(context.information_need.retrieval_query)
        else:
            current = context.previous_plans[-1]
            strategy = self._next_untried_strategy(
                current.strategy,
                previous_strategies,
                context.available_pipeline_names,
            )
            query = self._retry_query(context)
            top_k = self._retry_top_k(current.top_k)
            adjustments_list: list[str] = ["target_information_need"]
            if context.preferred_document is not None:
                adjustments_list.append("prefer_primary_document")
            if query != current.query:
                adjustments_list.append("expand_query")
            if top_k != current.top_k:
                adjustments_list.append("increase_top_k")
            if strategy is not current.strategy:
                adjustments_list.append("switch_pipeline")
            adjustments = tuple(adjustments_list)
            grade_status = context.previous_grade.status.value if context.previous_grade is not None else "unknown"
            base_rationale = (
                f"The previous attempt for {context.information_need.need_id} was graded {grade_status}. "
                "Re-plan only this information item using its own grader feedback and attempt history."
            )

        pipeline_name = self._pipeline_names[strategy]
        signature = (pipeline_name, query, top_k)
        previous_signatures = {plan.execution_signature for plan in context.previous_plans}
        if signature in previous_signatures:
            return InformationNeedPlanningStop(
                information_need_id=context.information_need.need_id,
                reason="no_effective_fallback",
                rationale=(
                    "The planner exhausted distinct query, top-k, and pipeline combinations for this information need; "
                    "another attempt would repeat earlier work."
                ),
            )

        return InformationNeedRetrievalPlan(
            information_need_id=context.information_need.need_id,
            strategy=strategy,
            selected_pipeline_name=pipeline_name,
            query=query,
            top_k=top_k,
            rationale=(
                f"{base_rationale} Execute attempt {attempt_number}/{context.max_attempts} with "
                f"{strategy.value} retrieval and top_k={top_k}."
            ),
            planner_name=self.name,
            based_on_query_type=context.classification.query_type,
            attempt_number=attempt_number,
            metadata_filter_hints=context.classification.metadata_filter_hints,
            requires_reranking=strategy is RetrievalStrategy.RERANK,
            adjustments=adjustments,
            document_constraint=context.classification.document_constraint,
            version_constraint=context.classification.version_constraint,
            date_constraints=context.classification.date_constraints,
            preferred_document=context.preferred_document,
        )

    def _select_strategy(
        self,
        question: str,
        classification: QueryClassification,
        *,
        information_need_count: int,
    ) -> tuple[RetrievalStrategy, str]:
        if information_need_count > 1:
            return (
                RetrievalStrategy.MULTI_QUERY,
                f"The independently produced decomposition contains {information_need_count} gradable information needs, "
                "so multi-query retrieval is selected for the compatibility whole-question plan.",
            )
        return self._select_information_need_strategy(question, classification)

    def _select_information_need_strategy(
        self,
        query: str,
        classification: QueryClassification,
    ) -> tuple[RetrievalStrategy, str]:
        if classification.query_type is QueryType.VERSION_SPECIFIC:
            return (
                RetrievalStrategy.HYBRID,
                "Version- and date-specific wording benefits from semantic and exact lexical matching.",
            )
        if classification.query_type is QueryType.COMPARISON:
            return (
                RetrievalStrategy.MULTI_QUERY,
                "This individual information need is comparative and benefits from query expansion and fused retrieval.",
            )
        if _RERANK_PATTERN.search(query) or classification.confidence < self._low_confidence_threshold:
            return (
                RetrievalStrategy.RERANK,
                "The information need asks for discriminative evidence or has low classification confidence, so candidates are reranked.",
            )
        if classification.query_type is QueryType.BROAD_EXPLANATION:
            if self._hierarchical_available and not classification.document_constraint.active:
                return (
                    RetrievalStrategy.HIERARCHICAL,
                    "This broad information need is not scoped to one named document, so document and section "
                    "summaries first route retrieval to the most relevant source areas before precise chunk lookup.",
                )
            if self._contextual_available:
                return (
                    RetrievalStrategy.CONTEXTUAL,
                    "This broad information need is scoped closely enough for contextualized chunks with surrounding document meaning.",
                )
            return (
                RetrievalStrategy.MULTI_QUERY,
                "Hierarchical and contextual retrieval are unavailable, so query expansion covers the broad information need.",
            )
        if classification.needs_metadata_filters:
            return (
                RetrievalStrategy.HYBRID,
                "Metadata-like document, author, section, or date wording benefits from exact lexical matching plus semantic recall.",
            )
        return (
            RetrievalStrategy.BASELINE,
            "This focused information need has no special filtering or ranking requirements, so dense retrieval is sufficient.",
        )

    def _available_initial_strategy(
        self,
        preferred: RetrievalStrategy,
        available_pipeline_names: tuple[str, ...],
    ) -> RetrievalStrategy:
        available = set(available_pipeline_names)
        if self._strategy_enabled(preferred) and self._pipeline_names[preferred] in available:
            return preferred
        if self._pipeline_names[RetrievalStrategy.BASELINE] in available:
            return RetrievalStrategy.BASELINE
        for candidate in RetrievalStrategy:
            if self._strategy_enabled(candidate) and self._pipeline_names[candidate] in available:
                return candidate
        raise ValueError("At least one configured retrieval pipeline must be available.")

    def _next_untried_strategy(
        self,
        current: RetrievalStrategy,
        attempted: tuple[RetrievalStrategy, ...],
        available_pipeline_names: tuple[str, ...],
    ) -> RetrievalStrategy:
        available = set(available_pipeline_names)
        attempted_set = set(attempted)
        for candidate in _FALLBACK_ORDER[current]:
            if not self._strategy_enabled(candidate):
                continue
            pipeline_name = self._pipeline_names[candidate]
            if pipeline_name in available and candidate not in attempted_set:
                return candidate
        return current

    def _strategy_enabled(self, strategy: RetrievalStrategy) -> bool:
        if strategy is RetrievalStrategy.CONTEXTUAL:
            return self._contextual_available
        if strategy is RetrievalStrategy.HIERARCHICAL:
            return self._hierarchical_available
        return True

    def _retry_query(self, context: InformationNeedPlanningContext) -> str:
        base = _normalized_query(context.information_need.retrieval_query)
        if not self._expand_query or context.previous_grade is None:
            return base
        requirement = _normalized_query(context.information_need.description)
        expanded = _merge_retrieval_phrases(base, requirement)
        return expanded[: self._max_query_chars].rstrip() or base

    def _retry_top_k(self, current_top_k: int) -> int:
        if current_top_k >= self._max_top_k or self._top_k_multiplier == 1.0:
            return current_top_k
        multiplied = math.ceil(current_top_k * self._top_k_multiplier)
        return min(self._max_top_k, max(current_top_k + 1, multiplied))


def _merge_retrieval_phrases(base: str, requirement: str) -> str:
    """Merge clean search terms without leaking planner instructions or grader prose."""

    base_terms = {
        match.group(0).casefold()
        for match in _RETRIEVAL_QUERY_TOKEN_PATTERN.finditer(base)
    }
    additional: list[str] = []
    for match in _RETRIEVAL_QUERY_TOKEN_PATTERN.finditer(requirement):
        token = match.group(0)
        normalized = token.casefold()
        if normalized in base_terms or normalized in _RETRIEVAL_QUERY_NOISE_TERMS:
            continue
        base_terms.add(normalized)
        additional.append(token)
    return " ".join((base, *additional)).strip()


def _require_pipeline_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("pipeline names must not be empty.")
    return normalized


def _normalized_query(value: str) -> str:
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise ValueError("retrieval query must not be empty.")
    return normalized


class RuleBasedClaimRetrievalPlanner:
    """Legacy focused claim planner retained for prior API compatibility."""

    name = "rule_based_claim_retrieval_planner"

    def __init__(self, *, max_claims_per_retry: int = 3, max_query_chars: int = 1_200) -> None:
        if max_claims_per_retry <= 0:
            raise ValueError("max_claims_per_retry must be positive.")
        if max_query_chars <= 0:
            raise ValueError("max_query_chars must be positive.")
        self._max_claims_per_retry = max_claims_per_retry
        self._max_query_chars = max_query_chars

    async def plan_claims(
        self,
        question: str,
        classification: QueryClassification,
        current_plan: RetrievalPlan,
        unresolved_claims: tuple[ClaimPlanningInput, ...],
    ) -> ClaimRetrievalPlan:
        normalized_question = " ".join(question.strip().split())
        if not normalized_question:
            raise ValueError("question must not be empty.")
        if not unresolved_claims:
            raise ValueError("At least one unresolved claim is required.")

        ordered = sorted(
            unresolved_claims,
            key=lambda claim: (
                0 if claim.support_status is ClaimSupportStatus.MISSING else 1,
                claim.coverage_score,
                claim.information_need_id,
            ),
        )
        selected = ordered[: self._max_claims_per_retry]
        deferred = ordered[self._max_claims_per_retry :]
        tasks = tuple(self._build_task(claim) for claim in selected)
        target_ids = ", ".join(task.information_need_id for task in tasks)
        rationale = (
            f"Re-plan only unresolved claims ({target_ids}) instead of broadening the complete question. "
            f"The original {current_plan.strategy.value} plan and {classification.query_type.value} classification "
            "remain context for fallback selection, while each claim receives an independent lookup query."
        )
        if deferred:
            rationale += f" {len(deferred)} additional unresolved claim(s) are deferred."
        return ClaimRetrievalPlan(
            tasks=tasks,
            rationale=rationale,
            planner_name=self.name,
            deferred_information_need_ids=tuple(claim.information_need_id for claim in deferred),
        )

    def _build_task(self, claim: ClaimPlanningInput) -> ClaimRetrievalTask:
        retrieval_query = _normalized_query(claim.retrieval_query)
        description = _normalized_query(claim.description)
        retrieval_query = _merge_retrieval_phrases(retrieval_query, description)
        retrieval_query = retrieval_query[: self._max_query_chars].rstrip()
        rationale = (
            f"Claim {claim.information_need_id} is {claim.support_status.value} at "
            f"coverage {claim.coverage_score:.2f}. Search it independently."
        )
        return ClaimRetrievalTask(
            information_need_id=claim.information_need_id,
            description=claim.description,
            retrieval_query=retrieval_query,
            prior_status=claim.support_status,
            prior_coverage_score=claim.coverage_score,
            prior_supporting_evidence_ranks=claim.supporting_evidence_ranks,
            grading_feedback=claim.grading_rationale,
            rationale=rationale,
        )
