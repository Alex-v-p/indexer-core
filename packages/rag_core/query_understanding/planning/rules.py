from __future__ import annotations

import re

from packages.rag_core.query_understanding.classification import QueryClassification, QueryType
from packages.rag_core.query_understanding.decomposition import InformationNeedDecomposition
from packages.rag_core.query_understanding.planning.models import (
    ClaimPlanningInput,
    ClaimRetrievalPlan,
    ClaimRetrievalTask,
    ClaimSupportStatus,
    RetrievalPlan,
    RetrievalStrategy,
)

_RERANK_PATTERN = re.compile(
    r"\b(most relevant|best evidence|strongest evidence|most important|rank|prioriti[sz]e|which .* best)\b",
    re.IGNORECASE,
)


class RuleBasedRetrievalPlanner:
    """Explainable policy that maps query understanding to a retrieval strategy.

    Classification and decomposition are produced by independent query-
    understanding stages. The planner consumes both outputs but does not own or
    execute either capability.
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
        low_confidence_threshold: float = 0.55,
        contextual_available: bool = True,
    ) -> None:
        if not 0.0 <= low_confidence_threshold <= 1.0:
            raise ValueError("low_confidence_threshold must be between 0 and 1.")
        self._pipeline_names = {
            RetrievalStrategy.BASELINE: _require_pipeline_name(baseline_pipeline_name),
            RetrievalStrategy.HYBRID: _require_pipeline_name(hybrid_pipeline_name),
            RetrievalStrategy.CONTEXTUAL: _require_pipeline_name(contextual_pipeline_name),
            RetrievalStrategy.MULTI_QUERY: _require_pipeline_name(multi_query_pipeline_name),
            RetrievalStrategy.RERANK: _require_pipeline_name(rerank_pipeline_name),
        }
        self._low_confidence_threshold = low_confidence_threshold
        self._contextual_available = contextual_available

    async def plan(
        self,
        question: str,
        classification: QueryClassification,
        decomposition: InformationNeedDecomposition,
    ) -> RetrievalPlan:
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
        )

    def _select_strategy(
        self,
        question: str,
        classification: QueryClassification,
        *,
        information_need_count: int,
    ) -> tuple[RetrievalStrategy, str]:
        if classification.query_type is QueryType.VERSION_SPECIFIC:
            return (
                RetrievalStrategy.HYBRID,
                "Version- and date-specific wording benefits from combining semantic retrieval with exact lexical matching. "
                "The detected metadata hints are preserved for the later version-aware retrieval step.",
            )

        if classification.query_type is QueryType.COMPARISON:
            return (
                RetrievalStrategy.MULTI_QUERY,
                "Comparison questions contain multiple evidence requirements, so query expansion can collect support for "
                "each side before fusing the results.",
            )

        if _RERANK_PATTERN.search(question) or classification.confidence < self._low_confidence_threshold:
            return (
                RetrievalStrategy.RERANK,
                "The query asks for especially discriminative evidence, or its classification confidence is low, so a "
                "hybrid candidate set is reranked before answer generation.",
            )

        if information_need_count > 1:
            return (
                RetrievalStrategy.MULTI_QUERY,
                f"The independently produced decomposition contains {information_need_count} gradable information needs, "
                "so multi-query retrieval is selected to improve coverage across all requested aspects.",
            )

        if classification.query_type is QueryType.BROAD_EXPLANATION:
            if self._contextual_available:
                return (
                    RetrievalStrategy.CONTEXTUAL,
                    "This broad but cohesive explanation benefits from document-aware contextualized chunks that retain "
                    "surrounding section and document meaning.",
                )
            return (
                RetrievalStrategy.MULTI_QUERY,
                "Contextual retrieval is unavailable, so query expansion is used to cover the broader explanation request.",
            )

        if classification.needs_metadata_filters:
            return (
                RetrievalStrategy.HYBRID,
                "The query contains document, section, author, file-type, or date constraints; hybrid retrieval improves "
                "matching of those exact terms while keeping semantic recall.",
            )

        return (
            RetrievalStrategy.BASELINE,
            "This is a focused factual lookup with one answer requirement and no special metadata or ranking needs, so "
            "dense-vector retrieval is the lowest-complexity suitable strategy.",
        )


def _require_pipeline_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("pipeline names must not be empty.")
    return normalized


class RuleBasedClaimRetrievalPlanner:
    """Create focused retry lookups from claim-level evidence feedback.

    The initial planner chooses a retrieval strategy for the complete question.
    This planner reuses that classification and plan as context, but only decides
    which unresolved claims need another lookup and what each lookup should ask.
    Pipeline escalation remains owned by the retry policy.
    """

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
            rationale += (
                f" {len(deferred)} additional unresolved claim(s) are deferred to a later bounded retry round."
            )
        return ClaimRetrievalPlan(
            tasks=tasks,
            rationale=rationale,
            planner_name=self.name,
            deferred_information_need_ids=tuple(claim.information_need_id for claim in deferred),
        )

    def _build_task(self, claim: ClaimPlanningInput) -> ClaimRetrievalTask:
        retrieval_query = " ".join(claim.retrieval_query.strip().split())
        description = " ".join(claim.description.strip().split())
        if description.lower() not in retrieval_query.lower():
            retrieval_query = f"{retrieval_query} | answer requirement: {description}"
        retrieval_query = retrieval_query[: self._max_query_chars].rstrip(" |")
        rationale = (
            f"Claim {claim.information_need_id} is {claim.support_status.value} at "
            f"coverage {claim.coverage_score:.2f}. Search it independently so evidence for already-supported "
            "claims can be preserved."
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
