from __future__ import annotations

import re

from packages.rag_core.query_understanding.classification import QueryClassification, QueryType
from packages.rag_core.query_understanding.decomposition import InformationNeedDecomposition
from packages.rag_core.query_understanding.planning.models import RetrievalPlan, RetrievalStrategy

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
