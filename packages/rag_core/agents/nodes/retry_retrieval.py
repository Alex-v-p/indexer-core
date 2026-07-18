from __future__ import annotations

from typing import Any

from packages.rag_core.agents.nodes.execute_retrieval_plan import ExecuteRetrievalPlanNode
from packages.rag_core.agents.nodes.grade_evidence import GradeEvidenceNode
from packages.rag_core.agents.state import QueryState
from packages.rag_core.query_understanding.decomposition import InformationNeed
from packages.rag_core.query_understanding.planning import (
    ClaimPlanningInput,
    ClaimRetrievalPlan,
    ClaimRetrievalPlanner,
    ClaimSupportStatus,
)
from packages.rag_core.retrieval.graders import (
    EvidenceGradingReport,
    EvidenceSufficiency,
    InformationNeedSupport,
)
from packages.rag_core.retrieval.models import EvidenceItem
from packages.rag_core.retrieval.retry import (
    ClaimLookupResult,
    RetryAction,
    RetrievalAttempt,
    RetrievalRetryContext,
    RetrievalRetryPolicy,
    RetrievalRetryReport,
)


class RetryRetrievalNode:
    """Run bounded, claim-level retrieval retries over accumulated evidence."""

    name = "retry_retrieval"
    step_type = "retrieval_retry"

    def __init__(
        self,
        *,
        retry_policy: RetrievalRetryPolicy,
        claim_retrieval_planner: ClaimRetrievalPlanner,
        retrieval_node: ExecuteRetrievalPlanNode,
        grading_node: GradeEvidenceNode,
        max_accumulated_evidence: int = 40,
    ) -> None:
        if max_accumulated_evidence <= 0:
            raise ValueError("max_accumulated_evidence must be positive.")
        self._retry_policy = retry_policy
        self._claim_retrieval_planner = claim_retrieval_planner
        self._retrieval_node = retrieval_node
        self._grading_node = grading_node
        self._max_accumulated_evidence = max_accumulated_evidence

    async def __call__(self, state: QueryState) -> QueryState:
        plan = state.effective_retrieval_plan
        grading = state.evidence_grading
        if plan is None:
            raise RuntimeError("Retry evaluation requires a retrieval plan.")
        if grading is None:
            raise RuntimeError("Retry evaluation requires evidence grading.")

        attempts = [
            self._snapshot_attempt(
                state,
                retry_number=0,
                new_evidence_count=len(state.retrieved_evidence),
            ),
        ]
        claim_plans: list[ClaimRetrievalPlan] = []
        stop_reason = None
        stop_rationale = None

        while True:
            current_plan = state.effective_retrieval_plan
            current_grading = state.evidence_grading
            if current_plan is None or current_grading is None:
                raise RuntimeError("Retry state lost its active retrieval plan or evidence grading report.")

            claim_plan = None
            if (
                not current_grading.sufficient
                and len(attempts) - 1 < self._retry_policy.max_retries
            ):
                claim_plan = await self._plan_unresolved_claims(state)

            context = RetrievalRetryContext(
                original_question=state.question,
                current_query=state.effective_retrieval_query,
                current_top_k=state.effective_retrieval_top_k,
                current_plan=current_plan,
                evidence_grading=current_grading,
                retries_used=len(attempts) - 1,
                attempted_strategies=tuple(attempt.strategy for attempt in attempts),
                available_pipeline_names=self._retrieval_node.available_pipeline_names,
                claim_retrieval_plan=claim_plan,
                unresolved_retrieval_queries=(
                    tuple(task.retrieval_query for task in claim_plan.tasks) if claim_plan is not None else ()
                ),
            )
            decision = self._retry_policy.decide(context)
            if not decision.should_retry:
                stop_reason = decision.stop_reason
                stop_rationale = decision.rationale
                break

            if decision.claim_retrieval_plan is None:
                raise RuntimeError("Claim-level retry decision is missing its claim retrieval plan.")
            claim_plans.append(decision.claim_retrieval_plan)

            previous_evidence = list(state.retrieved_evidence)
            previous_grading = current_grading
            state.active_retrieval_plan = decision.next_plan
            state.active_retrieval_top_k = decision.next_top_k
            claim_lookups, retrieved_for_claims = await self._execute_claim_lookups(
                state,
                decision.claim_retrieval_plan,
                retry_number=len(attempts),
                existing_evidence=previous_evidence,
            )
            merged = _merge_evidence(
                previous_evidence,
                retrieved_for_claims,
                previous_grading,
                max_items=self._max_accumulated_evidence,
            )
            state.retrieved_evidence = merged
            state.active_retrieval_query = decision.next_query
            state.evidence_grading = None
            state = await self._grading_node(state)
            previous_keys = {_evidence_key(item) for item in previous_evidence}
            new_evidence_count = sum(_evidence_key(item) not in previous_keys for item in merged)
            attempts.append(
                self._snapshot_attempt(
                    state,
                    retry_number=len(attempts),
                    actions=decision.actions,
                    decision_rationale=decision.rationale,
                    claim_retrieval_plan=decision.claim_retrieval_plan,
                    claim_lookups=claim_lookups,
                    new_evidence_count=new_evidence_count,
                ),
            )

        if stop_reason is None or stop_rationale is None:
            raise RuntimeError("Retry policy stopped without a reason and rationale.")

        report = RetrievalRetryReport(
            policy_name=self._retry_policy.name,
            max_retries=self._retry_policy.max_retries,
            attempts=tuple(attempts),
            stop_reason=stop_reason,
            stop_rationale=stop_rationale,
        )
        state.retrieval_retry = report
        state.metadata["claim_retrieval_plans"] = [plan.to_metadata() for plan in claim_plans]
        state.metadata["retrieval_retry"] = report.to_metadata()
        final_plan = state.effective_retrieval_plan
        if final_plan is not None:
            state.metadata["final_retrieval_plan"] = final_plan.to_metadata()
        return state

    async def _plan_unresolved_claims(self, state: QueryState) -> ClaimRetrievalPlan | None:
        grading = state.evidence_grading
        decomposition = state.information_need_decomposition
        classification = state.query_classification
        current_plan = state.effective_retrieval_plan
        if grading is None or grading.sufficient:
            return None
        if decomposition is None or classification is None or current_plan is None:
            raise RuntimeError("Claim-level retry planning requires classification, decomposition, plan, and grading.")

        inputs = _claim_planning_inputs(
            decomposition.information_needs,
            grading,
        )
        if not inputs:
            return None
        return await self._claim_retrieval_planner.plan_claims(
            state.question,
            classification,
            current_plan,
            inputs,
        )

    async def _execute_claim_lookups(
        self,
        state: QueryState,
        claim_plan: ClaimRetrievalPlan,
        *,
        retry_number: int,
        existing_evidence: list[EvidenceItem],
    ) -> tuple[tuple[ClaimLookupResult, ...], list[EvidenceItem]]:
        current_plan = state.effective_retrieval_plan
        if current_plan is None:
            raise RuntimeError("Claim lookup execution requires an active retrieval plan.")

        seen = {_evidence_key(item) for item in existing_evidence}
        retrieved_for_claims: list[EvidenceItem] = []
        lookups: list[ClaimLookupResult] = []
        execution_metadata: list[dict[str, Any]] = []

        for task in claim_plan.tasks:
            state.active_retrieval_query = task.retrieval_query
            state.retrieved_evidence = []
            state = await self._retrieval_node(state)
            lookup_evidence = list(state.retrieved_evidence)
            unique_added = 0
            for item in lookup_evidence:
                _annotate_claim_lookup(
                    item,
                    information_need_id=task.information_need_id,
                    retry_number=retry_number,
                    query=task.retrieval_query,
                )
                key = _evidence_key(item)
                if key not in seen:
                    seen.add(key)
                    unique_added += 1
                retrieved_for_claims.append(item)

            lookups.append(
                ClaimLookupResult(
                    information_need_id=task.information_need_id,
                    query=task.retrieval_query,
                    pipeline_name=current_plan.selected_pipeline_name,
                    strategy=current_plan.strategy,
                    top_k=state.effective_retrieval_top_k,
                    retrieved_count=len(lookup_evidence),
                    unique_evidence_added=unique_added,
                ),
            )
            execution = state.metadata.get("retrieval_plan_execution")
            if isinstance(execution, dict):
                execution_metadata.append(
                    {
                        **execution,
                        "information_need_id": task.information_need_id,
                        "retry_number": retry_number,
                    },
                )

        history = state.metadata.get("claim_retrieval_executions")
        if not isinstance(history, list):
            history = []
            state.metadata["claim_retrieval_executions"] = history
        history.extend(execution_metadata)
        return tuple(lookups), retrieved_for_claims

    @staticmethod
    def _snapshot_attempt(
        state: QueryState,
        *,
        retry_number: int,
        actions: tuple[RetryAction, ...] = (),
        decision_rationale: str | None = None,
        claim_retrieval_plan: ClaimRetrievalPlan | None = None,
        claim_lookups: tuple[ClaimLookupResult, ...] = (),
        new_evidence_count: int = 0,
    ) -> RetrievalAttempt:
        plan = state.effective_retrieval_plan
        grading = state.evidence_grading
        if plan is None or grading is None:
            raise RuntimeError("Cannot snapshot a retrieval attempt before plan execution and grading.")
        resolved, remaining = _claim_resolution_ids(state, grading)
        return RetrievalAttempt(
            attempt_number=retry_number + 1,
            retry_number=retry_number,
            query=state.effective_retrieval_query,
            top_k=state.effective_retrieval_top_k,
            retrieval_plan=plan,
            evidence_grading=grading,
            evidence_count=len(state.retrieved_evidence),
            actions=actions,
            decision_rationale=decision_rationale,
            claim_retrieval_plan=claim_retrieval_plan,
            claim_lookups=claim_lookups,
            new_evidence_count=new_evidence_count,
            accumulated_evidence_count=len(state.retrieved_evidence),
            resolved_information_need_ids=resolved,
            remaining_information_need_ids=remaining,
        )


def _claim_planning_inputs(
    information_needs: tuple[InformationNeed, ...],
    grading: EvidenceGradingReport,
) -> tuple[ClaimPlanningInput, ...]:
    grades_by_id = {grade.information_need_id: grade for grade in grading.information_need_grades}
    values: list[ClaimPlanningInput] = []
    for need in information_needs:
        if not need.required:
            continue
        grade = grades_by_id.get(need.need_id)
        if grade is not None and grade.status is InformationNeedSupport.SUPPORTED:
            continue
        if grade is None:
            status = (
                ClaimSupportStatus.MISSING
                if grading.status is EvidenceSufficiency.MISSING
                else ClaimSupportStatus.PARTIAL
            )
            coverage = grading.coverage_score
            rationale = grading.rationale
            ranks: tuple[int, ...] = ()
        else:
            status = ClaimSupportStatus(grade.status.value)
            coverage = grade.coverage_score
            rationale = grade.rationale
            ranks = grade.supporting_evidence_ranks
        values.append(
            ClaimPlanningInput(
                information_need_id=need.need_id,
                description=need.description,
                retrieval_query=need.retrieval_query,
                support_status=status,
                coverage_score=coverage,
                grading_rationale=rationale,
                supporting_evidence_ranks=ranks,
            ),
        )
    return tuple(values)


def _claim_resolution_ids(
    state: QueryState,
    grading: EvidenceGradingReport,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if grading.information_need_grades:
        resolved = tuple(
            grade.information_need_id
            for grade in grading.information_need_grades
            if grade.required and grade.status is InformationNeedSupport.SUPPORTED
        )
        remaining = tuple(
            grade.information_need_id
            for grade in grading.information_need_grades
            if grade.required and grade.status is not InformationNeedSupport.SUPPORTED
        )
        return resolved, remaining

    decomposition = state.information_need_decomposition
    if decomposition is None:
        return (), ()
    required_ids = tuple(need.need_id for need in decomposition.information_needs if need.required)
    return (required_ids, ()) if grading.sufficient else ((), required_ids)


def _merge_evidence(
    previous: list[EvidenceItem],
    retrieved: list[EvidenceItem],
    previous_grading: EvidenceGradingReport,
    *,
    max_items: int,
) -> list[EvidenceItem]:
    protected_ranks: list[int] = []
    for grade in previous_grading.information_need_grades:
        if grade.required and grade.supporting_evidence_ranks:
            rank = grade.supporting_evidence_ranks[0]
            if rank not in protected_ranks:
                protected_ranks.append(rank)

    protected_previous = [item for rank in protected_ranks for item in previous if item.rank == rank]
    protected_keys = {_evidence_key(item) for item in protected_previous}
    relevant_ranks = {grade.evidence_rank for grade in previous_grading.grades if grade.relevant}
    remaining_relevant = [
        item
        for item in previous
        if item.rank in relevant_ranks and _evidence_key(item) not in protected_keys
    ]
    # Previously graded irrelevant candidates are not carried into later attempts.
    # Their grades remain in the trace, while the bounded evidence pool is reserved
    # for relevant retained evidence and newly retrieved claim candidates.

    interleaved_new = _interleave_claim_evidence(retrieved)
    first_new, remaining_new = _split_first_evidence_per_claim(interleaved_new)
    ordered = [
        *protected_previous,
        *first_new,
        *remaining_relevant,
        *remaining_new,
    ]

    merged: list[EvidenceItem] = []
    by_key: dict[tuple[object, ...], EvidenceItem] = {}
    for item in ordered:
        key = _evidence_key(item)
        existing = by_key.get(key)
        if existing is not None:
            _merge_claim_annotations(existing, item)
            continue
        by_key[key] = item
        merged.append(item)
        if len(merged) >= max_items:
            break

    for rank, item in enumerate(merged, start=1):
        item.rank = rank
    return merged


def _interleave_claim_evidence(retrieved: list[EvidenceItem]) -> list[EvidenceItem]:
    """Round-robin claim results so one broad lookup cannot consume the entire evidence cap."""

    groups: dict[str, list[EvidenceItem]] = {}
    order: list[str] = []
    for item in retrieved:
        claim_id = _claim_id_for_evidence(item)
        if claim_id not in groups:
            groups[claim_id] = []
            order.append(claim_id)
        groups[claim_id].append(item)

    interleaved: list[EvidenceItem] = []
    index = 0
    while True:
        added = False
        for claim_id in order:
            group = groups[claim_id]
            if index < len(group):
                interleaved.append(group[index])
                added = True
        if not added:
            break
        index += 1
    return interleaved


def _split_first_evidence_per_claim(
    evidence: list[EvidenceItem],
) -> tuple[list[EvidenceItem], list[EvidenceItem]]:
    first: list[EvidenceItem] = []
    remaining: list[EvidenceItem] = []
    seen_claims: set[str] = set()
    for item in evidence:
        claim_id = _claim_id_for_evidence(item)
        if claim_id not in seen_claims:
            seen_claims.add(claim_id)
            first.append(item)
        else:
            remaining.append(item)
    return first, remaining


def _claim_id_for_evidence(item: EvidenceItem) -> str:
    annotations = item.metadata.get("claim_retrieval_lookups")
    if isinstance(annotations, list) and annotations and isinstance(annotations[-1], dict):
        return str(annotations[-1].get("information_need_id") or "unscoped")
    return "unscoped"


def _evidence_key(item: EvidenceItem) -> tuple[object, ...]:
    if item.qdrant_chunk_index_id is not None:
        return ("qdrant_chunk", str(item.qdrant_chunk_index_id))
    normalized_text = " ".join(item.text.split()).lower()
    return (
        "content",
        str(item.document_id) if item.document_id is not None else None,
        str(item.document_version_id) if item.document_version_id is not None else None,
        normalized_text,
    )


def _annotate_claim_lookup(
    item: EvidenceItem,
    *,
    information_need_id: str,
    retry_number: int,
    query: str,
) -> None:
    raw = item.metadata.setdefault("claim_retrieval_lookups", [])
    if not isinstance(raw, list):
        raw = []
        item.metadata["claim_retrieval_lookups"] = raw
    annotation = {
        "information_need_id": information_need_id,
        "retry_number": retry_number,
        "query": query,
    }
    if annotation not in raw:
        raw.append(annotation)


def _merge_claim_annotations(target: EvidenceItem, source: EvidenceItem) -> None:
    annotations = source.metadata.get("claim_retrieval_lookups")
    if not isinstance(annotations, list):
        return
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        information_need_id = annotation.get("information_need_id")
        query = annotation.get("query")
        retry_number = annotation.get("retry_number")
        if not isinstance(information_need_id, str) or not information_need_id.strip():
            continue
        if not isinstance(query, str) or not query.strip():
            continue
        if not isinstance(retry_number, int) or isinstance(retry_number, bool) or retry_number < 0:
            continue
        _annotate_claim_lookup(
            target,
            information_need_id=information_need_id,
            retry_number=retry_number,
            query=query,
        )
