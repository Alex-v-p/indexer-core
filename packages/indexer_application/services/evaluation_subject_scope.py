from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from packages.indexer_application.ports import UnitOfWork
from packages.indexer_application.services.query_subject_scope import (
    QuerySubjectScopeConfig,
    resolve_query_subject_scope_snapshot,
)
from packages.rag_core.evaluation.runner import SubjectScopeEvaluationRequest
from packages.rag_core.document_scope import ResolvedSubjectScopeSnapshot


class ApplicationSubjectScopeEvaluationProvider:
    """Resolve evaluation scope through the production application boundary."""

    def __init__(
        self,
        *,
        uow_factory: Callable[[], AbstractAsyncContextManager[UnitOfWork]],
        config: QuerySubjectScopeConfig = QuerySubjectScopeConfig(),
    ) -> None:
        self._uow_factory = uow_factory
        self._config = config

    async def resolve(
        self,
        request: SubjectScopeEvaluationRequest,
    ) -> ResolvedSubjectScopeSnapshot:
        async with self._uow_factory() as uow:
            requested_ids = list(request.requested_subject_ids)
            for name in request.requested_subject_names:
                matches = await uow.subjects.resolve_name(name)
                if len(matches) != 1:
                    raise ValueError(
                        f"Evaluation subject name {name!r} resolved to {len(matches)} active catalog entries; expected one.",
                    )
                requested_ids.append(matches[0].subject.id)
            return await resolve_query_subject_scope_snapshot(
                uow=uow,
                question=request.question,
                requested_subject_ids=tuple(dict.fromkeys(requested_ids)),
                coverage_mode=request.coverage_mode,
                config=self._config,
            )
