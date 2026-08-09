from __future__ import annotations

from typing import Mapping

from packages.rag_core.subjects import (
    StructuredSubjectModelResolutionProvider,
    SubjectCreateResolution,
    SubjectKind,
)


class StubStructuredProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.schemas: list[Mapping[str, object]] = []

    async def generate(self, prompt: str) -> str:
        del prompt
        return self.response

    async def generate_structured(
        self,
        prompt: str,
        *,
        response_schema: Mapping[str, object],
    ) -> str:
        assert "at most one durable organizing subject" in prompt
        self.schemas.append(response_schema)
        return self.response


async def test_structured_resolution_returns_one_validated_create_action() -> None:
    llm = StubStructuredProvider(
        '{"action":"create","kind":"project","name":"  Orion  ","confidence":0.91}',
    )

    proposal = await StructuredSubjectModelResolutionProvider(llm).resolve(
        "Orion delivery program status.", (),
    )

    assert proposal == SubjectCreateResolution(
        kind=SubjectKind.PROJECT,
        name="Orion",
        confidence=0.91,
    )
    assert "oneOf" in llm.schemas[0]


async def test_structured_resolution_allows_explicit_none_action() -> None:
    proposal = await StructuredSubjectModelResolutionProvider(
        StubStructuredProvider('{"action":"none"}'),
    ).resolve("Generic meeting minutes without a durable subject.", ())

    assert proposal is None
