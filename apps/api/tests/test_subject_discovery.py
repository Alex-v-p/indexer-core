from __future__ import annotations

from typing import Mapping

from packages.rag_core.subjects import (
    StructuredSubjectDiscoveryProvider,
    SubjectDiscoveryProposal,
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
        assert "at most one" in prompt
        self.schemas.append(response_schema)
        return self.response


async def test_structured_discovery_returns_one_validated_proposal() -> None:
    llm = StubStructuredProvider(
        '{"proposal":{"kind":"project","name":"  Orion  ","confidence":0.91}}',
    )

    proposal = await StructuredSubjectDiscoveryProvider(llm).discover(
        "Orion delivery program status.",
    )

    assert proposal == SubjectDiscoveryProposal(
        kind=SubjectKind.PROJECT,
        name="Orion",
        confidence=0.91,
    )
    proposal_schema = llm.schemas[0]["properties"]["proposal"]  # type: ignore[index]
    assert "oneOf" in proposal_schema  # type: ignore[operator]


async def test_structured_discovery_allows_explicit_no_proposal() -> None:
    proposal = await StructuredSubjectDiscoveryProvider(
        StubStructuredProvider('{"proposal":null}'),
    ).discover("Generic meeting minutes without a durable subject.")

    assert proposal is None
