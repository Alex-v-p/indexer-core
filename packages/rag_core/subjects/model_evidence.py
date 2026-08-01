from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Protocol

from packages.rag_core.ports import StructuredLLMProvider
from packages.rag_core.structured_output import (
    StructuredOutputError,
    StructuredValidationRule,
    generate_structured_output,
)
from packages.rag_core.subjects.classification import SubjectClassificationCandidate
from packages.rag_core.subjects.models import SubjectKind
from packages.rag_core.subjects.naming import SubjectName


class SubjectModelEvidenceError(RuntimeError):
    """Structured model evidence could not be generated safely."""


class _InvalidJSON(SubjectModelEvidenceError):
    pass


class _SchemaMismatch(SubjectModelEvidenceError):
    pass


class _SemanticMismatch(SubjectModelEvidenceError):
    pass


_VALIDATION_RULES = (
    StructuredValidationRule("invalid_json", (_InvalidJSON,)),
    StructuredValidationRule("schema_mismatch", (_SchemaMismatch,)),
    StructuredValidationRule("semantic_validation_failed", (_SemanticMismatch,)),
)


class SubjectModelEvidenceProvider(Protocol):
    async def score(
        self,
        summary: str,
        candidates: tuple[SubjectClassificationCandidate, ...],
    ) -> dict[uuid.UUID, float]: ...


@dataclass(frozen=True, slots=True)
class SubjectDiscoveryProposal:
    kind: SubjectKind
    name: str
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError("discovery name must be a string.")
        validated_name = SubjectName.from_value(self.name)
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("discovery confidence must be between 0 and 1.")
        object.__setattr__(self, "kind", SubjectKind(self.kind))
        object.__setattr__(self, "name", validated_name.value)
        object.__setattr__(self, "confidence", float(self.confidence))


class SubjectDiscoveryProvider(Protocol):
    async def discover(self, summary: str) -> SubjectDiscoveryProposal | None: ...


@dataclass(frozen=True, slots=True)
class StructuredSubjectModelEvidenceProvider:
    provider: StructuredLLMProvider
    max_summary_chars: int = 2_000
    max_repair_attempts: int = 1
    max_candidates_per_call: int = 20

    async def score(
        self,
        summary: str,
        candidates: tuple[SubjectClassificationCandidate, ...],
    ) -> dict[uuid.UUID, float]:
        bounded_summary = " ".join(summary.strip().split())[: self.max_summary_chars]
        if not bounded_summary or not candidates:
            return {}
        if self.max_candidates_per_call <= 0:
            raise ValueError("max_candidates_per_call must be positive.")
        scores: dict[uuid.UUID, float] = {}
        for start in range(0, len(candidates), self.max_candidates_per_call):
            batch = candidates[start : start + self.max_candidates_per_call]
            allowed = {candidate.subject_id for candidate in batch}
            prompt = _build_prompt(bounded_summary, batch)
            try:
                result = await generate_structured_output(
                    provider=self.provider,
                    prompt=prompt,
                    response_schema=_response_schema(len(batch)),
                    parser=lambda raw, allowed=allowed: _parse_scores(
                        raw,
                        allowed=allowed,
                    ),
                    validation_rules=_VALIDATION_RULES,
                    max_repair_attempts=self.max_repair_attempts,  # type: ignore[arg-type]
                )
            except StructuredOutputError as exc:
                raise SubjectModelEvidenceError(
                    "Structured subject model evidence generation failed."
                ) from exc
            scores.update(result.value)
        return scores


@dataclass(frozen=True, slots=True)
class StructuredSubjectDiscoveryProvider:
    provider: StructuredLLMProvider
    max_summary_chars: int = 2_000
    max_repair_attempts: int = 1

    async def discover(self, summary: str) -> SubjectDiscoveryProposal | None:
        bounded_summary = " ".join(summary.strip().split())[: self.max_summary_chars]
        if not bounded_summary:
            return None
        try:
            result = await generate_structured_output(
                provider=self.provider,
                prompt=_build_discovery_prompt(bounded_summary),
                response_schema=_discovery_response_schema(),
                parser=_parse_discovery,
                validation_rules=_VALIDATION_RULES,
                max_repair_attempts=self.max_repair_attempts,  # type: ignore[arg-type]
            )
        except StructuredOutputError as exc:
            raise SubjectModelEvidenceError(
                "Structured subject discovery generation failed."
            ) from exc
        return result.value


def _build_prompt(
    summary: str,
    candidates: tuple[SubjectClassificationCandidate, ...],
) -> str:
    catalogue = [
        {
            "subject_id": str(candidate.subject_id),
            "kind": candidate.kind.value,
            "name": candidate.canonical_name,
            "aliases": [alias[:255] for alias in candidate.aliases[:5]],
        }
        for candidate in candidates
    ]
    return (
        "Classify the document summary only against the supplied existing subjects. "
        "Do not invent subjects. Return confidence 0..1 only for supported matches.\n"
        f"Subjects: {json.dumps(catalogue, ensure_ascii=False)}\n"
        f"Document summary: {summary}"
    )


def _response_schema(max_items: int) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "matches": {
                "type": "array",
                "maxItems": max_items,
                "items": {
                    "type": "object",
                    "properties": {
                        "subject_id": {"type": "string", "format": "uuid"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["subject_id", "confidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["matches"],
        "additionalProperties": False,
    }


def _build_discovery_prompt(summary: str) -> str:
    return (
        "Discover at most one specific, durable subject named by this document summary. "
        "Return null when there is no clear project, topic, organization, or custom "
        "subject. Avoid generic document types and do not return multiple proposals.\n"
        f"Document summary: {summary}"
    )


def _discovery_response_schema() -> dict[str, object]:
    proposal = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": [kind.value for kind in SubjectKind]},
            "name": {"type": "string", "minLength": 1, "maxLength": 255},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["kind", "name", "confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"proposal": {"oneOf": [proposal, {"type": "null"}]}},
        "required": ["proposal"],
        "additionalProperties": False,
    }


def _parse_discovery(raw: str) -> SubjectDiscoveryProposal | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _InvalidJSON("Subject discovery response is not JSON.") from exc
    if not isinstance(payload, dict) or set(payload) != {"proposal"}:
        raise _SchemaMismatch("Subject discovery response fields do not match the schema.")
    proposal = payload["proposal"]
    if proposal is None:
        return None
    if not isinstance(proposal, dict) or set(proposal) != {"kind", "name", "confidence"}:
        raise _SchemaMismatch("Discovery proposal must contain kind, name, and confidence.")
    confidence = proposal["confidence"]
    if (
        not isinstance(proposal["kind"], str)
        or not isinstance(proposal["name"], str)
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
    ):
        raise _SchemaMismatch("Discovery proposal fields have invalid types.")
    try:
        return SubjectDiscoveryProposal(
            kind=SubjectKind(proposal["kind"]),
            name=proposal["name"],
            confidence=float(confidence),
        )
    except (TypeError, ValueError) as exc:
        raise _SemanticMismatch("Subject discovery proposal is invalid.") from exc


def _parse_scores(raw: str, *, allowed: set[uuid.UUID]) -> dict[uuid.UUID, float]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _InvalidJSON("Model evidence response is not JSON.") from exc
    if not isinstance(payload, dict) or set(payload) != {"matches"}:
        raise _SchemaMismatch("Model evidence response fields do not match the schema.")
    matches = payload["matches"]
    if not isinstance(matches, list) or len(matches) > len(allowed):
        raise _SchemaMismatch("matches must be a bounded array.")
    scores: dict[uuid.UUID, float] = {}
    for item in matches:
        if not isinstance(item, dict) or set(item) != {"subject_id", "confidence"}:
            raise _SchemaMismatch("Each model match must contain subject_id and confidence.")
        try:
            subject_id = uuid.UUID(item["subject_id"])
        except (TypeError, ValueError) as exc:
            raise _SchemaMismatch("subject_id must be a UUID.") from exc
        confidence = item["confidence"]
        if (
            subject_id not in allowed
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise _SemanticMismatch("Model evidence contains an unknown subject or confidence.")
        if subject_id in scores:
            raise _SemanticMismatch("Model evidence subject IDs must be unique.")
        scores[subject_id] = float(confidence)
    return scores
