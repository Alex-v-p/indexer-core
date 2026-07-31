from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace

from packages.rag_core.query_understanding.decomposition.models import InformationNeed

_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)

# These words describe how a request is phrased rather than what evidence a lane
# must retrieve. They are deliberately narrower than ordinary retrieval stop
# words: answer-type terms such as "overview", "purpose", "key", "points",
# and "contributor" remain part of the lane-specific intent signature.
_INTENT_NOISE_TERMS = frozenset(
    {
        "a",
        "an",
        "about",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "describe",
        "determine",
        "do",
        "does",
        "explain",
        "find",
        "for",
        "from",
        "give",
        "how",
        "identify",
        "in",
        "is",
        "it",
        "its",
        "list",
        "me",
        "of",
        "on",
        "or",
        "provide",
        "regarding",
        "tell",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "what",
        "which",
        "who",
        "why",
        "with",
    },
)

_QUERY_SCAFFOLD_TERMS = frozenset(
    {
        "a",
        "an",
        "about",
        "are",
        "describe",
        "determine",
        "did",
        "do",
        "does",
        "explain",
        "find",
        "give",
        "how",
        "identify",
        "is",
        "list",
        "me",
        "provide",
        "tell",
        "the",
        "was",
        "were",
        "what",
        "which",
        "who",
        "why",
    },
)

_DANGLING_CONNECTORS = frozenset(
    {"and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "to", "with"},
)


def isolate_information_need_queries(
    information_needs: Iterable[InformationNeed],
    *,
    max_query_chars: int,
) -> tuple[InformationNeed, ...]:
    """Remove sibling-lane intent leakage while preserving shared subject terms.

    Every lane is allowed to repeat project names, document aliases, products,
    systems, and other subject anchors. Only terms that belong to another
    lane's answer requirement are considered for removal.
    """

    needs = tuple(information_needs)
    if max_query_chars <= 0:
        raise ValueError("max_query_chars must be positive.")
    if len(needs) < 2:
        return needs

    repaired: list[InformationNeed] = []
    for need in needs:
        siblings = tuple(
            candidate for candidate in needs if candidate.need_id != need.need_id
        )
        query = isolate_retrieval_query_intent(
            need.retrieval_query,
            information_need=need,
            sibling_information_needs=siblings,
            max_query_chars=max_query_chars,
        )
        repaired.append(
            replace(need, retrieval_query=query) if query != need.retrieval_query else need,
        )
    return tuple(repaired)


def isolate_retrieval_query_intent(
    query: str,
    *,
    information_need: InformationNeed,
    sibling_information_needs: Iterable[InformationNeed] = (),
    max_query_chars: int,
) -> str:
    """Keep one query focused on its lane without removing subject aliases.

    The check is lexical and conservative. A sibling intent is removed only
    when the query covers most of that sibling's intent signature and includes
    terms that are not part of the current lane. This avoids treating shared
    anchors such as ``LLMguidance`` and ``LLM guidance project`` as overlap.
    """

    normalized_query = " ".join(query.strip().split())
    if not normalized_query:
        raise ValueError("query must not be empty.")
    if max_query_chars <= 0:
        raise ValueError("max_query_chars must be positive.")

    siblings = tuple(sibling_information_needs)
    if not siblings:
        return normalized_query[:max_query_chars].rstrip()

    all_needs = (information_need, *siblings)
    subject_tokens = _subject_tokens(all_needs)
    own_signature = _intent_signature(
        information_need.description,
        subject_tokens=subject_tokens,
    )
    own_terms = {token for _, token in own_signature}
    query_parts = [
        (match.group(0), _canonical(match.group(0)))
        for match in _TOKEN_PATTERN.finditer(normalized_query)
    ]
    query_terms = {token for _, token in query_parts}

    terms_to_remove: set[str] = set()
    for sibling in siblings:
        sibling_signature = _intent_signature(
            sibling.description,
            subject_tokens=subject_tokens,
        )
        sibling_terms = {token for _, token in sibling_signature}
        if not sibling_terms:
            continue
        exclusive_terms = (
            sibling_terms.difference(own_terms).difference(subject_tokens)
        )
        matched_exclusive = query_terms.intersection(exclusive_terms)
        if not matched_exclusive:
            continue

        matched_signature = query_terms.intersection(sibling_terms)
        coverage = len(matched_signature) / len(sibling_terms)
        # Multi-word intents require majority coverage so incidental vocabulary
        # is not removed. A one-word sibling intent is only isolated when that
        # exact answer type is absent from the current lane.
        if (
            len(sibling_terms) == 1
            or coverage >= 0.6
            or len(matched_exclusive) >= 2
        ):
            terms_to_remove.update(matched_exclusive)

    if not terms_to_remove:
        return normalized_query[:max_query_chars].rstrip()

    remaining = [
        (raw, canonical)
        for raw, canonical in query_parts
        if (
            canonical not in terms_to_remove
            or canonical in subject_tokens
            or canonical in own_terms
        )
    ]
    remaining = _trim_query_scaffolding(remaining)

    repaired = " ".join(raw for raw, _ in remaining).strip()
    repaired_terms = {canonical for _, canonical in remaining}
    if not repaired or (own_terms and not repaired_terms.intersection(own_terms)):
        repaired = _compose_lane_query(
            information_need,
            own_signature=own_signature,
            max_query_chars=max_query_chars,
        )

    return (
        repaired[:max_query_chars].rstrip()
        or normalized_query[:max_query_chars].rstrip()
    )


def _subject_tokens(information_needs: Iterable[InformationNeed]) -> set[str]:
    tokens: set[str] = set()
    for need in information_needs:
        tokens.update(
            _canonical(match.group(0))
            for match in _TOKEN_PATTERN.finditer(need.subject_context)
        )
    return tokens


def _intent_signature(
    description: str,
    *,
    subject_tokens: set[str],
) -> tuple[tuple[str, str], ...]:
    signature: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _TOKEN_PATTERN.finditer(description):
        raw = match.group(0)
        canonical = _canonical(raw)
        if canonical in subject_tokens or canonical in _INTENT_NOISE_TERMS or canonical in seen:
            continue
        signature.append((raw, canonical))
        seen.add(canonical)
    return tuple(signature)


def _trim_query_scaffolding(parts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    trimmed = list(parts)
    removable = _QUERY_SCAFFOLD_TERMS.union(_DANGLING_CONNECTORS)
    while trimmed and trimmed[0][1] in removable:
        trimmed.pop(0)
    while trimmed and trimmed[-1][1] in removable:
        trimmed.pop()
    return trimmed


def _compose_lane_query(
    information_need: InformationNeed,
    *,
    own_signature: tuple[tuple[str, str], ...],
    max_query_chars: int,
) -> str:
    subject = " ".join(information_need.subject_context.strip().split())
    intent = " ".join(raw for raw, _ in own_signature)
    merged = " ".join(part for part in (subject, intent) if part).strip()
    if not merged:
        merged = information_need.description
    return merged[:max_query_chars].rstrip()


def _canonical(token: str) -> str:
    return token.casefold()
