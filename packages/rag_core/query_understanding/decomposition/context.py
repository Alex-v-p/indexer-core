from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_ABOUT_SUBJECT_PATTERN = re.compile(
    r"\b(?:about|regarding|concerning)\s+(?:the\s+)?(?P<subject>.+?)(?=\s+(?:and|but|as well as)\b|[?.!]|$)",
    re.IGNORECASE,
)
_OF_SUBJECT_PATTERN = re.compile(
    r"\b(?:overview|summary|purpose|architecture|key points?|features?)\s+of\s+(?:the\s+)?(?P<subject>.+?)(?=\s+(?:and|but|as well as)\b|[?.!]|$)",
    re.IGNORECASE,
)
_TRAILING_REQUEST_PATTERN = re.compile(
    r"\s+(?:and\s+)?(?:the\s+)?(?:key points?|main points?|overview|summary|purpose|features?)\s*$",
    re.IGNORECASE,
)
_GENERIC_QUERY_TERMS = frozenset(
    {
        "about",
        "and",
        "describe",
        "details",
        "explain",
        "information",
        "key",
        "main",
        "of",
        "overview",
        "points",
        "project",
        "purpose",
        "summary",
        "system",
        "tell",
        "the",
        "what",
    },
)
_QUESTION_NOISE_TERMS = _GENERIC_QUERY_TERMS | frozenset(
    {
        "a",
        "an",
        "are",
        "does",
        "for",
        "how",
        "in",
        "is",
        "it",
        "me",
        "its",
        "to",
        "which",
        "who",
        "why",
    },
)


def infer_subject_context(question: str, *, max_chars: int = 120) -> str:
    """Extract a compact subject anchor that should survive decomposition.

    The heuristic is intentionally conservative. It first looks for explicit
    subject-introducing phrases (for example ``about X``), then falls back to
    distinctive question tokens such as project names, identifiers, and
    CamelCase terms. The result is a retrieval anchor, not a hard document
    constraint.
    """

    normalized = " ".join(question.strip().split())
    if not normalized:
        raise ValueError("question must not be empty.")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")

    for pattern in (_ABOUT_SUBJECT_PATTERN, _OF_SUBJECT_PATTERN):
        match = pattern.search(normalized)
        if match is None:
            continue
        subject = _clean_subject(match.group("subject"))
        if subject:
            return subject[:max_chars].rstrip()

    tokens = [match.group(0) for match in _TOKEN_PATTERN.finditer(normalized)]
    distinctive_indexes = [
        index
        for index, token in enumerate(tokens)
        if _is_distinctive_token(token) and token.casefold() not in _QUESTION_NOISE_TERMS
    ]
    if not distinctive_indexes:
        return ""

    first = distinctive_indexes[0]
    selected = [tokens[first]]
    if first + 1 < len(tokens):
        following = tokens[first + 1]
        if following.casefold() in {"project", "system", "platform", "service", "application", "document"}:
            selected.append(following)
    return " ".join(selected)[:max_chars].rstrip()


def contextualize_retrieval_query(
    retrieval_query: str,
    *,
    subject_context: str,
    max_chars: int,
) -> str:
    """Ensure a retrieval query contains the subject anchor when one is known."""

    query = " ".join(retrieval_query.strip().split())
    context = " ".join(subject_context.strip().split())
    if not query:
        raise ValueError("retrieval_query must not be empty.")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    if not context:
        return query[:max_chars].rstrip()

    query_tokens = {match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(query)}
    missing_context_tokens = [
        match.group(0)
        for match in _TOKEN_PATTERN.finditer(context)
        if match.group(0).casefold() not in query_tokens
    ]
    if not missing_context_tokens:
        return query[:max_chars].rstrip()

    merged = " ".join((*missing_context_tokens, *query.split())).strip()
    return merged[:max_chars].rstrip() or query[:max_chars].rstrip()


def _clean_subject(value: str) -> str:
    normalized = " ".join(value.strip(" ,;:-").split())
    normalized = _TRAILING_REQUEST_PATTERN.sub("", normalized).strip(" ,;:-")
    return normalized


def _is_distinctive_token(token: str) -> bool:
    if any(character.isdigit() for character in token):
        return True
    if "_" in token or "-" in token:
        return True
    if any(character.isupper() for character in token[1:]):
        return True
    return token.isupper() and len(token) > 1
