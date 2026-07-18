from __future__ import annotations

import re

from packages.rag_core.documents.naming import DocumentNameConstraint, normalize_document_name

_DOCUMENT_TYPE = r"(?:document|file|report|manual|policy|proposal|specification|spec|paper)"
_CUE = r"(?:use|using|consult|search|retrieve|consider|within|from|in|according\s+to|based\s+on)"
_FILENAME_PATTERN = re.compile(
    r"(?<![\w.-])([A-Za-z0-9][\w.-]*\.(?:pdf|md|markdown|txt|docx?|rtf|odt))(?![\w.-])",
    re.IGNORECASE,
)
_QUOTED_AFTER_CUE_PATTERN = re.compile(
    rf"\b{_CUE}\b\s+(?:only\s+|exclusively\s+)?(?:the\s+)?(?:{_DOCUMENT_TYPE}\s+)?"
    r"(?:named\s+|called\s+|titled\s+)?[\"“`](.+?)[\"”`]",
    re.IGNORECASE,
)
_TYPE_BEFORE_NAME_PATTERN = re.compile(
    rf"\b{_CUE}\b\s+(?:only\s+|exclusively\s+)?(?:the\s+)?{_DOCUMENT_TYPE}\s+"
    r"(?:named\s+|called\s+|titled\s+)?"
    r"([A-Za-z0-9][A-Za-z0-9_.() /-]{0,100}?)"
    r"(?=\s+(?:to|for|about|when|where|what|how|why|which|that)\b|[?!,;]|\.$|$)",
    re.IGNORECASE,
)
_NAME_BEFORE_TYPE_PATTERN = re.compile(
    rf"\b{_CUE}\b\s+(?:only\s+|exclusively\s+)?(?:the\s+)?"
    rf"(?!{_DOCUMENT_TYPE}\b)"
    r"([A-Za-z0-9][A-Za-z0-9_.() /-]{0,100}?)\s+"
    rf"{_DOCUMENT_TYPE}\b",
    re.IGNORECASE,
)
_VERSIONED_BASENAME_PATTERN = re.compile(
    r"(?<![\w.-])([A-Za-z][\w.-]*(?:[_-](?:draft|version|ver|v|revision|rev)[_-]?\d+))(?![\w.-])",
    re.IGNORECASE,
)
_VERSION_SELECTOR_PREFIX_PATTERN = re.compile(
    r"^(?:the\s+)?(?:"
    r"(?:(?:oldest|earliest|initial|original|first|latest|newest|current|most\s+recent|previous|prior|preceding)"
    r"(?:\s+available)?\s+(?:document\s+)?(?:versions?|revisions?|releases?|editions?))"
    r"|(?:(?:all|every|each)(?:\s+available)?\s+(?:document\s+)?(?:versions?|revisions?|releases?|editions?))"
    r"|(?:(?:all\s+)?(?:older|historical|superseded|non[- ]current)\s+"
    r"(?:document\s+)?(?:versions?|revisions?|releases?|editions?))"
    r"|(?:(?:version|revision|release|edition)\s*(?:number\s*)?\d+)"
    r")\s+of\s+(?:the\s+)?",
    re.IGNORECASE,
)
_GENERIC_REFERENCE_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "available",
        "chosen",
        "corresponding",
        "matching",
        "most",
        "relevant",
        "requested",
        "retrieved",
        "selected",
        "source",
        "appropriate",
        "applicable",
    },
)


class RuleBasedDocumentNameIntentDetector:
    """Conservatively extract explicit titles/filenames for strict filtering.

    Relative phrases such as ``the relevant document`` describe how retrieval
    should choose a source; they are not document identities and must never be
    promoted into hard name filters.
    """

    name = "rule_based_document_name_intent_detector"

    def detect(self, question: str) -> DocumentNameConstraint:
        normalized = " ".join(question.strip().split())
        if not normalized:
            raise ValueError("question must not be empty.")

        candidates: list[tuple[str, bool]] = []
        candidates.extend((match.group(1), True) for match in _QUOTED_AFTER_CUE_PATTERN.finditer(normalized))
        candidates.extend((match.group(1), True) for match in _FILENAME_PATTERN.finditer(normalized))
        candidates.extend((match.group(1), False) for match in _TYPE_BEFORE_NAME_PATTERN.finditer(normalized))
        candidates.extend((match.group(1), False) for match in _NAME_BEFORE_TYPE_PATTERN.finditer(normalized))
        candidates.extend((match.group(1), True) for match in _VERSIONED_BASENAME_PATTERN.finditer(normalized))

        unique: list[str] = []
        normalized_seen: set[str] = set()
        for candidate, explicit_identity in candidates:
            cleaned = _clean_candidate(candidate, strip_version_selector=not explicit_identity)
            normalized_candidate = normalize_document_name(cleaned)
            if not normalized_candidate:
                continue
            if not explicit_identity and _is_generic_document_reference(normalized_candidate):
                continue
            if normalized_candidate in normalized_seen:
                continue
            normalized_seen.add(normalized_candidate)
            unique.append(cleaned)

        if not unique:
            return DocumentNameConstraint(detector_name=self.name)

        return DocumentNameConstraint(
            names=tuple(unique),
            confidence=0.97 if any("." in value for value in unique) else 0.92,
            rationale=(
                "The query explicitly names one or more documents or filenames; retrieval must only use "
                "chunks whose source identity matches those names."
            ),
            detector_name=self.name,
        )


def detect_document_name_constraint(question: str) -> DocumentNameConstraint:
    return RuleBasedDocumentNameIntentDetector().detect(question)


def _clean_candidate(value: str, *, strip_version_selector: bool = False) -> str:
    cleaned = " ".join(value.strip().split())
    cleaned = cleaned.strip(" \"'`.,:;")
    if strip_version_selector:
        cleaned = _VERSION_SELECTOR_PREFIX_PATTERN.sub("", cleaned, count=1).strip()
    return cleaned.strip(" \"'`.,:;")


def _is_generic_document_reference(normalized_candidate: str) -> bool:
    tokens = normalized_candidate.split()
    return bool(tokens) and all(token in _GENERIC_REFERENCE_WORDS for token in tokens)
