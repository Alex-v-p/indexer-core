from __future__ import annotations

import re

from packages.rag_core.documents.versioning import DocumentVersionConstraint, VersionSelectionMode

_SPECIFIC_VERSION_PATTERN = re.compile(
    r"\b(?:version|revision|rev(?:ision)?|release|edition)\s*(?:number\s*)?(\d+)\b|\bv(\d+)(?:\.\d+)*\b",
    re.IGNORECASE,
)
_LATEST_PATTERN = re.compile(r"\b(latest|newest|current|most recent|up[- ]to[- ]date)\b", re.IGNORECASE)
_OLDEST_PATTERN = re.compile(
    r"\b(?:oldest|earliest|initial|original|first)\b.{0,32}\b(?:versions?|revisions?|releases?|editions?)\b"
    r"|\b(?:versions?|revisions?|releases?|editions?)\b.{0,32}\b(?:oldest|earliest|initial|original|first)\b"
    r"|\b(?:version|revision|release|edition)\s+(?:one|1)\b",
    re.IGNORECASE,
)
_PREVIOUS_PATTERN = re.compile(r"\b(previous|prior|preceding|one version back|last revision)\b", re.IGNORECASE)
_OLDER_VERSIONS_PATTERN = re.compile(
    r"\b(?:all\s+)?(?:older|old|historical|superseded|non[- ]current)\s+"
    r"(?:document\s+)?(?:versions|revisions|releases|editions)\b"
    r"|\ball\s+(?:previous|prior)\s+(?:document\s+)?(?:versions|revisions|releases|editions)\b",
    re.IGNORECASE,
)
_ALL_VERSIONS_PATTERN = re.compile(
    r"\b(?:all|every|each)\s+(?:document\s+)?(?:versions|revisions|releases|editions)\b",
    re.IGNORECASE,
)
_VERSION_COMPARISON_PATTERN = re.compile(
    r"\b(compare|comparison|contrast|difference|differences|versus|vs\.?)\b.*\b(version|revision|release|edition|current|latest|previous|prior)\b|"
    r"\b(version|revision|release|edition|current|latest|previous|prior)\b.*\b(compare|comparison|contrast|difference|differences|versus|vs\.?)\b",
    re.IGNORECASE,
)


class RuleBasedVersionIntentDetector:
    """Conservative detector for version qualifiers that map to indexed metadata."""

    name = "rule_based_version_intent_detector"

    def detect(self, question: str) -> DocumentVersionConstraint:
        normalized = " ".join(question.strip().split())
        if not normalized:
            raise ValueError("question must not be empty.")

        version_numbers = tuple(
            dict.fromkeys(
                int(first or second)
                for first, second in _SPECIFIC_VERSION_PATTERN.findall(normalized)
                if first or second
            ),
        )
        has_latest = _LATEST_PATTERN.search(normalized) is not None
        has_oldest = _OLDEST_PATTERN.search(normalized) is not None
        has_previous = _PREVIOUS_PATTERN.search(normalized) is not None
        has_older_versions = _OLDER_VERSIONS_PATTERN.search(normalized) is not None
        has_all_versions = _ALL_VERSIONS_PATTERN.search(normalized) is not None
        compares_versions = _VERSION_COMPARISON_PATTERN.search(normalized) is not None

        if version_numbers:
            return DocumentVersionConstraint(
                mode=VersionSelectionMode.SPECIFIC,
                version_numbers=version_numbers,
                confidence=0.96 if len(version_numbers) > 1 else 0.93,
                rationale=(
                    "The query names specific document version numbers; retrieval must filter to those versions."
                ),
                detector_name=self.name,
            )
        if has_latest and has_previous:
            return DocumentVersionConstraint(
                mode=VersionSelectionMode.LATEST_AND_PREVIOUS,
                confidence=0.94,
                rationale="The query explicitly asks for both the latest and immediately previous document versions.",
                detector_name=self.name,
            )
        if has_latest and has_oldest:
            return DocumentVersionConstraint(
                mode=VersionSelectionMode.OLDEST_AND_LATEST,
                confidence=0.95,
                rationale="The query explicitly asks for both the oldest and newest document versions.",
                detector_name=self.name,
            )
        if has_older_versions:
            return DocumentVersionConstraint(
                mode=VersionSelectionMode.ALL_EXCEPT_LATEST,
                confidence=0.91,
                rationale="The query asks for historical or superseded versions and excludes the current/latest version.",
                detector_name=self.name,
            )
        if has_all_versions:
            return DocumentVersionConstraint(
                mode=VersionSelectionMode.ALL_VERSIONS,
                confidence=0.93,
                rationale="The query explicitly asks for all indexed versions of the matching document(s).",
                detector_name=self.name,
            )
        if has_latest:
            return DocumentVersionConstraint(
                mode=VersionSelectionMode.LATEST,
                confidence=0.91,
                rationale="The query explicitly asks for the latest/current document version.",
                detector_name=self.name,
            )
        if has_oldest:
            return DocumentVersionConstraint(
                mode=VersionSelectionMode.OLDEST,
                confidence=0.91,
                rationale="The query explicitly asks for the oldest/original document version.",
                detector_name=self.name,
            )
        if has_previous:
            return DocumentVersionConstraint(
                mode=VersionSelectionMode.PREVIOUS,
                confidence=0.89,
                rationale="The query explicitly asks for the immediately previous document version.",
                detector_name=self.name,
            )
        if compares_versions:
            return DocumentVersionConstraint(
                mode=VersionSelectionMode.LATEST_AND_PREVIOUS,
                confidence=0.72,
                rationale=(
                    "The query compares versions without naming numbers; use the latest and immediately previous versions."
                ),
                detector_name=self.name,
            )
        return DocumentVersionConstraint(detector_name=self.name)


def detect_document_version_constraint(question: str) -> DocumentVersionConstraint:
    return RuleBasedVersionIntentDetector().detect(question)
