from enum import StrEnum


class DocumentTypeDecisionState(StrEnum):
    SUGGESTED = "suggested"
    ASSIGNED = "assigned"
    REJECTED = "rejected"


class ContentGroupAssignmentState(StrEnum):
    PENDING = "pending"
    UNRESOLVED = "unresolved"
    SUGGESTED = "suggested"
    ASSIGNED = "assigned"


class ClassificationSource(StrEnum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"


class ClassificationConfidenceBand(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ContentGroupNameMatchType(StrEnum):
    CANONICAL = "canonical"
    ALIAS = "alias"
