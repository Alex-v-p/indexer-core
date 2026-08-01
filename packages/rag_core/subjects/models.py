from enum import StrEnum


class SubjectKind(StrEnum):
    PROJECT = "project"
    TOPIC = "topic"
    ORGANIZATION = "organization"
    CUSTOM = "custom"


class DecisionState(StrEnum):
    SUGGESTED = "suggested"
    ASSIGNED = "assigned"
    REJECTED = "rejected"


class DecisionControlSource(StrEnum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"


class ConfidenceBand(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SubjectNameMatchType(StrEnum):
    CANONICAL = "canonical"
    ALIAS = "alias"
