from enum import StrEnum


def enum_values(enum_cls: type[StrEnum]) -> list[str]:
    """Use enum values instead of enum member names for database storage."""

    return [member.value for member in enum_cls]
