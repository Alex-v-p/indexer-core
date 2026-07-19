from packages.rag_core.documents import VersionSelectionMode
from packages.rag_core.query_understanding.versioning import detect_document_version_constraint


def test_ordinary_query_does_not_enable_version_preference() -> None:
    constraint = detect_document_version_constraint("What authentication mechanism does the API use?")

    assert constraint.mode is VersionSelectionMode.ALL
    assert constraint.active is False


def test_latest_version_intent_is_detected() -> None:
    constraint = detect_document_version_constraint("What does the latest architecture document say about retries?")

    assert constraint.mode is VersionSelectionMode.LATEST
    assert constraint.active is True


def test_specific_version_numbers_are_detected() -> None:
    constraint = detect_document_version_constraint("Compare version 2 with v4 of the deployment guide.")

    assert constraint.mode is VersionSelectionMode.SPECIFIC
    assert constraint.version_numbers == (2, 4)


def test_latest_and_previous_intent_is_detected() -> None:
    constraint = detect_document_version_constraint("What changed between the current and previous policy?")

    assert constraint.mode is VersionSelectionMode.LATEST_AND_PREVIOUS


def test_oldest_version_intent_is_detected() -> None:
    constraint = detect_document_version_constraint("Use the oldest version of the architecture guide.")

    assert constraint.mode is VersionSelectionMode.OLDEST
    assert constraint.active is True


def test_all_historical_versions_intent_is_detected() -> None:
    constraint = detect_document_version_constraint("Use all older versions of the policy.")

    assert constraint.mode is VersionSelectionMode.ALL_EXCEPT_LATEST


def test_all_versions_intent_is_explicitly_recorded() -> None:
    constraint = detect_document_version_constraint("Compare all versions of the policy.")

    assert constraint.mode is VersionSelectionMode.ALL_VERSIONS
    assert constraint.active is True


def test_oldest_and_newest_version_intent_is_detected() -> None:
    constraint = detect_document_version_constraint("Compare the oldest and newest versions.")

    assert constraint.mode is VersionSelectionMode.OLDEST_AND_LATEST
