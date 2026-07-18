from packages.rag_core.documents.version_families import (
    document_family_key,
    normalized_document_identity,
    same_document_family,
)


def test_draft_suffixes_map_to_same_document_family() -> None:
    assert document_family_key("Realization_Draft4.pdf") == "realization"
    assert document_family_key("Realization_Draft5.pdf") == "realization"
    assert same_document_family("Realization_Draft4.pdf", "Realization_Draft5.pdf") is True


def test_common_version_suffixes_are_removed_conservatively() -> None:
    assert same_document_family("Architecture-v2.md", "Architecture version 3.md") is True
    assert same_document_family("Policy revision 4", "Policy_rev5") is True


def test_draft_text_inside_a_word_is_not_treated_as_a_version_marker() -> None:
    assert document_family_key("Redraft4.pdf") == "redraft4"


def test_ordinary_trailing_numbers_are_preserved() -> None:
    assert document_family_key("Annual Report 2024.pdf") == "annual report 2024"
    assert same_document_family("Annual Report 2024.pdf", "Annual Report 2025.pdf") is False


def test_exact_identity_normalizes_extension_and_separators() -> None:
    assert normalized_document_identity("Realization_Draft4.pdf") == "realization draft4"
    assert normalized_document_identity("realization-draft4.md") == "realization draft4"
