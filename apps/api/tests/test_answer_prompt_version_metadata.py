from packages.rag_core.generation.prompt import source_metadata_bits
from packages.rag_core.retrieval.models import EvidenceItem


def test_answer_prompt_source_metadata_includes_document_version_label() -> None:
    item = EvidenceItem(
        rank=1,
        text="Versioned evidence.",
        metadata={
            "original_filename": "policy.md",
            "document_version_number": 3,
            "document_version_label": "v3",
        },
    )

    assert source_metadata_bits(item)[:2] == ["file=policy.md", "version=v3"]
