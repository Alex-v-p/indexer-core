from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentIngestionConfig:
    chunk_max_chars: int
    chunk_overlap_chars: int
    vector_collection_name: str
    original_vector_name: str = "original"
    contextual_vector_name: str = "contextual"
    hierarchy_vector_name: str = "hierarchy"
    contextualization_enabled: bool = True
    contextualization_fail_open: bool = False
    hierarchical_indexing_enabled: bool = False
    hierarchical_indexing_fail_open: bool = False
