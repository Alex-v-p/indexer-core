from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentIngestionConfig:
    chunk_max_chars: int
    chunk_overlap_chars: int
    vector_collection_name: str
