"""PostgreSQL repository adapters grouped by application aggregate.

The exports preserve the former ``postgres.repositories`` import surface while
allowing document, job, and query-run persistence to evolve independently.
"""

from packages.indexer_infrastructure.postgres.repositories.background_jobs import (
    SqlAlchemyBackgroundJobRepository,
)
from packages.indexer_infrastructure.postgres.repositories.documents import (
    SqlAlchemyDocumentRepository,
)
from packages.indexer_infrastructure.postgres.repositories.mappers import (
    storage_metadata,
    to_document_record,
    to_query_run_record,
)
from packages.indexer_infrastructure.postgres.repositories.query_runs import (
    SqlAlchemyQueryRunRepository,
)

__all__ = [
    "SqlAlchemyBackgroundJobRepository",
    "SqlAlchemyDocumentRepository",
    "SqlAlchemyQueryRunRepository",
    "storage_metadata",
    "to_document_record",
    "to_query_run_record",
]
