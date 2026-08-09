from __future__ import annotations

from packages.indexer_application.dto import BackgroundJobType
from packages.indexer_bootstrap.config import Settings
import pytest


def test_new_organization_enqueue_is_default_and_legacy_subject_enqueue_is_not() -> None:
    settings = Settings()
    assert settings.organization_classification_enabled is True
    assert settings.organization_classification_policy_version == "document-organization-policy/1.1"
    assert settings.organization_classification_semantic_reuse_threshold == 0.75
    assert settings.organization_classification_semantic_reuse_margin == 0.10
    assert settings.subject_classification_enabled is False
    assert BackgroundJobType.CLASSIFY_DOCUMENT_ORGANIZATION.value == "classify_document_organization"
    assert BackgroundJobType.CLASSIFY_DOCUMENT_SUBJECTS.value == "classify_document_subjects"


def test_organization_classification_fails_fast_without_model_provider() -> None:
    with pytest.raises(ValueError, match="model provider is disabled"):
        Settings(
            organization_classification_enabled=True,
            organization_classification_model_enabled=False,
        )
