from __future__ import annotations

from collections.abc import Mapping

import pytest

from app.composition import documents as api_documents
from app.composition import providers as api_providers
from packages.indexer_bootstrap.composition import documents, providers
from app.core.config import Settings


class _StructuredProvider:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def generate(self, prompt: str) -> str:
        return prompt

    async def generate_structured(
        self,
        prompt: str,
        *,
        response_schema: Mapping[str, object],
    ) -> str:
        return prompt


class _TextOnlyProvider:
    async def generate(self, prompt: str) -> str:
        return prompt


@pytest.mark.parametrize(
    "temperature",
    [float("nan"), float("inf"), float("-inf")],
)
def test_query_temperature_settings_reject_non_finite_values(temperature: float) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, ollama_query_temperature=temperature)


def test_query_language_model_receives_deterministic_settings(monkeypatch) -> None:
    constructed: list[_StructuredProvider] = []

    def build_provider(**kwargs) -> _StructuredProvider:
        provider = _StructuredProvider(**kwargs)
        constructed.append(provider)
        return provider

    monkeypatch.setattr(providers, "OllamaLLMProvider", build_provider)
    settings = Settings(
        _env_file=None,
        ollama_query_temperature=0,
        ollama_query_seed=0,
        ollama_query_max_output_tokens=1234,
        ollama_structured_max_output_tokens=4321,
    )

    provider = providers.build_language_model(settings)

    assert provider is constructed[0]
    assert constructed[0].kwargs["temperature"] == 0
    assert constructed[0].kwargs["seed"] == 0
    assert constructed[0].kwargs["max_output_tokens"] == 1234
    assert constructed[0].kwargs["structured_max_output_tokens"] == 4321


def test_ingestion_language_models_do_not_inherit_query_generation_options(monkeypatch) -> None:
    constructed: list[_StructuredProvider] = []

    def build_provider(**kwargs) -> _StructuredProvider:
        provider = _StructuredProvider(**kwargs)
        constructed.append(provider)
        return provider

    monkeypatch.setattr(documents, "OllamaLLMProvider", build_provider)
    settings = Settings(
        _env_file=None,
        contextualization_enabled=True,
        hierarchical_indexing_enabled=True,
    )

    documents.build_document_context_hierarchy_builder(settings)
    documents.build_chunk_contextualizer(
        settings,
        hierarchy_builder=object(),  # type: ignore[arg-type]
    )

    assert len(constructed) == 2
    for provider in constructed:
        assert "temperature" not in provider.kwargs
        assert "seed" not in provider.kwargs
        assert "max_output_tokens" not in provider.kwargs
        assert "structured_max_output_tokens" not in provider.kwargs


def test_structured_capability_miswiring_fails_fast() -> None:
    with pytest.raises(TypeError, match="does not support structured generation"):
        providers.require_structured_llm_provider(_TextOnlyProvider())


def test_query_language_model_builder_asserts_structured_capability(monkeypatch) -> None:
    monkeypatch.setattr(providers, "OllamaLLMProvider", lambda **kwargs: _TextOnlyProvider())

    with pytest.raises(TypeError, match="does not support structured generation"):
        providers.build_language_model(Settings(_env_file=None))


def test_query_generation_settings_load_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_QUERY_TEMPERATURE", "0")
    monkeypatch.setenv("OLLAMA_QUERY_SEED", "7")
    monkeypatch.setenv("OLLAMA_QUERY_MAX_OUTPUT_TOKENS", "1234")
    monkeypatch.setenv("OLLAMA_STRUCTURED_MAX_OUTPUT_TOKENS", "4321")

    settings = Settings(_env_file=None)

    assert settings.ollama_query_temperature == 0
    assert settings.ollama_query_seed == 7
    assert settings.ollama_query_max_output_tokens == 1234
    assert settings.ollama_structured_max_output_tokens == 4321


def test_document_builder_aliases_remain_available_from_provider_facade() -> None:
    assert api_providers.build_chunk_contextualizer is api_documents.build_chunk_contextualizer
    assert (
        api_providers.build_document_context_hierarchy_builder
        is api_documents.build_document_context_hierarchy_builder
    )
    assert api_providers.build_document_ingestion_config is api_documents.build_document_ingestion_config
    assert api_providers.build_document_object_store is api_documents.build_document_object_store
