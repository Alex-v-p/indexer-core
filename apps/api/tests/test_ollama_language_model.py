from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from packages.indexer_infrastructure.ollama import language_model
from packages.indexer_infrastructure.ollama.language_model import OllamaLLMProvider
from packages.rag_core.ports import LLMProviderError


@pytest.mark.parametrize(
    "temperature",
    [float("nan"), float("inf"), float("-inf")],
)
def test_provider_rejects_non_finite_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature must be finite"):
        OllamaLLMProvider(
            base_url="http://ollama",
            model="query-model",
            temperature=temperature,
        )


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def build_client(*args, **kwargs) -> httpx.AsyncClient:
        return async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(language_model.httpx, "AsyncClient", build_client)


@pytest.mark.asyncio
async def test_query_generation_sends_deterministic_text_options(monkeypatch) -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"response": "  stable answer  "})

    _install_transport(monkeypatch, handler)
    provider = OllamaLLMProvider(
        base_url="http://ollama",
        model="query-model",
        temperature=0,
        seed=0,
        max_output_tokens=2048,
        structured_max_output_tokens=4096,
    )

    answer = await provider.generate("Question")

    assert answer == "stable answer"
    assert requests == [
        {
            "model": "query-model",
            "prompt": "Question",
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": 0,
                "num_predict": 2048,
            },
        },
    ]


@pytest.mark.asyncio
async def test_legacy_free_text_generation_does_not_add_query_options(monkeypatch) -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"response": "context"})

    _install_transport(monkeypatch, handler)
    provider = OllamaLLMProvider(base_url="http://ollama", model="ingestion-model")

    await provider.generate("Contextualize")

    assert requests == [
        {
            "model": "ingestion-model",
            "prompt": "Contextualize",
            "stream": False,
        },
    ]


@pytest.mark.asyncio
async def test_structured_generation_sends_exact_schema_and_separate_bound(monkeypatch) -> None:
    requests: list[dict[str, object]] = []
    response_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "classification": {
                "type": "string",
                "enum": ["factual", "explanatory"],
            },
        },
        "required": ["classification"],
        "additionalProperties": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"response": '{"classification":"factual"}'})

    _install_transport(monkeypatch, handler)
    provider = OllamaLLMProvider(
        base_url="http://ollama",
        model="query-model",
        temperature=0,
        seed=0,
        max_output_tokens=2048,
        structured_max_output_tokens=4096,
    )

    answer = await provider.generate_structured(
        "Classify",
        response_schema=response_schema,
    )

    assert answer == '{"classification":"factual"}'
    assert requests[0]["format"] == response_schema
    assert requests[0]["options"] == {
        "temperature": 0,
        "seed": 0,
        "num_predict": 4096,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(503, text="unavailable"), "request failed"),
        (httpx.Response(200, content=b"not-json"), "invalid JSON"),
        (httpx.Response(200, json=[]), "invalid response envelope"),
        (httpx.Response(200, json={"response": 42}), "invalid response envelope"),
        (httpx.Response(200, json={"response": "  "}), "empty response"),
    ],
)
async def test_provider_normalizes_http_and_invalid_response_failures(
    monkeypatch,
    response: httpx.Response,
    message: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return response

    _install_transport(monkeypatch, handler)
    provider = OllamaLLMProvider(base_url="http://ollama", model="query-model")

    with pytest.raises(LLMProviderError, match=message):
        await provider.generate("Question")


@pytest.mark.asyncio
async def test_provider_normalizes_network_failures(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _install_transport(monkeypatch, handler)
    provider = OllamaLLMProvider(base_url="http://ollama", model="query-model")

    with pytest.raises(LLMProviderError, match="could not be completed"):
        await provider.generate("Question")
