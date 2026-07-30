from __future__ import annotations

import math
from collections.abc import Mapping

import httpx

from packages.rag_core.ports.language_models import LLMProviderError


class OllamaLLMProvider:
    """LLM provider backed by an Ollama HTTP service.

    In Docker Compose the API uses the internal URL http://ollama:11434.
    For local non-Docker development, use http://localhost:11434.

    Free-text generation options are opt-in so ingestion-time callers retain
    their existing Ollama behavior. Structured generation is always bounded
    and uses the configured deterministic query defaults.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        temperature: float | None = None,
        seed: int | None = None,
        max_output_tokens: int | None = None,
        structured_max_output_tokens: int = 4096,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty.")
        if temperature is not None and (not math.isfinite(temperature) or temperature < 0):
            raise ValueError("temperature must be finite and not negative.")
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive.")
        if structured_max_output_tokens <= 0:
            raise ValueError("structured_max_output_tokens must be positive.")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.seed = seed
        self.max_output_tokens = max_output_tokens
        self.structured_max_output_tokens = structured_max_output_tokens

    async def generate(self, prompt: str) -> str:
        request_payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        options = self._text_generation_options()
        if options:
            request_payload["options"] = options
        return await self._generate(request_payload)

    async def generate_structured(
        self,
        prompt: str,
        *,
        response_schema: Mapping[str, object],
    ) -> str:
        request_payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": dict(response_schema),
            "options": {
                "temperature": self.temperature if self.temperature is not None else 0,
                "seed": self.seed if self.seed is not None else 0,
                "num_predict": self.structured_max_output_tokens,
            },
        }
        return await self._generate(request_payload)

    def _text_generation_options(self) -> dict[str, int | float]:
        options: dict[str, int | float] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.seed is not None:
            options["seed"] = self.seed
        if self.max_output_tokens is not None:
            options["num_predict"] = self.max_output_tokens
        return options

    async def _generate(self, request_payload: dict[str, object]) -> str:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=request_payload,
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(f"Ollama generation request failed: {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError("Ollama generation request could not be completed.") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMProviderError("Ollama returned an invalid JSON response.") from exc
        if not isinstance(payload, dict):
            raise LLMProviderError("Ollama returned an invalid response envelope.")

        answer = payload.get("response")
        if answer is None or (isinstance(answer, str) and not answer.strip()):
            raise LLMProviderError("Ollama returned an empty response.")
        if not isinstance(answer, str):
            raise LLMProviderError("Ollama returned an invalid response envelope.")
        return answer.strip()
