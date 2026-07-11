from __future__ import annotations

import httpx

from packages.rag_core.ports.language_models import LLMProviderError


class OllamaLLMProvider:
    """LLM provider backed by an Ollama HTTP service.

    In Docker Compose the API uses the internal URL http://ollama:11434.
    For local non-Docker development, use http://localhost:11434.
    """

    def __init__(self, *, base_url: str, model: str, timeout_seconds: float = 120.0) -> None:
        if not model.strip():
            raise ValueError("model must not be empty.")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(f"Ollama generation request failed: {exc.response.text}") from exc

        payload = response.json()
        answer = payload.get("response")
        if not isinstance(answer, str) or not answer.strip():
            raise LLMProviderError("Ollama returned an empty response.")
        return answer.strip()
