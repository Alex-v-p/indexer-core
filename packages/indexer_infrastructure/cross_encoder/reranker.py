from __future__ import annotations

import asyncio
from collections.abc import Sequence
from threading import Lock
from typing import Any, Protocol

from packages.rag_core.retrieval.models import EvidenceItem
from packages.rag_core.retrieval.rerankers import RerankerError


class _CrossEncoderModel(Protocol):
    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Any:
        """Score query/passage pairs."""


class CrossEncoderReranker:
    """Local query/passage reranker backed by Sentence Transformers.

    The model is loaded lazily on the first reranking request. This keeps API
    startup lightweight and avoids downloading a model when only the baseline,
    hybrid, or Ollama-reranked pipelines are used.
    """

    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int = 16,
        max_length: int = 512,
        device: str = "cpu",
        cache_folder: str | None = None,
        model: _CrossEncoderModel | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        if not device.strip():
            raise ValueError("device must not be empty.")

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.cache_folder = cache_folder or None
        self._model = model
        self._model_lock = Lock()

    async def rerank(
        self,
        question: str,
        evidence: Sequence[EvidenceItem],
        *,
        top_k: int,
    ) -> list[EvidenceItem]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not question.strip():
            raise ValueError("question must not be empty.")
        if not evidence:
            return []

        indexed_candidates = list(enumerate(evidence))
        scores = await asyncio.to_thread(
            self._score_candidates,
            question.strip(),
            [item for _, item in indexed_candidates],
        )
        if len(scores) != len(indexed_candidates):
            raise RerankerError(
                "Cross-encoder returned an unexpected number of scores: "
                f"expected {len(indexed_candidates)}, received {len(scores)}.",
            )

        ordered = sorted(
            indexed_candidates,
            key=lambda candidate: (
                -scores[candidate[0]],
                _stable_original_rank(candidate[1], fallback_rank=candidate[0] + 1),
                candidate[0],
            ),
        )
        selected = ordered[: min(top_k, len(ordered))]
        return [
            _reranked_copy(
                item,
                rank=rank,
                rerank_score=scores[candidate_id],
                model=self.model_name,
            )
            for rank, (candidate_id, item) in enumerate(selected, start=1)
        ]

    def _score_candidates(self, question: str, evidence: Sequence[EvidenceItem]) -> list[float]:
        model = self._get_model()
        pairs = [(question, item.text.strip()) for item in evidence]
        try:
            raw_scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:  # provider libraries expose several runtime exception types
            raise RerankerError(f"Cross-encoder inference failed: {exc}") from exc
        return _normalize_scores(raw_scores)

    def _get_model(self) -> _CrossEncoderModel:
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                import torch
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RerankerError(
                    "Cross-encoder reranking requires the sentence-transformers dependency.",
                ) from exc

            kwargs: dict[str, Any] = {
                "max_length": self.max_length,
                "activation_fn": torch.nn.Sigmoid(),
            }
            if self.device.lower() != "auto":
                kwargs["device"] = self.device
            if self.cache_folder:
                kwargs["cache_folder"] = self.cache_folder

            try:
                self._model = CrossEncoder(self.model_name, **kwargs)
            except Exception as exc:  # download, configuration, and model-loading errors
                raise RerankerError(
                    f"Cross-encoder model {self.model_name!r} could not be loaded: {exc}",
                ) from exc
            return self._model


def _normalize_scores(raw_scores: Any) -> list[float]:
    values = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        values = [values]
    if not isinstance(values, (list, tuple)):
        raise RerankerError("Cross-encoder scores must be a sequence of numeric values.")

    normalized: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise RerankerError(
                    "Cross-encoder must produce one relevance score per candidate; "
                    f"candidate {index} returned {len(value)} values.",
                )
            value = value[0]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RerankerError(f"Cross-encoder score for candidate {index} must be numeric.")
        normalized.append(min(1.0, max(0.0, float(value))))
    return normalized


def _reranked_copy(item: EvidenceItem, *, rank: int, rerank_score: float, model: str) -> EvidenceItem:
    metadata = dict(item.metadata)
    metadata["rerank"] = {
        "provider": "cross_encoder",
        "model": model,
        "original_rank": item.rank,
        "original_score": item.score,
        "score": rerank_score,
        "score_source": "cross_encoder",
        "fallback_used": False,
    }
    return EvidenceItem(
        rank=rank,
        text=item.text,
        score=rerank_score,
        qdrant_chunk_index_id=item.qdrant_chunk_index_id,
        document_id=item.document_id,
        document_version_id=item.document_version_id,
        metadata=metadata,
    )


def _stable_original_rank(item: EvidenceItem, *, fallback_rank: int) -> int:
    return item.rank if item.rank > 0 else fallback_rank
