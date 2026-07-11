from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from packages.indexer_infrastructure.cross_encoder.reranker import CrossEncoderReranker, _normalize_scores
from packages.rag_core.retrieval import EvidenceItem
from packages.rag_core.retrieval.rerankers import RerankerError


class FakeCrossEncoder:
    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)
        self.calls: list[tuple[list[tuple[str, str]], int, bool]] = []

    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> list[float]:
        pairs = list(sentences)
        self.calls.append((pairs, batch_size, show_progress_bar))
        return self.scores[: len(pairs)]


async def test_cross_encoder_reranker_scores_pairs_and_preserves_retrieval_metadata() -> None:
    model = FakeCrossEncoder([0.2, 0.95, 0.6])
    reranker = CrossEncoderReranker(
        model_name="/models/cross-encoder",
        model_identifier="cross-encoder/test",
        batch_size=2,
        model=model,
    )
    evidence = [
        EvidenceItem(rank=1, text="weak", score=0.9, metadata={"retrieval_source": "hybrid"}),
        EvidenceItem(rank=2, text="best", score=0.5, metadata={"retrieval_source": "hybrid"}),
        EvidenceItem(rank=3, text="middle", score=0.7, metadata={"retrieval_source": "hybrid"}),
    ]

    reranked = await reranker.rerank("Which evidence is best?", evidence, top_k=2)

    assert model.calls == [
        (
            [
                ("Which evidence is best?", "weak"),
                ("Which evidence is best?", "best"),
                ("Which evidence is best?", "middle"),
            ],
            2,
            False,
        ),
    ]
    assert [item.text for item in reranked] == ["best", "middle"]
    assert [item.score for item in reranked] == [0.95, 0.6]
    assert reranked[0].metadata["retrieval_source"] == "hybrid"
    assert reranked[0].metadata["rerank"] == {
        "provider": "cross_encoder",
        "model": "cross-encoder/test",
        "original_rank": 2,
        "original_score": 0.5,
        "score": 0.95,
        "score_source": "cross_encoder",
        "fallback_used": False,
    }


async def test_cross_encoder_reranker_rejects_missing_scores() -> None:
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/test",
        model=FakeCrossEncoder([0.5]),
    )

    with pytest.raises(RerankerError, match="unexpected number of scores"):
        await reranker.rerank(
            "question",
            [EvidenceItem(rank=1, text="one"), EvidenceItem(rank=2, text="two")],
            top_k=2,
        )


def test_cross_encoder_score_normalization_handles_array_like_values() -> None:
    class ArrayLike:
        def tolist(self) -> list[list[float]]:
            return [[-0.1], [0.5], [1.1]]

    assert _normalize_scores(ArrayLike()) == [0.0, 0.5, 1.0]

    with pytest.raises(RerankerError, match="one relevance score"):
        _normalize_scores([[0.1, 0.9]])


def test_cross_encoder_rejects_missing_local_model_before_provider_loading(tmp_path: Path) -> None:
    reranker = CrossEncoderReranker(
        model_name=str(tmp_path / "missing-model"),
        model_identifier="cross-encoder/test",
        local_files_only=True,
    )

    with pytest.raises(RerankerError, match="cross-encoder-bootstrap"):
        reranker._get_model()


def test_cross_encoder_loader_uses_local_path_without_cache_or_hub_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    calls: list[tuple[str, dict[str, object]]] = []
    loaded_model = FakeCrossEncoder([0.5])

    def fake_cross_encoder(model_name: str, **kwargs: object) -> FakeCrossEncoder:
        calls.append((model_name, kwargs))
        return loaded_model

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", fake_cross_encoder)
    reranker = CrossEncoderReranker(
        model_name=str(model_path),
        model_identifier="cross-encoder/test",
        max_length=384,
        device="cpu",
        local_files_only=True,
    )

    assert reranker._get_model() is loaded_model
    assert reranker._get_model() is loaded_model
    assert len(calls) == 1
    assert calls[0][0] == str(model_path)
    assert calls[0][1]["local_files_only"] is True
    assert calls[0][1]["max_length"] == 384
    assert calls[0][1]["device"] == "cpu"
    assert "cache_folder" not in calls[0][1]
