from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.download_cross_encoder import download_cross_encoder_model


def _write_complete_model(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"weights")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_cross_encoder_bootstrap_downloads_once_then_reuses_local_snapshot(tmp_path: Path) -> None:
    target = tmp_path / "cross-encoder"
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        _write_complete_model(local_dir)
        return str(local_dir)

    result = download_cross_encoder_model(
        model_id="cross-encoder/test-model",
        revision="abc123",
        target_dir=target,
        snapshot_download_fn=fake_snapshot_download,
    )

    assert result == target.resolve()
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "cross-encoder/test-model"
    assert calls[0]["revision"] == "abc123"
    assert calls[0]["repo_type"] == "model"
    assert "model.safetensors" in calls[0]["allow_patterns"]
    assert "pytorch_model.bin" not in calls[0]["allow_patterns"]
    assert Path(calls[0]["local_dir"]).parent == target.parent.resolve()
    assert json.loads((target / ".indexer-cross-encoder.json").read_text(encoding="utf-8")) == {
        "model_id": "cross-encoder/test-model",
        "revision": "abc123",
    }

    def unexpected_network_call(**_: Any) -> str:
        raise AssertionError("completed local model should not invoke the downloader")

    reused = download_cross_encoder_model(
        model_id="cross-encoder/test-model",
        revision="abc123",
        target_dir=target,
        snapshot_download_fn=unexpected_network_call,
    )

    assert reused == target.resolve()


def test_cross_encoder_bootstrap_redownloads_when_revision_changes(tmp_path: Path) -> None:
    target = tmp_path / "cross-encoder"
    calls: list[str] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs["revision"])
        local_dir = Path(kwargs["local_dir"])
        _write_complete_model(local_dir)
        return str(local_dir)

    download_cross_encoder_model(
        model_id="cross-encoder/test-model",
        revision="revision-one",
        target_dir=target,
        snapshot_download_fn=fake_snapshot_download,
    )
    download_cross_encoder_model(
        model_id="cross-encoder/test-model",
        revision="revision-two",
        target_dir=target,
        snapshot_download_fn=fake_snapshot_download,
    )

    assert calls == ["revision-one", "revision-two"]
    assert json.loads((target / ".indexer-cross-encoder.json").read_text(encoding="utf-8"))["revision"] == (
        "revision-two"
    )


def test_cross_encoder_settings_default_to_volume_path_and_offline_loading() -> None:
    from app.core.config import Settings

    settings = Settings()

    assert settings.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert settings.cross_encoder_model_revision == "c5ee24cb16019beea0893ab7796b1df96625c6b8"
    assert settings.cross_encoder_model_path == "/root/.cache/huggingface/indexer/cross-encoder"
    assert settings.cross_encoder_local_files_only is True
    assert settings.cross_encoder_download_force is False


def test_cross_encoder_bootstrap_falls_back_to_pytorch_weights(tmp_path: Path) -> None:
    target = tmp_path / "cross-encoder"
    calls: list[list[str]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        allow_patterns = kwargs["allow_patterns"]
        calls.append(allow_patterns)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        (local_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        if "pytorch_model.bin" in allow_patterns:
            (local_dir / "pytorch_model.bin").write_bytes(b"weights")
        return str(local_dir)

    download_cross_encoder_model(
        model_id="cross-encoder/bin-only",
        revision="abc123",
        target_dir=target,
        snapshot_download_fn=fake_snapshot_download,
    )

    assert len(calls) == 2
    assert "model.safetensors" in calls[0]
    assert "pytorch_model.bin" in calls[1]
    assert (target / "pytorch_model.bin").is_file()


def test_cross_encoder_bootstrap_rejects_filesystem_root() -> None:
    with pytest.raises(ValueError, match="filesystem root"):
        download_cross_encoder_model(
            model_id="cross-encoder/test",
            revision="abc123",
            target_dir=Path("/"),
            snapshot_download_fn=lambda **_: "",
        )


def test_cross_encoder_bootstrap_uses_existing_hugging_face_cache_without_online_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    target = tmp_path / "cross-encoder"
    local_files_only_calls: list[bool] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        local_files_only_calls.append(kwargs["local_files_only"])
        if not kwargs["local_files_only"]:
            raise AssertionError("online download should not be used when the volume cache is complete")
        local_dir = Path(kwargs["local_dir"])
        _write_complete_model(local_dir)
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    download_cross_encoder_model(
        model_id="cross-encoder/cached-model",
        revision="abc123",
        target_dir=target,
    )

    assert local_files_only_calls == [True]
    assert (target / "model.safetensors").is_file()


def test_cross_encoder_bootstrap_downloads_online_only_after_local_cache_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    target = tmp_path / "cross-encoder"
    local_files_only_calls: list[bool] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        local_files_only_calls.append(kwargs["local_files_only"])
        if kwargs["local_files_only"]:
            raise FileNotFoundError("not cached")
        local_dir = Path(kwargs["local_dir"])
        _write_complete_model(local_dir)
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    download_cross_encoder_model(
        model_id="cross-encoder/new-model",
        revision="abc123",
        target_dir=target,
    )

    assert local_files_only_calls == [True, False]
    assert (target / "model.safetensors").is_file()


def test_cross_encoder_force_refresh_skips_local_cache_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    target = tmp_path / "cross-encoder"
    local_files_only_calls: list[bool] = []
    force_download_calls: list[bool] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        local_files_only_calls.append(kwargs["local_files_only"])
        force_download_calls.append(kwargs["force_download"])
        local_dir = Path(kwargs["local_dir"])
        _write_complete_model(local_dir)
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    download_cross_encoder_model(
        model_id="cross-encoder/forced-model",
        revision="abc123",
        target_dir=target,
        force=True,
    )

    assert local_files_only_calls == [False]
    assert force_download_calls == [True]
