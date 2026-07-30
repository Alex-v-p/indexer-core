from __future__ import annotations

import json
import logging
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPOSITORY_ROOT / "apps" / "api"
for import_root in (REPOSITORY_ROOT, API_ROOT):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from app.core.config import Settings, get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

_MANIFEST_NAME = ".indexer-cross-encoder.json"
_PREFERRED_MODEL_PATTERNS = (
    "config.json",
    "modules.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tokenizer.model",
)
_PYTORCH_WEIGHT_PATTERNS = (
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "pytorch_model.bin.index.json",
)


def download_cross_encoder_model(
    *,
    model_id: str,
    revision: str,
    target_dir: Path,
    force: bool = False,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> Path:
    """Populate a deterministic local model directory in a persistent volume.

    A manifest marks a complete download. Matching completed models are reused
    without invoking the Hugging Face client, so subsequent Compose startups do
    not perform network checks. Downloads are staged beside the target and then
    moved into place only after the snapshot is complete.
    """

    normalized_model_id = model_id.strip()
    normalized_revision = revision.strip() or "main"
    if not normalized_model_id:
        raise ValueError("model_id must not be empty.")

    target_dir = target_dir.resolve()
    if target_dir == Path(target_dir.anchor):
        raise ValueError("target_dir must not be the filesystem root.")

    if not force and _model_is_ready(
        target_dir,
        model_id=normalized_model_id,
        revision=normalized_revision,
    ):
        logger.info(
            "Cross-encoder model is already available locally; skipping download.",
            extra={"model_id": normalized_model_id, "target_dir": str(target_dir)},
        )
        return target_dir

    uses_hugging_face_downloader = snapshot_download_fn is None
    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = target_dir.with_name(f".{target_dir.name}.download")
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Preparing cross-encoder model in the persistent Docker volume.",
        extra={
            "model_id": normalized_model_id,
            "revision": normalized_revision,
            "target_dir": str(target_dir),
        },
    )

    try:
        restored_from_cache = False
        if uses_hugging_face_downloader and not force:
            restored_from_cache = _try_restore_from_local_hugging_face_cache(
                snapshot_download_fn,
                model_id=normalized_model_id,
                revision=normalized_revision,
                staging_dir=staging_dir,
            )

        if not restored_from_cache:
            logger.info("Downloading cross-encoder files from Hugging Face.")
            _download_model_files(
                snapshot_download_fn,
                model_id=normalized_model_id,
                revision=normalized_revision,
                staging_dir=staging_dir,
                local_files_only=False,
                force_download=force,
            )

        _write_manifest(
            staging_dir,
            model_id=normalized_model_id,
            revision=normalized_revision,
        )
        if not _model_files_exist(staging_dir):
            raise RuntimeError(
                "Downloaded snapshot is missing model configuration, weights, or tokenizer files.",
            )

        shutil.rmtree(target_dir, ignore_errors=True)
        staging_dir.replace(target_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    logger.info(
        "Cross-encoder model preparation completed.",
        extra={"model_id": normalized_model_id, "target_dir": str(target_dir)},
    )
    return target_dir


def _try_restore_from_local_hugging_face_cache(
    snapshot_download_fn: Callable[..., str],
    *,
    model_id: str,
    revision: str,
    staging_dir: Path,
) -> bool:
    try:
        _download_model_files(
            snapshot_download_fn,
            model_id=model_id,
            revision=revision,
            staging_dir=staging_dir,
            local_files_only=True,
            force_download=False,
        )
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)
        return False

    if _model_files_exist(staging_dir):
        logger.info("Restored cross-encoder files from the existing local Hugging Face cache.")
        return True

    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    return False


def _download_model_files(
    snapshot_download_fn: Callable[..., str],
    *,
    model_id: str,
    revision: str,
    staging_dir: Path,
    local_files_only: bool,
    force_download: bool,
) -> None:
    _download_snapshot_files(
        snapshot_download_fn,
        model_id=model_id,
        revision=revision,
        staging_dir=staging_dir,
        allow_patterns=_PREFERRED_MODEL_PATTERNS,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    if not _has_model_weights(staging_dir):
        _download_snapshot_files(
            snapshot_download_fn,
            model_id=model_id,
            revision=revision,
            staging_dir=staging_dir,
            allow_patterns=_PYTORCH_WEIGHT_PATTERNS,
            local_files_only=local_files_only,
            force_download=force_download,
        )


def _download_snapshot_files(
    snapshot_download_fn: Callable[..., str],
    *,
    model_id: str,
    revision: str,
    staging_dir: Path,
    allow_patterns: tuple[str, ...],
    local_files_only: bool,
    force_download: bool,
) -> None:
    snapshot_download_fn(
        repo_id=model_id,
        revision=revision,
        repo_type="model",
        local_dir=str(staging_dir),
        allow_patterns=list(allow_patterns),
        local_files_only=local_files_only,
        force_download=force_download,
    )


def _model_is_ready(target_dir: Path, *, model_id: str, revision: str) -> bool:
    if not _model_files_exist(target_dir):
        return False

    manifest_path = target_dir / _MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False

    return manifest == {
        "model_id": model_id,
        "revision": revision,
    }


def _model_files_exist(target_dir: Path) -> bool:
    if not target_dir.is_dir() or not (target_dir / "config.json").is_file():
        return False

    has_weights = _has_model_weights(target_dir)
    has_tokenizer = any(
        (target_dir / filename).is_file()
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "spiece.model",
            "sentencepiece.bpe.model",
        )
    )
    return has_weights and has_tokenizer


def _has_model_weights(target_dir: Path) -> bool:
    return any(target_dir.glob("*.safetensors")) or any(target_dir.glob("*.bin"))


def _write_manifest(target_dir: Path, *, model_id: str, revision: str) -> None:
    manifest: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
    }
    (target_dir / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def bootstrap_cross_encoder(settings: Settings) -> Path:
    return download_cross_encoder_model(
        model_id=settings.cross_encoder_model,
        revision=settings.cross_encoder_model_revision,
        target_dir=Path(settings.cross_encoder_model_path),
        force=settings.cross_encoder_download_force,
    )


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    bootstrap_cross_encoder(settings)


if __name__ == "__main__":
    main()
