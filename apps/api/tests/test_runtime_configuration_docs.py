import ast
from pathlib import Path


RUNTIME_SETTING_NAMES = (
    "TEMPORAL_QUERY_TIMEZONE",
    "RETRIEVAL_RETRY_MAX_RETRIES",
    "RETRIEVAL_RETRY_TOP_K_MULTIPLIER",
    "RETRIEVAL_RETRY_MAX_TOP_K",
    "RETRIEVAL_RETRY_EXPAND_QUERY",
    "RETRIEVAL_RETRY_MAX_QUERY_CHARS",
    "RETRIEVAL_RETRY_MAX_TOTAL_ATTEMPTS",
    "RETRIEVAL_RETRY_MAX_RECLASSIFICATIONS",
    "RETRIEVAL_RETRY_MAX_ACCUMULATED_EVIDENCE",
    "DB_POOL_SIZE",
    "DB_MAX_OVERFLOW",
    "DB_ECHO",
)


def test_api_runtime_settings_are_documented_and_forwarded_by_compose() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    runtime_defaults = _settings_defaults(repository_root)
    example_environment = (repository_root / ".env.example").read_text(encoding="utf-8")
    compose = (repository_root / "compose.yaml").read_text(encoding="utf-8")
    api_service = compose.split("  api:\n", maxsplit=1)[1].split("    ports:\n", maxsplit=1)[0]

    for name, default in runtime_defaults.items():
        assert f"{name}={default}" in example_environment
        assert f"      {name}: ${{{name}:-{default}}}" in api_service


def _settings_defaults(repository_root: Path) -> dict[str, str]:
    config_path = repository_root / "packages" / "indexer_bootstrap" / "config.py"
    module = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
    settings = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Settings"
    )
    fields = {
        node.target.id: node.value
        for node in settings.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    }

    defaults: dict[str, str] = {}
    for environment_name in RUNTIME_SETTING_NAMES:
        field_name = environment_name.lower()
        assert field_name in fields, f"Settings field {field_name!r} is missing"
        value = _literal_default(fields[field_name])
        defaults[environment_name] = str(value).lower() if isinstance(value, bool) else str(value)
    return defaults


def _literal_default(value: ast.expr) -> object:
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "Field":
        value = next(
            keyword.value
            for keyword in value.keywords
            if keyword.arg == "default"
        )
    return ast.literal_eval(value)


WORKER_SETTING_NAMES = (
    "BACKGROUND_WORKER_POLL_INTERVAL_SECONDS",
    "BACKGROUND_WORKER_CONCURRENCY",
    "BACKGROUND_WORKER_LOCK_TIMEOUT_SECONDS",
    "BACKGROUND_WORKER_HEARTBEAT_SECONDS",
    "BACKGROUND_WORKER_RETRY_BASE_SECONDS",
    "BACKGROUND_JOB_INGESTION_MAX_ATTEMPTS",
    "BACKGROUND_JOB_MAINTENANCE_MAX_ATTEMPTS",
    "BACKGROUND_JOB_EVALUATION_MAX_ATTEMPTS",
    "BACKGROUND_JOB_QUERY_MAX_ATTEMPTS",
    "BACKGROUND_JOB_QUERY_PRIORITY",
)


def test_background_worker_settings_are_documented_and_forwarded() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    runtime_defaults = _selected_settings_defaults(repository_root, WORKER_SETTING_NAMES)
    example_environment = (repository_root / ".env.example").read_text(encoding="utf-8")
    compose = (repository_root / "compose.yaml").read_text(encoding="utf-8")
    worker_service = compose.split("  worker:\n", maxsplit=1)[1].split("  web:\n", maxsplit=1)[0]

    assert 'command: ["python", "-m", "indexer_worker"]' in worker_service
    assert "dockerfile: infra/docker/worker.Dockerfile" in worker_service
    assert "image: indexer-core-worker:${IMAGE_TAG:-local}" in worker_service
    for name, default in runtime_defaults.items():
        assert f"{name}={default}" in example_environment
        assert f"      {name}: ${{{name}:-{default}}}" in worker_service


def _selected_settings_defaults(repository_root: Path, names: tuple[str, ...]) -> dict[str, str]:
    config_path = repository_root / "packages" / "indexer_bootstrap" / "config.py"
    module = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
    settings = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Settings"
    )
    fields = {
        node.target.id: node.value
        for node in settings.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    }
    defaults: dict[str, str] = {}
    for environment_name in names:
        field_name = environment_name.lower()
        value = _literal_default(fields[field_name])
        defaults[environment_name] = str(value).lower() if isinstance(value, bool) else str(value)
    return defaults
