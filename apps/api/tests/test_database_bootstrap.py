from alembic.config import Config

from app.core.config import Settings
from scripts import bootstrap_database


def test_bootstrap_settings_have_safe_defaults() -> None:
    settings = Settings()

    assert settings.bootstrap_db_max_attempts == 30
    assert settings.bootstrap_db_retry_seconds == 2.0


def test_run_migrations_targets_alembic_head(monkeypatch) -> None:
    calls: dict[str, object] = {}
    alembic_config = Config()

    monkeypatch.setattr(bootstrap_database, "get_alembic_config", lambda: alembic_config)

    def fake_upgrade(config: Config, revision: str) -> None:
        calls["config"] = config
        calls["revision"] = revision

    monkeypatch.setattr(bootstrap_database.command, "upgrade", fake_upgrade)

    bootstrap_database.run_migrations()

    assert calls == {
        "config": alembic_config,
        "revision": "head",
    }
