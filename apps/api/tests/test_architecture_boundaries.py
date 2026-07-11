from __future__ import annotations

import ast
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _imports_under(path: Path) -> set[str]:
    imports: set[str] = set()
    for source_file in path.rglob("*.py"):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
    return imports


def test_rag_core_does_not_depend_on_application_or_infrastructure() -> None:
    imports = _imports_under(REPOSITORY_ROOT / "packages" / "rag_core")

    assert not any(name.startswith("packages.indexer_application") for name in imports)
    assert not any(name.startswith("packages.indexer_infrastructure") for name in imports)
    assert not any(name.startswith("app") for name in imports)


def test_application_does_not_depend_on_api_or_concrete_infrastructure() -> None:
    imports = _imports_under(REPOSITORY_ROOT / "packages" / "indexer_application")

    assert not any(name.startswith("packages.indexer_infrastructure") for name in imports)
    assert not any(name.startswith("app") for name in imports)


def test_infrastructure_does_not_depend_on_api() -> None:
    imports = _imports_under(REPOSITORY_ROOT / "packages" / "indexer_infrastructure")

    assert not any(name.startswith("app") for name in imports)
