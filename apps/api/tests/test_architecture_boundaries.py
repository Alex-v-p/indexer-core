from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGES_ROOT = REPOSITORY_ROOT / "packages"
APPS_ROOT = REPOSITORY_ROOT / "apps"


@dataclass(frozen=True)
class Boundary:
    name: str
    path: Path
    forbidden_imports: tuple[str, ...]


def _imports_under(path: Path) -> dict[Path, set[str]]:
    imports_by_file: dict[Path, set[str]] = {}
    if not path.exists():
        return imports_by_file

    for source_file in path.rglob("*.py"):
        relative_parts = source_file.relative_to(path).parts
        if "tests" in relative_parts or "__pycache__" in relative_parts:
            continue

        imports: set[str] = set()
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        imports_by_file[source_file] = imports
    return imports_by_file


def _matches_prefix(import_name: str, prefix: str) -> bool:
    return import_name == prefix or import_name.startswith(f"{prefix}.")


def _violations(boundary: Boundary) -> list[str]:
    violations: list[str] = []
    for source_file, imports in _imports_under(boundary.path).items():
        for import_name in sorted(imports):
            if any(_matches_prefix(import_name, prefix) for prefix in boundary.forbidden_imports):
                relative_path = source_file.relative_to(REPOSITORY_ROOT)
                violations.append(f"{relative_path}: imports {import_name}")
    return violations


def _deployable_app_directories() -> tuple[Path, ...]:
    if not APPS_ROOT.exists():
        return ()
    return tuple(
        sorted(
            app_dir
            for app_dir in APPS_ROOT.iterdir()
            if app_dir.is_dir() and any(app_dir.rglob("*.py"))
        )
    )


def _top_level_python_packages(app_dir: Path) -> set[str]:
    package_names: set[str] = set()
    for import_root in (app_dir, app_dir / "src"):
        if not import_root.is_dir():
            continue
        package_names.update(
            child.name
            for child in import_root.iterdir()
            if child.is_dir() and (child / "__init__.py").is_file()
        )
    return package_names


def _assert_boundary(boundary: Boundary) -> None:
    violations = _violations(boundary)
    assert not violations, f"{boundary.name} dependency violations:\n" + "\n".join(violations)


def test_package_dependency_direction() -> None:
    deployable_packages = tuple(
        sorted(
            {
                package_name
                for app_dir in _deployable_app_directories()
                for package_name in _top_level_python_packages(app_dir)
            }
        )
    )
    entrypoint_imports = ("apps", "scripts", *deployable_packages)

    boundaries = (
        Boundary(
            name="rag_core",
            path=PACKAGES_ROOT / "rag_core",
            forbidden_imports=(
                "packages.indexer_application",
                "packages.indexer_infrastructure",
                *entrypoint_imports,
            ),
        ),
        Boundary(
            name="indexer_application",
            path=PACKAGES_ROOT / "indexer_application",
            forbidden_imports=("packages.indexer_infrastructure", *entrypoint_imports),
        ),
        Boundary(
            name="indexer_infrastructure",
            path=PACKAGES_ROOT / "indexer_infrastructure",
            forbidden_imports=entrypoint_imports,
        ),
    )

    for boundary in boundaries:
        _assert_boundary(boundary)


def test_deployable_apps_do_not_import_scripts_or_other_apps() -> None:
    app_directories = _deployable_app_directories()
    packages_by_app = {
        app_dir: _top_level_python_packages(app_dir) for app_dir in app_directories
    }

    for app_dir in app_directories:
        other_app_packages = {
            package_name
            for other_app, package_names in packages_by_app.items()
            if other_app != app_dir
            for package_name in package_names
        } - packages_by_app[app_dir]
        boundary = Boundary(
            name=f"deployable app {app_dir.name}",
            path=app_dir,
            forbidden_imports=("scripts", *(f"apps.{other.name}" for other in app_directories if other != app_dir), *sorted(other_app_packages)),
        )
        _assert_boundary(boundary)


def test_operational_and_non_python_trees_are_not_python_packages() -> None:
    forbidden_package_markers = [
        REPOSITORY_ROOT / "docs" / "__init__.py",
        REPOSITORY_ROOT / "scripts" / "__init__.py",
        *(APPS_ROOT / "web").rglob("__init__.py"),
    ]

    assert not [
        marker.relative_to(REPOSITORY_ROOT)
        for marker in forbidden_package_markers
        if marker.exists()
    ]


def test_empty_worker_scaffold_is_not_reintroduced_as_a_shared_package() -> None:
    assert not (PACKAGES_ROOT / "rag_worker").exists()


def test_shared_bootstrap_does_not_depend_on_deployable_apps() -> None:
    _assert_boundary(
        Boundary(
            name="indexer_bootstrap",
            path=PACKAGES_ROOT / "indexer_bootstrap",
            forbidden_imports=("apps", "app", "indexer_worker", "scripts"),
        ),
    )


def test_worker_is_an_independent_deployable_app() -> None:
    worker_root = APPS_ROOT / "worker"

    assert (worker_root / "indexer_worker" / "__main__.py").is_file()
    assert (worker_root / "indexer_worker" / "runtime.py").is_file()
    assert (worker_root / "indexer_worker" / "dispatcher.py").is_file()
    assert not (APPS_ROOT / "api" / "app" / "worker").exists()


def test_legacy_synchronous_ingestion_entrypoint_is_not_reintroduced() -> None:
    legacy_paths = (
        PACKAGES_ROOT / "indexer_application" / "commands" / "ingest_document.py",
        PACKAGES_ROOT / "indexer_application" / "services" / "document_ingestion.py",
        PACKAGES_ROOT / "indexer_application" / "services" / "ingestion" / "coordinator.py",
    )
    assert not [
        path.relative_to(REPOSITORY_ROOT)
        for path in legacy_paths
        if path.exists()
    ]

    dependency_source = (
        APPS_ROOT / "api" / "app" / "dependencies" / "application.py"
    ).read_text(encoding="utf-8")
    assert "get_ingest_document_handler" not in dependency_source
    assert "IngestDocumentHandler" not in dependency_source


def test_whole_document_deletion_entrypoint_is_not_reintroduced() -> None:
    legacy_paths = (
        PACKAGES_ROOT / "indexer_application" / "commands" / "enqueue_document_deletion.py",
        PACKAGES_ROOT / "indexer_application" / "services" / "background_jobs" / "deletion.py",
    )
    assert not [
        path.relative_to(REPOSITORY_ROOT)
        for path in legacy_paths
        if path.exists()
    ]

    route_source = (
        APPS_ROOT / "api" / "app" / "api" / "routes" / "documents.py"
    ).read_text(encoding="utf-8")
    assert '@router.delete(\n    "/{document_id}",' not in route_source
    assert '"/{document_id}/versions/{version_id}"' in route_source
