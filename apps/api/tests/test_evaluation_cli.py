from __future__ import annotations

import argparse
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_evaluation


def test_evaluation_cli_defaults_to_one_shot_mode() -> None:
    args = run_evaluation.build_parser().parse_args(
        ["datasets/eval_sets/baseline_demo.json"],
    )

    assert args.repetitions == 1


@pytest.mark.parametrize("value", ("0", "-1"))
def test_evaluation_cli_rejects_non_positive_repetitions(value: str) -> None:
    with pytest.raises(SystemExit):
        run_evaluation.build_parser().parse_args(
            ["datasets/eval_sets/baseline_demo.json", "--repetitions", value],
        )


def test_evaluation_cli_accepts_stability_repetitions() -> None:
    args = run_evaluation.build_parser().parse_args(
        ["datasets/eval_sets/baseline_demo.json", "--repetitions", "3"],
    )

    assert args.repetitions == 3
    assert isinstance(args, argparse.Namespace)


@pytest.mark.parametrize("repetitions", (None, 1))
async def test_run_from_args_preserves_one_shot_runner_writer_and_filename(
    repetitions: int | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    dataset = SimpleNamespace(name="legacy dataset")
    graph = object()
    legacy_report = object()

    class FakeEvaluationRunner:
        def __init__(self, *, graph: object, requested_pipeline_name: str | None) -> None:
            calls["runner_init"] = (graph, requested_pipeline_name)

        async def run(self, supplied_dataset: object, *, top_k_override: int | None) -> object:
            calls["runner_run"] = (supplied_dataset, top_k_override)
            return legacy_report

    class UnexpectedStabilityRunner:
        def __init__(self, **_: object) -> None:
            raise AssertionError("Stability runner must not be constructed in one-shot mode.")

    def write_legacy(
        report: object,
        path: Path,
        *,
        pretty: bool,
    ) -> Path:
        calls["legacy_writer"] = (report, path, pretty)
        return path

    def write_stability(*_: object, **__: object) -> Path:
        raise AssertionError("Stability writer must not run in one-shot mode.")

    _patch_cli_dependencies(
        monkeypatch,
        tmp_path=tmp_path,
        dataset=dataset,
        graph=graph,
    )
    monkeypatch.setattr(run_evaluation, "EvaluationRunner", FakeEvaluationRunner)
    monkeypatch.setattr(
        run_evaluation,
        "StabilityEvaluationRunner",
        UnexpectedStabilityRunner,
    )
    monkeypatch.setattr(run_evaluation, "write_evaluation_report", write_legacy)
    monkeypatch.setattr(
        run_evaluation,
        "write_stability_evaluation_report",
        write_stability,
    )
    arguments: dict[str, object] = {
        "dataset": Path("dataset.json"),
        "output": None,
        "top_k": 9,
        "pipeline": "baseline_rag",
        "compact": False,
    }
    if repetitions is not None:
        arguments["repetitions"] = repetitions

    report, output_path = await run_evaluation.run_from_args(
        argparse.Namespace(**arguments),
    )

    assert report is legacy_report
    assert calls["runner_init"] == (graph, "baseline_rag")
    assert calls["runner_run"] == (dataset, 9)
    assert calls["legacy_writer"] == (legacy_report, output_path, True)
    assert output_path.parent == tmp_path / "reports" / "evaluations"
    assert re.fullmatch(
        r"legacy-dataset-\d{8}T\d{6}Z\.json",
        output_path.name,
    )
    assert "-stability-v1-" not in output_path.name


async def test_run_from_args_uses_stability_runner_writer_and_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    dataset = SimpleNamespace(name="stability dataset")
    graph = object()

    class FakeStabilityReport:
        pass

    stability_report = FakeStabilityReport()

    class UnexpectedEvaluationRunner:
        def __init__(self, **_: object) -> None:
            raise AssertionError("One-shot runner must not be constructed in stability mode.")

    class FakeStabilityRunner:
        def __init__(self, *, graph: object, requested_pipeline_name: str | None) -> None:
            calls["runner_init"] = (graph, requested_pipeline_name)

        async def run(
            self,
            supplied_dataset: object,
            *,
            repetitions: int,
            top_k_override: int | None,
        ) -> object:
            calls["runner_run"] = (supplied_dataset, repetitions, top_k_override)
            return stability_report

    def write_legacy(*_: object, **__: object) -> Path:
        raise AssertionError("One-shot writer must not run in stability mode.")

    def write_stability(
        report: object,
        path: Path,
        *,
        pretty: bool,
    ) -> Path:
        calls["stability_writer"] = (report, path, pretty)
        return path

    _patch_cli_dependencies(
        monkeypatch,
        tmp_path=tmp_path,
        dataset=dataset,
        graph=graph,
    )
    monkeypatch.setattr(run_evaluation, "EvaluationRunner", UnexpectedEvaluationRunner)
    monkeypatch.setattr(run_evaluation, "StabilityEvaluationReport", FakeStabilityReport)
    monkeypatch.setattr(run_evaluation, "StabilityEvaluationRunner", FakeStabilityRunner)
    monkeypatch.setattr(run_evaluation, "write_evaluation_report", write_legacy)
    monkeypatch.setattr(
        run_evaluation,
        "write_stability_evaluation_report",
        write_stability,
    )
    arguments = argparse.Namespace(
        dataset=Path("dataset.json"),
        output=None,
        top_k=7,
        pipeline="agentic_rag",
        repetitions=3,
        compact=True,
    )

    report, output_path = await run_evaluation.run_from_args(arguments)

    assert report is stability_report
    assert calls["runner_init"] == (graph, "agentic_rag")
    assert calls["runner_run"] == (dataset, 3, 7)
    assert calls["stability_writer"] == (stability_report, output_path, False)
    assert output_path.parent == tmp_path / "reports" / "evaluations"
    assert re.fullmatch(
        r"stability-dataset-stability-v1-\d{8}T\d{6}Z\.json",
        output_path.name,
    )


def _patch_cli_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Path,
    dataset: object,
    graph: object,
) -> None:
    settings = object()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_evaluation, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        run_evaluation,
        "load_evaluation_dataset",
        lambda path: dataset,
    )
    monkeypatch.setattr(run_evaluation, "get_settings", lambda: settings)

    def build_graph(supplied_settings: object, *, pipeline_name: str | None) -> object:
        assert supplied_settings is settings
        assert pipeline_name in {"baseline_rag", "agentic_rag"}
        return graph

    monkeypatch.setattr(run_evaluation, "build_query_graph", build_graph)
