from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from packages.rag_core.evaluation.models import EvaluationReport
from packages.rag_core.evaluation.stability import StabilityEvaluationReport


def evaluation_report_to_dict(report: EvaluationReport) -> dict[str, Any]:
    """Convert a report into JSON-compatible primitives."""

    return asdict(report)


def write_evaluation_report(report: EvaluationReport, path: str | Path, *, pretty: bool = True) -> Path:
    """Write a report as UTF-8 JSON and return the resolved output path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evaluation_report_to_dict(report), indent=2 if pretty else None, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path.resolve()


def stability_evaluation_report_to_dict(
    report: StabilityEvaluationReport,
) -> dict[str, Any]:
    """Convert a stability report into JSON-compatible primitives."""

    return asdict(report)


def write_stability_evaluation_report(
    report: StabilityEvaluationReport,
    path: str | Path,
    *,
    pretty: bool = True,
) -> Path:
    """Write a separate stability report as UTF-8 JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            stability_evaluation_report_to_dict(report),
            indent=2 if pretty else None,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path.resolve()
