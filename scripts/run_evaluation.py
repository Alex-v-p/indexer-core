from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPOSITORY_ROOT / "apps" / "api"
for import_root in (REPOSITORY_ROOT, API_ROOT):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from app.core.config import get_settings  # noqa: E402
from app.composition import build_query_graph  # noqa: E402
from packages.rag_core.evaluation import (  # noqa: E402
    EvaluationDatasetError,
    EvaluationRunner,
    StabilityEvaluationReport,
    StabilityEvaluationRunner,
    load_evaluation_dataset,
    write_evaluation_report,
    write_stability_evaluation_report,
)
from packages.rag_core.evaluation.models import EvaluationReport, MetricValue  # noqa: E402
from packages.rag_core.pipelines import PipelineRegistryError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a JSON evaluation dataset through the configured full query graph.",
    )
    parser.add_argument("dataset", type=Path, help="Path to a version 1.0 evaluation dataset JSON file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Report path. Defaults to reports/evaluations/<dataset>-<timestamp>.json.",
    )
    parser.add_argument("--top-k", type=int, help="Override dataset and case top_k values for every case.")
    parser.add_argument(
        "--pipeline",
        help="Registered pipeline name. Defaults to DEFAULT_QUERY_PIPELINE.",
    )
    parser.add_argument(
        "--repetitions",
        type=_positive_repetition_count,
        default=1,
        help=(
            "Runs per case. The default of 1 preserves the one-shot evaluation; "
            "values >=2 write a separate repeated-query stability report."
        ),
    )
    parser.add_argument("--compact", action="store_true", help="Write compact JSON instead of indented JSON.")
    return parser


async def run_from_args(
    args: argparse.Namespace,
) -> tuple[EvaluationReport | StabilityEvaluationReport, Path]:
    os.chdir(REPOSITORY_ROOT)
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    dataset_path = args.dataset if args.dataset.is_absolute() else REPOSITORY_ROOT / args.dataset
    dataset = load_evaluation_dataset(dataset_path)
    settings = get_settings()
    graph = build_query_graph(settings, pipeline_name=args.pipeline)
    repetitions = getattr(args, "repetitions", 1)
    if repetitions >= 2:
        report: EvaluationReport | StabilityEvaluationReport = await StabilityEvaluationRunner(
            graph=graph,
            requested_pipeline_name=args.pipeline,
        ).run(
            dataset,
            repetitions=repetitions,
            top_k_override=args.top_k,
        )
    else:
        report = await EvaluationRunner(
            graph=graph,
            requested_pipeline_name=args.pipeline,
        ).run(dataset, top_k_override=args.top_k)

    output_path = args.output
    if output_path is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", dataset.name).strip("-") or "evaluation"
        filename = (
            f"{safe_name}-stability-v1-{timestamp}.json"
            if isinstance(report, StabilityEvaluationReport)
            else f"{safe_name}-{timestamp}.json"
        )
        output_path = REPOSITORY_ROOT / "reports" / "evaluations" / filename
    elif not output_path.is_absolute():
        output_path = REPOSITORY_ROOT / output_path

    if isinstance(report, StabilityEvaluationReport):
        written_path = write_stability_evaluation_report(
            report,
            output_path,
            pretty=not args.compact,
        )
    else:
        written_path = write_evaluation_report(report, output_path, pretty=not args.compact)
    return report, written_path


def main() -> int:
    args = build_parser().parse_args()
    try:
        report, output_path = asyncio.run(run_from_args(args))
    except (EvaluationDatasetError, PipelineRegistryError, ValueError) as exc:
        print(f"Evaluation configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Evaluation could not run: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset: {report.dataset_name} {report.dataset_version}")
    print(f"Pipeline: {report.pipeline_name} {report.pipeline_version}")
    if isinstance(report, StabilityEvaluationReport):
        print(f"Repetitions per case: {report.repetitions}")
        print(
            f"Attempts: {report.succeeded_attempts}/{report.total_attempts} succeeded, "
            f"{report.failed_attempts} failed",
        )
        print(f"Technical success rate: {_metric_text(report.metrics.technical_success_rate)}")
        print(f"Outcome consistency: {_metric_text(report.metrics.outcome_consistency)}")
        print(
            "Route signature consistency: "
            f"{_metric_text(report.metrics.route_signature_consistency)}",
        )
        print(
            "Evidence exact-set agreement: "
            f"{_metric_text(report.metrics.evidence_exact_set_agreement)}",
        )
        print(
            "Evidence mean pairwise Jaccard: "
            f"{_metric_text(report.metrics.evidence_mean_pairwise_jaccard)}",
        )
        print(
            "Normalized answer exact-match rate: "
            f"{_metric_text(report.metrics.normalized_answer_exact_match_rate)}",
        )
        print(
            "Normalized answer token Jaccard: "
            f"{_metric_text(report.metrics.normalized_answer_mean_pairwise_token_jaccard)}",
        )
        print(
            "Presentation signature consistency: "
            f"{_metric_text(report.metrics.presentation_signature_consistency)}",
        )
        exit_code = 1 if report.failed_attempts else 0
    else:
        print(
            f"Cases: {report.succeeded_cases}/{report.total_cases} succeeded, "
            f"{report.failed_cases} failed",
        )
        print(f"Recall@k: {_metric_text(report.metrics.recall_at_k)}")
        print(f"MRR: {_metric_text(report.metrics.mrr)}")
        print(f"Citation hit rate: {_metric_text(report.metrics.citation_hit_rate)}")
        print(f"Answer faithfulness: {_metric_text(report.metrics.answer_faithfulness)}")
        exit_code = 1 if report.failed_cases else 0
    print(f"Report: {output_path}")
    return exit_code


def _metric_text(metric: MetricValue) -> str:
    if metric.value is None:
        return metric.status
    return f"{metric.value:.4f}"


def _positive_repetition_count(value: str) -> int:
    repetitions = int(value)
    if repetitions < 1:
        raise argparse.ArgumentTypeError("--repetitions must be 1 or greater.")
    return repetitions


if __name__ == "__main__":
    raise SystemExit(main())
