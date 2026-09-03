from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from xhs_eval.generate import generate_predictions
from xhs_eval.io import load_examples, write_json, write_jsonl
from xhs_eval.judge import evaluate_with_judge, load_rubric
from xhs_eval.metrics import compute_metrics
from xhs_eval.providers import build_provider, resolve_path
from xhs_eval.report import render_report, write_report

LOGGER = logging.getLogger(__name__)


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    base_dir = config_path.parent.parent
    return config, base_dir


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    config, base_dir = load_config(config_path)
    dataset_path = resolve_path(config["dataset_path"], base_dir)
    rubric_path = resolve_path(config["rubric_path"], base_dir)
    output_config = config["outputs"]
    prediction_path = resolve_path(output_config["predictions"], base_dir)
    metric_path = resolve_path(output_config["metrics"], base_dir)
    judgement_path = resolve_path(output_config["judgements"], base_dir)
    report_path = resolve_path(output_config["report"], base_dir)

    LOGGER.info("Loading dataset from %s", dataset_path)
    examples = load_examples(dataset_path)

    LOGGER.info("Generating %d predictions", len(examples))
    generator = build_provider(config["generator"], base_dir=base_dir)
    predictions = generate_predictions(examples, generator)
    write_jsonl(prediction_path, predictions)

    LOGGER.info("Computing automatic metrics")
    metric_rows, metric_summary = compute_metrics(examples, predictions)
    write_json(
        metric_path,
        {
            "summary": metric_summary,
            "samples": [row.model_dump(mode="json") for row in metric_rows],
        },
    )

    LOGGER.info("Running LLM-as-Judge")
    rubric = load_rubric(rubric_path)
    judge_provider = build_provider(config["judge"], base_dir=base_dir)
    judgements = evaluate_with_judge(examples, predictions, rubric, judge_provider)
    write_jsonl(judgement_path, judgements)

    LOGGER.info("Rendering report")
    report = render_report(
        examples,
        predictions,
        metric_rows,
        judgements,
        run_name=config.get("run_name", "XHS Copy Eval"),
        dataset_path=str(config["dataset_path"]),
        rubric_path=str(config["rubric_path"]),
    )
    write_report(report_path, report)

    result = {
        "sample_count": len(examples),
        "predictions": str(prediction_path),
        "metrics": str(metric_path),
        "judgements": str(judgement_path),
        "report": str(report_path),
        "judge_pass_rate": round(sum(row.passed for row in judgements) / len(judgements), 4),
    }
    LOGGER.info("Completed: %s", json.dumps(result, ensure_ascii=False))
    return result
