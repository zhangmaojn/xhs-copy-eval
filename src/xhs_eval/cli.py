from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from xhs_eval.io import load_examples, load_predictions, write_json
from xhs_eval.metrics import compute_metrics
from xhs_eval.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xhs-eval",
        description="中文社交文案端到端评测工具",
    )
    parser.add_argument("--verbose", action="store_true", help="输出详细日志")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="校验数据集 schema 与 ID 唯一性")
    validate.add_argument("--dataset", required=True, type=Path)

    metrics = subparsers.add_parser("metrics", help="计算自动指标")
    metrics.add_argument("--dataset", required=True, type=Path)
    metrics.add_argument("--predictions", required=True, type=Path)
    metrics.add_argument("--output", required=True, type=Path)

    run = subparsers.add_parser("run", help="执行生成、指标、Judge、报告完整流水线")
    run.add_argument("--config", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.command == "validate":
        examples = load_examples(args.dataset)
        categories = sorted({row.category for row in examples})
        print(
            json.dumps(
                {"valid": True, "samples": len(examples), "categories": categories},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "metrics":
        examples = load_examples(args.dataset)
        predictions = load_predictions(args.predictions)
        rows, summary = compute_metrics(examples, predictions)
        write_json(
            args.output,
            {
                "summary": summary,
                "samples": [row.model_dump(mode="json") for row in rows],
            },
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "run":
        print(json.dumps(run_pipeline(args.config), ensure_ascii=False, indent=2))
        return

    parser.error(f"unsupported command: {args.command}")
