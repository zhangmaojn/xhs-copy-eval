#!/usr/bin/env python3
"""Compute automatic metrics for an existing prediction file."""

import argparse
import json
from pathlib import Path

from xhs_eval.io import load_examples, load_predictions, write_json
from xhs_eval.metrics import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows, summary = compute_metrics(load_examples(args.dataset), load_predictions(args.predictions))
    write_json(
        args.output,
        {
            "summary": summary,
            "samples": [row.model_dump(mode="json") for row in rows],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
