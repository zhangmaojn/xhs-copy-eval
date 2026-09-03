#!/usr/bin/env python3
"""Optional Hugging Face evaluate example (install the `data` extra first)."""

import argparse
from pathlib import Path

import evaluate

from xhs_eval.io import load_examples, load_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()

    examples = load_examples(args.dataset)
    predictions = {row.id: row.output for row in load_predictions(args.predictions)}
    rouge = evaluate.load("rouge")
    result = rouge.compute(
        predictions=[predictions[row.id] for row in examples],
        references=[row.reference for row in examples],
        use_stemmer=False,
    )
    print(result)


if __name__ == "__main__":
    main()
