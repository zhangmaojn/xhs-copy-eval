#!/usr/bin/env python3
"""Create deterministic, category-stratified JSONL splits."""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from xhs_eval.io import load_examples, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--validation", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.train < 1 or not 0 <= args.validation < 1:
        parser.error("ratios must be within [0, 1]")
    if args.train + args.validation >= 1:
        parser.error("train + validation must be less than 1")

    grouped: dict[str, list] = defaultdict(list)
    for row in load_examples(args.dataset):
        grouped[row.category].append(row)

    randomizer = random.Random(args.seed)
    splits = {"train": [], "validation": [], "test": []}
    for rows in grouped.values():
        randomizer.shuffle(rows)
        train_end = int(len(rows) * args.train)
        validation_end = train_end + int(len(rows) * args.validation)
        for split, subset in (
            ("train", rows[:train_end]),
            ("validation", rows[train_end:validation_end]),
            ("test", rows[validation_end:]),
        ):
            splits[split].extend(row.model_copy(update={"split": split}) for row in subset)

    for split, rows in splits.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", sorted(rows, key=lambda row: row.id))
        print(f"{split}: {len(rows)}")


if __name__ == "__main__":
    main()
