#!/usr/bin/env python3
"""Load and inspect an evaluation dataset."""

import argparse
from collections import Counter
from pathlib import Path

from xhs_eval.io import load_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--head", type=int, default=3)
    args = parser.parse_args()

    rows = load_examples(args.dataset)
    print(f"samples={len(rows)}")
    print(f"categories={dict(Counter(row.category for row in rows))}")
    print(f"difficulties={dict(Counter(row.difficulty for row in rows))}")
    for row in rows[: args.head]:
        print(f"- {row.id} | {row.category} | {row.brief}")


if __name__ == "__main__":
    main()
