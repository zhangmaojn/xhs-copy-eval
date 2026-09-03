#!/usr/bin/env python3
"""Print the lowest-scoring samples, optionally filtered by attribution."""

import argparse
from pathlib import Path

from xhs_eval.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("judgements", type=Path)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--attribution")
    args = parser.parse_args()

    rows = read_jsonl(args.judgements)
    if args.attribution:
        rows = [row for row in rows if args.attribution in row.get("attributions", [])]
    rows.sort(key=lambda row: float(row["overall"]))
    for row in rows[: args.top]:
        labels = "、".join(row.get("attributions", [])) or "通过"
        print(f"{row['id']}\t{row['overall']:.2f}\t{labels}\t{row['rationale']}")


if __name__ == "__main__":
    main()
