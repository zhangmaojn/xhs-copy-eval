#!/usr/bin/env python3
"""Profile category x difficulty distribution with pandas."""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    frame = pd.read_json(args.dataset, lines=True)
    table = pd.crosstab(frame["category"], frame["difficulty"], margins=True)
    print(table.to_string())
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv)


if __name__ == "__main__":
    main()
