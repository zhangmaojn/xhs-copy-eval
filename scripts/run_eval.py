#!/usr/bin/env python3
"""Run the complete evaluation pipeline from a YAML config."""

import argparse
import json
from pathlib import Path

from xhs_eval.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    args = parser.parse_args()
    print(json.dumps(run_pipeline(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
