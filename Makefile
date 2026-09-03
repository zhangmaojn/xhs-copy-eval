.PHONY: setup test lint demo validate clean

setup:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check .

validate:
	uv run xhs-eval validate --dataset data/xhs_eval.jsonl

demo:
	uv run xhs-eval run --config configs/demo.yaml

clean:
	rm -f artifacts/*.json artifacts/*.jsonl

