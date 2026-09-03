from collections import Counter
from pathlib import Path

from xhs_eval.io import load_examples

ROOT = Path(__file__).parents[1]


def test_published_dataset_is_balanced() -> None:
    examples = load_examples(ROOT / "data/xhs_eval.jsonl")
    assert len(examples) == 18
    assert Counter(row.difficulty for row in examples) == {
        "easy": 6,
        "medium": 6,
        "hard": 6,
    }
    assert set(Counter(row.category for row in examples).values()) == {3}
