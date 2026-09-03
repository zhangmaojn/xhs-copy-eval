import json

import pytest

from xhs_eval.io import DataValidationError, load_examples


def test_duplicate_ids_are_rejected(tmp_path) -> None:
    row = {
        "id": "same",
        "category": "test",
        "difficulty": "easy",
        "brief": "write",
        "reference": "answer",
    }
    source = tmp_path / "data.jsonl"
    source.write_text(
        json.dumps(row, ensure_ascii=False) + "\n" + json.dumps(row, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="duplicate ids"):
        load_examples(source)
