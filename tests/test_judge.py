import json

import pytest

from xhs_eval.judge import load_rubric, parse_judge_response

DIMENSIONS = {
    "instruction_following",
    "factual_grounding",
    "style_authenticity",
    "usefulness",
    "safety",
}


def test_parse_json_inside_markdown_fence() -> None:
    payload = {
        "scores": {name: 4 for name in DIMENSIONS},
        "rationale": "good",
        "evidence": ["evidence"],
    }
    parsed = parse_judge_response(f"```json\n{json.dumps(payload)}\n```", DIMENSIONS)
    assert parsed["scores"]["safety"] == 4.0


def test_out_of_range_score_is_rejected() -> None:
    payload = {"scores": {name: 6 for name in DIMENSIONS}}
    with pytest.raises(ValueError, match="outside"):
        parse_judge_response(json.dumps(payload), DIMENSIONS)


def test_rubric_weights_must_sum_to_one(tmp_path) -> None:
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text(
        "dimensions:\n  quality:\n    weight: 0.7\n    description: quality\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_rubric(rubric)
