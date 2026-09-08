from pathlib import Path

from xhs_eval.models import Prediction
from xhs_eval.product_catalog import load_product_catalog
from xhs_eval.skill_ab import (
    CandidateScore,
    PairwiseJudgement,
    balance_position_passes,
    build_product_examples,
    paired_bootstrap_ci,
    position_bias_audit,
    reverse_orders,
    summarize_ab,
)

CATALOG = Path("data/products/alibaba_products_20260908.jsonl")


def test_builds_one_ab_example_per_category() -> None:
    examples = build_product_examples(load_product_catalog(CATALOG))

    assert len(examples) == 20
    assert len({row.category for row in examples}) == 20
    assert all(row.checks.min_hashtags == 2 for row in examples)
    assert all(len(row.checks.required_terms) == 2 for row in examples)


def test_paired_bootstrap_is_deterministic() -> None:
    first = paired_bootstrap_ci([0.1, 0.2, 0.3], samples=1_000, seed=7)
    second = paired_bootstrap_ci([0.1, 0.2, 0.3], samples=1_000, seed=7)

    assert first == second
    assert first[0] <= 0.2 <= first[1]


def test_summary_uses_automatic_metric_summary_keys() -> None:
    example = build_product_examples(load_product_catalog(CATALOG))[0]
    prediction = Prediction(id=example.id, output=example.reference, model="test")
    scores = {
        "instruction_following": 4.0,
        "factual_grounding": 4.0,
        "style_authenticity": 4.0,
        "usefulness": 4.0,
        "safety": 4.0,
    }
    judgement = PairwiseJudgement(
        id=example.id,
        order={"X": "control", "Y": "treatment"},
        control=CandidateScore(scores=scores, overall=4.0),
        treatment=CandidateScore(scores=scores, overall=4.0),
        winner="tie",
        rationale="equal",
        judge_model="test",
    )

    summary = summarize_ab([example], [prediction], [prediction], [judgement])

    assert 0 <= summary["control"]["constraint_score"] <= 1
    assert 0 <= summary["treatment"]["constraint_score"] <= 1


def test_position_reversal_cancels_first_position_bias() -> None:
    dimensions = {
        "instruction_following": 4.0,
        "factual_grounding": 4.0,
        "style_authenticity": 4.0,
        "usefulness": 4.0,
        "safety": 4.0,
    }
    favored = {key: 5.0 for key in dimensions}
    first = PairwiseJudgement(
        id="one",
        order={"X": "control", "Y": "treatment"},
        control=CandidateScore(scores=favored, overall=5.0),
        treatment=CandidateScore(scores=dimensions, overall=4.0),
        winner="control",
        rationale="X wins",
        judge_model="test",
    )
    second = PairwiseJudgement(
        id="one",
        order={"X": "treatment", "Y": "control"},
        control=CandidateScore(scores=dimensions, overall=4.0),
        treatment=CandidateScore(scores=favored, overall=5.0),
        winner="treatment",
        rationale="X wins again",
        judge_model="test",
    )

    assert reverse_orders([first])["one"] == {"X": "treatment", "Y": "control"}
    balanced = balance_position_passes([first], [second])[0]
    assert balanced.control.overall == balanced.treatment.overall == 4.5
    assert balanced.winner == "tie"
    assert position_bias_audit([first], [second])["pass_1"]["x_wins"] == 1
