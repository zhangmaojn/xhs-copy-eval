import pytest

from xhs_eval.metrics import char_bleu, evaluate_constraints, rouge_l_f1
from xhs_eval.models import ConstraintChecks


def test_identical_text_has_perfect_similarity() -> None:
    assert rouge_l_f1("早八护肤", "早八护肤") == pytest.approx(1.0)
    assert char_bleu("早八护肤", "早八护肤") == pytest.approx(1.0)


def test_constraint_checks_report_individual_failures() -> None:
    checks = ConstraintChecks(
        required_terms=["早八"],
        forbidden_terms=["根治"],
        min_chars=4,
        max_chars=20,
        min_hashtags=1,
    )
    score, details = evaluate_constraints("早八护肤 #日常", checks)
    assert score == pytest.approx(1.0)
    assert all(details.values())


def test_forbidden_term_fails() -> None:
    checks = ConstraintChecks(forbidden_terms=["根治"])
    score, details = evaluate_constraints("三天根治", checks)
    assert score == 0.0
    assert details["forbidden:根治"] is False


def test_no_overlap_has_zero_rouge() -> None:
    assert rouge_l_f1("甲乙", "丙丁") == 0.0
