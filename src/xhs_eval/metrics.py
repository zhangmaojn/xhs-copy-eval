from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from xhs_eval.models import AutomaticMetric, ConstraintChecks, EvalExample, Prediction

HASHTAG_PATTERN = re.compile(r"#[^#\s]+")
EMOJI_PATTERN = re.compile(
    "[\U0001f1e0-\U0001f1ff\U0001f300-\U0001faff\U00002600-\U000027bf]",
    flags=re.UNICODE,
)
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub("", text).lower()


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = list(normalize_text(prediction))
    ref = list(normalize_text(reference))
    if not pred or not ref:
        return 0.0
    previous = [0] * (len(ref) + 1)
    for pred_char in pred:
        current = [0]
        for index, ref_char in enumerate(ref, start=1):
            if pred_char == ref_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def char_bleu(prediction: str, reference: str, max_order: int = 4) -> float:
    pred = list(normalize_text(prediction))
    ref = list(normalize_text(reference))
    if not pred or not ref:
        return 0.0
    precisions: list[float] = []
    effective_order = min(max_order, len(pred), len(ref))
    for order in range(1, effective_order + 1):
        pred_counts = Counter(tuple(pred[i : i + order]) for i in range(len(pred) - order + 1))
        ref_counts = Counter(tuple(ref[i : i + order]) for i in range(len(ref) - order + 1))
        matches = sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
        total = sum(pred_counts.values())
        precisions.append((matches + 1) / (total + 1))
    log_precision = sum(math.log(value) for value in precisions) / effective_order
    brevity_penalty = 1.0 if len(pred) > len(ref) else math.exp(1 - len(ref) / len(pred))
    return brevity_penalty * math.exp(log_precision)


def evaluate_constraints(text: str, checks: ConstraintChecks) -> tuple[float, dict[str, bool]]:
    normalized = normalize_text(text)
    details: dict[str, bool] = {}
    for term in checks.required_terms:
        details[f"required:{term}"] = normalize_text(term) in normalized
    for term in checks.forbidden_terms:
        details[f"forbidden:{term}"] = normalize_text(term) not in normalized
    char_count = len(normalized)
    if checks.min_chars is not None:
        details[f"min_chars:{checks.min_chars}"] = char_count >= checks.min_chars
    if checks.max_chars is not None:
        details[f"max_chars:{checks.max_chars}"] = char_count <= checks.max_chars
    if checks.require_emoji:
        details["require_emoji"] = bool(EMOJI_PATTERN.search(text))
    hashtag_count = len(HASHTAG_PATTERN.findall(text))
    if checks.min_hashtags:
        details[f"min_hashtags:{checks.min_hashtags}"] = hashtag_count >= checks.min_hashtags
    if checks.max_hashtags is not None:
        details[f"max_hashtags:{checks.max_hashtags}"] = hashtag_count <= checks.max_hashtags
    if not details:
        return 1.0, details
    return sum(details.values()) / len(details), details


def compute_metrics(
    examples: Sequence[EvalExample], predictions: Sequence[Prediction]
) -> tuple[list[AutomaticMetric], dict[str, Any]]:
    example_by_id = {example.id: example for example in examples}
    prediction_by_id = {prediction.id: prediction for prediction in predictions}
    missing = sorted(set(example_by_id) - set(prediction_by_id))
    extra = sorted(set(prediction_by_id) - set(example_by_id))
    if missing or extra:
        raise ValueError(f"prediction ids do not match dataset; missing={missing}, extra={extra}")

    rows: list[AutomaticMetric] = []
    for example in examples:
        prediction = prediction_by_id[example.id]
        constraint_score, details = evaluate_constraints(prediction.output, example.checks)
        rows.append(
            AutomaticMetric(
                id=example.id,
                char_count=len(normalize_text(prediction.output)),
                rouge_l_f1=round(rouge_l_f1(prediction.output, example.reference), 4),
                char_bleu=round(char_bleu(prediction.output, example.reference), 4),
                constraint_score=round(constraint_score, 4),
                constraint_details=details,
            )
        )

    def mean(field: str) -> float:
        values = [float(getattr(row, field)) for row in rows]
        return round(sum(values) / len(values), 4) if values else 0.0

    summary = {
        "sample_count": len(rows),
        "mean_rouge_l_f1": mean("rouge_l_f1"),
        "mean_char_bleu": mean("char_bleu"),
        "mean_constraint_score": mean("constraint_score"),
    }
    return rows, summary
