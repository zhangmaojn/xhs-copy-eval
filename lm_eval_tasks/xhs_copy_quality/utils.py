from __future__ import annotations

from typing import Any


def doc_to_text(doc: dict[str, Any]) -> str:
    facts = "\n".join(f"- {item}" for item in doc.get("facts", []))
    constraints = "\n".join(f"- {item}" for item in doc.get("constraints", []))
    return (
        f"任务：{doc['brief']}\n\n可用事实：\n{facts}\n\n必须满足的约束：\n{constraints}\n\n文案："
    )


def _normalize(text: str) -> str:
    return "".join(text.lower().split())


def _rouge_l(prediction: str, reference: str) -> float:
    pred = list(_normalize(prediction))
    ref = list(_normalize(reference))
    if not pred or not ref:
        return 0.0
    previous = [0] * (len(ref) + 1)
    for pred_char in pred:
        current = [0]
        for index, ref_char in enumerate(ref, start=1):
            current.append(
                previous[index - 1] + 1
                if pred_char == ref_char
                else max(previous[index], current[-1])
            )
        previous = current
    lcs = previous[-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _constraint_score(text: str, checks: dict[str, Any]) -> float:
    normalized = _normalize(text)
    results: list[bool] = []
    results.extend(_normalize(term) in normalized for term in checks.get("required_terms", []))
    results.extend(_normalize(term) not in normalized for term in checks.get("forbidden_terms", []))
    length = len(normalized)
    if checks.get("min_chars") is not None:
        results.append(length >= int(checks["min_chars"]))
    if checks.get("max_chars") is not None:
        results.append(length <= int(checks["max_chars"]))
    hashtag_count = text.count("#")
    if checks.get("min_hashtags"):
        results.append(hashtag_count >= int(checks["min_hashtags"]))
    if checks.get("max_hashtags") is not None:
        results.append(hashtag_count <= int(checks["max_hashtags"]))
    return sum(results) / len(results) if results else 1.0


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    prediction = results[0]
    return {
        "char_rouge_l": _rouge_l(prediction, doc["reference"]),
        "constraint_score": _constraint_score(prediction, doc.get("checks", {})),
    }
