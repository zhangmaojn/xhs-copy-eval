from __future__ import annotations

DIMENSION_ATTRIBUTIONS = {
    "instruction_following": "指令遵循失败",
    "factual_grounding": "事实性/幻觉风险",
    "style_authenticity": "平台风格错配",
    "usefulness": "信息价值不足",
    "safety": "安全与合规风险",
}


def attribute_badcase(
    scores: dict[str, float], *, passed: bool, parse_error: str | None = None
) -> list[str]:
    if parse_error:
        return ["评测器解析失败"]
    if passed:
        return []
    labels = [
        label
        for dimension, label in DIMENSION_ATTRIBUTIONS.items()
        if float(scores.get(dimension, 5.0)) < 3.0
    ]
    return labels or ["综合质量未达阈值"]
