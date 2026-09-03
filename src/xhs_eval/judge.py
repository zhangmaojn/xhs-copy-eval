from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from xhs_eval.attribution import attribute_badcase
from xhs_eval.models import EvalExample, JudgeRecord, Prediction
from xhs_eval.providers import TextProvider

JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def load_rubric(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("dimensions"), dict):
        raise ValueError("rubric must contain a dimensions mapping")
    weights: list[float] = []
    for name, item in value["dimensions"].items():
        if not isinstance(item, dict) or "weight" not in item or "description" not in item:
            raise ValueError(f"rubric dimension {name!r} requires weight and description")
        weight = float(item["weight"])
        if weight <= 0:
            raise ValueError(f"rubric dimension {name!r} must have a positive weight")
        weights.append(weight)
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"rubric weights must sum to 1.0, got {sum(weights):.6f}")
    return value


def build_judge_prompt(example: EvalExample, prediction: Prediction, rubric: dict[str, Any]) -> str:
    dimension_lines = []
    for key, item in rubric["dimensions"].items():
        dimension_lines.append(f"- {key}（权重 {item['weight']}）：{item['description']}")
    facts = "\n".join(f"- {item}" for item in example.facts) or "- 无"
    constraints = "\n".join(f"- {item}" for item in example.constraints) or "- 无"
    score_example = json.dumps({key: 4.0 for key in rubric["dimensions"]}, ensure_ascii=False)
    return f"""请作为独立评测员，根据量表评价候选文案。不要因为候选与参考答案措辞不同而扣分。

任务：{example.brief}
已知事实：
{facts}
约束：
{constraints}

候选文案：
{prediction.output}

参考文案（仅帮助理解任务，不要求字面一致）：
{example.reference}

评分维度（每项 1-5 分，可使用一位小数）：
{chr(10).join(dimension_lines)}

只输出一个 JSON 对象，格式如下：
{{
  "scores": {score_example},
  "rationale": "简洁说明主要优缺点",
  "evidence": ["引用候选中的短语或指出缺失的约束"]
}}
"""


def parse_judge_response(raw: str, dimensions: set[str]) -> dict[str, Any]:
    match = JSON_BLOCK.search(raw)
    if not match:
        raise ValueError("judge response does not contain a JSON object")
    payload = json.loads(match.group(0))
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("judge response is missing scores")
    if set(scores) != dimensions:
        missing = sorted(dimensions - set(scores))
        extra = sorted(set(scores) - dimensions)
        raise ValueError(f"score dimensions mismatch; missing={missing}, extra={extra}")
    normalized_scores: dict[str, float] = {}
    for name, value in scores.items():
        score = float(value)
        if not 1.0 <= score <= 5.0:
            raise ValueError(f"score for {name} is outside [1, 5]")
        normalized_scores[name] = score
    return {
        "scores": normalized_scores,
        "rationale": str(payload.get("rationale", "")).strip(),
        "evidence": [str(item) for item in payload.get("evidence", [])],
    }


def evaluate_with_judge(
    examples: Sequence[EvalExample],
    predictions: Sequence[Prediction],
    rubric: dict[str, Any],
    provider: TextProvider,
) -> list[JudgeRecord]:
    prediction_by_id = {prediction.id: prediction for prediction in predictions}
    dimensions = set(rubric["dimensions"])
    threshold = float(rubric.get("pass_threshold", 3.5))
    records: list[JudgeRecord] = []

    for example in examples:
        prediction = prediction_by_id[example.id]
        raw = provider.generate(
            [
                {
                    "role": "system",
                    "content": "你是严格、一致的中文生成质量评测员，只返回有效 JSON。",
                },
                {"role": "user", "content": build_judge_prompt(example, prediction, rubric)},
            ],
            sample_id=example.id,
        )
        try:
            parsed = parse_judge_response(raw, dimensions)
            scores = parsed["scores"]
            overall = sum(
                scores[key] * float(rubric["dimensions"][key]["weight"]) for key in dimensions
            )
            passed = overall >= threshold
            parse_error = None
            rationale = parsed["rationale"]
            evidence = parsed["evidence"]
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            scores = {key: 1.0 for key in dimensions}
            overall = 1.0
            passed = False
            parse_error = str(exc)
            rationale = "评测器输出无法解析，需人工复核。"
            evidence = [raw[:240]]
        records.append(
            JudgeRecord(
                id=example.id,
                scores=scores,
                overall=round(overall, 4),
                passed=passed,
                rationale=rationale,
                evidence=evidence,
                attributions=attribute_badcase(scores, passed=passed, parse_error=parse_error),
                judge_model=provider.name,
                parse_error=parse_error,
            )
        )
    return records
