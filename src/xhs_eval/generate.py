from __future__ import annotations

import time
from collections.abc import Sequence

from xhs_eval.models import EvalExample, Prediction
from xhs_eval.providers import TextProvider

SYSTEM_PROMPT = """你是中文社交媒体文案作者。严格依据用户提供的事实和约束写作。
不要补充未经提供的产品功效、价格、地点或经历。只输出最终文案。"""


def build_generation_prompt(example: EvalExample) -> str:
    facts = "\n".join(f"- {item}" for item in example.facts) or "- 无额外事实"
    constraints = "\n".join(f"- {item}" for item in example.constraints) or "- 无额外约束"
    return f"任务：{example.brief}\n\n可用事实：\n{facts}\n\n必须满足的约束：\n{constraints}"


def generate_predictions(
    examples: Sequence[EvalExample], provider: TextProvider
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for example in examples:
        started = time.perf_counter()
        output = provider.generate(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_generation_prompt(example)},
            ],
            sample_id=example.id,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        predictions.append(
            Prediction(
                id=example.id,
                output=output.strip(),
                model=provider.name,
                latency_ms=latency_ms,
            )
        )
    return predictions
