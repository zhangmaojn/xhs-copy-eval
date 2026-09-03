from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from xhs_eval.models import AutomaticMetric, EvalExample, JudgeRecord, Prediction


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(
    examples: Sequence[EvalExample],
    predictions: Sequence[Prediction],
    automatic_metrics: Sequence[AutomaticMetric],
    judgements: Sequence[JudgeRecord],
    *,
    run_name: str,
    dataset_path: str,
    rubric_path: str,
) -> str:
    example_by_id = {row.id: row for row in examples}
    metric_by_id = {row.id: row for row in automatic_metrics}
    prediction_by_id = {row.id: row for row in predictions}
    pass_count = sum(row.passed for row in judgements)
    dimensions = list(judgements[0].scores) if judgements else []

    category_scores: dict[str, list[float]] = defaultdict(list)
    difficulty_scores: dict[str, list[float]] = defaultdict(list)
    attribution_counts: Counter[str] = Counter()
    for judgement in judgements:
        example = example_by_id[judgement.id]
        category_scores[example.category].append(judgement.overall)
        difficulty_scores[example.difficulty].append(judgement.overall)
        attribution_counts.update(judgement.attributions)

    lines = [
        f"# {run_name} 评测报告",
        "",
        f"> 生成时间：{datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"> 数据集：`{dataset_path}` · Rubric：`{rubric_path}`",
        "",
        "## 1. 执行摘要",
        "",
        f"- 样本数：**{len(examples)}**",
        f"- LLM-as-Judge 通过率：**{pass_count / len(judgements):.1%}**"
        if judgements
        else "- LLM-as-Judge 通过率：N/A",
        f"- 平均综合分：**{_mean([row.overall for row in judgements]):.2f} / 5**",
        f"- 平均约束满足率：**{_mean([row.constraint_score for row in automatic_metrics]):.1%}**",
        f"- 平均 ROUGE-L：**{_mean([row.rouge_l_f1 for row in automatic_metrics]):.3f}**",
        f"- 平均 Char-BLEU：**{_mean([row.char_bleu for row in automatic_metrics]):.3f}**",
        "",
        "自动指标用于稳定地发现格式/约束问题；开放式文案的主结论以 Rubric 打分和人工抽检为准。",
        "",
        "## 2. Rubric 维度",
        "",
        "| 维度 | 平均分 |",
        "|---|---:|",
    ]
    for dimension in dimensions:
        lines.append(
            f"| `{dimension}` | {_mean([row.scores[dimension] for row in judgements]):.2f} |"
        )

    lines.extend(["", "## 3. 分组表现", "", "### 按内容类别", ""])
    lines.extend(["| 类别 | 样本数 | 平均分 |", "|---|---:|---:|"])
    for category, scores in sorted(category_scores.items()):
        lines.append(f"| {_cell(category)} | {len(scores)} | {_mean(scores):.2f} |")

    lines.extend(["", "### 按难度", "", "| 难度 | 样本数 | 平均分 |", "|---|---:|---:|"])
    for difficulty in ("easy", "medium", "hard"):
        scores = difficulty_scores.get(difficulty, [])
        if scores:
            lines.append(f"| {difficulty} | {len(scores)} | {_mean(scores):.2f} |")

    lines.extend(
        [
            "",
            "## 4. Badcase 归因",
            "",
            "| 原因 | 数量 |",
            "|---|---:|",
        ]
    )
    if attribution_counts:
        for label, count in attribution_counts.most_common():
            lines.append(f"| {_cell(label)} | {count} |")
    else:
        lines.append("| 无未通过样本 | 0 |")

    lines.extend(
        [
            "",
            "## 5. Top 失败样本",
            "",
            "| ID | 类别 | 综合分 | 约束满足率 | 归因 | 说明 |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    failures = sorted((row for row in judgements if not row.passed), key=lambda row: row.overall)
    for row in failures[:10]:
        example = example_by_id[row.id]
        metric = metric_by_id[row.id]
        lines.append(
            f"| `{row.id}` | {_cell(example.category)} | {row.overall:.2f} | "
            f"{metric.constraint_score:.0%} | {_cell('、'.join(row.attributions))} | "
            f"{_cell(row.rationale)} |"
        )
    if not failures:
        lines.append("| - | - | - | - | - | 本次无失败样本 |")

    lines.extend(["", "## 6. 样本级审计表", ""])
    lines.extend(
        [
            "| ID | 难度 | Judge | ROUGE-L | Char-BLEU | 约束 | 模型 |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for judgement in sorted(judgements, key=lambda row: row.id):
        example = example_by_id[judgement.id]
        metric = metric_by_id[judgement.id]
        prediction = prediction_by_id[judgement.id]
        lines.append(
            f"| `{judgement.id}` | {example.difficulty} | {judgement.overall:.2f} | "
            f"{metric.rouge_l_f1:.3f} | {metric.char_bleu:.3f} | "
            f"{metric.constraint_score:.0%} | {_cell(prediction.model)} |"
        )

    lines.extend(
        [
            "",
            "## 7. 结论与下一轮动作",
            "",
            "1. 优先修复数量最多、业务风险最高的失败归因，再复跑同一固定测试集。",
            "2. 对边界分（3.2–3.8）和 Judge 分歧样本做双人复核，避免自动评测偏差。",
            "3. 新发现的真实线上失败只进入候选池；去重、复核并版本化后再并入测试集。",
            "4. 记录模型、提示词、温度、数据集版本和 Judge 版本，保证纵向对比有效。",
            "",
            "## 8. 局限",
            "",
            "- 当前数据是用于工程演示的合成小样本，不代表真实平台流量分布。",
            "- LLM-as-Judge 可能存在位置、长度、风格偏好和自我偏好；正式结论需校准人工一致性。",
            "- ROUGE/BLEU 只衡量表面相似度，不应单独用于开放式文案质量判断。",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
