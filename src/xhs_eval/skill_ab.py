from __future__ import annotations

import json
import random
import re
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from xhs_eval.generate import build_generation_prompt
from xhs_eval.io import write_json, write_jsonl
from xhs_eval.metrics import compute_metrics
from xhs_eval.models import EvalExample, Prediction
from xhs_eval.product_catalog import ProductRecord
from xhs_eval.providers import TextProvider

JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

CONTROL_SYSTEM_PROMPT = """你是中文社交媒体文案作者。严格依据用户提供的事实和约束写作。
不要补充未经提供的产品功效、价格、地点或经历。只输出最终文案。"""


class CandidateScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: dict[str, float]
    overall: float


class PairwiseJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    order: dict[str, str]
    control: CandidateScore
    treatment: CandidateScore
    winner: str
    rationale: str
    evidence: list[str] = Field(default_factory=list)
    judge_model: str
    parse_error: str | None = None


def build_product_examples(
    records: Sequence[ProductRecord], *, one_per_category: bool = True
) -> list[EvalExample]:
    selected: list[ProductRecord] = []
    seen_categories: set[str] = set()
    for record in records:
        if one_per_category and record.category in seen_categories:
            continue
        selected.append(record)
        seen_categories.add(record.category)

    examples: list[EvalExample] = []
    for record in selected:
        claims = "；".join(record.selling_points) if record.selling_points else "无额外卡片卖点"
        facts = [
            f"商品ID：{record.product_id}",
            f"品类：{record.category}",
            f"卖家商品标题（未独立核验）：{record.product_name}",
            f"采集时页面批发价：{record.price_text}",
            f"采集时页面信息：{record.min_order_text}",
            f"供应商页面名称：{record.supplier_name}",
            "采集日期：2026-09-08",
            f"卡片卖点（卖家来源主张，未独立核验）：{claims}",
        ]
        constraints = [
            "只输出最终小红书文案，不解释创作过程",
            "第一行写20字以内的标题",
            "面向正在做商品选品的小店主，正文简洁、客观、可扫读",
            f"原样保留价格文本“{record.price_text}”",
            f"原样保留起订量文本“{record.min_order_text}”",
            "明确价格和起订量是采集时页面显示的批发信息",
            "不虚构个人体验、材质、功效、认证、库存或零售价",
            "结尾使用2-4个相关话题标签",
        ]
        reference = (
            f"{record.category}选品，先核对这几项\n\n"
            f"这款商品在 2026-09-08 的 Alibaba.com 页面显示：批发价"
            f"{record.price_text}，{record.min_order_text}。供应商页面名称为"
            f"{record.supplier_name}。标题和卡片卖点属于卖家来源信息，正式选品前建议回到"
            "商品页复核材质、规格、当前价格与库存。\n\n"
            f"#{record.category} #小店选品 #商品资料核验"
        )
        examples.append(
            EvalExample(
                id=f"product-{record.product_id}",
                category=record.category,
                difficulty="medium",
                brief="写一篇小红书商品选品介绍，帮助小店主快速理解页面信息。",
                facts=facts,
                constraints=constraints,
                checks={
                    "required_terms": [record.price_text, record.min_order_text],
                    "forbidden_terms": [
                        "保证有效",
                        "全网最低",
                        "人人适合",
                        "亲测有效",
                        "闭眼入不踩雷",
                    ],
                    "min_chars": 120,
                    "max_chars": 360,
                    "min_hashtags": 2,
                    "max_hashtags": 4,
                },
                reference=reference,
                split="test",
                tags=["real-product", "catalog-grounded", record.category],
            )
        )
    return examples


def load_treatment_system_prompt(skill_dir: str | Path) -> str:
    root = Path(skill_dir)
    sources = [
        root / "SKILL.md",
        root / "references" / "xiaohongshu.md",
        root / "references" / "product-catalog.md",
    ]
    sections = []
    for source in sources:
        sections.append(f"## {source.name}\n\n{source.read_text(encoding='utf-8')}")
    return (
        "你是使用指定文案 Skill 的中文社交媒体文案作者。"
        "以下内容是本次必须遵守的 Skill 指令。只输出最终文案。\n\n" + "\n\n".join(sections)
    )


def generate_arm(
    examples: Sequence[EvalExample],
    provider: TextProvider,
    *,
    system_prompt: str,
    arm: str,
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for index, example in enumerate(examples, start=1):
        started = time.perf_counter()
        output = provider.generate(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_generation_prompt(example)},
            ],
            sample_id=f"{example.id}-{arm}",
        )
        predictions.append(
            Prediction(
                id=example.id,
                output=output.strip(),
                model=provider.name,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                metadata={"arm": arm},
            )
        )
        print(f"generated arm={arm} sample={index}/{len(examples)}", flush=True)
    return predictions


def _judge_prompt(
    example: EvalExample,
    candidate_x: str,
    candidate_y: str,
    rubric: dict[str, Any],
) -> str:
    facts = "\n".join(f"- {item}" for item in example.facts)
    constraints = "\n".join(f"- {item}" for item in example.constraints)
    dimensions = "\n".join(
        f"- {key}: {item['description']}（权重 {item['weight']}）"
        for key, item in rubric["dimensions"].items()
    )
    score_template = {key: 4.0 for key in rubric["dimensions"]}
    return f"""你是独立中文文案评测员。候选 X/Y 的来源已匿名且顺序随机。
严格依据同一任务、事实和约束分别评分，不因长度更长或措辞接近参考文案而偏爱某方。

任务：{example.brief}
事实：
{facts}
约束：
{constraints}

候选 X：
{candidate_x}

候选 Y：
{candidate_y}

评分维度（每项 1-5 分，可使用一位小数）：
{dimensions}

只输出 JSON：
{{
  "candidate_x": {{"scores": {json.dumps(score_template, ensure_ascii=False)}}},
  "candidate_y": {{"scores": {json.dumps(score_template, ensure_ascii=False)}}},
  "rationale": "比较两者最关键的优缺点",
  "evidence": ["来自候选的短证据"]
}}
"""


def _normalize_scores(
    payload: dict[str, Any], rubric: dict[str, Any]
) -> tuple[dict[str, float], float]:
    raw_scores = payload.get("scores")
    dimensions = set(rubric["dimensions"])
    if not isinstance(raw_scores, dict) or set(raw_scores) != dimensions:
        raise ValueError("candidate score dimensions do not match rubric")
    scores: dict[str, float] = {}
    for key, value in raw_scores.items():
        score = float(value)
        if not 1 <= score <= 5:
            raise ValueError(f"score outside [1,5]: {key}={score}")
        scores[key] = score
    overall = sum(scores[key] * float(rubric["dimensions"][key]["weight"]) for key in scores)
    return scores, round(overall, 4)


def judge_pairs(
    examples: Sequence[EvalExample],
    control: Sequence[Prediction],
    treatment: Sequence[Prediction],
    provider: TextProvider,
    rubric: dict[str, Any],
    *,
    seed: int = 20260908,
    forced_orders: dict[str, dict[str, str]] | None = None,
    run_label: str = "judge",
) -> list[PairwiseJudgement]:
    rng = random.Random(seed)
    control_by_id = {row.id: row for row in control}
    treatment_by_id = {row.id: row for row in treatment}
    records: list[PairwiseJudgement] = []
    dimensions = list(rubric["dimensions"])

    for index, example in enumerate(examples, start=1):
        control_output = control_by_id[example.id].output
        treatment_output = treatment_by_id[example.id].output
        if forced_orders is not None:
            order = forced_orders[example.id]
            if set(order) != {"X", "Y"} or set(order.values()) != {
                "control",
                "treatment",
            }:
                raise ValueError(f"invalid forced order for {example.id}: {order}")
            candidate_x = control_output if order["X"] == "control" else treatment_output
            candidate_y = control_output if order["Y"] == "control" else treatment_output
        elif rng.random() < 0.5:
            order = {"X": "control", "Y": "treatment"}
            candidate_x, candidate_y = control_output, treatment_output
        else:
            order = {"X": "treatment", "Y": "control"}
            candidate_x, candidate_y = treatment_output, control_output

        raw = provider.generate(
            [
                {
                    "role": "system",
                    "content": "你是严格、一致的盲评员，只返回一个有效 JSON 对象。",
                },
                {
                    "role": "user",
                    "content": _judge_prompt(example, candidate_x, candidate_y, rubric),
                },
            ],
            sample_id=f"{example.id}-{run_label}",
        )
        try:
            match = JSON_BLOCK.search(raw)
            if not match:
                raise ValueError("judge response has no JSON object")
            payload = json.loads(match.group(0))
            x_scores, x_overall = _normalize_scores(payload["candidate_x"], rubric)
            y_scores, y_overall = _normalize_scores(payload["candidate_y"], rubric)
            by_arm = {
                order["X"]: CandidateScore(scores=x_scores, overall=x_overall),
                order["Y"]: CandidateScore(scores=y_scores, overall=y_overall),
            }
            delta = by_arm["treatment"].overall - by_arm["control"].overall
            winner = "treatment" if delta > 0.05 else "control" if delta < -0.05 else "tie"
            record = PairwiseJudgement(
                id=example.id,
                order=order,
                control=by_arm["control"],
                treatment=by_arm["treatment"],
                winner=winner,
                rationale=str(payload.get("rationale", "")).strip(),
                evidence=[str(item) for item in payload.get("evidence", [])],
                judge_model=provider.name,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            fallback = CandidateScore(
                scores={dimension: 1.0 for dimension in dimensions}, overall=1.0
            )
            record = PairwiseJudgement(
                id=example.id,
                order=order,
                control=fallback,
                treatment=fallback,
                winner="tie",
                rationale="Judge 输出无法解析，需人工复核。",
                evidence=[raw[:300]],
                judge_model=provider.name,
                parse_error=str(exc),
            )
        records.append(record)
        print(f"judged sample={index}/{len(examples)}", flush=True)
    return records


def reverse_orders(
    judgements: Sequence[PairwiseJudgement],
) -> dict[str, dict[str, str]]:
    return {row.id: {"X": row.order["Y"], "Y": row.order["X"]} for row in judgements}


def _average_candidate_scores(first: CandidateScore, second: CandidateScore) -> CandidateScore:
    dimensions = list(first.scores)
    scores = {
        dimension: round((first.scores[dimension] + second.scores[dimension]) / 2, 4)
        for dimension in dimensions
    }
    return CandidateScore(
        scores=scores,
        overall=round((first.overall + second.overall) / 2, 4),
    )


def balance_position_passes(
    first: Sequence[PairwiseJudgement], second: Sequence[PairwiseJudgement]
) -> list[PairwiseJudgement]:
    second_by_id = {row.id: row for row in second}
    balanced: list[PairwiseJudgement] = []
    for first_row in first:
        second_row = second_by_id[first_row.id]
        if second_row.order != {
            "X": first_row.order["Y"],
            "Y": first_row.order["X"],
        }:
            raise ValueError(f"second pass did not reverse positions for {first_row.id}")
        if set(second_row.control.scores) != set(first_row.control.scores):
            raise ValueError(f"score dimensions differ for {first_row.id}")
        control = _average_candidate_scores(first_row.control, second_row.control)
        treatment = _average_candidate_scores(first_row.treatment, second_row.treatment)
        delta = treatment.overall - control.overall
        winner = "treatment" if delta > 0.05 else "control" if delta < -0.05 else "tie"
        errors = [value for value in (first_row.parse_error, second_row.parse_error) if value]
        balanced.append(
            PairwiseJudgement(
                id=first_row.id,
                order={"method": "average-of-both-positions"},
                control=control,
                treatment=treatment,
                winner=winner,
                rationale=(f"位置1：{first_row.rationale} 位置2：{second_row.rationale}"),
                evidence=[*first_row.evidence, *second_row.evidence],
                judge_model=first_row.judge_model,
                parse_error="; ".join(errors) if errors else None,
            )
        )
    return balanced


def position_bias_audit(
    first: Sequence[PairwiseJudgement], second: Sequence[PairwiseJudgement]
) -> dict[str, Any]:
    def counts(rows: Sequence[PairwiseJudgement]) -> dict[str, int]:
        return {
            "x_wins": sum(row.winner == row.order.get("X") for row in rows),
            "y_wins": sum(row.winner == row.order.get("Y") for row in rows),
            "ties": sum(row.winner == "tie" for row in rows),
        }

    return {
        "pass_1": counts(first),
        "pass_2_reversed": counts(second),
        "correction": "final scores average each candidate across X and Y positions",
    }


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def paired_bootstrap_ci(
    deltas: Sequence[float], *, samples: int = 10_000, seed: int = 20260908
) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(_mean([rng.choice(deltas) for _ in deltas]))
    estimates.sort()
    return estimates[int(samples * 0.025)], estimates[int(samples * 0.975)]


def summarize_ab(
    examples: Sequence[EvalExample],
    control: Sequence[Prediction],
    treatment: Sequence[Prediction],
    judgements: Sequence[PairwiseJudgement],
) -> dict[str, Any]:
    control_metrics, control_auto = compute_metrics(examples, control)
    treatment_metrics, treatment_auto = compute_metrics(examples, treatment)
    dimensions = list(judgements[0].control.scores) if judgements else []
    deltas = [row.treatment.overall - row.control.overall for row in judgements]
    ci_low, ci_high = paired_bootstrap_ci(deltas) if deltas else (0.0, 0.0)
    threshold = 3.5
    outcomes = Counter(row.winner for row in judgements)
    parse_errors = sum(row.parse_error is not None for row in judgements)
    dimension_means = {
        dimension: {
            "control": round(_mean([row.control.scores[dimension] for row in judgements]), 4),
            "treatment": round(_mean([row.treatment.scores[dimension] for row in judgements]), 4),
        }
        for dimension in dimensions
    }

    def constraint_failures(rows: Sequence[Any]) -> dict[str, int]:
        failures: Counter[str] = Counter()
        for row in rows:
            failures.update(key for key, passed in row.constraint_details.items() if not passed)
        return dict(failures.most_common())

    control_constraint_score = control_auto["mean_constraint_score"]
    treatment_constraint_score = treatment_auto["mean_constraint_score"]
    guardrail_pass = (
        treatment_constraint_score >= control_constraint_score
        and dimension_means.get("factual_grounding", {}).get("treatment", 0)
        >= dimension_means.get("factual_grounding", {}).get("control", 0)
        and dimension_means.get("safety", {}).get("treatment", 0)
        >= dimension_means.get("safety", {}).get("control", 0)
    )
    return {
        "sample_count": len(examples),
        "hypothesis": (
            "在相同商品事实、模型和输出约束下，加载文案 Skill 会提高平台适配、"
            "信息价值和主张边界，同时不降低事实一致性。"
        ),
        "control": {
            "mean_overall": round(_mean([row.control.overall for row in judgements]), 4),
            "pass_rate": round(
                _mean([float(row.control.overall >= threshold) for row in judgements]), 4
            ),
            "constraint_score": control_constraint_score,
            "constraint_failures": constraint_failures(control_metrics),
            "capture_context_mentions": sum("采集时" in row.output for row in control),
            "mean_latency_ms": round(_mean([row.latency_ms or 0 for row in control]), 3),
        },
        "treatment": {
            "mean_overall": round(_mean([row.treatment.overall for row in judgements]), 4),
            "pass_rate": round(
                _mean([float(row.treatment.overall >= threshold) for row in judgements]), 4
            ),
            "constraint_score": treatment_constraint_score,
            "constraint_failures": constraint_failures(treatment_metrics),
            "capture_context_mentions": sum("采集时" in row.output for row in treatment),
            "mean_latency_ms": round(_mean([row.latency_ms or 0 for row in treatment]), 3),
        },
        "delta": {
            "mean_overall": round(_mean(deltas), 4),
            "paired_bootstrap_95_ci": [round(ci_low, 4), round(ci_high, 4)],
        },
        "paired_outcomes": {
            "treatment_wins": outcomes["treatment"],
            "ties": outcomes["tie"],
            "control_wins": outcomes["control"],
        },
        "dimension_means": dimension_means,
        "guardrail": {
            "passed": guardrail_pass,
            "decision": "proceed_to_larger_test" if guardrail_pass else "revise_and_retest",
        },
        "judge_parse_errors": parse_errors,
        "automatic_metric_rows": {
            "control": [row.model_dump(mode="json") for row in control_metrics],
            "treatment": [row.model_dump(mode="json") for row in treatment_metrics],
        },
    }


def render_ab_report(
    examples: Sequence[EvalExample],
    control: Sequence[Prediction],
    treatment: Sequence[Prediction],
    judgements: Sequence[PairwiseJudgement],
    summary: dict[str, Any],
    *,
    generator_model: str,
    judge_model: str,
) -> str:
    example_by_id = {row.id: row for row in examples}
    control_by_id = {row.id: row for row in control}
    treatment_by_id = {row.id: row for row in treatment}
    c = summary["control"]
    t = summary["treatment"]
    d = summary["delta"]
    outcomes = summary["paired_outcomes"]
    position_audit = summary.get("position_bias_audit")
    guardrail = summary["guardrail"]
    lines = [
        "# 真实商品文案 Skill A/B Smoke Test",
        "",
        "> 运行日期：2026-09-08  ",
        "> 数据：Alibaba.com 商品快照 v20260908（20 类 × 每类 1 条）",
        "",
        "## 结论",
        "",
        f"- Treatment（加载 Skill）平均分：**{t['mean_overall']:.2f}/5**；"
        f"Control（普通提示）平均分：**{c['mean_overall']:.2f}/5**。",
        f"- 配对差值：**{d['mean_overall']:+.2f}**；描述性 bootstrap 95% 区间："
        f"**[{d['paired_bootstrap_95_ci'][0]:+.2f}, {d['paired_bootstrap_95_ci'][1]:+.2f}]**。",
        f"- 配对胜/平/负：**{outcomes['treatment_wins']} / {outcomes['ties']} / "
        f"{outcomes['control_wins']}**。",
        f"- 通过率：Control **{c['pass_rate']:.0%}**，Treatment **{t['pass_rate']:.0%}**；"
        f"硬约束满足率：Control **{c['constraint_score']:.0%}**，"
        f"Treatment **{t['constraint_score']:.0%}**。",
        f"- Guardrail：**{'通过' if guardrail['passed'] else '未通过'}**；决策："
        f"**{guardrail['decision']}**。",
        "",
        "这是 20 条本地模型试跑，只能说明这批样本上的方向，不能宣称普遍或线上业务提升。",
        "",
        "## 实验设计",
        "",
        f"- 生成模型：`{generator_model}`；两组模型、温度、token 上限、事实和约束相同。",
        "- Control：中性事实约束提示；Treatment：额外加载版本化 Skill、"
        "小红书指南和商品库事实边界。",
        f"- 盲评模型：`{judge_model}`；候选匿名为 X/Y，并按固定随机种子交换位置。",
        "- 主指标：五维加权综合分；辅助指标：通过率、硬约束满足率、配对胜平负。",
        "- Guardrail：事实一致性与安全合规不得下降。",
    ]
    if position_audit:
        pass_1 = position_audit["pass_1"]
        pass_2 = position_audit["pass_2_reversed"]
        lines.extend(
            [
                "- 位置偏差审计：首次盲评 X/Y 胜出数为 "
                f"{pass_1['x_wins']}/{pass_1['y_wins']}；反转后为 "
                f"{pass_2['x_wins']}/{pass_2['y_wins']}。",
                "- 最终分数对同一候选位于 X 和 Y 时的得分取平均，"
                "不使用存在位置偏差的单次结果直接下结论。",
            ]
        )
    lines.extend(
        [
            "",
            "## 分维度结果",
            "",
            "| 维度 | Control | Treatment | 差值 |",
            "|---|---:|---:|---:|",
        ]
    )
    for dimension, values in summary["dimension_means"].items():
        delta = values["treatment"] - values["control"]
        lines.append(
            f"| `{dimension}` | {values['control']:.2f} | "
            f"{values['treatment']:.2f} | {delta:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## 自动约束审计",
            "",
            f"- 明确写出“采集时”：Control {c['capture_context_mentions']}/{len(examples)}，"
            f"Treatment {t['capture_context_mentions']}/{len(examples)}。",
            "- Control 失败项："
            + "；".join(f"{key} × {count}" for key, count in c["constraint_failures"].items()),
            "- Treatment 失败项："
            + "；".join(f"{key} × {count}" for key, count in t["constraint_failures"].items()),
            "",
            "Treatment 的 Judge 综合分略高，但硬约束满足率下降，因此本轮不接受 Skill v1 "
            "为胜出版本。尤其需要修复标签数量、起订量原文保留和快照语境。",
        ]
    )

    ranked = sorted(
        judgements,
        key=lambda row: row.treatment.overall - row.control.overall,
        reverse=True,
    )
    lines.extend(
        [
            "",
            "## 样本级结果",
            "",
            "| 品类 | Control | Treatment | 差值 | 判定 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in ranked:
        category = example_by_id[row.id].category
        delta = row.treatment.overall - row.control.overall
        lines.append(
            f"| {category} | {row.control.overall:.2f} | {row.treatment.overall:.2f} | "
            f"{delta:+.2f} | {row.winner} |"
        )

    lines.extend(["", "## 对照示例", ""])
    example_rows = [ranked[0], ranked[-1]] if len(ranked) >= 2 else ranked
    for row in example_rows:
        example = example_by_id[row.id]
        lines.extend(
            [
                f"### {example.category} · `{row.id}`",
                "",
                f"Judge：Control {row.control.overall:.2f} / Treatment "
                f"{row.treatment.overall:.2f}。{row.rationale}",
                "",
                "**Control**",
                "",
                control_by_id[row.id].output,
                "",
                "**Treatment**",
                "",
                treatment_by_id[row.id].output,
                "",
            ]
        )

    lines.extend(
        [
            "## 局限与下一轮",
            "",
            f"- Judge 解析失败：{summary['judge_parse_errors']} 条。",
            "- 单一 4B Judge 可能有位置、长度和文风偏好；正式结论需要人工双标校准。",
            "- 两轮 Judge 都在 20/20 样本偏好 X；双位置平均只能抵消一阶位置效应，"
            "不能证明该 Judge 已可靠校准。",
            "- 每类只有 1 条，区间只是描述性不确定性，不替代预注册的大样本实验。",
            "- Control 先运行且包含冷启动，延迟数据受运行顺序混淆，不用于判断 Skill 更快。",
            "- 商品库是批发搜索页快照；价格、库存、规格和卖家主张在真实发布前必须复核。",
            "- 下一轮建议固定这 20 条作为开发集，另抽 60–100 条未见测试集，"
            "并加入至少两名人工评审。",
            "",
        ]
    )
    return "\n".join(lines)


def write_ab_artifacts(
    output_dir: str | Path,
    *,
    examples: Sequence[EvalExample],
    control: Sequence[Prediction],
    treatment: Sequence[Prediction],
    judgements: Sequence[PairwiseJudgement],
    summary: dict[str, Any],
    report: str,
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_jsonl(target / "dataset.jsonl", examples)
    write_jsonl(target / "control_predictions.jsonl", control)
    write_jsonl(target / "treatment_predictions.jsonl", treatment)
    write_jsonl(target / "pairwise_judgements_balanced.jsonl", judgements)
    serializable_summary = {
        key: value for key, value in summary.items() if key != "automatic_metric_rows"
    }
    write_json(target / "summary.json", serializable_summary)
    write_json(target / "automatic_metrics.json", summary["automatic_metric_rows"])
    (target / "report.md").write_text(report, encoding="utf-8")
