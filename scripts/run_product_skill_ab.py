from __future__ import annotations

import argparse
import os
from pathlib import Path

from xhs_eval.io import load_models, load_predictions, write_jsonl
from xhs_eval.judge import load_rubric
from xhs_eval.product_catalog import load_product_catalog
from xhs_eval.providers import OpenAICompatibleProvider
from xhs_eval.skill_ab import (
    CONTROL_SYSTEM_PROMPT,
    PairwiseJudgement,
    balance_position_passes,
    build_product_examples,
    generate_arm,
    judge_pairs,
    load_treatment_system_prompt,
    position_bias_audit,
    render_ab_report,
    reverse_orders,
    summarize_ab,
    write_ab_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行真实商品文案 Skill A/B smoke test")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/products/alibaba_products_20260908.jsonl"),
    )
    parser.add_argument("--skill-dir", type=Path, default=Path("skills/xiaohongshu-ins-copy"))
    parser.add_argument("--rubric", type=Path, default=Path("rubrics/xhs_copy_v1.yaml"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluations/product_skill_ab_20260908")
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--generator-model", default="qwen2.5:7b")
    parser.add_argument("--judge-model", default="gemma3:4b")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("OLLAMA_API_KEY", "ollama-local")
    examples = build_product_examples(load_product_catalog(args.catalog))
    rubric = load_rubric(args.rubric)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "dataset.jsonl", examples)

    generator = OpenAICompatibleProvider(
        model=args.generator_model,
        base_url=args.base_url,
        api_key_env="OLLAMA_API_KEY",
        temperature=0.2,
        max_tokens=700,
        timeout_seconds=180,
        max_retries=1,
    )
    judge = OpenAICompatibleProvider(
        model=args.judge_model,
        base_url=args.base_url,
        api_key_env="OLLAMA_API_KEY",
        temperature=0,
        max_tokens=900,
        timeout_seconds=180,
        max_retries=1,
        response_format_json=True,
    )

    control_path = args.output_dir / "control_predictions.jsonl"
    if control_path.exists():
        control = load_predictions(control_path)
        print(f"resumed control={len(control)}", flush=True)
    else:
        control = generate_arm(
            examples, generator, system_prompt=CONTROL_SYSTEM_PROMPT, arm="control"
        )
        write_jsonl(control_path, control)

    treatment_path = args.output_dir / "treatment_predictions.jsonl"
    if treatment_path.exists():
        treatment = load_predictions(treatment_path)
        print(f"resumed treatment={len(treatment)}", flush=True)
    else:
        treatment = generate_arm(
            examples,
            generator,
            system_prompt=load_treatment_system_prompt(args.skill_dir),
            arm="treatment",
        )
        write_jsonl(treatment_path, treatment)

    first_judgement_path = args.output_dir / "pairwise_judgements_pass1.jsonl"
    if first_judgement_path.exists():
        first_judgements = load_models(first_judgement_path, PairwiseJudgement)
        print(f"resumed first_judgements={len(first_judgements)}", flush=True)
    else:
        first_judgements = judge_pairs(examples, control, treatment, judge, rubric)
        write_jsonl(first_judgement_path, first_judgements)

    reversed_judgement_path = args.output_dir / "pairwise_judgements_reversed.jsonl"
    if reversed_judgement_path.exists():
        reversed_judgements = load_models(reversed_judgement_path, PairwiseJudgement)
        print(f"resumed reversed_judgements={len(reversed_judgements)}", flush=True)
    else:
        reversed_judgements = judge_pairs(
            examples,
            control,
            treatment,
            judge,
            rubric,
            forced_orders=reverse_orders(first_judgements),
            run_label="judge-reversed",
        )
        write_jsonl(reversed_judgement_path, reversed_judgements)

    judgements = balance_position_passes(first_judgements, reversed_judgements)
    write_jsonl(args.output_dir / "pairwise_judgements_balanced.jsonl", judgements)
    summary = summarize_ab(examples, control, treatment, judgements)
    summary["position_bias_audit"] = position_bias_audit(first_judgements, reversed_judgements)
    report = render_ab_report(
        examples,
        control,
        treatment,
        judgements,
        summary,
        generator_model=args.generator_model,
        judge_model=args.judge_model,
    )
    write_ab_artifacts(
        args.output_dir,
        examples=examples,
        control=control,
        treatment=treatment,
        judgements=judgements,
        summary=summary,
        report=report,
    )
    print(f"completed output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
