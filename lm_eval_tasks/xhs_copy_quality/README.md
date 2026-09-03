# xhs_copy_quality task

Local generation task for the 18-sample synthetic Chinese social-copy dataset. It reports
character-level ROUGE-L and deterministic constraint compliance inside
`lm-evaluation-harness`. Use the repository's `xhs-eval` pipeline for the calibrated
LLM-as-Judge and bad-case report.

From the repository root:

```bash
lm-eval validate --tasks xhs_copy_quality --include_path ./lm_eval_tasks
```

