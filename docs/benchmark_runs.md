# 标准 Benchmark 运行记录

标准任务用于证明会使用框架；小模型 + `--limit` 的 smoke test 不应包装成有意义的模型排名。

## Run metadata

| 字段 | 值 |
|---|---|
| 日期 | 2026-09-03 |
| lm-eval 版本 | 0.4.13（`importlib.metadata.version("lm-eval")`） |
| Git commit | 首次提交前的本地验证 |
| 模型与 revision | `dummy`（仅链路验证，无模型能力含义） |
| 后端 / 设备 | lm-eval dummy backend / CPU |
| tasks | hellaswag, arc_easy |
| few-shot / limit | 0-shot / 每任务 5 条 |
| batch size / seed | 1 / 默认 seeds: 0, 1234, 1234, 1234 |
| output path | `results/standard-smoke/`（本地产物，不提交） |

## 命令

```bash
uv run lm-eval run --model dummy \
  --tasks hellaswag arc_easy --limit 5 --batch_size 1 \
  --output_path results/standard-smoke --log_samples
```

## Smoke test 结果

| Task | Metric | Value | Stderr |
|---|---|---:|---:|
| arc_easy | acc | 0.4 | 0.2449 |
| arc_easy | acc_norm | 0.4 | 0.2449 |
| hellaswag | acc | 0.0 | 0.0000 |
| hellaswag | acc_norm | 0.4 | 0.2449 |

这些随机/dummy 数字不可用于比较模型。有效产出是：两个数据集成功加载、任务上下文成功构建、
40 个 log-likelihood 请求完成、聚合与逐样本文件成功写出。

## 自定义 task smoke test

```bash
uv run lm-eval validate --tasks xhs_copy_quality --include_path ./lm_eval_tasks
uv run lm-eval run --model dummy --tasks xhs_copy_quality \
  --include_path ./lm_eval_tasks --limit 2 --batch_size 1 \
  --output_path results/xhs-custom-smoke --log_samples
```

校验输出为 `All tasks found and valid`；运行成功完成 2 个 `generate_until` 请求，聚合了
`char_rouge_l` 与 `constraint_score`。dummy 指标没有模型能力含义。

## 读结果时要讲清楚

- 指标名与方向，例如 accuracy / normalized accuracy。
- 是否 few-shot、是否 chat template、样本数和随机种子。
- 任务版本、模型 revision、框架 commit 和硬件。
- smoke test 只代表链路通过；正式比较需要完整集和一致配置。
