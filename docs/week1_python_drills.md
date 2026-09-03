# 第 1 周 Python 评测脚本训练

每天控制在 60–90 分钟：先不看仓库实现写一遍，再与本项目脚本对照，最后加一个边界测试。

| 天 | 题目 | 完成标准 | 对照文件 |
|---|---|---|---|
| Day 1 | 加载 JSONL 并校验唯一 ID | 错行要带文件名和行号 | `io.py`, `load_data.py` |
| Day 2 | 按 category 统计 easy/medium/hard | 同时输出表格和 CSV | `class_distribution.py` |
| Day 3 | 固定 seed 做分层 train/val/test | 同类样本尽量均匀，比例参数化 | `split_dataset.py` |
| Day 4 | 实现字符级 ROUGE-L | 空字符串、完全相同、重复字符有测试 | `metrics.py` |
| Day 5 | 读取预测并算每类指标 | 检查 missing / extra ID，不按行号盲合并 | `compute_metrics.py` |
| Day 6 | 解析 Judge JSON 并筛 Top badcase | 处理 Markdown fence、越界分和字段缺失 | `judge.py`, `top_failures.py` |
| Day 7 | 用 argparse 串成 CLI | 有配置、日志、输出目录和退出错误 | `cli.py`, `pipeline.py` |

## 现场写题自测

> 给定 `judgements.jsonl`，输出各类别通过率、最低分 5 条和归因频次；若预测 ID 与数据集不一致，
> 程序必须报出具体缺失/多余 ID。

限时 35 分钟。评分点：正确性 50%、异常处理 20%、函数拆分 15%、CLI/日志 10%、测试 5%。

