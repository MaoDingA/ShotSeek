# ShotSeek 评测与版本门禁

ShotSeek 把回归稳定性、独立素材泛化和长视频规模验证分开报告。75 秒黄金样片的
40 条查询用于防止功能倒退，不作为长视频泛化结论。

## 指标

每次 Benchmark 固定输出：

- Recall@1、Recall@3、MRR；
- 负例误报 Query 数与 Hit 数；
- Verifier precision、证据支持率、直接证据率；
- 查询 P50、P95 和最大延迟；
- 有人工时间标注时的 Temporal IoU、开始/结束边界误差；
- 分类指标、逐条结果、数据集/SQLite SHA-256 和代码提交；
- 同一代码与数据库上的确定性回放结果。

默认参赛门槛为 Recall@1 ≥ 0.65、Recall@3 ≥ 0.80、证据支持率 ≥ 0.85、
直接证据率 ≥ 0.85、P95 ≤ 3 秒、边界中位误差 ≤ 1.5 秒，以及负例误报为零。

## 运行黄金回归

```bash
python scripts/run_benchmark.py \
  --database runs/m1c/latest/search.sqlite3 \
  --queries eval/m1_queries.jsonl \
  --queries eval/m2_queries.jsonl \
  --output runs/benchmark/regression-v1 \
  --split regression-golden-v1 \
  --require-pass
```

输出目录包含：

```text
evaluation.json
results.json
report.md
report.html
traces/
```

报告默认只运行本地 Rule Planner 和 Rule Verifier，`network_calls=0`。如后续增加
StepFun Planner/Verifier 对照实验，必须单独命名 split 与运行目录，不能覆盖本地
确定性基线。

## 数据集纪律

- Development 可用于发现问题和调参。
- Holdout 冻结后只用于版本门禁，不根据结果修改系统或标注。
- Longform 必须来自连续 30–43 分钟素材，并至少覆盖对白、视觉动作、人物地点、
  多模态、序数/前后关系与困难负例。
- 报告失败是有效结果；只有同时提交失败 Case，性能数字才具有可审计性。
