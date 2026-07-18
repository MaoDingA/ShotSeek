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

## 当前可审计结果

| Split | 状态 | Query | R@1 | R@3 | Verifier P | P95 | 负例误报 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 黄金回归 | PASS | 40 | 100.0% | 100.0% | 87.0% | 57.2 ms | 0 |
| Development v1 | PASS | 15 | 100.0% | 100.0% | 100.0% | 37.8 ms | 0 |
| Holdout v1 | FAIL | 15 | 66.7% | 66.7% | 88.9% | 36.7 ms | 0 |
| Longform v1 | FAIL | 20 | 73.3% | 73.3% | 93.8% | 63.0 ms | 0 |
| Holdout v2 首次运行 | FAIL | 15 | 16.7% | 16.7% | 100.0% | 9.0 ms | 0 |

总体状态是 `MIXED - GENERALIZATION GATES NOT MET`。Holdout v2 只返回两个结果，
且两个都正确；这说明当前 Rule Verifier 的边界是 precision-first：不会为了覆盖率
放宽直接证据要求，但对跨素材的实体别名、复合动作和序数表达过于保守。

36:58 连续素材与 02:26 独立素材均通过真实 StepFun Production Runtime 到达
`READY`。前者用时 1784.916 秒，生成 216 个 Scene；后者用时 144.770 秒，
生成 18 个 Scene。两次运行的视觉与 ASR 产物均为 `LIVE`。

生成 Markdown、HTML 与 PDF 汇总报告：

```bash
.venv/bin/python scripts/build_competition_report.py
```

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
- Holdout v2 首次结果必须永久保留；如果将它用于开发，需改名 Development v2，
  下一次门禁必须使用完全未见的 Holdout v3。
- Longform 必须来自连续 30–43 分钟素材，并至少覆盖对白、视觉动作、人物地点、
  多模态、序数/前后关系与困难负例。
- 报告失败是有效结果；只有同时提交失败 Case，性能数字才具有可审计性。
