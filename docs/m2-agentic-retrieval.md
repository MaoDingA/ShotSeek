# M2 Agentic Retrieval 与证据验证

M2 在不修改 `scene-v1` 时间线资产的前提下，把 M1 的确定性搜索升级为可审计的 Agent 检索链：

```text
Query Planner
→ FTS5 Top 20 宽召回
→ 确定性 before / after / during / between
→ 直接证据与反证验证
→ 序数运算
→ 可解释重排
→ Top 3 + Agent Trace
```

## QuerySpec v2

Query Planner 只解析查询，不允许选择 Scene、时间码、帧或 Shot。结构化结果包含人物、动作、物体、地点、精确对白、时间关系、序数、否定约束、证据偏好和 Top K。

简单查询使用规则规划器；复杂查询可调用 Step 3.7 Flash。缓存命中时离线复用；模型不可用时回退到确定性规则，并在 Trace 中保留 `LIVE / CACHED / FALLBACK / RULE` 状态。

## Evidence Verifier

Verifier 只接收一个 QuerySpec 和一个候选 Scene，只能输出 `supported / unsupported / uncertain`。最终结果必须满足：

1. 每个正向约束都有结构化证据；
2. 没有命中否定约束；
3. 至少存在一条直接视觉或对白证据；
4. Scene 的证据引用真实存在；
5. 时间关系和序数由本地确定性代码完成。

StepFun 验证器不能把规则层判定为 `unsupported` 的候选升级为 `supported`，也不能改写 Scene ID、时间码、帧或 Shot。

## Agent Trace

每次搜索保留规范化 QuerySpec、Planner 路由与耗时、Top 20 候选、时间锚点、验证数量、直接证据、分项得分和最终 Scene。运行态 Trace 写入被忽略的 `runs/`。

## 只读 API

```text
GET  /health
GET  /videos
GET  /videos/{video_id}
GET  /videos/{video_id}/scenes
GET  /scenes/{scene_id}
POST /search
GET  /traces/{trace_id}
GET  /metrics
```

API 不提供上传、删除、重建索引或媒体处理端点。默认禁止网络调用；只有服务端显式设置 `SHOTSEEK_ALLOW_NETWORK=1` 时，复杂规划或候选验证才允许使用 StepFun。

## 40 条离线评测

评测集由 15 条 M1 查询和 25 条 M2 查询组成：

| 新增类别 | 数量 |
| --- | ---: |
| 中文同义表达 | 8 |
| 英文同义表达 | 4 |
| 复杂时间关系 | 5 |
| 第二次 / 最后一次 | 3 |
| 否定约束 | 2 |
| 困难负例 | 3 |

2026-07-17 在 DGX Spark 上的可复现实测：

| 指标 | 结果 |
| --- | ---: |
| QuerySpec 有效率 | 1.000 |
| Planner 准确率 | 1.000 |
| Candidate Recall@20 | 1.000 |
| Recall@1 | 1.000 |
| Recall@3 | 1.000 |
| MRR | 1.000 |
| Evidence Support Rate | 1.000 |
| Exact Dialogue Recall@1 | 1.000 |
| 负例高置信误报 | 0 |
| Broken Evidence Reference | 0 |
| 离线网络调用 | 0 |

Verifier Precision 为 0.870，按人工可接受 Scene 集合计算；所有实际返回项均为 `supported` 且具有直接证据。该指标与 Evidence Support Rate 分开报告。

## 消融结果

| 系统 | Recall@1 | Recall@3 | MRR | 负例误报 |
| --- | ---: | ---: | ---: | ---: |
| M1 FTS 基线 | 0.429 | 0.429 | 0.429 | 0 |
| M2 宽召回，不验证 | 0.771 | 0.943 | 0.843 | 7 |
| M2 完整 Agent | 1.000 | 1.000 | 1.000 | 0 |

同义词和时间规划提高召回，证据验证负责消除宽召回带来的错误候选。

## 复现

```bash
.venv/bin/python scripts/run_m2_evaluation.py
.venv/bin/python scripts/verify_m2_completion.py
```

输出位于 `runs/m2/evaluation-v1/`，包含 `evaluation.json`、`ablation.json`、`results.json` 和 `traces/`。评测运行两遍并比较 QuerySpec 与最终 Scene 序列；延迟不参与确定性哈希。
