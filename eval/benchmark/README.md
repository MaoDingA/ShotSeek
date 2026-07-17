# Benchmark 数据集契约

本目录只保存可公开复现的查询标注与版本清单。媒体、SQLite 和每次运行报告仍写入
被忽略的 `runs/`，不进入公开仓库。

## 四个数据面

| Split | 用途 | 是否允许据此调参 |
|---|---|---:|
| Regression | 保护 M0–M2 的 75 秒黄金样片能力 | 是，但不能宣称泛化 |
| Development | 独立素材上的日常开发评测 | 是 |
| Holdout | 冻结后只在版本门禁时运行 | 否 |
| Longform | 连续 30–43 分钟素材的规模与边界评测 | 否 |

`manifest.json` 是当前数据面的真实状态。未标注完成的 split 必须保留
`pending_annotation`，不能用 Regression 指标代替。

## JSONL Case

每行是一个严格 Case：

```json
{
  "query_id": "longform_001",
  "category": "visual_action",
  "text": "找到女人推开白色门的镜头",
  "acceptable_scene_ids": ["scene_0042"],
  "reference_start_ms": 812340,
  "reference_end_ms": 829880,
  "notes": "人工双人复核"
}
```

负例的 `acceptable_scene_ids` 必须为空。时间标注可省略；一旦填写，开始和结束
必须同时存在，且使用原片毫秒时间。

## 冻结规则

1. 每个 split 的 `query_id` 全局唯一。
2. Holdout 和 Longform 冻结后不得根据检索结果修改措辞或答案；发现标注错误时
   新建数据集版本，并在清单记录原因。
3. 查询文件、数据库、代码提交均写入报告 SHA-256；没有这些来源信息的数字不能
   用于答辩。
4. LIVE、CACHED、fixture 必须保留原状态；缓存结果不能伪装为实时模型调用。
5. 任何版本都同时保留失败 Case，不只展示成功示例。
