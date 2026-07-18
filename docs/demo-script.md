# ShotSeek 四分钟演示脚本

目标不是展示最多功能，而是让评委在四分钟内确认四件事：真实处理、准确跳转、
证据可解释、结果可交付。

## 演示前检查

- Runtime `/health` 为 `ok`，Registry integrity 为 `ok`；
- 60–90 秒真实演示素材尚未处理，用于现场上传；
- 36:58 长视频项目已处于 `READY`，用于规模展示；
- 浏览器缩放 100%，播放器、Evidence Drawer 与导出按钮可见；
- StepFun 额度和网络正常，同时保留明确标记的 `CACHED` 降级项目；
- EDL、JSON、SRT、XML 下载目录为空，便于现场确认新文件；
- 不在屏幕、终端历史或日志中显示 API Key。

## 0:00–0:25 定位

旁白：

> ShotSeek 不是视频聊天，而是影视场景定位。用户用自然语言描述画面或对白，
> 系统返回原片的准确镜头边界、直接证据和可交付的剪辑文件。

画面停留在工作台，快速指出播放器、搜索框、时间线和结果区。

## 0:25–1:20 真实处理

拖入 60–90 秒 MP4，展示：

```text
Upload → Probe → Transcode → Shot Detection → StepFun Vision
       → StepAudio ASR → Alignment → Scene → Index → READY
```

说明：

- DGX Spark 负责本地媒体、镜头网格、证据时间线、索引和 Runtime；
- Step 3.7 Flash 负责结构化视觉事件；
- StepAudio 2.5 ASR 负责对白和时间戳；
- `LIVE/CACHED/PARTIAL/FAILED` 是真实状态，缓存不会伪装为实时调用。

## 1:20–2:35 三类查询

对 75 秒黄金样片依次搜索：

1. 精确对白：`"Memory override in progress"`；
2. 中文视觉：`有人正瞄着带镜步枪`；
3. 序数约束：`second robotic hand`。

每次命中后点击“立即跳转”，确认播放器跳到对应入点。第三条查询展开 Agent Trace：

```text
Query Planner → Top 20 Recall → Temporal / Ordinal
→ Evidence Verifier → Shot Boundary → Final
```

强调序数和前后关系由确定性时间运算完成，不由模型猜测。

## 2:35–3:20 展开证据

展开一个结果的 Evidence Drawer，依次指出：

- Visual：画面事件与关键帧；
- Dialogue：对白文本和时间戳；
- Boundary：模型原始区间与 shot-first 最终区间；
- Trace：Planner、Retriever、Verifier 的输入输出；
- Confidence：分项得分与直接证据。

旁白：

> StepFun 告诉系统“发生了什么”，ShotSeek 用真实镜头网格确定“准确发生在哪”。
> 模型时间只是候选，最终边界不能被模型随意改写。

## 3:20–3:45 长视频与交付

切换到已完成的 36:58 项目：

- 状态 `READY`；
- 216 个 Scene；
- 真实视觉与 ASR 均为 `LIVE`；
- 总运行 1784.916 秒，RTF 约 0.805。

选择一个结果，依次下载 CMX3600 EDL 和 JSON，说明 SRT/XML 使用同一帧级时间基。

## 3:45–4:00 真实边界

最后展示评测页：

> 黄金回归 40 条查询全部命中，证明系统稳定；独立 Holdout 和 Longform 的泛化
> 门禁仍有失败。当前系统选择“没有证据就不返回”，因此没有负例误报，但会漏答。
> 这是下一轮模型验证器要解决的明确边界。

收尾：

> ShotSeek 把自然语言映射到真实镜头，每次命中都给证据与边界。

## 降级策略

- 真实接口失败：切换到明确显示 `CACHED` 的同素材项目，不隐藏失败状态；
- 长视频项目不可用：展示 Runtime 报告与 SQLite 完整性，再使用黄金样片完成闭环；
- 网络中断：只演示本地搜索、Evidence Drawer、播放器和导出；
- 播放器编解码异常：切换代理视频，搜索结果与时间码不变。

任何降级都必须口头说明，不能把 Fixture、缓存或预生成结果称为实时处理。
