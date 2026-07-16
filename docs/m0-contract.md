# M0：接口与时间线契约验证

## 目标

M0 用一条命令证明 StepFun 视频理解与 StepAudio ASR 可以产生可校验的多模态证据，并由 DGX Spark 本地代码归一化到同一条原片毫秒时间线。

```bash
python scripts/run_m0_probe.py \
  --live \
  --video samples/golden.mp4 \
  --audio-url "$GOLDEN_AUDIO_URL"
```

离线回归不读取密钥，也不访问网络：

```bash
python scripts/run_m0_probe.py --fixture --video samples/golden.mp4
```

## 输入

- 项目目录树内的 MP4 视频；
- 可选的项目内视频分片清单；每段必须是公网 HTTP(S) URL、最长 10 秒，并连续覆盖完整原片；
- Live 模式下公网可访问的 MP3/WAV/OGG/PCM 音频 URL；异步文件与 SSE 两种 ASR 通道共用该输入；
- Live 模式下通过环境变量提供的 StepFun API Key；
- `live` 或 `fixture` 运行模式。

## 输出

```text
runs/m0/<run_id>/
├── manifest.json
├── raw/
│   ├── stepfun_file.json
│   ├── vision_response.json
│   └── asr_response.json
├── normalized/
│   ├── visual_events.json
│   ├── utterances.json
│   └── evidence_timeline.json
└── run_report.json
```

所有运行产物位于仓库目录内的 `runs/`，并由 Git 忽略。模型原始响应与标准化事实严格分离。

## 接口通道

M0 显式区分四条接口通道，不能用一个 Base URL 覆盖全部服务：

- Files API：`https://api.stepfun.com/v1`；
- Chat Completions：`https://api.stepfun.com/step_plan/v1`；
- 异步文件 ASR：`https://api.stepfun.com/v1`；
- Step Plan SSE ASR：`https://api.stepfun.com/step_plan/v1`。

对应环境变量为 `STEPFUN_FILES_BASE_URL`、`STEPFUN_CHAT_BASE_URL`、`STEPFUN_ASR_BASE_URL` 和 `STEPFUN_SSE_ASR_BASE_URL`。`async_file` 是完整验收默认通道；`sse` 是当前套餐可用的时间戳诊断通道。

官方文档：

- [Step Plan 音频接入](https://platform.stepfun.com/docs/zh/step-plan/integrations/audio-api)
- [SSE ASR](https://platform.stepfun.com/docs/zh/api-reference/audio/asr-sse)
- [异步文件 ASR](https://platform.stepfun.com/docs/zh/api-reference/audio/asr)

### Step Plan SSE 时间戳诊断

SSE ASR 必须显式设置 `enable_timestamp=true`；该字段默认关闭。可复用同一视频 SHA256 的真实视觉运行，只实时执行 ASR 与统一时间线：

```bash
python scripts/run_m0_probe.py \
  --live \
  --video samples/golden-stepfun.mp4 \
  --vision-cache-run runs/m0/<vision_run_id> \
  --asr-transport sse
```

缓存命中、接口耗时和每项硬门槛都会写入 `run_report.json`。SSE 当前不返回说话人，因此该路径不会被误判为完整 M0 通过。

### 直连分片诊断入口

当前标准 Files API 额度不可用时，可以提供以下格式的清单：

```json
{
  "chunks": [
    {
      "chunk_id": "chunk_000",
      "source_start_ms": 0,
      "source_end_ms": 10000,
      "url": "https://example.invalid/chunk-000.mp4"
    }
  ]
}
```

探针会校验分片 ID 唯一、顺序连续、完整覆盖原片且每段不超过 10 秒。运行产物只保存 `<provided>`，不保存真实 URL。该入口明确标记为 `direct_url_chunks`，只用于验证视频理解和全局时间偏移，不能替代 Files API 验收。

## 冻结契约

- 视觉时间使用 `approx_start_ms/approx_end_ms`，不得伪装成最终镜头边界；
- ASR 使用服务返回的句级和字词级毫秒时间戳；
- SSE 请求固定 `enable_timestamp=true`；无有效时间戳时直接失败；
- 统一证据使用 `start_ms/end_ms`，并保留 `source_ref`；
- 视觉边界标记为 `approximate`，ASR 边界标记为 `asr_timestamp`；
- 时间必须满足 `0 <= start_ms < end_ms <= video_duration_ms`；
- Chunk 偏移使用确定性纯函数计算，即使 M0 的偏移为 0 也必须通过测试；
- Fixture 模式禁止读取密钥或发出网络请求，并对同一输入产生确定性输出；
- 429、5xx 和临时网络错误最多重试 3 次并指数退避；
- 视频请求固定 `stream=false`、`reasoning_effort=low`，默认输出预算 4096 tokens；
- 结构化正文为空或无法解析时，以 8192 tokens 额外重试一次，并保留两次原始响应；
- 模型偶发返回数组/字符串类型漂移时，由本地 Schema 层规范化后再入库；
- 每个 Live 阶段成功后立即原子落盘，后续失败时报告 `partial` 并保留已完成阶段；
- `run_report.json` 固定记录真实阶段耗时、缓存命中、逐项 Gate 和 `m0_complete`。

## M0 不包含

- 前端、播放器和 Agent UI；
- 生产级自动长视频切片和任务队列；M0 仅接受预先生成的固定短分片清单；
- 镜头检测、shot-first 和边界审计；
- 向量库、混合检索和 Query Agent；
- EDL/剪辑软件交付；
- 43 分钟剧集处理；
- OpenBMB 工程整体迁移。

## 验收条件

1. Files API 成功上传小于 128MB 的 MP4；
2. 视频模型返回严格结构化的可观察事件；
3. StepAudio ASR 返回分句及毫秒时间戳；
4. StepAudio ASR 返回说话人信息；
5. 所有时间在原片范围内且无负时长；
6. 视觉和对白证据进入同一时间线并可追溯来源；
7. 原始响应与标准化结果分开留档；
8. 脱敏 Fixture 不含密钥、真实 URL、任务 ID 或本地绝对路径；
9. Fixture 模式完全离线且全部测试通过。

Live 报告会逐项输出上述 Gate。只有所有 Gate 为 `true` 时，`m0_complete` 才为 `true`、状态才为 `pass`；任何绕过 Files、缺失时间戳或缺失说话人的诊断路径都保持 `partial`。

## 实测记录（2026-07-16）

### Fixture 基线

- 黄金样片：75,000 ms，21,782,705 bytes，24 fps；
- 视频 SHA256：`9a11b716f750bd61f081c47f2195ca3fdacf8b098891d862c273bfd172c50aa8`；
- 标准化视觉事件：4；
- 标准化 ASR 分句：6；
- 统一证据：10；
- 自动化测试：38 passed；
- Fixture 运行状态：`pass`；
- 公网 ASR 音频：https://github.com/MaoDingA/ShotSeek/releases/download/m0-golden-audio-v1/golden.mp3。

### Live 能力矩阵

| 能力 | 结果 | 判定 |
| --- | --- | --- |
| Step Plan 文本与 JSON Mode | HTTP 200 | 通过 |
| `step-3.7-flash` 图片输入 | HTTP 200 | 通过 |
| `step-3.7-flash` 原生视频输入 | 1～10 秒稳定；75 秒按 8 段得到 23 个事件 | 通过 |
| Step Plan SSE ASR 转写 | HTTP 200，可返回文本 | 通过 |
| SSE ASR 分句时间戳 | 显式 `enable_timestamp=true` 后返回有效毫秒区间 | 通过 |
| SSE ASR 说话人 | 无说话人字段 | 不通过 |
| 标准 Files API | HTTP 402 quota exceeded | BLOCKED |
| 标准异步文件 ASR | HTTP 402 quota exceeded | BLOCKED |

根据 StepFun 技术支持建议，视频请求必须显式传入 `"stream": false`，并先用 1～2 秒超短视频验证。1 秒、2 秒和 2 秒结构化事件请求均为 HTTP 200。时长边界复测如下：

| 片段时长 | 结果 |
| ---: | --- |
| 3 秒 | 通过，1 个事件 |
| 5 秒 | 通过，2 个事件 |
| 10 秒 | 通过；4096-token 输出预算下得到 5 个事件 |
| 15 秒 | HTTP 200，5 个事件；出现 `location` 数组漂移，本地规范化后通过 |
| 20 秒 | HTTP 500 `engine_exception` |
| 30 秒 | HTTP 500 `engine_exception` |

因此 M0 将 10 秒作为保守分片上限。Provider 固定显式非流式请求、`reasoning_effort=low` 和默认 `max_tokens=4096`；若结构化正文为空或不可解析，会以 8192 tokens 重试一次。

### 75 秒真实视频链路

- Run ID：`20260716T094713.480866Z`；
- 输入：75 秒黄金样片，7 个 10 秒片段和 1 个 5 秒片段；
- 视频分片：8/8 成功；
- 标准化视觉事件：23；
- 视觉调用总耗时：171,134 ms；
- 每条事件保留 `chunk_id` 和 `source_start_ms`，可确定性映射回原片时间；
- 随后的标准异步 ASR 提交返回 HTTP 402，因此运行状态正确记录为 `partial`，已完成的视频产物全部保留。

### ASR 时间戳复测

早期 SSE 请求没有显式设置 `enable_timestamp=true`，因此得到的 `start_time/end_time` 为 0。按照官方接口说明补上该字段后：

- 12 秒 PCM：HTTP 200，9 个增量事件中 7 个具有正时长时间戳；
- 75 秒黄金音频：HTTP 200，标准化为 10 个分句、60 个带时间戳片段，覆盖 `5,600—73,840 ms`；
- 即使附带 `enable_speaker_info=true`，SSE 响应仍没有说话人字段；
- 标准异步文件 ASR 仍返回 HTTP 402 quota exceeded。

### 当前最完整 Live 运行

- Run ID：`20260716T101727.546493Z`；
- 输入：75 秒黄金样片；复用同一视频 SHA256 的真实视觉运行，仅 ASR 为本次实时调用；
- 视觉事件：23；
- ASR 分句：10，带时间戳片段 60；
- 统一证据：33，包含 `visual` 与 `dialogue` 两类；
- 时间范围：全部通过 `0 <= start_ms < end_ms <= 75000` 校验；
- 运行产物：清单、三份原始响应、三份标准化数据和报告共 8 个 JSON；
- 通过 Gate：公开授权、文件大小、结构化视觉、ASR 时间戳、统一时间线、时间范围、原始/标准化分离；
- 未通过 Gate：`files_api_upload`、`speaker_info`；
- 状态：`partial`，`m0_complete=false`。

当前 M0 Live 仍保持 **BLOCKED**，原因已收敛为两个外部硬门槛：标准 Files API 上传证据，以及异步 ASR 说话人信息。SSE 真实时间戳和统一时间线已经通过，但不能用 SSE 替代完整 Files + speaker 验收。
