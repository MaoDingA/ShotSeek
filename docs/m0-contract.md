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
- Live 模式下公网可访问的 MP3/WAV/OGG/PCM 音频 URL；
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

M0 显式区分三条接口通道，不能用一个 Base URL 覆盖全部服务：

- Files API：`https://api.stepfun.com/v1`；
- Chat Completions：`https://api.stepfun.com/step_plan/v1`；
- 异步文件 ASR：`https://api.stepfun.com/v1`。

对应环境变量为 `STEPFUN_FILES_BASE_URL`、`STEPFUN_CHAT_BASE_URL` 和 `STEPFUN_ASR_BASE_URL`。

## 冻结契约

- 视觉时间使用 `approx_start_ms/approx_end_ms`，不得伪装成最终镜头边界；
- ASR 使用服务返回的句级和字词级毫秒时间戳；
- 统一证据使用 `start_ms/end_ms`，并保留 `source_ref`；
- 视觉边界标记为 `approximate`，ASR 边界标记为 `asr_timestamp`；
- 时间必须满足 `0 <= start_ms < end_ms <= video_duration_ms`；
- Chunk 偏移使用确定性纯函数计算，即使 M0 的偏移为 0 也必须通过测试；
- Fixture 模式禁止读取密钥或发出网络请求，并对同一输入产生确定性输出；
- 429、5xx 和临时网络错误最多重试 3 次并指数退避；
- 每个 Live 阶段成功后立即原子落盘，后续失败时报告 `partial` 并保留已完成阶段。

## M0 不包含

- 前端、播放器和 Agent UI；
- 长视频切片和生产任务队列；
- 镜头检测、shot-first 和边界审计；
- 向量库、混合检索和 Query Agent；
- EDL/剪辑软件交付；
- 43 分钟剧集处理；
- OpenBMB 工程整体迁移。

## 验收条件

1. Files API 成功上传小于 128MB 的 MP4；
2. 视频模型返回严格结构化的可观察事件；
3. StepAudio ASR 返回分句及毫秒时间戳；
4. 所有时间在原片范围内且无负时长；
5. 视觉和对白证据进入同一时间线并可追溯来源；
6. 原始响应与标准化结果分开留档；
7. 脱敏 Fixture 不含密钥、真实 URL、任务 ID 或本地绝对路径；
8. Fixture 模式完全离线且全部测试通过。

## 实测记录

### Fixture 基线（2026-07-16）

- 黄金样片：75,000 ms，21,782,705 bytes，24 fps；
- 视频 SHA256：`9a11b716f750bd61f081c47f2195ca3fdacf8b098891d862c273bfd172c50aa8`；
- 标准化视觉事件：4；
- 标准化 ASR 分句：6；
- 统一证据：10；
- 自动化测试：25 passed；
- Fixture 运行状态：`pass`；
- 公网 ASR 音频：https://github.com/MaoDingA/ShotSeek/releases/download/m0-golden-audio-v1/golden.mp3；

### Live 能力诊断（2026-07-16）

当前密钥已完成脱敏实测，结论如下：

| 能力 | 结果 | 判定 |
| --- | --- | --- |
| Step Plan 文本与 JSON Mode | HTTP 200 | 通过 |
| `step-3.7-flash` 图片输入 | HTTP 200 | 通过 |
| `step-3.7-flash` 原生视频输入 | HTTP 500 `engine_exception` | BLOCKED |
| Step Plan SSE ASR 转写 | HTTP 200，可返回文本 | 仅基础能力通过 |
| SSE ASR 分句时间戳 | `start_time/end_time` 均为 0 | 不通过 |
| SSE ASR 说话人 | 无说话人字段 | 不通过 |
| 标准 Files API | HTTP 402 quota exceeded | BLOCKED |
| 标准异步文件 ASR | HTTP 402 quota exceeded | BLOCKED |

视频输入已分别使用 75 秒和 5 秒 H.264/AAC（或无音频）MP4 测试，并验证了公网直链、`video/mp4`、`faststart`、`Range: bytes` 和 206 响应，结果仍为服务端 `engine_exception`。因此当前 M0 Live 状态必须保持 **BLOCKED**，不能用 SSE 零时间戳或本地推导时间窗伪装成硬门槛通过。

解除阻塞需要：标准开放平台 Files/异步文件 ASR 可用额度，以及 StepFun 原生视频服务恢复或官方确认可用调用方式。解除后重新运行原始 `--live` 命令即可，工程无需再混用 Base URL。
