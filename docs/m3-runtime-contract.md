# M3 Production Runtime 契约

本文定义 ShotSeek 生产运行时的公开、可测试契约。运行时生成物默认位于
`data/runtime/`，并被 Git 忽略。

## 路径与安全边界

- 项目根目录是所有持久化写入的唯一边界。
- 上传、临时文件、缓存、SQLite、代理视频和模型原始响应都位于项目树内。
- 上传接口直接流式写入项目内的临时文件，不依赖系统临时目录。
- 文件名会去除目录成分并限制为视频扩展名。
- 视频按 SHA-256 内容寻址；重复上传不会重复存储。

## Job 状态机

```text
CREATED → QUEUED → PROBING → TRANSCODING → EXTRACTING_AUDIO
        → DETECTING_SHOTS → CHUNKING → ANALYZING_VISUAL
        → ANALYZING_ASR → ALIGNING → BUILDING_SCENES
        → INDEXING → READY
```

异常状态为 `RETRYING`、`PARTIAL`、`FAILED` 和 `CANCELLED`。每次状态或进度
变化都追加到 `job_event`，不会覆盖历史。服务重启时，未结束任务回到
`QUEUED`，并从持久化的 `resume_state` 继续。

运行时只有一个媒体 Worker，避免多个转码任务争用单路硬件编解码器。阶段内部
可报告 `completed_units / total_units`；取消请求在阶段之间及进度回调时检查。

## HTTP API

### 上传并创建任务

```http
POST /api/v1/jobs?filename=episode.mp4
Content-Type: video/mp4

<原始视频字节>
```

返回 `202`。相同内容存在活动任务或已完成任务时，返回原任务，并将
`job_reused` 设为 `true`。

### 查询与控制

```text
GET  /api/v1/jobs
GET  /api/v1/jobs/{job_id}
GET  /api/v1/jobs/{job_id}/events
GET  /api/v1/jobs/{job_id}/result
POST /api/v1/jobs/{job_id}/cancel

GET  /api/v1/videos
GET  /api/v1/videos/{video_id}
GET  /api/v1/videos/{video_id}/media
GET  /api/v1/videos/{video_id}/scenes
GET  /api/v1/videos/{video_id}/scenes/{scene_id}
GET  /api/v1/videos/{video_id}/scenes/{scene_id}/preview
GET  /api/v1/videos/{video_id}/scenes/{scene_id}/evidence
POST /api/v1/videos/{video_id}/search
```

`events` 使用 Server-Sent Events，事件 ID 对应 SQLite 的递增 `event_id`；客户
端断线后可使用 `after` 继续读取。非 `READY/PARTIAL` 任务访问结果返回 `409`。

`media` 支持标准单段 HTTP Range 请求并返回 `206`、`Accept-Ranges`、
`Content-Range` 和准确的 `Content-Length`，浏览器可以直接拖动长视频。Scene
预览由时间线中点抽帧生成，不依赖前端硬编码图片。

## 媒体与证据流水线

每个视频保存独立目录：

```text
data/runtime/videos/{video_id}/
├── media/       # source_info、720p CFR proxy、16 kHz MP3
├── chunks/      # 小于 128 MiB 的 10 秒 MP4 与 manifest
├── raw/         # StepFun 原始响应
├── evidence/    # VisualEvent、Utterance
├── timeline/    # shot grid、对齐证据、Scene、审计
├── index/       # SQLite FTS5
└── traces/      # Planner → Retriever → Verifier 轨迹
```

代理优先使用 `h264_nvenc`，不可用时明确回退 `libx264`。视觉切片严格限制在
1–10 秒，默认 10 秒；单片安全上限为 110 MiB。视觉缓存键包含片段内容、模型、
Prompt 版本、推理档位和 Schema 版本。ASR 缓存键包含音频内容、模型和 Schema。

模型时间只作为候选时间。最终 Scene 必须通过镜头网格执行 `shot_first` 对齐，
并保存原始区间、最终区间和帧差。

## LIVE、CACHED 与 fixture

- `live`：真实调用 StepFun Files、Step 3.7 和 StepAudio 2.5 ASR。
- `fixture`：只用于测试和离线演示，使用仓库内脱敏响应，状态必须显示为
  `CACHED`，不得伪装为实时调用。
- 同一个 live 内容缓存命中后也显示 `CACHED`。
- 无音轨视频会明确记录 `audio_absent`，不会伪造对白。

## 启动

确定性 fixture 模式：

```bash
shotseek-runtime --project-root /home/phenom8000/model/spark --mode fixture
```

真实 StepFun 模式：

```bash
export STEPFUN_API_KEY='<secret>'
shotseek-runtime --project-root /home/phenom8000/model/spark --mode live
```

密钥只从环境读取，不写入 Job、Artifact、日志或模型缓存。

## 工作台构建与浏览器验收

React、TypeScript 与 Vite 源码位于 `apps/web/`。生产构建输出到
`shotseek/runtime/static/`，FastAPI 在根路径直接托管，因此部署 Runtime 不要求
目标机器另外运行 Node 服务。

```bash
cd apps/web
npm install
npm run typecheck
npm run build
```

在 Runtime 已启动且存在 READY 视频时，可执行真实 Chromium 端到端检查：

```bash
npm run e2e
```

测试必须完成查询输入、结果卡、关键帧、证据抽屉和 shot-first 边界页验证。

## 当前验收门

- 状态转换非法时拒绝。
- Worker 失败后只重试当前阶段。
- 重启后不重复已完成阶段。
- 上传内容寻址并且不会写出项目根目录。
- 真实 FFmpeg 媒体阶段能生成代理、音频、镜头、切片和索引。
- 搜索必须绑定 `video_id`，证据接口必须返回 shot-first 原始与最终边界。
