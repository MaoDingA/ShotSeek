# ShotSeek 部署指南

本文面向单机 DGX Spark 或兼容的 Linux 工作站。ShotSeek 的所有持久化数据都写在
项目目录内；上传媒体、SQLite、缓存、模型响应和运行日志不会写到仓库之外。

## 1. 环境要求

- Python 3.11 或更高版本；
- FFmpeg 和 ffprobe；
- 可访问 StepFun API；
- 至少能够解码输入视频并编码 H.264/AAC；
- Node.js 仅在修改并重新构建工作台时需要。

先检查媒体能力：

```bash
python3 --version
ffmpeg -version
ffprobe -version
ffmpeg -hide_banner -encoders | grep -E 'h264_nvenc|libx264'
```

运行时优先使用 `h264_nvenc`；不可用时会明确回退到 `libx264`。回退不会改变
时间线契约，但媒体阶段速度可能下降。

## 2. 安装

在项目根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev,competition]"
cp .env.example .env
```

`.env` 已被 Git 忽略。填写 `STEPFUN_API_KEY`，不要把真实密钥写入命令、日志、
截图或提交记录。

加载环境变量：

```bash
set -a
. ./.env
set +a
```

## 3. 启动 Production Runtime

真实 StepFun 模式：

```bash
.venv/bin/shotseek-runtime \
  --project-root "$(pwd)" \
  --mode live \
  --host 0.0.0.0 \
  --port 8000 \
  --chunk-duration-seconds 30 \
  --vision-workers 3
```

浏览器打开 `http://127.0.0.1:8000`。FastAPI 同时托管工作台和 `/api/v1` API，
生产部署不需要单独运行 Node 服务。

离线演示或测试：

```bash
.venv/bin/shotseek-runtime --project-root "$(pwd)" --mode fixture
```

Fixture 模式不会访问 StepFun，并且模型产物必须显示为 `CACHED`。它只用于
确定性回放，不能作为真实接口验收证据。

## 4. 长视频参数

默认 30 秒视觉切片、3 个网络 Worker。减少长片 API 请求数时可以使用：

```bash
.venv/bin/shotseek-runtime \
  --project-root "$(pwd)" \
  --mode live \
  --chunk-duration-seconds 60 \
  --vision-workers 3
```

`--vision-workers` 限制为 1–4。媒体 Worker 保持单实例，避免多个转码任务争用
硬件编解码器。已标准化为 H.264、720 高、25fps CFR、AAC/无音频的 MP4 可显式
使用 `--proxy-passthrough`；不满足契约时会拒绝直通。

## 5. 健康检查

```bash
curl -fsS http://127.0.0.1:8000/health
```

返回结果应满足：

- `status = ok`；
- `service = shotseek-runtime`；
- `worker_enabled = true`；
- `registry.integrity_check = ok`。

上传接口直接接收视频字节：

```bash
curl --fail --data-binary @episode.mp4 \
  "http://127.0.0.1:8000/api/v1/jobs?filename=episode.mp4"
```

任务进度使用 SSE：

```bash
curl -N "http://127.0.0.1:8000/api/v1/jobs/JOB_ID/events"
```

## 6. 数据目录与备份

默认 Runtime 根目录为 `data/runtime/`：

```text
data/runtime/
├── runtime.sqlite3
├── uploads/
├── cache/
└── videos/{video_id}/
    ├── media/
    ├── chunks/
    ├── raw/
    ├── evidence/
    ├── timeline/
    ├── index/
    ├── previews/
    └── traces/
```

停止服务后备份整个 `data/runtime/` 即可保留 Job、视频索引和证据。目录被 Git
忽略，不应通过源码仓库分发媒体或模型原始响应。

## 7. 工作台构建

仓库已经包含生产静态文件。修改 `apps/web/` 后执行：

```bash
cd apps/web
npm ci
npm run typecheck
npm run build
```

构建产物写入 `shotseek/runtime/static/`。Runtime 已有 READY 视频时可运行
`npm run e2e` 做浏览器闭环检查。

## 8. 常见故障

- `401/403`：检查密钥是否属于正确 StepFun 通道，不要打印密钥。
- `429`：降低 `--vision-workers`；Runtime 会保留真实重试状态。
- Files 仍在处理：保持轮询，不要重复上传同一内容。
- 无音轨：系统记录 `audio_absent`，视觉检索仍可继续。
- 视频时长不一致：系统以视频流帧数校验 CFR，不使用 AAC padding 后的容器时长。
- Job 中断：重启 Runtime 后会从持久化 `resume_state` 恢复。

Runtime 当前未提供多租户身份认证。绑定 `0.0.0.0` 时只应部署在可信网络或放在
带身份认证和 TLS 的反向代理后。

## 9. 部署前只读诊断

安装后、现场演示前或陌生机器部署时先运行：

```bash
.venv/bin/shotseek doctor
```

默认模式：

- 不访问公网，只允许读取本机 `127.0.0.1` 的 Runtime 健康接口；
- 不解析 `.env`，只判断 `STEPFUN_API_KEY` 是否已由进程环境提供；
- 不显示密钥、Authorization 或环境变量值；
- 不创建、删除或修改项目数据；
- 不启动、停止或终止任何服务和进程；
- 不调用 `sudo`，不下载依赖，不自动修复问题。

常用模式：

```bash
# 显示各检查项的脱敏细节
.venv/bin/shotseek doctor --verbose

# 稳定机器可读 JSON
.venv/bin/shotseek doctor --json

# 枚举能力之外，再做一次真实 1 秒 NVENC 合成编码
.venv/bin/shotseek doctor --deep

# 一次低成本 StepFun 文本连通性请求，不上传视频、不启动 ASR
.venv/bin/shotseek doctor --live
```

`--deep` 是唯一会临时写文件的本地检查。输出位于项目 `tmp/doctor-nvenc-*`，探针
结束后自动删除；它不会修改媒体、SQLite、缓存或 Runtime 状态。默认模式只枚举
NVENC/NVDEC，且不会把“编码器名称存在”等同于硬件实际可用。

`--live` 必须在密钥已经导入当前进程后显式执行。请求固定使用 `stream=false`、低
推理和极小输出，只验证 Chat Completions；不使用 Files、视频理解或 ASR。Live
失败会标为 `FAIL`，但不会改变任何离线产物或 Runtime 数据。

磁盘阈值可以通过参数配置：

```bash
.venv/bin/shotseek doctor \
  --disk-warn-gb 20 \
  --disk-fail-gb 5 \
  --runtime-port 8000 \
  --frontend-port 5173 \
  --debug-port 8877
```

等价环境变量为：

```text
SHOTSEEK_DOCTOR_DISK_WARN_GB
SHOTSEEK_DOCTOR_DISK_FAIL_GB
SHOTSEEK_DOCTOR_TIMEOUT_SECONDS
SHOTSEEK_FRONTEND_PORT
```

每个检查项都有唯一 `check_id`，状态只使用：

```text
PASS
WARN
FAIL
SKIP
```

最终状态：

```text
pass
pass_with_warnings
fail
```

退出码仅在最终状态为 `fail` 时为 1；警告和跳过仍返回 0，便于部署脚本区分硬失败。
Runtime 未启动时 `/health` 和静态托管检查为 `SKIP`，而不是让整次 Doctor 失败。
Doctor 只负责诊断；未来如果增加修复能力，应使用独立、显式确认的 `repair` 命令。
