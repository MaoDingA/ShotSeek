# ShotSeek

> 一句话，定位长视频里的任意镜头。

ShotSeek 是面向影视后期团队的长视频镜头定位 Agent。它使用 StepFun 多模态模型理解画面、对白与文字，并在 NVIDIA DGX Spark 上完成视频处理和场景索引，让剪辑师能够用自然语言快速找到目标画面。

## 它能做什么

输入一句话：

> 找到女主第一次发现尸体的场景。

ShotSeek 会返回对应的时间码、关键帧、片段预览和匹配依据，并让播放器直接跳转到目标位置。

## 核心能力

- **自然语言找镜头**：用剧情、动作、对白、人物或物体描述目标画面。
- **长视频场景理解**：融合视频内容、语音和画面文字建立结构化时间线。
- **精准时间码定位**：搜索结果直接对应原始视频中的入点和出点。
- **多模态证据**：通过关键帧、对白和文字说明为什么命中。
- **后期友好交付**：支持导出 JSON、XML 和 SRT 等结构化结果。

## 工作方式

~~~mermaid
flowchart LR
    A["上传长视频<br/>可选剧本或字幕"] --> B["DGX Spark<br/>转码、切片、抽帧"]
    B --> C["StepFun<br/>视频理解与语音识别"]
    C --> D["场景时间线<br/>证据对齐与索引"]
    D --> E["自然语言搜索"]
    E --> F["时间码、画面与导出"]
~~~

## 技术栈

### StepFun

- **Step 3.7 Flash**：视频理解、画面文字识别、中文推理与 Agent 编排。
- **StepAudio 2.5 ASR**：对白识别、分句和时间戳。

### NVIDIA DGX Spark

- GPU 加速的视频解码、转码、切片和抽帧。
- 长视频任务调度、场景索引、结果缓存与交付。
- 面向影视素材的本地高性能工作站。

## 使用场景

- 从几十分钟素材中寻找指定剧情节点。
- 根据一句对白定位对应画面。
- 查找包含特定人物、动作、道具或地点的镜头。
- 为剪辑、审片和素材整理建立可检索的视频时间线。

## 项目状态

ShotSeek 正在持续开发中，代码、演示和使用说明将陆续开放。

## M0 契约探针

当前版本已经提供 StepFun 视频理解、StepAudio ASR 与统一毫秒时间线的最小契约探针。离线模式使用脱敏 Fixture，不需要 API Key 或网络：

```bash
python3 scripts/prepare_golden_sample.py
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python scripts/run_m0_probe.py --fixture --video samples/golden.mp4
.venv/bin/python -m pytest -q
```

Live 模式需要在项目内复制 `.env.example` 为 `.env`，填入 `STEPFUN_API_KEY`。黄金样片音频已经作为带署名的公开 Release 资产提供：

```bash
cp .env.example .env
# 编辑 .env，填入 STEPFUN_API_KEY
.venv/bin/python scripts/run_m0_probe.py \
  --live \
  --video samples/golden.mp4
```

默认 `async_file` 通道用于完整 M0 验收，因为它能够返回说话人信息。Step Plan 套餐内的 SSE 通道可用于验证真实 ASR 时间戳；请求会显式发送 `enable_timestamp=true`：

```bash
.venv/bin/python scripts/run_m0_probe.py \
  --live \
  --video samples/golden-stepfun.mp4 \
  --vision-cache-run runs/m0/<vision_run_id> \
  --asr-transport sse
```

当标准 Files API 与异步 ASR 额度恢复后，可以复用同一黄金视频已经获得的真实视觉结果，同时强制执行一次新的 Files 上传。运行清单会明确记录 `files_api_plus_vision_cache`，不会把缓存伪装成新的模型调用：

```bash
.venv/bin/python scripts/run_m0_probe.py \
  --live \
  --video samples/golden-stepfun.mp4 \
  --vision-cache-run runs/m0/<vision_run_id> \
  --require-files-upload
```

这个入口只复用视觉响应；Files 上传和异步 ASR 仍是本次运行的真实请求。

SSE 通道已经取得有效毫秒时间戳，但当前响应不包含说话人字段，因此不会被探针误判为完整 M0 通过。

当标准 Files API 暂时没有额度时，也可以用项目目录内的 JSON 清单提供连续、完整覆盖原片的公网视频分片：

```bash
.venv/bin/python scripts/run_m0_probe.py \
  --live \
  --video samples/golden-stepfun.mp4 \
  --video-chunks .tmp/m0-chunks/manifest.json
```

当前实测稳定分片上限取 10 秒。该入口会在运行清单中明确标记为 `direct_url_chunks`，不会伪装成 Files API 上传成功，也不能替代 M0 对 Files API 的正式验收。

每次运行都会在 `runs/m0/<run_id>/` 中生成原始响应、标准化证据、统一时间线和运行报告；该目录不会提交到 Git。Live 调用对限流和临时服务错误进行有界重试，并在后续阶段失败时保留已完成阶段的真实响应。

可以用一条独立命令按老师的 15 项清单重新审计 Live 运行。命令只有在所有硬门槛、Fixture 来源、测试和 README 状态同时通过时才返回 0：

```bash
.venv/bin/python scripts/verify_m0_completion.py \
  --run runs/m0/<run_id>
```

标准 Files 和异步 ASR 成功后，再从完整 Live 运行自动生成脱敏 Fixture，脚本会拒绝任何 `partial` 运行：

```bash
.venv/bin/python scripts/update_m0_fixtures.py \
  --run runs/m0/<run_id>
```

当前默认 Fixture 的视觉与 SSE ASR 来自真实调用并已脱敏；Files 和异步 ASR 仍明确标记为契约样本，不会伪装成已完成的 Live 证据。

| M0 能力 | 状态 |
| --- | --- |
| 开放授权黄金样片 | 已完成 |
| 毫秒时间线 Schema | 已完成 |
| Files / 视频 / ASR 接口适配 | 已完成 |
| 离线 Fixture 与契约测试 | 已完成，48 passed |
| 公网 ASR 黄金音频 | 已完成 |
| StepFun 鉴权与基础能力验证 | 已完成 |
| Step 3.7 原生视频 | 已通过：1～10 秒稳定；75 秒样片按 8 段处理得到 23 个事件 |
| Step Plan SSE ASR 时间戳 | 已通过：75 秒音频得到 10 个分句、60 个带时间戳片段 |
| 多模态统一时间线 | 已通过：23 条视觉事件 + 10 条对白 = 33 条证据 |
| M0 Live 硬门槛 | BLOCKED：当前账户的标准 Files API 与异步文件 ASR 返回 402，尚缺正式上传证据和说话人信息 |
| shot-first 镜头校准 | 下一阶段 |
