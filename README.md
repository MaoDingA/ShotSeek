# ShotSeek

> 用一句话，定位长视频里的准确镜头。

ShotSeek 是面向影视后期的证据对齐场景检索工具。它使用 StepFun 理解画面和对白，在 NVIDIA DGX Spark 上建立帧级、镜头级证据时间线，让自然语言查询返回可验证的时间码、镜头边界和证据引用。

## 核心能力

- 按对白、人物、动作、物体或地点寻找镜头。
- 将模型的近似时间映射到原片 Shot Grid。
- 每个结果保留视觉、对白和镜头引用，可追溯到来源。
- 支持精确对白、视觉、多模态、序数和前后关系查询。
- 查询阶段完全本地运行；StepFun 响应可缓存并离线重放。

## 技术链路

~~~mermaid
flowchart LR
    A[视频] --> B[StepFun 视频理解和 ASR]
    B --> C[DGX Spark Shot Grid]
    C --> D[证据对齐 Scene]
    D --> E[SQLite FTS5 和时间规则]
    E --> F[时间码和证据]
~~~

- **Step 3.7 Flash**：结构化视觉事件。
- **StepAudio 2.5 ASR**：对白、说话人和毫秒时间戳。
- **FFmpeg / DGX Spark**：媒体探测、镜头切点和帧级时间基。
- **SQLite FTS5**：确定性文字、多字段和时间关系检索。

## 当前状态

| 阶段 | 状态 | 验证结果 |
| --- | --- | --- |
| M0 接口与时间线契约 | 完成 | StepFun Files、视频理解、异步 ASR 真实验收通过 |
| M1A Shot Grid 与证据对齐 | 完成 | 25 个 Shot，23/23 视觉事件，7/7 ASR，8 条人工边界审计 |
| M1B Scene Schema | 完成 | 23 个 Scene，证据与 Shot 引用错误为 0 |
| M1C 场景检索 | 完成 | 15/15 查询；Recall@1/3 为 1.0；负例高置信误报为 0 |

当前仓库提供可复现的“接口 → 时间线 → Scene → 搜索”核心。工作台 UI 和长视频产品化不在本版本范围内。

## 快速开始

需要 Python 3.11+ 和 FFmpeg。

~~~bash
python3 scripts/prepare_golden_sample.py
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

.venv/bin/python scripts/run_m0_probe.py --fixture --video samples/golden.mp4

.venv/bin/python scripts/run_m1a.py
.venv/bin/python scripts/run_m1b.py
.venv/bin/python scripts/run_m1c.py

.venv/bin/python scripts/verify_m1_completion.py --output runs/m1/completion-report.json
~~~

M1 使用已脱敏的真实 StepFun Fixture，运行时不会发起网络请求。运行产物写入 runs/，不会提交到 Git。

## 真实 StepFun 验证

复制环境文件并填写自己的密钥：

~~~bash
cp .env.example .env
.venv/bin/python scripts/run_m0_probe.py --live --video samples/golden.mp4
~~~

.env、媒体、数据库、运行报告和内部文档均已忽略。请勿提交密钥或原始素材。

## 测试

~~~bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_repository_hygiene.py
~~~

黄金样片来自 Blender Foundation 的 *Tears of Steel*；来源和许可信息见 samples/README.md。
