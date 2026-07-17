# ShotSeek

> 用一句话，定位长视频里的准确镜头。

ShotSeek 是面向影视后期的证据对齐场景检索工具。它使用 StepFun 理解画面、对白与查询意图，在 NVIDIA DGX Spark 上建立帧级、镜头级证据时间线，让自然语言查询返回可验证的时间码、完整镜头边界和证据引用。

## 能做什么

- 按对白、人物、动作、物体、地点或中文同义表达寻找镜头。
- 理解“之前、之后、期间、第二次、最后一次”等时间约束。
- 将模型的近似时间映射到原片 Shot Grid，而不是直接相信模型时间码。
- 对 Top 20 候选逐项核对画面、对白、反证和边界，再返回 Top 3。
- 为每次搜索保留 Agent Trace、分项得分和直接证据。
- 通过只读 API 提供视频、场景、搜索、Trace 和 Metrics。

## 技术链路

~~~mermaid
flowchart LR
    A[视频] --> B[StepFun 视频理解和 ASR]
    B --> C[DGX Spark Shot Grid]
    C --> D[证据对齐 Scene]
    Q[自然语言查询] --> P[Query Planner]
    D --> R[FTS5 Top 20 宽召回]
    P --> R
    R --> T[确定性时间与序数运算]
    T --> V[Evidence Verifier]
    V --> O[Top 3 时间码与证据]
~~~

- **Step 3.7 Flash**：结构化视觉事件、复杂 QuerySpec 和候选证据复核。
- **StepAudio 2.5 ASR**：对白、说话人和毫秒时间戳。
- **FFmpeg / DGX Spark**：媒体探测、镜头切点与帧级时间基。
- **SQLite FTS5**：本地宽召回；时间与序数由确定性代码计算。
- **FastAPI**：只读搜索和诊断接口。

## 当前状态

| 阶段 | 状态 | 已验证结果 |
| --- | --- | --- |
| M0 接口与时间线契约 | 完成 | StepFun Files、视频理解、异步 ASR 真实验收通过 |
| M1 Shot Grid、Scene 与检索 | 完成 | 25 个 Shot、23 个 Scene、证据引用错误 0 |
| M2A Query Planner | 完成 | Rule / StepFun / Cache / Fallback；真实 StepFun 与离线重放通过 |
| M2B Agentic Retrieval | 完成 | Top 20 召回、时间关系、序数、反证与候选验证闭环 |
| M2C 只读 API | 完成 | Health、Video、Scene、Search、Trace、Metrics |
| M2 评测 | 完成 | 40/40；Recall@1/3、MRR、Candidate Recall@20 均为 1.0 |

40 条评测包含原有 15 条查询，以及中文同义词、英文同义词、复杂时间关系、序数、否定约束和困难负例。完整 Agent 的负例高置信误报为 0，离线回放结果确定一致。详细契约与消融结果见 [M2 Agentic Retrieval](docs/m2-agentic-retrieval.md)。

## 快速开始

需要 Python 3.11+ 和 FFmpeg。

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

.venv/bin/python scripts/run_m0_probe.py --fixture --video samples/golden.mp4
.venv/bin/python scripts/run_m1a.py
.venv/bin/python scripts/run_m1b.py
.venv/bin/python scripts/run_m1c.py
.venv/bin/python scripts/run_m2_evaluation.py
~~~

启动只读 API：

~~~bash
.venv/bin/python scripts/serve_api.py
~~~

主要接口：

~~~text
GET  /health
GET  /videos
GET  /videos/{video_id}
GET  /videos/{video_id}/scenes
GET  /scenes/{scene_id}
POST /search
GET  /traces/{trace_id}
GET  /metrics
~~~

所有运行产物写入被忽略的 runs/。默认 API 与评测均不访问网络。

## 真实 StepFun 验证

复制环境文件并填写自己的密钥：

~~~bash
cp .env.example .env
.venv/bin/python scripts/run_m0_probe.py --live --video samples/golden.mp4
~~~

M2 已分别完成复杂查询规划和候选验证的真实 StepFun 调用，并将脱敏响应保存为离线 Fixture。模型只解析查询或验证给定候选，不负责直接选择最终 Scene，也不能改写时间码和镜头边界。

.env、媒体、数据库、运行报告和内部文档均已忽略。请勿提交密钥或原始素材。

## 验收

~~~bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_m2_evaluation.py
.venv/bin/python scripts/verify_m2_completion.py
.venv/bin/python scripts/check_repository_hygiene.py
~~~

黄金样片来自 Blender Foundation 的 *Tears of Steel*；来源和许可信息见 [samples/README.md](samples/README.md)。

## 当前边界

本版本聚焦“查询 → 准确场景 → 直接证据 → 时间码”的后端核心，不包含前端工作台、多集管理、自动剪片、视频生成、人脸实名识别或 DaVinci Resolve 直接控制。
