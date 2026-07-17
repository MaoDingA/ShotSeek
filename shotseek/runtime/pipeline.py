"""Idempotent production media pipeline used by the Runtime Worker."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from shotseek.m0 import ensure_within_project, probe_video
from shotseek.media.probe import probe_video_contract
from shotseek.media.schema import (
    AlignedVisualEvent,
    ContextualizedUtterance,
    Shot,
    VideoContract,
)
from shotseek.media.shots import build_shot_grid, detect_shot_boundaries
from shotseek.providers.stepfun import DEFAULT_ASR_MODEL, DEFAULT_VISION_MODEL
from shotseek.providers.stepfun.asr import normalize_asr_response
from shotseek.providers.stepfun.asr_sse import run_sse_asr_bytes
from shotseek.providers.stepfun.files import upload_video
from shotseek.providers.stepfun.vision import (
    VISION_PROMPT_VERSION,
    VISION_SCHEMA_VERSION,
    analyze_video,
    normalize_vision_bundle,
)
from shotseek.retrieval.sqlite_index import build_index
from shotseek.runtime.paths import RuntimePaths
from shotseek.runtime.registry import RuntimeRegistry
from shotseek.runtime.schema import JobRecord, JobState, VideoRecord
from shotseek.runtime.worker import ProgressCallback, StageResult
from shotseek.scenes.builder import build_scenes, validate_scene_references
from shotseek.scenes.schema import Scene
from shotseek.schemas import Utterance, VisualEvent
from shotseek.timeline.alignment import align_visual_events, contextualize_utterances

PIPELINE_VERSION = "m3-media-pipeline-v1"
ASR_SCHEMA_VERSION = "stepfun-sse-asr-v1"


@dataclass(frozen=True)
class PipelineSettings:
    mode: Literal["live", "fixture"] = "fixture"
    api_key: str | None = None
    proxy_height: int = 720
    proxy_fps: int = 25
    proxy_bitrate: str = "3M"
    chunk_duration_ms: int = 10_000
    chunk_max_bytes: int = 110 * 1024 * 1024
    shot_threshold: float = 0.30
    shot_min_gap_frames: int = 6
    reasoning_effort: Literal["low", "medium"] = "low"
    vision_model: str = DEFAULT_VISION_MODEL
    asr_model: str = DEFAULT_ASR_MODEL

    def __post_init__(self) -> None:
        if self.mode == "live" and not (self.api_key or "").strip():
            raise ValueError("live pipeline requires a StepFun API key")
        if not 1_000 <= self.chunk_duration_ms <= 10_000:
            raise ValueError("StepFun chunk duration must be 1-10 seconds")
        if self.proxy_fps <= 0 or self.proxy_height <= 0:
            raise ValueError("invalid proxy settings")


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_key(*values: Any) -> str:
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ProductionPipeline:
    def __init__(
        self,
        *,
        paths: RuntimePaths,
        registry: RuntimeRegistry,
        settings: PipelineSettings,
    ) -> None:
        self.paths = paths
        self.registry = registry
        self.settings = settings
        self.paths.ensure()

    def _root(self, video_id: str) -> Path:
        root = self.paths.video_root(video_id)
        for name in ("media", "chunks", "evidence", "timeline", "index", "raw"):
            (root / name).mkdir(parents=True, exist_ok=True)
        return root

    def _source(self, video: VideoRecord) -> Path:
        return ensure_within_project(
            self.paths.project_root,
            self.paths.project_root / video.source_path,
        )

    def _relative(self, path: Path) -> str:
        return str(ensure_within_project(self.paths.project_root, path).relative_to(self.paths.project_root))

    def _record_artifact(
        self,
        video: VideoRecord,
        *,
        kind: str,
        path: Path,
        status: str = "GENERATED",
        model: str | None = None,
        prompt_version: str | None = None,
        key_parts: tuple[Any, ...] = (),
    ) -> None:
        self.registry.add_artifact(
            video_id=video.video_id,
            kind=kind,
            path=self._relative(path),
            cache_key=_cache_key(video.sha256, kind, PIPELINE_VERSION, *key_parts),
            schema_version=PIPELINE_VERSION,
            status=status,
            provider="stepfun" if model else "shotseek",
            model=model,
            prompt_version=prompt_version,
        )

    def _run_ffmpeg(self, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", *arguments],
            cwd=self.paths.project_root,
            check=True,
            capture_output=True,
            text=True,
        )

    def run_stage(
        self,
        *,
        job: JobRecord,
        video: VideoRecord,
        stage: JobState,
        progress: ProgressCallback,
    ) -> StageResult:
        handlers = {
            JobState.PROBING: self._probe,
            JobState.TRANSCODING: self._transcode,
            JobState.EXTRACTING_AUDIO: self._extract_audio,
            JobState.DETECTING_SHOTS: self._detect_shots,
            JobState.CHUNKING: self._chunk,
            JobState.ANALYZING_VISUAL: self._analyze_visual,
            JobState.ANALYZING_ASR: self._analyze_asr,
            JobState.ALIGNING: self._align,
            JobState.BUILDING_SCENES: self._build_scenes,
            JobState.INDEXING: self._index,
        }
        if stage not in handlers:
            raise ValueError(f"unsupported pipeline stage: {stage.value}")
        return handlers[stage](job, video, progress)

    def _probe(
        self, _: JobRecord, video: VideoRecord, progress: ProgressCallback
    ) -> StageResult:
        progress(0, 1, "读取媒体参数")
        info = probe_video(self.paths.project_root, self._source(video))
        output = self._root(video.video_id) / "media" / "source_info.json"
        _dump_json(output, info.model_dump(mode="json"))
        self._record_artifact(video, kind="source_info", path=output)
        progress(1, 1, "媒体参数读取完成")
        return StageResult(
            message="媒体参数读取完成",
            video_updates={
                "duration_ms": info.duration_ms,
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "status": "PROCESSING",
            },
        )

    def _transcode(
        self, _: JobRecord, video: VideoRecord, progress: ProgressCallback
    ) -> StageResult:
        root = self._root(video.video_id)
        source = self._source(video)
        proxy = root / "media" / "proxy.mp4"
        partial = root / "media" / ".proxy.part.mp4"
        meta = root / "media" / "proxy_info.json"
        expected_key = _cache_key(
            video.sha256,
            PIPELINE_VERSION,
            self.settings.proxy_height,
            self.settings.proxy_fps,
            self.settings.proxy_bitrate,
        )
        backend = "cached"
        if not proxy.is_file() or not meta.is_file() or _load_json(meta).get("cache_key") != expected_key:
            partial.unlink(missing_ok=True)
            progress(0, 1, "生成 720p CFR 代理视频")
            common = [
                "-y", "-i", str(source),
                "-map", "0:v:0", "-map", "0:a:0?",
                "-vf", f"scale=-2:{self.settings.proxy_height},fps={self.settings.proxy_fps}",
                "-c:a", "aac", "-b:a", "96k",
                "-movflags", "+faststart",
            ]
            gop = self.settings.proxy_fps * (self.settings.chunk_duration_ms // 1000)
            try:
                if os.environ.get("SHOTSEEK_DISABLE_NVENC") == "1":
                    raise RuntimeError("NVENC disabled")
                self._run_ffmpeg([
                    *common,
                    "-c:v", "h264_nvenc", "-preset", "p4",
                    "-b:v", self.settings.proxy_bitrate,
                    "-g", str(gop), "-forced-idr", "1",
                    str(partial),
                ])
                backend = "h264_nvenc"
            except (subprocess.CalledProcessError, RuntimeError):
                partial.unlink(missing_ok=True)
                self._run_ffmpeg([
                    *common,
                    "-c:v", "libx264", "-preset", "veryfast",
                    "-b:v", self.settings.proxy_bitrate,
                    "-g", str(gop), "-keyint_min", str(gop),
                    "-sc_threshold", "0",
                    str(partial),
                ])
                backend = "libx264"
            partial.replace(proxy)
            contract = probe_video_contract(self.paths.project_root, proxy)
            _dump_json(
                meta,
                {
                    "schema_version": PIPELINE_VERSION,
                    "cache_key": expected_key,
                    "backend": backend,
                    "video_contract": contract.model_dump(mode="json"),
                },
            )
        contract = VideoContract.model_validate(_load_json(meta)["video_contract"])
        self._record_artifact(video, kind="proxy_video", path=proxy, key_parts=(expected_key,))
        progress(1, 1, f"代理视频完成（{backend}）")
        return StageResult(
            message=f"代理视频完成（{backend}）",
            video_updates={
                "proxy_path": self._relative(proxy),
                "duration_ms": contract.duration_ms,
                "width": contract.width,
                "height": contract.height,
                "fps": contract.fps_num / contract.fps_den,
            },
        )

    def _extract_audio(
        self, _: JobRecord, video: VideoRecord, progress: ProgressCallback
    ) -> StageResult:
        root = self._root(video.video_id)
        source_info = _load_json(root / "media" / "source_info.json")
        audio = root / "media" / "audio.mp3"
        if source_info.get("audio_codec") is None:
            marker = root / "media" / "audio_absent.json"
            _dump_json(marker, {"schema_version": PIPELINE_VERSION, "audio": False})
            self._record_artifact(video, kind="audio_absent", path=marker)
            progress(1, 1, "输入视频没有音轨")
            return StageResult(message="输入视频没有音轨")
        partial = root / "media" / ".audio.part.mp3"
        if not audio.is_file():
            progress(0, 1, "提取单声道音频")
            partial.unlink(missing_ok=True)
            self._run_ffmpeg([
                "-y", "-i", str(root / "media" / "proxy.mp4"),
                "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "libmp3lame", "-b:a", "64k", str(partial),
            ])
            partial.replace(audio)
        self._record_artifact(video, kind="audio", path=audio)
        progress(1, 1, "音频提取完成")
        return StageResult(
            message="音频提取完成",
            video_updates={"audio_path": self._relative(audio)},
        )

    def _detect_shots(
        self, _: JobRecord, video: VideoRecord, progress: ProgressCallback
    ) -> StageResult:
        root = self._root(video.video_id)
        proxy = root / "media" / "proxy.mp4"
        contract = VideoContract.model_validate(
            _load_json(root / "media" / "proxy_info.json")["video_contract"]
        )
        progress(0, 1, "检测镜头切点")
        boundaries = detect_shot_boundaries(
            self.paths.project_root,
            proxy,
            contract,
            threshold=self.settings.shot_threshold,
            min_gap_frames=self.settings.shot_min_gap_frames,
        )
        shots = build_shot_grid(contract, boundaries)
        boundary_path = root / "timeline" / "shot_boundaries.json"
        shots_path = root / "timeline" / "shots.json"
        _dump_json(boundary_path, [item.model_dump(mode="json") for item in boundaries])
        _dump_json(shots_path, [item.model_dump(mode="json") for item in shots])
        self._record_artifact(video, kind="shot_grid", path=shots_path)
        progress(1, 1, f"检测到 {len(shots)} 个镜头")
        return StageResult(message=f"镜头检测完成：{len(shots)} 个镜头")

    def _chunk(
        self, _: JobRecord, video: VideoRecord, progress: ProgressCallback
    ) -> StageResult:
        root = self._root(video.video_id)
        chunks_dir = root / "chunks"
        manifest_path = chunks_dir / "manifest.json"
        proxy_info = _load_json(root / "media" / "proxy_info.json")
        duration_ms = int(proxy_info["video_contract"]["duration_ms"])
        expected_key = _cache_key(video.sha256, PIPELINE_VERSION, self.settings.chunk_duration_ms)
        cached = manifest_path.is_file() and _load_json(manifest_path).get("cache_key") == expected_key
        if not cached:
            for old in chunks_dir.glob("chunk_*.mp4"):
                old.unlink()
            progress(0, 1, "切分 StepFun 视觉片段")
            self._run_ffmpeg([
                "-y", "-i", str(root / "media" / "proxy.mp4"),
                "-map", "0:v:0", "-an", "-c", "copy",
                "-f", "segment",
                "-segment_time", f"{self.settings.chunk_duration_ms / 1000:.3f}",
                "-reset_timestamps", "1",
                str(chunks_dir / "chunk_%05d.mp4"),
            ])
            files = sorted(chunks_dir.glob("chunk_*.mp4"))
            if not files:
                raise RuntimeError("ffmpeg did not produce video chunks")
            chunks = []
            for index, path in enumerate(files):
                size = path.stat().st_size
                if size >= self.settings.chunk_max_bytes:
                    raise ValueError(f"StepFun chunk exceeds safety limit: {path.name}")
                start_ms = index * self.settings.chunk_duration_ms
                end_ms = min(duration_ms, start_ms + self.settings.chunk_duration_ms)
                chunks.append({
                    "chunk_id": f"chunk_{index:05d}",
                    "path": self._relative(path),
                    "source_start_ms": start_ms,
                    "source_end_ms": end_ms,
                    "bytes": size,
                    "sha256": _sha256_file(path),
                })
            _dump_json(
                manifest_path,
                {
                    "schema_version": PIPELINE_VERSION,
                    "cache_key": expected_key,
                    "duration_ms": duration_ms,
                    "chunk_duration_ms": self.settings.chunk_duration_ms,
                    "chunks": chunks,
                },
            )
        manifest = _load_json(manifest_path)
        self._record_artifact(video, kind="chunk_manifest", path=manifest_path, key_parts=(expected_key,))
        progress(1, 1, f"视觉切片完成：{len(manifest['chunks'])} 个")
        return StageResult(message=f"视觉切片完成：{len(manifest['chunks'])} 个")

    @staticmethod
    def _clip_events(events: list[VisualEvent], duration_ms: int) -> list[VisualEvent]:
        clipped: list[VisualEvent] = []
        for event in events:
            global_start = event.source_start_ms + event.approx_start_ms
            global_end = min(duration_ms, event.source_start_ms + event.approx_end_ms)
            if global_start >= duration_ms or global_end <= global_start:
                continue
            payload = event.model_dump(mode="json")
            payload["approx_end_ms"] = global_end - event.source_start_ms
            clipped.append(VisualEvent.model_validate(payload))
        return clipped

    def _analyze_visual(
        self, _: JobRecord, video: VideoRecord, progress: ProgressCallback
    ) -> StageResult:
        root = self._root(video.video_id)
        manifest = _load_json(root / "chunks" / "manifest.json")
        duration_ms = int(manifest["duration_ms"])
        raw_items: list[dict[str, Any]] = []
        events: list[VisualEvent] = []
        cache_hits = 0
        chunks = manifest["chunks"]
        if self.settings.mode == "fixture":
            fixture = _load_json(
                self.paths.project_root / "tests" / "fixtures" / "stepfun" / "vision_response.sample.json"
            )
            events = normalize_vision_bundle(fixture, model=self.settings.vision_model)
            raw_items = [{"mode": "fixture", "response": fixture}]
            cache_hits = len(chunks)
            progress(len(chunks), len(chunks), "使用脱敏 StepFun 视觉 fixture")
        else:
            cache_dir = self.paths.root / "cache" / "visual"
            cache_dir.mkdir(parents=True, exist_ok=True)
            for index, chunk in enumerate(chunks, start=1):
                key = _cache_key(
                    chunk["sha256"],
                    self.settings.vision_model,
                    VISION_PROMPT_VERSION,
                    self.settings.reasoning_effort,
                    VISION_SCHEMA_VERSION,
                )
                cache_path = cache_dir / f"{key}.json"
                if cache_path.is_file():
                    cached = _load_json(cache_path)
                    chunk_events = [VisualEvent.model_validate(item) for item in cached["events"]]
                    raw = cached["raw"]
                    cache_hits += 1
                else:
                    chunk_path = ensure_within_project(
                        self.paths.project_root,
                        self.paths.project_root / chunk["path"],
                    )
                    uploaded, upload_raw = upload_video(
                        chunk_path,
                        api_key=self.settings.api_key or "",
                    )
                    chunk_events, vision_raw = analyze_video(
                        uploaded.file_uri,
                        api_key=self.settings.api_key or "",
                        model=self.settings.vision_model,
                        reasoning_effort=self.settings.reasoning_effort,
                        chunk_id_override=chunk["chunk_id"],
                        source_start_ms=int(chunk["source_start_ms"]),
                    )
                    raw = {"upload": upload_raw, "vision": vision_raw}
                    _dump_json(
                        cache_path,
                        {
                            "schema_version": VISION_SCHEMA_VERSION,
                            "events": [item.model_dump(mode="json") for item in chunk_events],
                            "raw": raw,
                        },
                    )
                events.extend(chunk_events)
                raw_items.append({"chunk_id": chunk["chunk_id"], "response": raw})
                progress(index, len(chunks), f"视觉分析 {index}/{len(chunks)}")
        events = self._clip_events(events, duration_ms)
        if not events:
            raise RuntimeError("StepFun vision returned no in-range events")
        events_path = root / "evidence" / "visual_events.json"
        raw_path = root / "raw" / "vision.json"
        _dump_json(events_path, [item.model_dump(mode="json") for item in events])
        _dump_json(raw_path, {"mode": self.settings.mode, "chunks": raw_items})
        status = "CACHED" if cache_hits == len(chunks) else "LIVE"
        self._record_artifact(
            video,
            kind="visual_events",
            path=events_path,
            status=status,
            model=self.settings.vision_model,
            prompt_version=VISION_PROMPT_VERSION,
            key_parts=(manifest["cache_key"], self.settings.reasoning_effort),
        )
        return StageResult(message=f"视觉事件完成：{len(events)} 条（{status}）")

    @staticmethod
    def _clip_utterances(items: list[Utterance], duration_ms: int) -> list[Utterance]:
        result: list[Utterance] = []
        for item in items:
            if item.start_ms >= duration_ms:
                continue
            end_ms = min(item.end_ms, duration_ms)
            if end_ms <= item.start_ms:
                continue
            payload = item.model_dump(mode="json")
            payload["end_ms"] = end_ms
            payload["words"] = [
                word for word in payload["words"] if word["end_ms"] <= end_ms
            ]
            result.append(Utterance.model_validate(payload))
        return result

    def _analyze_asr(
        self, _: JobRecord, video: VideoRecord, progress: ProgressCallback
    ) -> StageResult:
        root = self._root(video.video_id)
        duration_ms = int(_load_json(root / "chunks" / "manifest.json")["duration_ms"])
        audio = root / "media" / "audio.mp3"
        raw: dict[str, Any]
        status: str
        if not audio.is_file():
            utterances: list[Utterance] = []
            raw = {"mode": "no_audio"}
            status = "GENERATED"
            progress(1, 1, "无音轨，跳过 ASR")
        elif self.settings.mode == "fixture":
            fixture = _load_json(
                self.paths.project_root / "tests" / "fixtures" / "stepfun" / "asr_response.sample.json"
            )
            utterances = normalize_asr_response(fixture["result"])
            raw = {"mode": "fixture", "response": fixture}
            status = "CACHED"
            progress(1, 1, "使用脱敏 StepFun ASR fixture")
        else:
            key = _cache_key(_sha256_file(audio), self.settings.asr_model, ASR_SCHEMA_VERSION)
            cache_path = self.paths.root / "cache" / "asr" / f"{key}.json"
            if cache_path.is_file():
                cached = _load_json(cache_path)
                utterances = [Utterance.model_validate(item) for item in cached["utterances"]]
                raw = cached["raw"]
                status = "CACHED"
            else:
                progress(0, 1, "调用 StepFun ASR")
                utterances, raw = run_sse_asr_bytes(
                    audio.read_bytes(),
                    audio_format="mp3",
                    api_key=self.settings.api_key or "",
                    model=self.settings.asr_model,
                )
                _dump_json(
                    cache_path,
                    {
                        "schema_version": ASR_SCHEMA_VERSION,
                        "utterances": [item.model_dump(mode="json") for item in utterances],
                        "raw": raw,
                    },
                )
                status = "LIVE"
            progress(1, 1, f"ASR 完成（{status}）")
        utterances = self._clip_utterances(utterances, duration_ms)
        utterances_path = root / "evidence" / "utterances.json"
        raw_path = root / "raw" / "asr.json"
        _dump_json(utterances_path, [item.model_dump(mode="json") for item in utterances])
        _dump_json(raw_path, raw)
        self._record_artifact(
            video,
            kind="utterances",
            path=utterances_path,
            status=status,
            model=self.settings.asr_model,
            key_parts=(duration_ms,),
        )
        return StageResult(message=f"ASR 完成：{len(utterances)} 条（{status}）")

    def _align(
        self, _: JobRecord, video: VideoRecord, progress: ProgressCallback
    ) -> StageResult:
        root = self._root(video.video_id)
        contract = VideoContract.model_validate(
            _load_json(root / "media" / "proxy_info.json")["video_contract"]
        )
        shots = [Shot.model_validate(item) for item in _load_json(root / "timeline" / "shots.json")]
        events = [VisualEvent.model_validate(item) for item in _load_json(root / "evidence" / "visual_events.json")]
        utterances = [Utterance.model_validate(item) for item in _load_json(root / "evidence" / "utterances.json")]
        fps = Fraction(contract.fps_num, contract.fps_den)
        aligned = align_visual_events(events, shots, fps=fps, frame_count=contract.frame_count)
        contextualized = contextualize_utterances(
            utterances, shots, fps=fps, frame_count=contract.frame_count
        )
        aligned_path = root / "timeline" / "aligned_visual_events.json"
        dialogue_path = root / "timeline" / "contextualized_utterances.json"
        _dump_json(aligned_path, [item.model_dump(mode="json") for item in aligned])
        _dump_json(dialogue_path, [item.model_dump(mode="json") for item in contextualized])
        self._record_artifact(video, kind="aligned_timeline", path=aligned_path)
        progress(1, 1, "证据已映射到真实镜头边界")
        return StageResult(message=f"时间线对齐完成：{len(aligned)} 个事件")

    def _build_scenes(
        self, _: JobRecord, video: VideoRecord, progress: ProgressCallback
    ) -> StageResult:
        root = self._root(video.video_id)
        aligned = [AlignedVisualEvent.model_validate(item) for item in _load_json(root / "timeline" / "aligned_visual_events.json")]
        utterances = [ContextualizedUtterance.model_validate(item) for item in _load_json(root / "timeline" / "contextualized_utterances.json")]
        shots = [Shot.model_validate(item) for item in _load_json(root / "timeline" / "shots.json")]
        scenes = build_scenes(aligned, utterances)
        audit = validate_scene_references(scenes, aligned, utterances, shots)
        if not audit["pass"]:
            raise RuntimeError("scene reference audit failed")
        scenes_path = root / "timeline" / "scenes.json"
        audit_path = root / "timeline" / "scene_audit.json"
        _dump_json(scenes_path, [item.model_dump(mode="json") for item in scenes])
        _dump_json(audit_path, audit)
        self._record_artifact(video, kind="scenes", path=scenes_path)
        progress(1, 1, f"构建 {len(scenes)} 个证据场景")
        return StageResult(
            message=f"Scene 构建完成：{len(scenes)} 个",
            video_updates={"scene_count": len(scenes)},
        )

    def _index(
        self, _: JobRecord, video: VideoRecord, progress: ProgressCallback
    ) -> StageResult:
        root = self._root(video.video_id)
        scenes = [Scene.model_validate(item) for item in _load_json(root / "timeline" / "scenes.json")]
        utterances = [ContextualizedUtterance.model_validate(item) for item in _load_json(root / "timeline" / "contextualized_utterances.json")]
        database = root / "index" / "search.sqlite3"
        metrics = build_index(database, scenes, utterances)
        metrics_path = root / "index" / "metrics.json"
        _dump_json(metrics_path, metrics)
        if not metrics["integrity_check_pass"]:
            raise RuntimeError("search index integrity check failed")
        self._record_artifact(video, kind="search_index", path=database)
        progress(1, 1, "搜索索引构建完成")
        return StageResult(
            message=f"索引完成：{len(scenes)} 个场景",
            video_updates={
                "scene_count": len(scenes),
                "search_db_path": self._relative(database),
            },
        )
