import hashlib
import io
import json
import shutil
import threading
import time
from pathlib import Path

from shotseek.runtime import JobState, RuntimePaths, RuntimeRegistry, store_upload
from shotseek.runtime.pipeline import PipelineSettings, ProductionPipeline
from shotseek.schemas import UploadedFile, VisualEvent


def test_live_visual_chunks_use_bounded_parallel_workers(
    tmp_path: Path, monkeypatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    runtime_root = root / "runs" / "tests" / "runtime-visual-parallel" / tmp_path.name
    shutil.rmtree(runtime_root, ignore_errors=True)
    paths = RuntimePaths(root, runtime_root)
    stored = store_upload(paths, io.BytesIO(b"parallel-video"), "parallel.mp4")
    registry = RuntimeRegistry(paths.registry)
    video, _ = registry.register_video(
        sha256=stored.sha256,
        original_filename=stored.original_filename,
        source_path=str(stored.path.relative_to(root)),
        bytes=stored.bytes,
    )
    job = registry.create_job(video.video_id)
    pipeline = ProductionPipeline(
        paths=paths,
        registry=registry,
        settings=PipelineSettings(
            mode="live",
            api_key="fixture-key",
            vision_workers=3,
        ),
    )
    video_root = paths.video_root(video.video_id)
    chunks_dir = video_root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for index in range(4):
        path = chunks_dir / f"chunk_{index:05d}.mp4"
        content = f"chunk-{index}".encode()
        path.write_bytes(content)
        chunks.append(
            {
                "chunk_id": f"chunk_{index:05d}",
                "path": str(path.relative_to(root)),
                "source_start_ms": index * 10_000,
                "source_end_ms": (index + 1) * 10_000,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    (chunks_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "test",
                "cache_key": "parallel-cache",
                "duration_ms": 40_000,
                "chunk_duration_ms": 10_000,
                "chunks": chunks,
            }
        ),
        encoding="utf-8",
    )

    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def fake_upload(path, **kwargs):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return (
            UploadedFile(
                file_id=path.stem,
                file_uri=f"stepfile://{path.stem}",
                filename=path.name,
                bytes=path.stat().st_size,
                sha256=digest,
                status="processed",
            ),
            {"status": "processed"},
        )

    def fake_analyze(file_uri, **kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.08)
            chunk_id = kwargs["chunk_id_override"]
            source_start_ms = kwargs["source_start_ms"]
            return (
                [
                    VisualEvent(
                        event_id=f"{chunk_id}:visual_0001",
                        approx_start_ms=0,
                        approx_end_ms=1_000,
                        summary=f"event for {chunk_id}",
                        confidence=0.9,
                        model=kwargs["model"],
                        chunk_id=chunk_id,
                        source_start_ms=source_start_ms,
                    )
                ],
                {"file_uri": file_uri},
            )
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr("shotseek.runtime.pipeline.upload_video", fake_upload)
    monkeypatch.setattr("shotseek.runtime.pipeline.analyze_video", fake_analyze)
    updates = []
    result = pipeline.run_stage(
        job=job,
        video=video,
        stage=JobState.ANALYZING_VISUAL,
        progress=lambda completed, total, message: updates.append((completed, total)),
    )
    assert maximum_active == 3
    assert updates[-1] == (4, 4)
    assert "LIVE" in result.message
    events = json.loads(
        (video_root / "evidence" / "visual_events.json").read_text(encoding="utf-8")
    )
    assert [item["source_start_ms"] for item in events] == [0, 10_000, 20_000, 30_000]
