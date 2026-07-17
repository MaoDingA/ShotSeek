import io
import shutil
from pathlib import Path

from shotseek.runtime import (
    JobState,
    RuntimePaths,
    RuntimeRegistry,
    RuntimeWorker,
    StageResult,
    store_upload,
)
from shotseek.runtime.schema import STAGE_STATES


class RecordingExecutor:
    def __init__(self, *, fail_once_at: JobState | None = None) -> None:
        self.calls: list[JobState] = []
        self.fail_once_at = fail_once_at
        self.failed = False

    def run_stage(self, *, job, video, stage, progress):
        self.calls.append(stage)
        progress(1, 2, f"{stage.value} 1/2")
        if stage == self.fail_once_at and not self.failed:
            self.failed = True
            raise RuntimeError("synthetic transient failure")
        progress(2, 2, f"{stage.value} 2/2")
        updates = {
            "duration_ms": 1000,
            "width": 640,
            "height": 360,
            "fps": 25.0,
        }
        return StageResult(
            message=f"{stage.value} 完成",
            video_updates=updates if stage == JobState.PROBING else {},
        )


def _queued_job(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    runtime_root = root / "runs" / "tests" / "runtime-worker" / tmp_path.name
    shutil.rmtree(runtime_root, ignore_errors=True)
    paths = RuntimePaths(root, runtime_root)
    upload = store_upload(paths, io.BytesIO(b"worker-video"), "demo.mp4")
    registry = RuntimeRegistry(paths.registry)
    video, _ = registry.register_video(
        sha256=upload.sha256,
        original_filename=upload.original_filename,
        source_path=str(upload.path.relative_to(root)),
        bytes=upload.bytes,
    )
    job = registry.create_job(video.video_id)
    registry.transition(job.job_id, JobState.QUEUED)
    return registry, video, job


def test_worker_completes_all_stages_and_marks_video_ready(tmp_path: Path) -> None:
    registry, video, job = _queued_job(tmp_path)
    executor = RecordingExecutor()
    result = RuntimeWorker(registry, executor).run_once()
    assert result is not None
    assert result.state == JobState.READY
    assert result.progress == 1.0
    assert executor.calls == list(STAGE_STATES)
    assert registry.get_video(video.video_id).status == "READY"
    assert registry.get_job(job.job_id).state == JobState.READY


def test_worker_retries_failed_stage_without_repeating_previous_stages(
    tmp_path: Path,
) -> None:
    registry, _, job = _queued_job(tmp_path)
    executor = RecordingExecutor(fail_once_at=JobState.CHUNKING)
    result = RuntimeWorker(registry, executor, max_retries=2).run_once()
    assert result.state == JobState.READY
    assert result.retry_count == 1
    assert executor.calls.count(JobState.CHUNKING) == 2
    assert executor.calls.count(JobState.PROBING) == 1
    assert any(
        event.state == JobState.RETRYING for event in registry.events(job.job_id)
    )


def test_worker_resumes_from_persisted_stage(tmp_path: Path) -> None:
    registry, _, job = _queued_job(tmp_path)
    registry.claim_next_job()
    registry.transition(
        job.job_id,
        JobState.CHUNKING,
        progress=0.4,
        message="synthetic restart",
        force=True,
    )
    registry.recover_incomplete_jobs()
    executor = RecordingExecutor()
    result = RuntimeWorker(registry, executor).run_once()
    assert result.state == JobState.READY
    assert executor.calls[0] == JobState.CHUNKING
    assert JobState.PROBING not in executor.calls
