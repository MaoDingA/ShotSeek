import hashlib
import io
import shutil
from pathlib import Path

import pytest

from shotseek.runtime import JobState, RuntimePaths, RuntimeRegistry, store_upload


def _registry(tmp_path: Path) -> tuple[RuntimePaths, RuntimeRegistry]:
    root = Path(__file__).resolve().parents[1]
    runtime = root / "runs" / "tests" / "runtime-contract" / tmp_path.name
    shutil.rmtree(runtime, ignore_errors=True)
    paths = RuntimePaths(root, runtime)
    paths.ensure()
    database = runtime / "runtime.sqlite3"
    database.unlink(missing_ok=True)
    return paths, RuntimeRegistry(database)


def test_upload_storage_is_content_addressed_and_project_locked(tmp_path: Path) -> None:
    paths, _ = _registry(tmp_path)
    payload = b"not-a-real-video-but-valid-storage-contract"
    first = store_upload(paths, io.BytesIO(payload), "sample.mp4")
    second = store_upload(paths, io.BytesIO(payload), "sample.mp4")
    assert first.sha256 == hashlib.sha256(payload).hexdigest()
    assert first.path == second.path
    assert first.created is True
    assert second.created is False
    assert first.path.resolve().is_relative_to(paths.project_root)


def test_runtime_registry_enforces_state_machine_and_records_events(
    tmp_path: Path,
) -> None:
    paths, registry = _registry(tmp_path)
    upload = store_upload(paths, io.BytesIO(b"video"), "demo.mp4")
    video, created = registry.register_video(
        sha256=upload.sha256,
        original_filename=upload.original_filename,
        source_path=str(upload.path.relative_to(paths.project_root)),
        bytes=upload.bytes,
    )
    assert created is True
    duplicate, duplicate_created = registry.register_video(
        sha256=upload.sha256,
        original_filename=upload.original_filename,
        source_path=str(upload.path.relative_to(paths.project_root)),
        bytes=upload.bytes,
    )
    assert duplicate == video
    assert duplicate_created is False

    job = registry.create_job(video.video_id)
    job = registry.transition(job.job_id, JobState.QUEUED, message="queued")
    job = registry.transition(job.job_id, JobState.PROBING, message="probing")
    job = registry.update_progress(
        job.job_id,
        completed_units=1,
        total_units=2,
        progress=0.1,
        message="probe 1/2",
    )
    assert job.completed_units == 1
    assert len(registry.events(job.job_id)) == 4
    with pytest.raises(ValueError):
        registry.transition(job.job_id, JobState.READY)


def test_cancel_and_restart_recovery_are_literal(tmp_path: Path) -> None:
    paths, registry = _registry(tmp_path)
    upload = store_upload(paths, io.BytesIO(b"another-video"), "demo.mov")
    video, _ = registry.register_video(
        sha256=upload.sha256,
        original_filename=upload.original_filename,
        source_path=str(upload.path.relative_to(paths.project_root)),
        bytes=upload.bytes,
    )
    first = registry.create_job(video.video_id)
    registry.transition(first.job_id, JobState.QUEUED)
    registry.transition(first.job_id, JobState.PROBING)
    recovered = registry.recover_incomplete_jobs()
    assert recovered[0].state == JobState.QUEUED
    assert recovered[0].resume_state == JobState.PROBING

    second = registry.create_job(video.video_id)
    registry.transition(second.job_id, JobState.QUEUED)
    registry.request_cancel(second.job_id)
    assert registry.cancel_if_requested(second.job_id) is True
    assert registry.get_job(second.job_id).state == JobState.CANCELLED
