"""Single-worker runtime orchestration with retries, cancellation and resume."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from shotseek.runtime.registry import RuntimeRegistry
from shotseek.runtime.schema import JobRecord, JobState, STAGE_STATES, VideoRecord


@dataclass(frozen=True)
class StageResult:
    message: str
    completed_units: int = 1
    total_units: int = 1
    video_updates: dict[str, Any] = field(default_factory=dict)


ProgressCallback = Callable[[int, int, str], None]


class StageExecutor(Protocol):
    def run_stage(
        self,
        *,
        job: JobRecord,
        video: VideoRecord,
        stage: JobState,
        progress: ProgressCallback,
    ) -> StageResult: ...


class RuntimeWorker:
    """Runs exactly one media job at a time and persists every boundary."""

    def __init__(
        self,
        registry: RuntimeRegistry,
        executor: StageExecutor,
        *,
        max_retries: int = 2,
        poll_interval: float = 0.25,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.registry = registry
        self.executor = executor
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    @staticmethod
    def _stage_progress(stage: JobState, fraction: float = 1.0) -> float:
        index = STAGE_STATES.index(stage)
        bounded = min(1.0, max(0.0, fraction))
        return (index + bounded) / len(STAGE_STATES)

    @staticmethod
    def _next_state(stage: JobState) -> JobState:
        index = STAGE_STATES.index(stage)
        if index + 1 == len(STAGE_STATES):
            return JobState.READY
        return STAGE_STATES[index + 1]

    def _progress_callback(self, job_id: str, stage: JobState) -> ProgressCallback:
        def update(completed: int, total: int, message: str) -> None:
            if total < 0 or completed < 0 or (total and completed > total):
                raise ValueError("invalid stage progress")
            fraction = completed / total if total else 0.0
            self.registry.update_progress(
                job_id,
                completed_units=completed,
                total_units=total,
                progress=self._stage_progress(stage, fraction),
                message=message,
            )
            self.registry.cancel_if_requested(job_id)

        return update

    def _execute_claimed(self, claimed: JobRecord) -> JobRecord:
        current = claimed
        while current.state in STAGE_STATES:
            if self.registry.cancel_if_requested(current.job_id):
                result = self.registry.get_job(current.job_id)
                if result is None:
                    raise KeyError(current.job_id)
                return result
            video = self.registry.get_video(current.video_id)
            if video is None:
                return self.registry.transition(
                    current.job_id,
                    JobState.FAILED,
                    message="视频记录不存在",
                    error_code="VIDEO_NOT_FOUND",
                    force=True,
                )
            stage = current.state
            try:
                result = self.executor.run_stage(
                    job=current,
                    video=video,
                    stage=stage,
                    progress=self._progress_callback(current.job_id, stage),
                )
                if result.video_updates:
                    self.registry.update_video(video.video_id, **result.video_updates)
                if self.registry.cancel_if_requested(current.job_id):
                    cancelled = self.registry.get_job(current.job_id)
                    if cancelled is None:
                        raise KeyError(current.job_id)
                    return cancelled
                next_state = self._next_state(stage)
                current = self.registry.transition(
                    current.job_id,
                    next_state,
                    progress=self._stage_progress(stage),
                    completed_units=result.completed_units,
                    total_units=result.total_units,
                    message=result.message,
                )
            except Exception as error:  # stage boundary is the durability boundary
                latest = self.registry.get_job(current.job_id) or current
                if latest.state == JobState.CANCELLED:
                    return latest
                if latest.retry_count >= self.max_retries:
                    self.registry.update_video(video.video_id, status="FAILED")
                    return self.registry.transition(
                        current.job_id,
                        JobState.FAILED,
                        message=f"{stage.value} 失败: {type(error).__name__}",
                        error_code=f"{stage.value}_FAILED",
                        force=True,
                    )
                self.registry.transition(
                    current.job_id,
                    JobState.RETRYING,
                    message=f"{stage.value} 失败，准备重试",
                    error_code=f"{stage.value}_RETRY",
                    resume_state=stage,
                    increment_retry=True,
                    force=True,
                )
                current = self.registry.transition(
                    current.job_id,
                    stage,
                    message=f"重试 {stage.value}",
                    force=True,
                )
        if current.state == JobState.READY:
            self.registry.update_video(current.video_id, status="READY")
        return current

    def run_once(self) -> JobRecord | None:
        claimed = self.registry.claim_next_job()
        if claimed is None:
            return None
        return self._execute_claimed(claimed)

    def run_forever(self) -> None:
        while not self._stop.is_set():
            try:
                result = self.run_once()
                self.last_error = None
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"
                self._stop.wait(self.poll_interval)
                continue
            if result is None:
                self._stop.wait(self.poll_interval)

    def start(self, *, recover: bool = True) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if recover:
            self.registry.recover_incomplete_jobs()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run_forever,
            name="shotseek-runtime-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None
