"""StepFun Files API adapter for M0 video uploads."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import httpx

from shotseek.schemas import UploadedFile

from . import DEFAULT_FILES_BASE_URL
from .http import request_with_retry

MAX_STORAGE_BYTES = 128 * 1024 * 1024
READY_STATUSES = {"success", "processed"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _headers(api_key: str) -> dict[str, str]:
    if not api_key.strip():
        raise ValueError("StepFun API key is required")
    return {"Authorization": f"Bearer {api_key}"}


def upload_video(
    path: Path,
    *,
    api_key: str,
    base_url: str = DEFAULT_FILES_BASE_URL,
    timeout_s: float = 180.0,
    poll_interval_s: float = 2.0,
    client: httpx.Client | None = None,
    retry_attempts: int = 3,
    retry_base_delay_s: float = 0.5,
) -> tuple[UploadedFile, dict[str, Any]]:
    """Upload one MP4 using purpose=storage and wait until it is usable."""
    video_path = path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if video_path.suffix.lower() != ".mp4":
        raise ValueError("StepFun storage video input must be an MP4")
    size = video_path.stat().st_size
    if size <= 0:
        raise ValueError("video is empty")
    if size >= MAX_STORAGE_BYTES:
        raise ValueError("video must be smaller than 128 MiB")

    owns_client = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(timeout_s))
    try:
        def send_upload() -> httpx.Response:
            with video_path.open("rb") as handle:
                return http.post(
                    f"{base_url.rstrip('/')}/files",
                    headers=_headers(api_key),
                    data={"purpose": "storage"},
                    files={"file": (video_path.name, handle, "video/mp4")},
                )

        response = request_with_retry(
            send_upload,
            max_attempts=retry_attempts,
            base_delay_s=retry_base_delay_s,
        )
        upload_raw = response.json()
        file_id = str(upload_raw.get("id", "")).strip()
        if not file_id:
            raise ValueError("StepFun Files response did not include id")

        final_raw = upload_raw
        deadline = time.monotonic() + timeout_s
        while str(final_raw.get("status", "")).lower() not in READY_STATUSES:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"StepFun file {file_id} was not ready before timeout")
            time.sleep(poll_interval_s)
            status_response = request_with_retry(
                lambda: http.get(
                    f"{base_url.rstrip('/')}/files/{file_id}",
                    headers=_headers(api_key),
                ),
                max_attempts=retry_attempts,
                base_delay_s=retry_base_delay_s,
            )
            final_raw = status_response.json()

        uploaded = UploadedFile(
            file_id=file_id,
            file_uri=f"stepfile://{file_id}",
            filename=str(final_raw.get("filename") or video_path.name),
            bytes=int(final_raw.get("bytes") or size),
            sha256=file_sha256(video_path),
            status=str(final_raw.get("status") or "processed"),
        )
        return uploaded, {"upload": upload_raw, "final": final_raw}
    finally:
        if owns_client:
            http.close()
