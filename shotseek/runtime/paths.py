"""Project-root locked paths and content-addressed upload storage."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, BinaryIO

from shotseek.m0 import ensure_within_project

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
ALLOWED_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm"})


@dataclass(frozen=True)
class StoredUpload:
    path: Path
    sha256: str
    bytes: int
    original_filename: str
    created: bool


class RuntimePaths:
    def __init__(self, project_root: Path, runtime_root: Path | None = None) -> None:
        self.project_root = project_root.resolve()
        self.root = ensure_within_project(
            self.project_root,
            runtime_root or self.project_root / "data" / "runtime",
        )

    @property
    def registry(self) -> Path:
        return self.root / "runtime.sqlite3"

    @property
    def uploads(self) -> Path:
        return self.root / "uploads"

    @property
    def videos(self) -> Path:
        return self.root / "videos"

    def video_root(self, video_id: str) -> Path:
        if not re.fullmatch(r"video_[a-f0-9]{16}", video_id):
            raise ValueError("invalid video_id")
        return ensure_within_project(self.project_root, self.videos / video_id)

    def ensure(self) -> None:
        for path in (self.root, self.uploads, self.videos):
            path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(value: str) -> str:
    name = Path(value or "upload.mp4").name
    cleaned = SAFE_NAME.sub("_", name).strip("._")
    if not cleaned:
        cleaned = "upload.mp4"
    suffix = Path(cleaned).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"unsupported video extension: {suffix or '<none>'}")
    return cleaned[:180]


def store_upload(
    paths: RuntimePaths,
    source: BinaryIO,
    filename: str,
    *,
    max_bytes: int = 4 * 1024 * 1024 * 1024,
) -> StoredUpload:
    paths.ensure()
    safe_name = sanitize_filename(filename)
    temporary = ensure_within_project(
        paths.project_root,
        paths.uploads / f".upload-{os.getpid()}-{id(source)}.tmp",
    )
    digest = hashlib.sha256()
    total = 0
    try:
        with temporary.open("wb") as handle:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise ValueError("uploaded video exceeds runtime size limit")
                digest.update(block)
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        if total <= 0:
            raise ValueError("uploaded video is empty")
        sha256 = digest.hexdigest()
        destination = ensure_within_project(
            paths.project_root,
            paths.uploads / f"{sha256}{Path(safe_name).suffix.lower()}",
        )
        created = not destination.exists()
        if created:
            temporary.replace(destination)
        else:
            temporary.unlink(missing_ok=True)
        return StoredUpload(
            path=destination,
            sha256=sha256,
            bytes=total,
            original_filename=safe_name,
            created=created,
        )
    finally:
        temporary.unlink(missing_ok=True)


async def store_upload_stream(
    paths: RuntimePaths,
    blocks: AsyncIterator[bytes],
    filename: str,
    *,
    max_bytes: int = 4 * 1024 * 1024 * 1024,
) -> StoredUpload:
    """Persist an HTTP body without framework temp files outside the project."""
    paths.ensure()
    safe_name = sanitize_filename(filename)
    temporary = ensure_within_project(
        paths.project_root,
        paths.uploads / f".upload-{os.getpid()}-{id(blocks)}.tmp",
    )
    digest = hashlib.sha256()
    total = 0
    try:
        with temporary.open("wb") as handle:
            async for block in blocks:
                if not block:
                    continue
                total += len(block)
                if total > max_bytes:
                    raise ValueError("uploaded video exceeds runtime size limit")
                digest.update(block)
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        if total <= 0:
            raise ValueError("uploaded video is empty")
        sha256 = digest.hexdigest()
        destination = ensure_within_project(
            paths.project_root,
            paths.uploads / f"{sha256}{Path(safe_name).suffix.lower()}",
        )
        created = not destination.exists()
        if created:
            temporary.replace(destination)
        else:
            temporary.unlink(missing_ok=True)
        return StoredUpload(
            path=destination,
            sha256=sha256,
            bytes=total,
            original_filename=safe_name,
            created=created,
        )
    finally:
        temporary.unlink(missing_ok=True)
