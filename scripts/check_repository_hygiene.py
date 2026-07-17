#!/usr/bin/env python3
"""Fail CI when secrets, runtime products, or media are tracked."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PREFIXES = (
    "doc/",
    "runs/",
    "data/",
    "uploads/",
    "outputs/",
    "artifacts/",
)
FORBIDDEN_SUFFIXES = (
    ".avi",
    ".db",
    ".flac",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".sqlite",
    ".sqlite3",
    ".wav",
)
SENSITIVE_PATTERNS = (
    re.compile(rb"Authorization\s*:\s*Bearer\s+[A-Za-z0-9._-]{16,}", re.IGNORECASE),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(rb"X-Amz-(?:Credential|Signature)=", re.IGNORECASE),
)


def tracked_files() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
    )
    return [
        item.decode("utf-8")
        for item in completed.stdout.split(b"\0")
        if item
    ]


def check_repository() -> dict[str, object]:
    violations: list[dict[str, str]] = []
    files = tracked_files()
    for relative in files:
        lowered = relative.lower()
        if relative == ".env" or (
            relative.startswith(".env.") and relative != ".env.example"
        ):
            violations.append({"path": relative, "reason": "tracked_secret_env"})
        if lowered.startswith(FORBIDDEN_PREFIXES):
            violations.append({"path": relative, "reason": "tracked_runtime_path"})
        if lowered.endswith(FORBIDDEN_SUFFIXES):
            violations.append({"path": relative, "reason": "tracked_media_or_database"})

        path = PROJECT_ROOT / relative
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        for pattern in SENSITIVE_PATTERNS:
            if pattern.search(data):
                violations.append(
                    {"path": relative, "reason": f"sensitive_pattern:{pattern.pattern!r}"}
                )

    return {
        "status": "pass" if not violations else "failed",
        "tracked_file_count": len(files),
        "violation_count": len(violations),
        "violations": violations,
    }


def main() -> int:
    report = check_repository()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
