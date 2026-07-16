"""StepFun asynchronous file ASR adapter for M0."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

import httpx

from shotseek.schemas import Utterance, WordTimestamp

from . import DEFAULT_ASR_BASE_URL, DEFAULT_ASR_MODEL
from .http import request_with_retry


def _headers(api_key: str) -> dict[str, str]:
    if not api_key.strip():
        raise ValueError("StepFun API key is required")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def validate_public_audio_url(audio_url: str) -> None:
    parsed = urlparse(audio_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("ASR audio URL must be a public HTTP(S) URL")
    hostname = (parsed.hostname or "").lower()
    if hostname in {"localhost", "127.0.0.1", "::1"} or hostname.endswith(".local"):
        raise ValueError("ASR audio URL cannot point to localhost")


def infer_audio_format(audio_url: str) -> str:
    suffix = PurePosixPath(urlparse(audio_url).path).suffix.lower().lstrip(".")
    if suffix in {"wav", "mp3", "ogg", "pcm"}:
        return suffix
    return "mp3"


def submit_asr(
    audio_url: str,
    *,
    api_key: str,
    model: str = DEFAULT_ASR_MODEL,
    base_url: str = DEFAULT_ASR_BASE_URL,
    channel: int = 1,
    client: httpx.Client | None = None,
    retry_attempts: int = 3,
    retry_base_delay_s: float = 0.5,
) -> tuple[str, dict[str, Any]]:
    validate_public_audio_url(audio_url)
    if channel not in {1, 2}:
        raise ValueError("audio channel must be 1 or 2")
    request_body = {
        "audio": {
            "format": infer_audio_format(audio_url),
            "channel": channel,
            "url": audio_url,
        },
        "request": {
            "model_name": model,
            "enable_channel_split": False,
            "show_utterances": True,
            "enable_speaker_info": True,
        },
    }
    owns_client = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(60.0))
    try:
        response = request_with_retry(
            lambda: http.post(
                f"{base_url.rstrip('/')}/audio/asr/file/submit",
                headers=_headers(api_key),
                json=request_body,
            ),
            max_attempts=retry_attempts,
            base_delay_s=retry_base_delay_s,
        )
        raw = response.json()
        task_id = str(raw.get("task_id", "")).strip()
        if not task_id:
            raise ValueError("ASR submit response did not include task_id")
        return task_id, raw
    finally:
        if owns_client:
            http.close()


def wait_for_asr(
    task_id: str,
    *,
    api_key: str,
    base_url: str = DEFAULT_ASR_BASE_URL,
    poll_interval_s: float = 2.0,
    timeout_s: float = 600.0,
    client: httpx.Client | None = None,
    retry_attempts: int = 3,
    retry_base_delay_s: float = 0.5,
) -> dict[str, Any]:
    if not task_id.strip():
        raise ValueError("task_id is required")
    owns_client = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(60.0))
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            response = request_with_retry(
                lambda: http.post(
                    f"{base_url.rstrip('/')}/audio/asr/file/query",
                    headers=_headers(api_key),
                    json={"task_id": task_id},
                ),
                max_attempts=retry_attempts,
                base_delay_s=retry_base_delay_s,
            )
            raw = response.json()
            status = str(raw.get("status", "")).upper()
            if status == "RUNNING":
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"ASR task {task_id} exceeded {timeout_s} seconds")
                time.sleep(poll_interval_s)
                continue
            if "result" in raw:
                return raw
            raise RuntimeError(f"ASR task ended without result; status={status or 'unknown'}")
    finally:
        if owns_client:
            http.close()


def normalize_asr_response(raw: dict[str, Any]) -> list[Utterance]:
    result = raw.get("result")
    if not isinstance(result, list) or not result:
        raise ValueError("ASR response result must be a non-empty array")
    utterance_items: list[Any] = []
    for result_item in result:
        if isinstance(result_item, dict):
            candidate = result_item.get("utterances")
            if isinstance(candidate, list):
                utterance_items.extend(candidate)

    utterances: list[Utterance] = []
    for index, item in enumerate(utterance_items, start=1):
        if not isinstance(item, dict):
            raise ValueError("every ASR utterance must be an object")
        speaker = item.get("speaker")
        speaker_id = None
        if isinstance(speaker, dict) and speaker.get("id") is not None:
            speaker_id = str(speaker["id"])
        words = [
            WordTimestamp(
                text=str(word["text"]),
                start_ms=int(word["start_time"]),
                end_ms=int(word["end_time"]),
            )
            for word in item.get("words", [])
            if isinstance(word, dict)
            and str(word.get("text", "")).strip()
            and "start_time" in word
            and "end_time" in word
        ]
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        utterances.append(
            Utterance(
                utterance_id=f"utterance_{index:04d}",
                start_ms=int(item["start_time"]),
                end_ms=int(item["end_time"]),
                text=text,
                speaker_id=speaker_id,
                words=words,
            )
        )
    if not utterances:
        raise ValueError("ASR response did not contain usable utterances")
    return utterances


def run_asr(
    audio_url: str,
    *,
    api_key: str,
    model: str = DEFAULT_ASR_MODEL,
    base_url: str = DEFAULT_ASR_BASE_URL,
    channel: int = 1,
    poll_interval_s: float = 2.0,
    timeout_s: float = 600.0,
    retry_attempts: int = 3,
    retry_base_delay_s: float = 0.5,
    on_submit: Callable[[dict[str, Any]], None] | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[Utterance], dict[str, Any]]:
    with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
        task_id, submit_raw = submit_asr(
            audio_url,
            api_key=api_key,
            model=model,
            base_url=base_url,
            channel=channel,
            client=client,
            retry_attempts=retry_attempts,
            retry_base_delay_s=retry_base_delay_s,
        )
        if on_submit is not None:
            on_submit(submit_raw)
        result_raw = wait_for_asr(
            task_id,
            api_key=api_key,
            base_url=base_url,
            poll_interval_s=poll_interval_s,
            timeout_s=timeout_s,
            client=client,
            retry_attempts=retry_attempts,
            retry_base_delay_s=retry_base_delay_s,
        )
        if on_result is not None:
            on_result(result_raw)
    return normalize_asr_response(result_raw), {"submit": submit_raw, "result": result_raw}
