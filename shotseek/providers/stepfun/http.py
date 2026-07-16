"""Shared bounded retry behavior for StepFun HTTP calls."""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def request_with_retry(
    operation: Callable[[], httpx.Response],
    *,
    max_attempts: int = 3,
    base_delay_s: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """Execute an HTTP operation with bounded exponential backoff."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if base_delay_s < 0:
        raise ValueError("base_delay_s must be non-negative")

    last_transport_error: httpx.TransportError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = operation()
        except httpx.TransportError as exc:
            last_transport_error = exc
            if attempt == max_attempts:
                raise
            sleep(base_delay_s * (2 ** (attempt - 1)))
            continue

        if response.status_code not in RETRYABLE_STATUS_CODES or attempt == max_attempts:
            response.raise_for_status()
            return response

        retry_after = response.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after is not None else base_delay_s * (2 ** (attempt - 1))
        except ValueError:
            delay = base_delay_s * (2 ** (attempt - 1))
        response.close()
        sleep(max(0.0, delay))

    if last_transport_error is not None:
        raise last_transport_error
    raise RuntimeError("retry loop ended without a response")
