from __future__ import annotations

import httpx
import pytest

from shotseek.providers.stepfun.http import request_with_retry


def test_retryable_status_is_retried_without_waiting() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"ok": True})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = request_with_retry(
            lambda: client.get("https://api.example.invalid/test"),
            base_delay_s=0,
        )
    assert response.json() == {"ok": True}
    assert calls == 2


def test_transport_error_is_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("temporary failure", request=request)
        return httpx.Response(200, json={"ok": True})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = request_with_retry(
            lambda: client.get("https://api.example.invalid/test"),
            base_delay_s=0,
        )
    assert response.status_code == 200
    assert calls == 2


def test_non_retryable_status_fails_immediately() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": "unauthorized"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            request_with_retry(
                lambda: client.get("https://api.example.invalid/test"),
                base_delay_s=0,
            )
    assert calls == 1
