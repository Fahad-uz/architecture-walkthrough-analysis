from __future__ import annotations

import httpx
from google.genai import types
from google.genai.errors import ClientError, ServerError
import pytest

from architecture_walkthrough.ai.retry import (
    bounded_http_options,
    call_with_backoff,
    is_transient_error,
)


def test_http_options_bound_transport_and_disable_sdk_retries() -> None:
    options = bounded_http_options(types, timeout_seconds=12.5)

    assert options.timeout == 12_500
    assert options.retry_options is not None
    assert options.retry_options.attempts == 1


def test_transient_errors_are_recognized() -> None:
    assert is_transient_error(RuntimeError("503 UNAVAILABLE: high demand"))
    assert is_transient_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert is_transient_error(RuntimeError("request timed out"))
    assert is_transient_error(httpx.ReadTimeout(""))
    assert is_transient_error(httpx.ConnectError(""))
    assert is_transient_error(ServerError(504, {"error": {"message": "deadline"}}))
    assert not is_transient_error(ClientError(401, {"error": {"message": "invalid key"}}))
    assert not is_transient_error(RuntimeError("processed 500 wall candidates"))
    assert not is_transient_error(RuntimeError("401 UNAUTHENTICATED"))


def test_backoff_retries_transient_then_succeeds(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("architecture_walkthrough.ai.retry.time.sleep", sleeps.append)
    calls = {"count": 0}

    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("503 UNAVAILABLE")
        return "ok"

    assert call_with_backoff(flaky, attempts=3, base_delay_seconds=1.0) == "ok"
    assert calls["count"] == 3
    assert sleeps == [1.0, 2.0]


def test_non_transient_errors_raise_immediately(monkeypatch) -> None:
    monkeypatch.setattr("architecture_walkthrough.ai.retry.time.sleep", lambda _s: None)
    calls = {"count": 0}

    def broken() -> str:
        calls["count"] += 1
        raise RuntimeError("401 UNAUTHENTICATED")

    with pytest.raises(RuntimeError, match="401"):
        call_with_backoff(broken, attempts=3)
    assert calls["count"] == 1


def test_transient_error_raises_after_final_attempt(monkeypatch) -> None:
    monkeypatch.setattr("architecture_walkthrough.ai.retry.time.sleep", lambda _s: None)

    def always_busy() -> str:
        raise RuntimeError("503 UNAVAILABLE")

    with pytest.raises(RuntimeError, match="503"):
        call_with_backoff(always_busy, attempts=2)
