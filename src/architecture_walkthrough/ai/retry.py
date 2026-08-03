from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from google.genai.errors import APIError

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

# Free-tier Gemini regularly sheds load with transient HTTP failures; these are
# worth one bounded application-level retry.
TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
TRANSIENT_MARKERS = (
    "unavailable",
    "resource_exhausted",
    "high demand",
    "bad gateway",
    "gateway timeout",
    "timed out",
    "timeout",
)


def bounded_http_options(types_module: Any, timeout_seconds: float) -> Any:
    """Build google-genai options with one bounded transport attempt.

    Pinning SDK attempts to one keeps application-level backoff as the single
    retry authority and protects against nested retries in future SDK behavior.
    """

    return types_module.HttpOptions(
        timeout=max(1, round(timeout_seconds * 1000)),
        retry_options=types_module.HttpRetryOptions(attempts=1),
    )


def is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, APIError):
        return exc.code in TRANSIENT_STATUS_CODES
    text = str(exc).lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def call_with_backoff(
    operation: Callable[[], T],
    attempts: int = 3,
    base_delay_seconds: float = 10.0,
) -> T:
    """Run `operation`, retrying transient Gemini failures with backoff."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - filtered by is_transient_error
            if not is_transient_error(exc) or attempt == attempts - 1:
                raise
            delay = base_delay_seconds * (2**attempt)
            LOGGER.warning(
                "transient Gemini error (attempt %d/%d), retrying in %.0fs: %s",
                attempt + 1,
                attempts,
                delay,
                str(exc)[:160],
            )
            last = exc
            time.sleep(delay)
    raise last if last else RuntimeError("retry loop exited without result")
