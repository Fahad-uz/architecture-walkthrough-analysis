from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

# Free-tier Gemini regularly sheds load with 503 UNAVAILABLE / 429 rate limits;
# these are transient and worth a couple of patient retries.
TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "high demand")


def is_transient_error(exc: Exception) -> bool:
    text = str(exc)
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
