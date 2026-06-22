from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleConverter:
    pixels_per_metre: float

    def __post_init__(self) -> None:
        if self.pixels_per_metre <= 0:
            raise ValueError("pixels_per_metre must be positive")

    def px_to_m(self, value_px: float) -> float:
        return value_px / self.pixels_per_metre

    def m_to_px(self, value_m: float) -> float:
        return value_m * self.pixels_per_metre


_MEASURE_RE = re.compile(r"(?P<a>\d+(?:\.\d+)?)\s*(?:x|×|by)\s*(?P<b>\d+(?:\.\d+)?)\s*(?P<u>cm|m)?", re.I)


def parse_dimension_pair(text: str) -> tuple[float, float, str] | None:
    match = _MEASURE_RE.search(text)
    if not match:
        return None
    unit = (match.group("u") or "cm").lower()
    a = float(match.group("a"))
    b = float(match.group("b"))
    if unit == "cm":
        a /= 100.0
        b /= 100.0
    if not (0.2 <= a <= 100 and 0.2 <= b <= 100):
        raise ValueError(f"unreasonable architectural dimension: {text!r}")
    return a, b, "m"
