from __future__ import annotations

from dataclasses import dataclass

from architecture_walkthrough.vision.ocr import parse_dimension_pair as parse_ocr_dimension_pair


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


def parse_dimension_pair(text: str) -> tuple[float, float, str] | None:
    parsed = parse_ocr_dimension_pair(text)
    if parsed is None:
        return None
    return parsed.width_m, parsed.height_m, "m"
