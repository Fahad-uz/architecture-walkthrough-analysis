from __future__ import annotations

from dataclasses import dataclass

from architecture_walkthrough.geometry.scale import parse_dimension_pair


@dataclass(frozen=True)
class ExtractedMeasurement:
    width_m: float
    height_m: float
    confidence: float
    source_text: str


def parse_measurement_text(text: str) -> ExtractedMeasurement | None:
    parsed = parse_dimension_pair(text)
    if parsed is None:
        return None
    width_m, height_m, _ = parsed
    return ExtractedMeasurement(width_m=width_m, height_m=height_m, confidence=0.5, source_text=text)
