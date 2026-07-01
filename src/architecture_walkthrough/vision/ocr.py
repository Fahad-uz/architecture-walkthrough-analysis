from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class OCRText:
    text: str
    polygon: list[tuple[float, float]]
    confidence: float
    normalized_text: str
    semantic_type: str


@dataclass(frozen=True)
class DimensionPair:
    width_m: float
    height_m: float
    source_text: str
    unit: str


DIMENSION_RE = re.compile(
    r"(?P<a>\d+(?:\.\d+)?)\s*(?P<ua>cm|m)?\s*(?:x|X|\*|by|×)\s*(?P<b>\d+(?:\.\d+)?)\s*(?P<ub>cm|m)?",
    re.IGNORECASE,
)

ROOM_KEYWORDS = {
    "living",
    "kitchen",
    "dining",
    "bedroom",
    "toilet",
    "bath",
    "balcony",
    "lift",
    "entrance",
    "stair",
    "staircase",
    "shelf",
}


def normalize_ocr_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").replace("×", "x").strip().split())


def classify_text(text: str) -> str:
    normalized = normalize_ocr_text(text).lower()
    if parse_dimension_pair(normalized) is not None:
        return "dimension"
    if any(keyword in normalized for keyword in ROOM_KEYWORDS):
        if "stair" in normalized:
            return "stair_label"
        if "lift" in normalized:
            return "lift_label"
        if "balcony" in normalized:
            return "balcony_label"
        return "room_label"
    return "text"


def _interpret_value(value: float, explicit_unit: str | None) -> tuple[float, str]:
    unit = (explicit_unit or "").lower()
    if unit == "m":
        metres = value
        used = "m"
    elif unit == "cm":
        metres = value / 100.0
        used = "cm"
    elif value >= 100:
        metres = value / 100.0
        used = "context_cm"
    elif value > 20:
        metres = value / 100.0
        used = "context_cm"
    else:
        metres = value
        used = "context_m"
    return metres, used


def parse_dimension_pair(text: str) -> DimensionPair | None:
    match = DIMENSION_RE.search(normalize_ocr_text(text))
    if not match:
        return None
    a = float(match.group("a"))
    b = float(match.group("b"))
    unit_a = match.group("ua")
    unit_b = match.group("ub") or unit_a
    width_m, used_a = _interpret_value(a, unit_a or unit_b)
    height_m, used_b = _interpret_value(b, unit_b or unit_a)
    if not (0.20 <= width_m <= 80.0 and 0.20 <= height_m <= 80.0):
        raise ValueError(f"unreasonable architectural dimension: {text!r}")
    return DimensionPair(width_m=width_m, height_m=height_m, source_text=text, unit=used_a if used_a == used_b else "mixed")


class OCRBackend:
    def recognize(self, image_path: Path) -> list[OCRText]:
        raise NotImplementedError


class NoopOCRBackend(OCRBackend):
    def recognize(self, image_path: Path) -> list[OCRText]:
        return []


class TesseractOCRBackend(OCRBackend):
    def recognize(self, image_path: Path) -> list[OCRText]:
        try:
            import pytesseract  # type: ignore[import-not-found]
        except ImportError:
            return []
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return []
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        results: list[OCRText] = []
        for index, text in enumerate(data.get("text", [])):
            normalized = normalize_ocr_text(text)
            if not normalized:
                continue
            try:
                confidence = float(data["conf"][index]) / 100.0
            except (ValueError, KeyError):
                confidence = 0.0
            x = float(data["left"][index])
            y = float(data["top"][index])
            w = float(data["width"][index])
            h = float(data["height"][index])
            polygon = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            results.append(
                OCRText(
                    text=text,
                    polygon=polygon,
                    confidence=max(0.0, min(1.0, confidence)),
                    normalized_text=normalized,
                    semantic_type=classify_text(normalized),
                )
            )
        return results


def get_ocr_backend(name: str = "auto") -> OCRBackend:
    if name.lower() in {"auto", "tesseract"}:
        return TesseractOCRBackend()
    return NoopOCRBackend()


def run_ocr(image_path: Path, backend_name: str = "auto", min_confidence: float = 0.35) -> list[OCRText]:
    backend = get_ocr_backend(backend_name)
    return [item for item in backend.recognize(image_path) if item.confidence >= min_confidence]
