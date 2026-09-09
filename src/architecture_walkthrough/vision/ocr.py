from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    unit = used_a if used_a == used_b else "mixed"
    return DimensionPair(width_m=width_m, height_m=height_m, source_text=text, unit=unit)


class OCRBackend:
    def recognize(self, image_path: Path) -> list[OCRText]:
        raise NotImplementedError


class NoopOCRBackend(OCRBackend):
    def recognize(self, image_path: Path) -> list[OCRText]:
        return []


def _result_items(result: Any) -> list[OCRText]:
    """Convert a RapidOCR result without coupling the pipeline to NumPy arrays."""
    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is None or texts is None or scores is None:
        return []

    items: list[OCRText] = []
    for box, text, score in zip(boxes, texts, scores, strict=False):
        normalized = normalize_ocr_text(str(text))
        if not normalized:
            continue
        polygon = [(float(point[0]), float(point[1])) for point in box]
        if len(polygon) < 4:
            continue
        confidence = max(0.0, min(1.0, float(score)))
        items.append(
            OCRText(
                text=str(text),
                polygon=polygon,
                confidence=confidence,
                normalized_text=normalized,
                semantic_type=classify_text(normalized),
            )
        )
    return items


class RapidOCRBackend(OCRBackend):
    """Self-contained ONNX OCR with detection and line-level recognition.

    RapidOCR ships its recognition models in its wheel, so unlike Tesseract it
    does not require a separately installed native executable. Detecting a
    full line is important here: architectural dimensions such as ``300X400``
    must remain one token so they can be associated with their room polygon.
    """

    def recognize(self, image_path: Path) -> list[OCRText]:
        # Disable the native telemetry uploader before importing ONNX Runtime.
        # The API alone runs after native initialization and can leave an
        # initialization-event HTTP worker alive during interpreter shutdown
        # (ONNX Runtime 1.29 on macOS crashes in that worker's destroyed mutex).
        # OCR is local supporting evidence and does not need runtime telemetry.
        os.environ["ORT_DISABLE_TELEMETRY"] = "1"
        try:
            import onnxruntime

            onnxruntime.disable_telemetry_events()
            from rapidocr import RapidOCR  # type: ignore[import-not-found]
        except ImportError:
            return []

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return []
        try:
            engine = RapidOCR(
                params={
                    "Global.log_level": "warning",
                    "Global.use_cls": False,
                }
            )
            result = engine(image, use_cls=False, text_score=0.0)
        except (ImportError, OSError, RuntimeError, ValueError):
            # OCR is supporting evidence. A damaged/unsupported model must not
            # abort local geometry reconstruction; auto mode can still try the
            # next available backend.
            return []
        return _result_items(result)


class TesseractOCRBackend(OCRBackend):
    def recognize(self, image_path: Path) -> list[OCRText]:
        try:
            import pytesseract  # type: ignore[import-not-found]
        except ImportError:
            return []
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return []
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except (OSError, RuntimeError):
            return []
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


class AutoOCRBackend(OCRBackend):
    def recognize(self, image_path: Path) -> list[OCRText]:
        for backend in (RapidOCRBackend(), TesseractOCRBackend()):
            results = backend.recognize(image_path)
            if results:
                return results
        return []


def get_ocr_backend(name: str = "auto") -> OCRBackend:
    normalized = name.lower()
    if normalized == "auto":
        return AutoOCRBackend()
    if normalized == "rapidocr":
        return RapidOCRBackend()
    if normalized == "tesseract":
        return TesseractOCRBackend()
    return NoopOCRBackend()


def run_ocr(
    image_path: Path,
    backend_name: str = "auto",
    min_confidence: float = 0.35,
) -> list[OCRText]:
    if backend_name.lower() == "auto":
        candidate_backends: tuple[OCRBackend, ...] = (
            RapidOCRBackend(),
            TesseractOCRBackend(),
        )
        for candidate_backend in candidate_backends:
            accepted = [
                item
                for item in candidate_backend.recognize(image_path)
                if item.confidence >= min_confidence
            ]
            if accepted:
                return accepted
        return []
    backend = get_ocr_backend(backend_name)
    return [item for item in backend.recognize(image_path) if item.confidence >= min_confidence]
