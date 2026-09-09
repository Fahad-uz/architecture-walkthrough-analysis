from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from architecture_walkthrough.vision.ocr import (
    AutoOCRBackend,
    OCRText,
    RapidOCRBackend,
    TesseractOCRBackend,
    classify_text,
    get_ocr_backend,
    parse_dimension_pair,
    run_ocr,
)
from architecture_walkthrough.vision.furniture_detection import detect_furniture_from_image


def _image(path: Path) -> None:
    cv2.imwrite(str(path), np.full((80, 160, 3), 255, dtype=np.uint8))


def test_rapidocr_preserves_line_boxes_and_dimension_tokens(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "plan.png"
    _image(image_path)

    class FakeRapidOCR:
        def __init__(self, **_kwargs) -> None:
            pass

        def __call__(self, _image, **_kwargs):
            return SimpleNamespace(
                boxes=np.asarray(
                    [
                        [[10, 20], [90, 20], [90, 40], [10, 40]],
                        [[100, 45], [150, 45], [150, 65], [100, 65]],
                    ],
                    dtype=np.float32,
                ),
                txts=("BEDROOM", "300X290"),
                scores=(0.98, 0.96),
            )

    monkeypatch.setitem(sys.modules, "rapidocr", SimpleNamespace(RapidOCR=FakeRapidOCR))

    results = RapidOCRBackend().recognize(image_path)

    assert [item.normalized_text for item in results] == ["BEDROOM", "300X290"]
    assert results[0].polygon == [
        (10.0, 20.0),
        (90.0, 20.0),
        (90.0, 40.0),
        (10.0, 40.0),
    ]
    assert results[0].semantic_type == "room_label"
    assert results[1].semantic_type == "dimension"
    assert parse_dimension_pair(results[1].normalized_text).height_m == 2.9


def test_local_ocr_disables_native_telemetry_before_runtime_initializes(
    tmp_path: Path, monkeypatch,
) -> None:
    image_path = tmp_path / "plan.png"
    _image(image_path)
    events: list[str] = []
    original_import = builtins.__import__

    def observe_import(name, *args, **kwargs):
        if name == "onnxruntime":
            # The environment opt-out must precede the import, since the
            # native library can start its uploader before returning to Python.
            assert os.environ.get("ORT_DISABLE_TELEMETRY") == "1"
            events.append("import_runtime")
        return original_import(name, *args, **kwargs)

    class FakeRapidOCR:
        def __init__(self, **_kwargs) -> None:
            assert events == ["import_runtime", "disable_telemetry"]
            events.append("create_session")

        def __call__(self, _image, **_kwargs):
            return SimpleNamespace(boxes=None, txts=None, scores=None)

    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        disable_telemetry_events=lambda: events.append("disable_telemetry"),
    ))
    monkeypatch.setitem(sys.modules, "rapidocr", SimpleNamespace(RapidOCR=FakeRapidOCR))
    monkeypatch.setattr(builtins, "__import__", observe_import)

    assert RapidOCRBackend().recognize(image_path) == []
    assert events == ["import_runtime", "disable_telemetry", "create_session"]


def test_auto_backend_prefers_rapidocr_and_falls_back_to_tesseract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected = OCRText(
        text="KITCHEN",
        polygon=[(0, 0), (2, 0), (2, 1), (0, 1)],
        confidence=0.9,
        normalized_text="KITCHEN",
        semantic_type=classify_text("KITCHEN"),
    )
    monkeypatch.setattr(RapidOCRBackend, "recognize", lambda _self, _path: [])
    monkeypatch.setattr(TesseractOCRBackend, "recognize", lambda _self, _path: [expected])

    assert AutoOCRBackend().recognize(tmp_path / "missing.png") == [expected]
    assert isinstance(get_ocr_backend("auto"), AutoOCRBackend)
    assert isinstance(get_ocr_backend("rapidocr"), RapidOCRBackend)


def test_auto_backend_falls_back_when_rapidocr_results_are_below_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    weak = OCRText(
        text="noise",
        polygon=[(0, 0), (2, 0), (2, 1), (0, 1)],
        confidence=0.1,
        normalized_text="noise",
        semantic_type="text",
    )
    expected = OCRText(
        text="DINING",
        polygon=[(0, 0), (2, 0), (2, 1), (0, 1)],
        confidence=0.9,
        normalized_text="DINING",
        semantic_type="room_label",
    )
    monkeypatch.setattr(RapidOCRBackend, "recognize", lambda _self, _path: [weak])
    monkeypatch.setattr(TesseractOCRBackend, "recognize", lambda _self, _path: [expected])

    assert run_ocr(tmp_path / "missing.png", "auto", min_confidence=0.35) == [expected]


def test_dimension_parser_accepts_unicode_multiplication_sign() -> None:
    parsed = parse_dimension_pair("3.00 m × 4.00 m")

    assert parsed is not None
    assert parsed.width_m == 3.0
    assert parsed.height_m == 4.0


def test_furniture_detection_rejects_colored_text_plaque(tmp_path: Path) -> None:
    image_path = tmp_path / "colored_plan.png"
    image = np.full((200, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (30, 40), (120, 90), (0, 128, 128), -1)
    cv2.rectangle(image, (220, 40), (310, 90), (0, 128, 128), -1)
    cv2.imwrite(str(image_path), image)
    label = OCRText(
        text="LIVING 599X340",
        polygon=[(45, 50), (110, 50), (110, 80), (45, 80)],
        confidence=0.99,
        normalized_text="LIVING 599X340",
        semantic_type="dimension",
    )

    without_ocr = detect_furniture_from_image(image_path, 100.0, 200, max_side=400)
    with_ocr = detect_furniture_from_image(
        image_path,
        100.0,
        200,
        max_side=400,
        text_regions=[label],
    )

    assert len(without_ocr) == 2
    assert len(with_ocr) == 1
    assert with_ocr[0].center.x > 2.0
