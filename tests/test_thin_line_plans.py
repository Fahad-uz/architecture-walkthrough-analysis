from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from architecture_walkthrough.vision.plan_roi import detect_plan_roi
from architecture_walkthrough.vision.preprocessing import (
    preprocess_array,
    strip_border_bars,
    structural_ink_mask,
)
from architecture_walkthrough.vision.wall_detection import detect_wall_bands


def _thin_line_plan() -> np.ndarray:
    """CAD-style plan: 1 px anti-aliased grey walls + a dark colored fill."""
    image = np.full((600, 800, 3), 255, dtype=np.uint8)
    grey = (120, 120, 120)
    # Room rectangle drawn with 1 px lines (upper-left area).
    cv2.rectangle(image, (80, 80), (700, 500), grey, 1)
    cv2.line(image, (400, 80), (400, 500), grey, 1)
    # Dark red "kitchen counter" fill: dark but saturated - not structure.
    cv2.rectangle(image, (500, 300), (620, 460), (0, 0, 120), -1)
    # Page frame bar hugging the top edge, like a screenshot artifact.
    cv2.rectangle(image, (0, 0), (799, 12), (30, 30, 30), -1)
    return image


def test_structural_ink_mask_keeps_grey_lines_and_drops_colored_fills() -> None:
    image = _thin_line_plan()
    mask = structural_ink_mask(image, dark_threshold=170, max_saturation=80)
    assert mask[300, 80] > 0  # thin grey wall line
    assert mask[380, 560] == 0  # dark red fill is furniture, not wall
    assert mask[5, 400] == 0  # page frame bar stripped


def test_strip_border_bars_keeps_interior_walls() -> None:
    mask = np.zeros((400, 600), dtype=np.uint8)
    cv2.rectangle(mask, (0, 0), (599, 8), 255, -1)  # top page bar
    cv2.line(mask, (50, 50), (50, 350), 255, 3)  # interior wall
    cleaned = strip_border_bars(mask)
    assert cleaned[4, 300] == 0
    assert cleaned[200, 50] > 0


def test_thin_line_plan_yields_wall_bands(tmp_path: Path) -> None:
    image = _thin_line_plan()
    result = preprocess_array(image, tmp_path, max_side=1800)
    detection = detect_wall_bands(
        result.layers["horizontal_wall_band"],
        result.layers["vertical_wall_band"],
        min_length_ratio=0.02,
    )
    # Expect at least the four rectangle sides plus the divider.
    assert len(detection.walls) >= 5
    horizontals = [w for w in detection.walls if abs(w.end.y - w.start.y) < abs(w.end.x - w.start.x)]
    verticals = [w for w in detection.walls if abs(w.end.y - w.start.y) >= abs(w.end.x - w.start.x)]
    assert len(horizontals) >= 2 and len(verticals) >= 3


def test_roi_covers_full_thin_line_plan() -> None:
    image = _thin_line_plan()
    result = detect_plan_roi(image, padding_ratio=0.0)
    rect = result.roi.rect
    assert rect.x <= 85 and rect.y <= 85
    assert rect.x + rect.width >= 695
    assert rect.y + rect.height >= 495
