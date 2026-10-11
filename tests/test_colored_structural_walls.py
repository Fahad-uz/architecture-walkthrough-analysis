from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from architecture_walkthrough.vision.plan_roi import detect_plan_roi
from architecture_walkthrough.vision.preprocessing import preprocess_array, structural_ink_mask


def _canvas(height: int = 600, width: int = 800) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


@pytest.mark.parametrize("color", [(25, 120, 25), (170, 55, 20), (20, 30, 180)])
@pytest.mark.parametrize("scale", [1, 2])
def test_recovers_connected_colored_walls_without_selecting_a_hue(color, scale) -> None:
    image = _canvas(600 * scale, 800 * scale)
    cv2.rectangle(image, (80 * scale, 80 * scale), (720 * scale, 520 * scale), color, 10 * scale)
    cv2.line(image, (400 * scale, 80 * scale), (400 * scale, 320 * scale), color, 8 * scale)
    # The doorway remains a gap, even when both ends belong to the network.
    cv2.rectangle(image, (395 * scale, 180 * scale), (405 * scale, 230 * scale), (255, 255, 255), -1)

    mask = structural_ink_mask(image, dark_threshold=170, max_saturation=80)

    assert mask[300 * scale, 80 * scale] == 255
    assert mask[80 * scale, 600 * scale] == 255
    assert mask[150 * scale, 400 * scale] == 255
    assert mask[200 * scale, 400 * scale] == 0
    assert mask[300 * scale, 600 * scale] == 0


@pytest.mark.parametrize("shape", ["room_fill", "counter_fill", "furniture_outline", "long_bar", "thin_grid"])
def test_rejects_colored_room_and_furniture_counterexamples(shape: str) -> None:
    image = _canvas()
    color = (20, 30, 180)
    if shape == "room_fill":
        cv2.rectangle(image, (80, 80), (720, 520), color, -1)
    elif shape == "counter_fill":
        cv2.rectangle(image, (100, 100), (300, 150), color, -1)
        cv2.rectangle(image, (100, 100), (150, 270), color, -1)
    elif shape == "furniture_outline":
        cv2.rectangle(image, (200, 200), (380, 310), color, 5)
        cv2.line(image, (200, 230), (380, 230), color, 5)
    elif shape == "long_bar":
        cv2.line(image, (80, 80), (720, 80), color, 10)
    else:
        # Even an unusually large furniture grid has no bold wall bands.
        cv2.rectangle(image, (80, 80), (720, 520), color, 1)
        for x in (200, 400, 600):
            cv2.line(image, (x, 80), (x, 520), color, 1)
    assert not structural_ink_mask(image, 170, 80).any()


def test_keeps_furniture_excluded_beside_same_color_wall_network() -> None:
    image = _canvas()
    color = (20, 30, 180)
    cv2.rectangle(image, (80, 80), (720, 520), color, 10)
    cv2.rectangle(image, (160, 160), (300, 270), color, -1)
    cv2.rectangle(image, (460, 250), (600, 370), color, 2)

    mask = structural_ink_mask(image, 170, 80)

    assert mask[300, 80] == 255
    assert not mask[150:280, 150:310].any()
    assert not mask[240:380, 450:610].any()


def test_neutral_ink_unchanged_when_saturation_recovery_disabled() -> None:
    image = _canvas()
    cv2.rectangle(image, (80, 80), (720, 520), (60, 60, 60), 5)
    cv2.putText(image, "ROOM", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    expected = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 170, 255, cv2.THRESH_BINARY_INV)[1]
    assert np.array_equal(structural_ink_mask(image, 170, 80), expected)
    assert np.array_equal(structural_ink_mask(image, 170, 255), expected)


def test_colored_wall_evidence_reaches_roi_and_wall_bands(tmp_path: Path) -> None:
    image = _canvas()
    cv2.rectangle(image, (80, 80), (720, 520), (180, 60, 20), 10)
    roi = detect_plan_roi(image, padding_ratio=0)
    assert roi.roi.rect.x <= 80
    assert roi.roi.rect.y <= 80
    assert roi.roi.rect.x + roi.roi.rect.width >= 720
    assert roi.roi.rect.y + roi.roi.rect.height >= 520

    result = preprocess_array(image, tmp_path)
    horizontal = cv2.imread(str(result.layers["horizontal_wall_band"]), cv2.IMREAD_GRAYSCALE)
    vertical = cv2.imread(str(result.layers["vertical_wall_band"]), cv2.IMREAD_GRAYSCALE)
    assert horizontal[80, 400] == 255
    assert vertical[300, 80] == 255
