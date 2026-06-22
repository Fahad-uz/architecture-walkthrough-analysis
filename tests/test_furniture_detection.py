from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from architecture_walkthrough.vision.furniture_detection import detect_furniture_from_image


def test_detect_furniture_from_colored_floorplan(tmp_path: Path) -> None:
    image = np.full((500, 700, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (80, 80), (610, 420), (230, 225, 215), -1)
    cv2.rectangle(image, (120, 120), (240, 260), (63, 116, 116), -1)  # olive bed/sofa-like object
    cv2.rectangle(image, (330, 170), (460, 220), (45, 75, 125), -1)  # wood table/counter-like object
    cv2.circle(image, (560, 120), 22, (40, 130, 45), -1)  # plant
    path = tmp_path / "plan.png"
    assert cv2.imwrite(str(path), image)

    furniture = detect_furniture_from_image(path, pixels_per_metre=50.0, image_height_px=500)

    assert len(furniture) >= 3
    categories = {item.category for item in furniture}
    assert "plant" in categories
    assert any(category in categories for category in {"bed", "sofa", "chair"})
