from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from architecture_walkthrough.geometry.models import Point2D
from architecture_walkthrough.vision.furniture_detection import (
    LocalObjectFootprint,
    colored_object_mask_from_image,
    detect_colored_object_footprints,
    detect_furniture_from_image,
    ground_ai_furniture_semantics,
)


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

    mask = colored_object_mask_from_image(path)
    assert mask.shape == image.shape[:2]
    assert mask[180, 380] == 255
    assert mask[20, 20] == 0

    footprints = detect_colored_object_footprints(path)
    assert len(footprints) >= 3
    assert all(item.width_px >= item.depth_px for item in footprints)


def test_large_connected_color_region_is_not_furniture(tmp_path: Path) -> None:
    image = np.full((500, 700, 3), 240, dtype=np.uint8)
    cv2.rectangle(image, (30, 40), (450, 300), (65, 110, 110), -1)
    cv2.rectangle(image, (520, 80), (640, 190), (65, 110, 110), -1)
    path = tmp_path / "large_fill.png"
    assert cv2.imwrite(str(path), image)

    footprints = detect_colored_object_footprints(path)

    assert len(footprints) == 1
    assert footprints[0].center.x > 500
    assert footprints[0].category == "bed"


def test_u_shaped_counter_is_decomposed_into_local_bars(tmp_path: Path) -> None:
    image = np.full((600, 800, 3), 245, dtype=np.uint8)
    counter_color = (35, 45, 125)
    cv2.rectangle(image, (450, 180), (485, 480), counter_color, -1)
    cv2.rectangle(image, (700, 180), (735, 480), counter_color, -1)
    cv2.rectangle(image, (450, 445), (735, 480), counter_color, -1)
    path = tmp_path / "counter.png"
    assert cv2.imwrite(str(path), image)

    counters = [
        item
        for item in detect_colored_object_footprints(path)
        if item.category == "kitchen_counter"
    ]

    assert len(counters) >= 3
    assert any(abs(item.rotation_deg) < 1 for item in counters)
    assert sum(abs(abs(item.rotation_deg) - 90) < 1 for item in counters) >= 2


def test_ai_furniture_only_renames_matched_local_footprints() -> None:
    footprints = [
        LocalObjectFootprint(
            category="furniture",
            center=Point2D(x=100, y=100),
            width_px=80,
            depth_px=50,
            rotation_deg=0,
        )
    ]
    hints = SimpleNamespace(
        furniture=[
            SimpleNamespace(
                category="sofa",
                confidence=0.9,
                center=SimpleNamespace(x=0.10, y=0.10),
                width=0.08,
                depth=0.05,
            ),
            SimpleNamespace(
                category="bed",
                confidence=0.95,
                center=SimpleNamespace(x=0.80, y=0.80),
                width=0.15,
                depth=0.20,
            ),
        ]
    )

    grounded = ground_ai_furniture_semantics(
        footprints,
        hints,
        image_width_px=1000,
        image_height_px=1000,
        pixels_per_metre=100,
    )

    assert len(grounded.furniture) == 1
    assert grounded.furniture[0].category == "sofa"
    assert grounded.matched_hint_count == 1
    assert grounded.rejected_hint_count == 1


def test_ai_chair_hint_does_not_overwrite_local_dining_table() -> None:
    footprints = [
        LocalObjectFootprint(
            category="dining_table",
            center=Point2D(x=547.5, y=311.0),
            width_px=142,
            depth_px=59,
            rotation_deg=-90,
        )
    ]
    hints = SimpleNamespace(
        furniture=[
            SimpleNamespace(
                category="dining table",
                confidence=0.9,
                center=SimpleNamespace(x=0.516, y=0.389),
                width=0.08,
                depth=0.133,
            ),
            SimpleNamespace(
                category="chair",
                confidence=0.9,
                center=SimpleNamespace(x=0.557, y=0.389),
                width=0.036,
                depth=0.04,
            ),
        ]
    )

    grounded = ground_ai_furniture_semantics(
        footprints,
        hints,
        image_width_px=942,
        image_height_px=785,
        pixels_per_metre=82.224,
    )

    assert grounded.furniture[0].category == "dining_table"
    assert grounded.matched_hint_count == 0
    assert grounded.rejected_hint_count == 2


def test_ai_can_refine_a_local_category_within_the_same_family() -> None:
    footprints = [
        LocalObjectFootprint(
            category="dining_table",
            center=Point2D(x=100, y=100),
            width_px=80,
            depth_px=50,
            rotation_deg=0,
        )
    ]
    hints = SimpleNamespace(
        furniture=[
            SimpleNamespace(
                category="coffee table",
                confidence=0.9,
                center=SimpleNamespace(x=0.10, y=0.10),
                width=0.08,
                depth=0.05,
            )
        ]
    )

    grounded = ground_ai_furniture_semantics(
        footprints,
        hints,
        image_width_px=1000,
        image_height_px=1000,
        pixels_per_metre=100,
    )

    assert grounded.furniture[0].category == "coffee_table"
    assert grounded.matched_hint_count == 1
    assert grounded.rejected_hint_count == 0
