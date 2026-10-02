from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box

from architecture_walkthrough.vision.furniture_detection import (
    detect_colored_object_footprints,
    detect_furniture_from_image,
    ground_ai_furniture_semantics,
)


@pytest.mark.parametrize("image_angle", [30.0, -25.0])
def test_rotated_furniture_keeps_its_measured_footprint_after_y_axis_flip(
    tmp_path: Path, image_angle: float,
) -> None:
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    corners = cv2.boxPoints(((200.0, 200.0), (120.0, 50.0), image_angle))
    cv2.fillConvexPoly(image, np.round(corners).astype(np.int32), (160, 110, 60))
    source = tmp_path / "rotated-sofa.png"
    assert cv2.imwrite(str(source), image)
    footprints = detect_colored_object_footprints(source)
    assert len(footprints) == 1
    footprint = footprints[0]
    measured = cv2.boxPoints((
        (footprint.center.x, footprint.center.y),
        (footprint.width_px, footprint.depth_px),
        footprint.rotation_deg,
    ))
    metric_evidence = Polygon([(float(x) / 100, (400 - float(y)) / 100) for x, y in measured])

    direct = detect_furniture_from_image(source, pixels_per_metre=100, image_height_px=400)
    grounded = ground_ai_furniture_semantics(
        footprints, None, image_width_px=400, image_height_px=400, pixels_per_metre=100,
    ).furniture
    for placements in (direct, grounded):
        assert len(placements) == 1
        placement = placements[0]
        rendered = box(-placement.width_m / 2, -placement.depth_m / 2,
                       placement.width_m / 2, placement.depth_m / 2)
        rendered = rotate(rendered, placement.rotation_deg, origin=(0, 0))
        rendered = translate(rendered, placement.center.x, placement.center.y)
        assert rendered.hausdorff_distance(metric_evidence) < 1e-5
