from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from architecture_walkthrough.vision.preprocessing import preprocess_array
from architecture_walkthrough.vision.wall_detection import (
    WallBand, _pair_parallel_wall_faces, _suppress_repetitive_detail_bands,
    detect_wall_bands,
)


def _band(name: str, orientation: str, coordinate: float, start: float, end: float,
          thickness: float = 4.0) -> WallBand:
    line = ((start, coordinate, end, coordinate) if orientation == "h"
            else (coordinate, start, coordinate, end))
    rect = ((start, coordinate - thickness / 2, end - start, thickness)
            if orientation == "h" else (coordinate - thickness / 2, start, thickness, end - start))
    return WallBand(name, orientation, rect, line, thickness, 1.0)


def test_outline_pair_clips_dimension_extension_tails_to_shared_wall_evidence() -> None:
    paired = _pair_parallel_wall_faces([
        _band("outer_with_measurement_tails", "h", 160, 120, 780),
        _band("inner_face", "h", 180, 200, 700),
    ], (900, 900))
    assert len(paired) == 1
    assert paired[0].centerline == pytest.approx((190, 170, 710, 170))


def test_outline_pair_accepts_short_face_contained_by_extended_outer_face() -> None:
    paired = _pair_parallel_wall_faces([
        _band("outer", "h", 700, 500, 830),
        _band("inner", "h", 716, 520, 750),
    ], (1000, 1000))
    assert len(paired) == 1
    assert paired[0].centerline == pytest.approx((512, 708, 758, 708))


def test_matching_face_endpoints_do_not_extend_into_a_doorway() -> None:
    paired = _pair_parallel_wall_faces([
        _band("face_a", "h", 160, 200, 500),
        _band("face_b", "h", 180, 200, 500),
    ], (900, 900))
    assert paired[0].centerline == pytest.approx((200, 170, 500, 170))


def test_outlined_plan_keeps_physical_perimeter_without_dimension_walls(tmp_path: Path) -> None:
    image = np.full((700, 900, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (180, 160), (720, 620), (0, 0, 0), 2)
    cv2.rectangle(image, (200, 180), (700, 600), (0, 0, 0), 2)
    cv2.rectangle(image, (450, 180), (460, 600), (0, 0, 0), -1)
    # Measurement baselines and extension tails drawn on outer wall faces.
    cv2.line(image, (180, 110), (720, 110), (0, 0, 0), 2)
    cv2.line(image, (180, 110), (180, 160), (0, 0, 0), 2)
    cv2.line(image, (720, 110), (720, 160), (0, 0, 0), 2)
    for x in (120, 780):
        cv2.line(image, (x, 160), (x, 620), (0, 0, 0), 2)
    cv2.line(image, (120, 160), (780, 160), (0, 0, 0), 2)
    cv2.line(image, (120, 620), (780, 620), (0, 0, 0), 2)

    layers = preprocess_array(image, tmp_path, max_side=900).layers
    result = detect_wall_bands(layers["horizontal_wall_band"], layers["vertical_wall_band"])
    assert len(result.walls) == 5
    xs = [point.x for wall in result.walls for point in (wall.start, wall.end)]
    ys = [point.y for wall in result.walls for point in (wall.start, wall.end)]
    assert (min(xs), max(xs)) == pytest.approx((190, 710), abs=5)
    assert (min(ys), max(ys)) == pytest.approx((170, 610), abs=5)


def test_double_line_stair_stringer_is_not_a_room_partition() -> None:
    treads = [_band(f"tread_{x}", "v", x, 200, 360) for x in range(200, 377, 22)]
    stringer = _band("fused_double_stringer", "h", 280, 200, 376, thickness=10)
    boundary = _band("stair_boundary", "h", 200, 180, 400, thickness=16)
    partition = _band("substantial_partition", "h", 330, 180, 400, thickness=24)
    kept, rejected, regions = _suppress_repetitive_detail_bands(
        [*treads, stringer, boundary, partition], (1156, 910),
    )
    assert regions and regions[0].is_stair_like()
    assert stringer in rejected
    assert boundary in kept and partition in kept
