from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from architecture_walkthrough.geometry.furniture_layout import fit_furniture_to_rooms
from architecture_walkthrough.geometry.models import (
    DoorOpening,
    FurniturePlacement,
    Point2D,
    RoomPolygon,
    WallSegment,
    WindowOpening,
)
from architecture_walkthrough.geometry.reconstruction import reconstruct_walls
from architecture_walkthrough.geometry.room_extraction import extract_rooms_from_walls
from architecture_walkthrough.geometry.scale_solver import ScaleConstraint, solve_scale
from architecture_walkthrough.vision.ocr import OCRText, classify_text, parse_dimension_pair
from architecture_walkthrough.vision.opening_detection import attach_openings_to_walls
from architecture_walkthrough.vision.plan_roi import detect_plan_roi
from architecture_walkthrough.vision.wall_detection import detect_wall_bands


def test_roi_detection_unions_all_structural_regions() -> None:
    # Thin-walled plans fragment into several components; the ROI must cover
    # the union of drawn structure, never just the largest blob.
    image = np.full((300, 240, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (220, 150), (0, 0, 0), 6)
    cv2.rectangle(image, (20, 220), (220, 285), (0, 0, 0), 2)
    result = detect_plan_roi(image, padding_ratio=0.0)
    assert result.roi.source == "auto"
    rect = result.roi.rect
    assert rect.y <= 20
    assert rect.y + rect.height >= 280


def test_dimension_parser_handles_cm_and_metres() -> None:
    assert parse_dimension_pair("599X340").width_m == pytest.approx(5.99)
    assert parse_dimension_pair("3.00 m x 4.00 m").height_m == pytest.approx(4.0)
    assert classify_text("Kitchen 300X400") == "dimension"
    with pytest.raises(ValueError):
        parse_dimension_pair("100000x2m")


def test_scale_solver_rejects_outlier() -> None:
    result = solve_scale(
        [
            ScaleConstraint(id="a", source="room", measured_px=(600, 300), expected_m=(6, 3)),
            ScaleConstraint(id="b", source="room", measured_px=(500, 250), expected_m=(5, 2.5)),
            ScaleConstraint(id="bad", source="room", measured_px=(900, 100), expected_m=(3, 3)),
        ]
    )
    assert result.pixels_per_metre == pytest.approx(100)
    assert any(item.id == "bad" for item in result.rejected_constraints)


def test_wall_band_detection_generates_single_centerlines(tmp_path: Path) -> None:
    horizontal = np.zeros((160, 220), dtype=np.uint8)
    vertical = np.zeros_like(horizontal)
    cv2.rectangle(horizontal, (20, 20), (200, 29), 255, -1)
    cv2.rectangle(horizontal, (20, 130), (200, 139), 255, -1)
    cv2.rectangle(vertical, (20, 20), (29, 140), 255, -1)
    cv2.rectangle(vertical, (190, 20), (199, 140), 255, -1)
    h_path = tmp_path / "h.png"
    v_path = tmp_path / "v.png"
    cv2.imwrite(str(h_path), horizontal)
    cv2.imwrite(str(v_path), vertical)

    result = detect_wall_bands(h_path, v_path, min_length_ratio=0.1)

    assert len(result.walls) == 4
    assert all(wall.evidence_source == "wall_band" for wall in result.walls)


def test_reconstruction_merges_collinear_and_snaps_intersections() -> None:
    walls = [
        WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=2, y=0.02)),
        WallSegment(start=Point2D(x=2.05, y=0.01), end=Point2D(x=4, y=0)),
        WallSegment(start=Point2D(x=4.02, y=-0.1), end=Point2D(x=4.01, y=2)),
    ]
    result = reconstruct_walls(walls, estimated_thickness_m=0.12)
    assert len(result.walls) == 2
    assert all(wall.id for wall in result.walls)


def test_room_extraction_and_label_assignment() -> None:
    walls = [
        WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=4, y=0), id="w0"),
        WallSegment(start=Point2D(x=4, y=0), end=Point2D(x=4, y=3), id="w1"),
        WallSegment(start=Point2D(x=4, y=3), end=Point2D(x=0, y=3), id="w2"),
        WallSegment(start=Point2D(x=0, y=3), end=Point2D(x=0, y=0), id="w3"),
    ]
    labels = [
        OCRText(
            text="Living",
            normalized_text="Living",
            semantic_type="room_label",
            confidence=0.9,
            polygon=[(180, 130), (220, 130), (220, 150), (180, 150)],
        )
    ]
    result = extract_rooms_from_walls(walls, labels, pixels_per_metre=100)
    assert len(result.rooms) == 1
    assert result.rooms[0].name == "Living"


def test_openings_project_to_nearest_wall() -> None:
    walls = [WallSegment(id="w0", start=Point2D(x=0, y=0), end=Point2D(x=4, y=0))]
    result = attach_openings_to_walls(
        walls,
        [DoorOpening(center=Point2D(x=2, y=0.1))],
        [WindowOpening(center=Point2D(x=3, y=0.1))],
        tolerance_m=0.2,
    )
    assert result.doors[0].wall_id == "w0"
    assert result.doors[0].center.y == 0
    assert result.windows[0].wall_id == "w0"


def test_furniture_is_fit_inside_room_bounds() -> None:
    room = RoomPolygon(
        points=[
            Point2D(x=0, y=0),
            Point2D(x=2, y=0),
            Point2D(x=2, y=2),
            Point2D(x=0, y=2),
        ]
    )
    fitted = fit_furniture_to_rooms(
        [FurniturePlacement(category="bed", center=Point2D(x=4, y=4), width_m=3, depth_m=3)],
        [room],
    )
    assert len(fitted) == 1
    assert 0 <= fitted[0].center.x <= 2
    assert 0 <= fitted[0].center.y <= 2
    assert fitted[0].width_m <= 1.8
    assert fitted[0].depth_m <= 1.8
