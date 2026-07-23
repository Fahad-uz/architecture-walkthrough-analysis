from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box

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
from architecture_walkthrough.vision.wall_detection import RepetitiveDetailRegion, detect_wall_bands
from architecture_walkthrough.pipeline import _balconies_from_patterned_regions, _classify_special_elements


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


def test_wall_band_detection_pairs_thin_parallel_wall_faces(tmp_path: Path) -> None:
    horizontal = np.zeros((180, 260), dtype=np.uint8)
    vertical = np.zeros_like(horizontal)
    cv2.rectangle(horizontal, (25, 70), (225, 73), 255, -1)
    cv2.rectangle(horizontal, (25, 82), (225, 85), 255, -1)
    h_path = tmp_path / "h_hollow.png"
    v_path = tmp_path / "v_hollow.png"
    cv2.imwrite(str(h_path), horizontal)
    cv2.imwrite(str(v_path), vertical)

    result = detect_wall_bands(h_path, v_path, min_length_ratio=0.05)

    assert len(result.bands) == 1
    assert result.bands[0].centerline[1] == pytest.approx(78.0)
    assert result.bands[0].thickness_px >= 15.0


def test_wall_band_merge_does_not_bridge_reversed_disjoint_spans(tmp_path: Path) -> None:
    """Adjacent coordinate buckets must not turn far-apart runs into one wall.

    This reproduces the real-plan ordering that joined a right-side bedroom
    wall to a clipped stair tread on the far left, creating a 750 px phantom
    wall across the whole floor plan.
    """

    horizontal = np.zeros((800, 800), dtype=np.uint8)
    vertical = np.zeros_like(horizontal)
    cv2.rectangle(horizontal, (507, 277), (749, 287), 255, -1)
    cv2.rectangle(horizontal, (0, 285), (31, 287), 255, -1)
    cv2.rectangle(vertical, (727, 527), (729, 578), 255, -1)
    cv2.rectangle(vertical, (731, 406), (734, 483), 255, -1)
    h_path = tmp_path / "h_disjoint.png"
    v_path = tmp_path / "v_disjoint.png"
    cv2.imwrite(str(h_path), horizontal)
    cv2.imwrite(str(v_path), vertical)

    result = detect_wall_bands(h_path, v_path, min_length_ratio=0.02)
    horizontals = [wall for wall in result.walls if abs(wall.end.y - wall.start.y) < abs(wall.end.x - wall.start.x)]
    verticals = [wall for wall in result.walls if abs(wall.end.y - wall.start.y) >= abs(wall.end.x - wall.start.x)]

    assert len(horizontals) == 2
    assert len(verticals) == 2
    assert max(wall.start.distance_to(wall.end) for wall in horizontals) < 300
    assert max(wall.start.distance_to(wall.end) for wall in verticals) < 100


def test_wall_band_detection_suppresses_stair_treads_and_centerline(tmp_path: Path) -> None:
    horizontal = np.zeros((360, 520), dtype=np.uint8)
    vertical = np.zeros_like(horizontal)
    for y in range(80, 221, 20):
        cv2.rectangle(horizontal, (40, y), (180, y + 3), 255, -1)
    # Thin line through the middle is the stair stringer, not a wall.  The two
    # perimeter faces and a separate room wall must survive.
    cv2.rectangle(vertical, (39, 75), (43, 230), 255, -1)
    cv2.rectangle(vertical, (178, 75), (182, 230), 255, -1)
    cv2.rectangle(vertical, (108, 80), (111, 223), 255, -1)
    cv2.rectangle(horizontal, (250, 90), (470, 102), 255, -1)
    h_path = tmp_path / "h_stairs.png"
    v_path = tmp_path / "v_stairs.png"
    cv2.imwrite(str(h_path), horizontal)
    cv2.imwrite(str(v_path), vertical)

    result = detect_wall_bands(h_path, v_path, min_length_ratio=0.025)

    assert len(result.repetitive_detail_regions) == 1
    assert result.repetitive_detail_regions[0].line_count == 8
    assert len(result.rejected_detail_bands) == 9
    assert not any(100 <= wall.start.x <= 120 and wall.start.x == wall.end.x for wall in result.walls)
    assert any(wall.start.x <= 45 and wall.start.x == wall.end.x for wall in result.walls)
    assert any(wall.start.x >= 175 and wall.start.x == wall.end.x for wall in result.walls)
    assert any(wall.start.x >= 240 and wall.start.y == wall.end.y for wall in result.walls)

    specials = _classify_special_elements(result.repetitive_detail_regions, 50.0, 360)
    assert len(specials) == 1
    assert specials[0].kind == "staircase"
    assert specials[0].metadata["step_count"] == 8
    assert specials[0].width_m == pytest.approx(2.82)
    assert specials[0].depth_m == pytest.approx(3.2)


def test_wide_shallow_hatching_is_not_classified_as_staircase(tmp_path: Path) -> None:
    horizontal = np.zeros((220, 620), dtype=np.uint8)
    vertical = np.zeros_like(horizontal)
    for y in range(40, 101, 12):
        cv2.rectangle(horizontal, (40, y), (560, y + 3), 255, -1)
    h_path = tmp_path / "h_hatch.png"
    v_path = tmp_path / "v_hatch.png"
    cv2.imwrite(str(h_path), horizontal)
    cv2.imwrite(str(v_path), vertical)

    result = detect_wall_bands(h_path, v_path, min_length_ratio=0.02)

    assert len(result.repetitive_detail_regions) == 1
    assert not result.repetitive_detail_regions[0].is_stair_like()
    assert _classify_special_elements(result.repetitive_detail_regions, 50.0, 220) == []


def test_patterned_region_with_local_label_becomes_balcony() -> None:
    regions = [
        RepetitiveDetailRegion(
            orientation="h",
            rect=(40.0, 20.0, 180.0, 45.0),
            spacing_px=9.0,
            line_count=7,
        ),
        RepetitiveDetailRegion(
            orientation="h",
            rect=(20.0, 90.0, 120.0, 110.0),
            spacing_px=14.0,
            line_count=8,
        ),
    ]
    labels = [
        OCRText(
            text="BALCONY",
            normalized_text="BALCONY",
            semantic_type="balcony_label",
            confidence=0.96,
            polygon=[(100, 34), (150, 34), (150, 48), (100, 48)],
        ),
        OCRText(
            text="DINING",
            normalized_text="DINING",
            semantic_type="room_label",
            confidence=0.99,
            polygon=[(50, 105), (90, 105), (90, 120), (50, 120)],
        ),
    ]

    balconies = _balconies_from_patterned_regions(
        regions,
        labels,
        pixels_per_metre=50.0,
        image_height_px=240,
    )

    assert len(balconies) == 1
    assert balconies[0].name == "BALCONY"
    assert balconies[0].evidence_source == "patterned_region+ocr_balcony_label"
    assert balconies[0].points[0] == Point2D(x=0.8, y=3.5)
    assert balconies[0].points[2] == Point2D(x=4.4, y=4.4)


def test_patterned_outer_strip_recovers_boundary_and_rejects_mortar_wall(tmp_path: Path) -> None:
    horizontal = np.zeros((200, 300), dtype=np.uint8)
    vertical = np.zeros_like(horizontal)
    # One connected component: the actual top envelope plus a shallow patterned
    # balcony strip that would otherwise exceed the maximum wall thickness.
    cv2.rectangle(horizontal, (20, 5), (280, 7), 255, -1)
    for y in range(7, 50, 9):
        cv2.rectangle(horizontal, (60, y), (240, y + 2), 255, -1)
    for x in range(60, 241, 30):
        cv2.rectangle(horizontal, (x, 7), (x + 2, 49), 255, -1)
    # A mortar joint that survives the vertical morphology pass is not a wall.
    cv2.rectangle(vertical, (118, 8), (121, 49), 255, -1)
    h_path = tmp_path / "h_pattern.png"
    v_path = tmp_path / "v_pattern.png"
    cv2.imwrite(str(h_path), horizontal)
    cv2.imwrite(str(v_path), vertical)

    result = detect_wall_bands(h_path, v_path, min_length_ratio=0.02)

    assert any(
        wall.start.y == pytest.approx(5.5, abs=2.0)
        and min(wall.start.x, wall.end.x) <= 25
        and max(wall.start.x, wall.end.x) >= 275
        for wall in result.walls
    )
    assert not any(
        wall.start.x == wall.end.x and 110 <= wall.start.x <= 130
        for wall in result.walls
    )
    assert any(not region.is_stair_like() for region in result.repetitive_detail_regions)


def test_reconstruction_merges_collinear_and_snaps_intersections() -> None:
    walls = [
        WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=2, y=0.02)),
        WallSegment(start=Point2D(x=2.05, y=0.01), end=Point2D(x=4, y=0)),
        WallSegment(start=Point2D(x=4.02, y=-0.1), end=Point2D(x=4.01, y=2)),
    ]
    result = reconstruct_walls(walls, estimated_thickness_m=0.12)
    assert len(result.walls) == 2
    assert all(wall.id for wall in result.walls)


def test_reconstruction_does_not_bridge_reversed_disjoint_spans() -> None:
    walls = [
        WallSegment(id="right", start=Point2D(x=5.0, y=0.80), end=Point2D(x=7.5, y=0.80)),
        WallSegment(id="left", start=Point2D(x=0.0, y=0.95), end=Point2D(x=0.32, y=0.95)),
    ]

    result = reconstruct_walls(walls, estimated_thickness_m=0.12)

    assert len(result.walls) == 2
    assert max(wall.start.distance_to(wall.end) for wall in result.walls) == pytest.approx(2.5)


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


def test_furniture_near_room_edge_is_fit_inside_room_bounds() -> None:
    room = RoomPolygon(
        points=[
            Point2D(x=0, y=0),
            Point2D(x=2, y=0),
            Point2D(x=2, y=2),
            Point2D(x=0, y=2),
        ]
    )
    fitted = fit_furniture_to_rooms(
        [FurniturePlacement(category="bed", center=Point2D(x=2.2, y=1), width_m=3, depth_m=3)],
        [room],
    )
    assert len(fitted) == 1
    assert 0 <= fitted[0].center.x <= 2
    assert 0 <= fitted[0].center.y <= 2
    assert fitted[0].width_m <= 1.8
    assert fitted[0].depth_m <= 1.8


def test_furniture_far_from_every_room_is_rejected_instead_of_teleported() -> None:
    room = RoomPolygon(
        points=[
            Point2D(x=0, y=0),
            Point2D(x=2, y=0),
            Point2D(x=2, y=2),
            Point2D(x=0, y=2),
        ]
    )
    fitted = fit_furniture_to_rooms(
        [FurniturePlacement(category="bed", center=Point2D(x=4, y=4), width_m=1.8, depth_m=2.0)],
        [room],
    )
    assert fitted == []


def test_grounded_furniture_can_survive_in_an_open_plan_without_room_face() -> None:
    room = RoomPolygon(
        points=[
            Point2D(x=0, y=0),
            Point2D(x=2, y=0),
            Point2D(x=2, y=2),
            Point2D(x=0, y=2),
        ]
    )
    item = FurniturePlacement(
        category="sofa",
        center=Point2D(x=4, y=2),
        width_m=1.8,
        depth_m=0.8,
    )

    fitted = fit_furniture_to_rooms([item], [room], preserve_unassigned=True)

    assert fitted == [item]


def test_rotated_furniture_footprint_stays_inside_concave_room() -> None:
    room = RoomPolygon(
        points=[
            Point2D(x=0, y=0),
            Point2D(x=4, y=0),
            Point2D(x=4, y=1.5),
            Point2D(x=2, y=1.5),
            Point2D(x=2, y=4),
            Point2D(x=0, y=4),
        ]
    )
    fitted = fit_furniture_to_rooms(
        [
            FurniturePlacement(
                category="sofa",
                center=Point2D(x=1.8, y=1.6),
                width_m=1.8,
                depth_m=0.8,
                rotation_deg=35,
            )
        ],
        [room],
    )

    assert len(fitted) == 1
    item = fitted[0]
    footprint = rotate(
        box(-item.width_m / 2, -item.depth_m / 2, item.width_m / 2, item.depth_m / 2),
        item.rotation_deg,
        origin=(0, 0),
    )
    footprint = translate(footprint, item.center.x, item.center.y)
    room_shape = Polygon([(point.x, point.y) for point in room.points]).buffer(-0.08)
    assert room_shape.covers(footprint)
