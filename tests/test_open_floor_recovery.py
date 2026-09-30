from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from architecture_walkthrough.geometry.models import (
    DoorOpening,
    Point2D,
    RoomPolygon,
    WallSegment,
    WindowOpening,
)
from architecture_walkthrough.geometry.open_floor_recovery import recover_open_labeled_floors
from architecture_walkthrough.geometry.room_extraction import extract_rooms_from_walls
from architecture_walkthrough.vision.ocr import OCRText


def wall(name: str, x0: float, y0: float, x1: float, y1: float) -> WallSegment:
    return WallSegment(id=name, start=Point2D(x=x0, y=y0), end=Point2D(x=x1, y=y1))


def label(name: str, x: float = 4, y: float = 4, confidence: float = 0.98) -> OCRText:
    px, py = x * 100, 1000 - y * 100
    return OCRText(name, [(px-1, py-1), (px+1, py-1), (px+1, py+1), (px-1, py+1)], confidence, name, "room_label")


def open_room() -> list[WallSegment]:
    return [
        wall("left", 0, 0, 0, 8), wall("top", 0, 8, 8, 8), wall("right", 8, 0, 8, 8),
        wall("front_left", 0, 0, 2.5, 0), wall("front_right", 5.5, 0, 8, 0),
    ]


def test_unique_labeled_open_floor_recovers_without_changing_walls_or_openings() -> None:
    walls = open_room()
    before = [item.model_dump() for item in walls]
    doors = [DoorOpening(id="door", center=Point2D(x=0, y=4), wall_id="left", offset_m=4)]
    windows = [WindowOpening(id="window", center=Point2D(x=4, y=8), wall_id="top", offset_m=4)]
    opening_before = [item.model_dump() for item in [*doors, *windows]]
    result = recover_open_labeled_floors(
        walls, [], [label("Living and Dining")], 100, 1000, doors=doors, windows=windows,
    )
    assert len(result.rooms) == 1
    assert result.rooms[0].name == "Living and Dining"
    assert result.rooms[0].evidence_source == "inferred_open_floor_boundary"
    assert result.rooms[0].confidence <= 0.65
    assert Polygon([(p.x, p.y) for p in result.rooms[0].points]).area == pytest.approx(64)
    assert result.issues[0].code == "inferred_open_floor_boundary"
    assert result.candidates[0]["physical_wall_added"] is False
    assert [item.model_dump() for item in walls] == before
    assert [item.model_dump() for item in [*doors, *windows]] == opening_before


@pytest.mark.parametrize("labels", [[], [label("Living", 12, 4)], [label("Living", confidence=0.5)], [label("15' x 23'")], [label(" ")]])
def test_no_recovery_without_unenclosed_confident_room_name(labels: list[OCRText]) -> None:
    result = recover_open_labeled_floors(open_room(), [], labels, 100, 1000)
    assert not result.rooms
    assert not result.issues


def test_competing_aligned_boundaries_are_left_for_review() -> None:
    walls = [*open_room(), wall("inner_left", 0, 1, 2.5, 1), wall("inner_right", 5.5, 1, 8, 1)]
    for ordered in (walls, list(reversed(walls))):
        result = recover_open_labeled_floors(ordered, [], [label("Living")], 100, 1000)
        assert not result.rooms
        assert not result.issues
        assert not result.candidates


def test_multiple_room_names_in_one_unresolved_face_are_not_combined() -> None:
    result = recover_open_labeled_floors(open_room(), [], [label("Living", 3, 4), label("Kitchen", 6, 4)], 100, 1000)
    assert not result.rooms


def test_already_enclosed_room_and_its_identity_are_preserved() -> None:
    walls = [*open_room(), wall("divider", 0, 6, 8, 6)]
    labels = [label("Living", 4, 3), label("Bedroom", 4, 7)]
    existing = extract_rooms_from_walls(walls, labels, 100, image_height_px=1000, bridge_ambiguous_openings=True).rooms
    assert len(existing) == 1
    result = recover_open_labeled_floors(walls, existing, labels, 100, 1000)
    assert len(result.rooms) == 2
    assert result.rooms[0] is existing[0]
    assert result.rooms[1].name == "Living"
    assert Polygon([(p.x, p.y) for p in result.rooms[1].points]).area == pytest.approx(48)


def test_existing_room_area_is_never_consumed_by_proposed_floor() -> None:
    existing = RoomPolygon(
        id="reviewed", name="Reviewed floor",
        points=[Point2D(x=x, y=y) for x, y in [(0, 0), (2, 0), (2, 2), (0, 2)]],
        evidence_source="manual",
    )
    before = existing.model_dump()
    result = recover_open_labeled_floors(open_room(), [existing], [label("Living")], 100, 1000)
    assert result.rooms == [existing]
    assert result.rooms[0] is existing
    assert existing.model_dump() == before
    assert not result.issues
    assert not result.candidates


def test_unrelated_outside_gap_cannot_bound_the_missing_labeled_floor() -> None:
    walls = [item for item in open_room() if not (item.id or "").startswith("front")]
    walls.extend([
        wall("outside_left", 0, -2, 2.5, -2),
        wall("outside_right", 5.5, -2, 8, -2),
    ])
    result = recover_open_labeled_floors(walls, [], [label("Living")], 100, 1000)
    assert not result.rooms
    assert not result.issues
    assert not result.candidates


def test_low_confidence_label_cannot_rename_recovered_floor() -> None:
    result = recover_open_labeled_floors(
        open_room(), [], [label("Study", confidence=0.4), label("Living", 3, 4)], 100, 1000,
    )
    assert len(result.rooms) == 1
    assert result.rooms[0].name == "Living"
    assert result.candidates[0]["room_name"] == "Living"


@pytest.mark.parametrize("name", ["Balcony", "TERRACE", "Living / Balcony", "Open terrace"])
def test_outdoor_labels_do_not_enter_room_only_floor_recovery(name: str) -> None:
    result = recover_open_labeled_floors(open_room(), [], [label(name)], 100, 1000)
    assert not result.rooms
    assert not result.issues
    assert not result.candidates


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_scale_cannot_propose_floor(scale: float) -> None:
    assert not recover_open_labeled_floors(open_room(), [], [label("Living")], scale, 1000).rooms


def test_recovery_requires_only_one_broad_boundary() -> None:
    walls = [item for item in open_room() if item.id != "top"]
    walls.extend([wall("back_left", 0, 8, 2.5, 8), wall("back_right", 5.5, 8, 8, 8)])
    assert not recover_open_labeled_floors(walls, [], [label("Living")], 100, 1000).rooms


def test_plan_relative_gap_limit_prevents_missing_half_building_side() -> None:
    walls = [item for item in open_room() if not (item.id or "").startswith("front")]
    walls.extend([wall("front_left", 0, 0, 1.5, 0), wall("front_right", 6.5, 0, 8, 0)])
    assert not recover_open_labeled_floors(walls, [], [label("Living")], 100, 1000).rooms
