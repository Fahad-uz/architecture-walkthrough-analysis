from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from architecture_walkthrough.geometry.migration import migrate_floorplan_payload
from architecture_walkthrough.geometry.models import (
    SCHEMA_VERSION,
    DoorOpening,
    FloorPlanModel,
    Point2D,
    WallSegment,
    WindowOpening,
)
from architecture_walkthrough.scene.wall_builder import opening_from_door, opening_from_window

FIXTURES = Path(__file__).parent / "fixtures"


def test_door_interval_requires_both_offsets() -> None:
    with pytest.raises(ValidationError):
        DoorOpening(center=Point2D(x=0, y=0), start_offset_m=1.0)


def test_door_interval_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        DoorOpening(center=Point2D(x=0, y=0), start_offset_m=2.0, end_offset_m=1.0)


def test_interval_falls_back_to_midpoint_representation() -> None:
    door = DoorOpening(center=Point2D(x=0, y=0), offset_m=2.0, width_m=0.9)
    assert door.interval(door.width_m) == pytest.approx((1.55, 2.45))
    explicit = WindowOpening(center=Point2D(x=0, y=0), start_offset_m=1.0, end_offset_m=2.4)
    assert explicit.interval(explicit.width_m) == (1.0, 2.4)


def test_migration_upgrades_v2_openings_and_rooms() -> None:
    payload = {
        "schema_version": "2.0",
        "walls": [{"id": "w0", "start": {"x": 0, "y": 0}, "end": {"x": 4, "y": 0}}],
        "doors": [{"center": {"x": 2, "y": 0}, "wall_id": "w0", "offset_m": 2.0, "width_m": 0.9}],
        "windows": [{"center": {"x": 3, "y": 0}, "wall_id": "w0", "offset_m": 3.0, "width_m": 1.2}],
        "rooms": [{"id": "room_000", "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}]}],
    }
    migrated = migrate_floorplan_payload(payload)
    assert migrated["schema_version"] == SCHEMA_VERSION
    assert migrated["doors"][0]["start_offset_m"] == pytest.approx(1.55)
    assert migrated["doors"][0]["end_offset_m"] == pytest.approx(2.45)
    assert migrated["windows"][0]["start_offset_m"] == pytest.approx(2.4)
    assert migrated["rooms"][0]["face_id"] == "room_000"
    model = FloorPlanModel.model_validate(migrated)
    assert model.doors[0].interval(model.doors[0].width_m) == pytest.approx((1.55, 2.45))


def test_migration_leaves_current_schema_untouched() -> None:
    payload = {"schema_version": SCHEMA_VERSION, "doors": [{"center": {"x": 1, "y": 0}, "offset_m": 1.0}]}
    migrated = migrate_floorplan_payload(payload)
    assert "start_offset_m" not in migrated["doors"][0]


def test_legacy_fixture_loads_via_migration(tmp_path: Path) -> None:
    legacy = json.loads((FIXTURES / "sample_floorplan.json").read_text(encoding="utf-8"))
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    model = FloorPlanModel.load_json(path)
    assert model.schema_version == SCHEMA_VERSION


def test_two_bedroom_fixture_is_valid_v3() -> None:
    model = FloorPlanModel.load_json(FIXTURES / "sample_two_bedroom_flat.json")
    assert model.schema_version == SCHEMA_VERSION
    assert len(model.walls) == 8
    assert len(model.rooms) == 5
    assert all(room.face_id for room in model.rooms)
    wall_ids = {wall.id for wall in model.walls}
    for opening in [*model.doors, *model.windows]:
        assert opening.wall_id in wall_ids
        interval = opening.interval(opening.width_m)
        assert interval is not None
        start, end = interval
        wall = next(wall for wall in model.walls if wall.id == opening.wall_id)
        assert 0.0 <= start < end <= wall.start.distance_to(wall.end) + 1e-6


def test_wall_builder_prefers_explicit_interval() -> None:
    wall = WallSegment(id="w0", start=Point2D(x=0, y=0), end=Point2D(x=6, y=0))
    door = DoorOpening(center=Point2D(x=99, y=99), wall_id="w0", start_offset_m=1.0, end_offset_m=2.2, width_m=0.9)
    opening = opening_from_door(wall, door)
    assert opening.start_m == pytest.approx(1.0)
    assert opening.end_m == pytest.approx(2.2)
    window = WindowOpening(center=Point2D(x=99, y=99), wall_id="w0", start_offset_m=3.0, end_offset_m=4.4)
    window_opening = opening_from_window(wall, window)
    assert window_opening.width_m == pytest.approx(1.4)
