from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from architecture_walkthrough.config import AISettings, AppConfig
from architecture_walkthrough.geometry.floorplan import load_corrected_floorplan
from architecture_walkthrough.geometry.models import DoorOpening, FloorPlanModel, Point2D, WallSegment
from architecture_walkthrough.geometry.wall_graph import collinear_gaps, enumerate_faces
from architecture_walkthrough.pipeline import _manual_pixels_per_metre, analyze_image


def _wall(name: str, x1: float, y1: float, x2: float, y2: float) -> WallSegment:
    return WallSegment(id=name, start=Point2D(x=x1, y=y1), end=Point2D(x=x2, y=y2))


def test_overlapping_wall_fragments_do_not_manufacture_door_gaps() -> None:
    walls = [
        _wall("long", 0, 0, 10, 0),
        _wall("nested", 2, 0, 3, 0),
        _wall("nested_later", 4, 0, 5, 0),
        _wall("next", 11, 0, 14, 0),
    ]
    gaps = collinear_gaps(walls, coord_tol=0.1)
    assert len(gaps) == 1
    assert gaps[0]["wall_a"] == "long"
    assert gaps[0]["wall_b"] == "next"
    assert gaps[0]["length_m"] == pytest.approx(1.0)


def test_nearby_door_does_not_confirm_an_unrelated_wall_gap() -> None:
    walls = [
        _wall("bottom", 0, 0, 4, 0),
        _wall("right", 4, 0, 4, 4),
        _wall("top", 4, 4, 0, 4),
        _wall("left", 0, 4, 0, 0),
        _wall("divider_a", 2, 0, 2, 1.5),
        _wall("divider_b", 2, 2.4, 2, 4),
    ]
    door = DoorOpening(
        center=Point2D(x=2, y=0.95), wall_id="divider_a",
        start_offset_m=0.5, end_offset_m=1.4, width_m=0.9,
    )
    result = enumerate_faces(walls, doors=[door])
    assert len(result.faces) == 1
    assert len(result.unclosed_gaps) == 1
    assert not result.closures


def test_confirmed_gap_joins_measured_endpoints_with_small_centerline_jitter() -> None:
    walls = [
        _wall("bottom", 0, 0, 4, 0),
        _wall("right", 4, 0, 4, 4),
        _wall("top", 4, 4, 0, 4),
        _wall("left", 0, 4, 0, 0),
        _wall("divider_a", 2, 0, 2, 1.5),
        _wall("divider_b", 2.08, 2.4, 2.08, 4),
    ]
    door = DoorOpening(
        center=Point2D(x=2, y=1.95), wall_id="divider_a",
        start_offset_m=1.5, end_offset_m=2.4, width_m=0.9,
    )
    result = enumerate_faces(walls, doors=[door])
    assert len(result.faces) == 2
    assert not result.unclosed_gaps
    assert sum(face.polygon.area for face in result.faces) == pytest.approx(16.0)


def test_legacy_index_openings_keep_their_wall_on_load_and_roundtrip(tmp_path: Path) -> None:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_high_detail_floorplan.json"))
    assert model.doors[0].wall_id == model.walls[4].id
    assert model.windows[0].wall_id == model.walls[2].id
    assert len({wall.id for wall in model.walls}) == len(model.walls)
    path = tmp_path / "model.json"
    model.save_json(path)
    assert FloorPlanModel.load_json(path) == model


def test_explicit_numeric_wall_id_wins_over_legacy_position() -> None:
    model = FloorPlanModel(
        walls=[
            _wall("1", 0, 0, 4, 0),
            WallSegment(start=Point2D(x=4, y=0), end=Point2D(x=4, y=4)),
        ],
        doors=[DoorOpening(center=Point2D(x=2, y=0), wall_id="1", offset_m=2)],
    )
    assert model.doors[0].wall_id == model.walls[0].id == "1"
    assert model.walls[1].id != "1"


def test_editor_read_preserves_precise_wall_and_opening_geometry(tmp_path: Path) -> None:
    model = FloorPlanModel(
        walls=[_wall("measured", 0.023, 0.012, 4.037, 0.012)],
        doors=[DoorOpening(
            center=Point2D(x=1.973, y=0.012), wall_id="measured",
            start_offset_m=1.5, end_offset_m=2.4,
        )],
    )
    path = tmp_path / "floorplan.corrected.json"
    model.save_json(path)
    assert load_corrected_floorplan(path) == model


@pytest.mark.parametrize("scale", [0, -0.01, float("nan"), float("inf")])
def test_manual_scale_rejects_nonfinite_and_nonpositive_values(scale: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        _manual_pixels_per_metre(scale)


def test_three_room_plan_keeps_metric_layout_when_analysis_resizes_it(tmp_path: Path) -> None:
    image = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 100, 700, 500), outline="black", width=12)
    draw.line((400, 100, 400, 500), fill="black", width=10)
    draw.line((400, 300, 700, 300), fill="black", width=10)
    path = tmp_path / "three_rooms.png"
    image.save(path)
    config = AppConfig(ai=AISettings(gemini_enabled=False))
    config.ocr.enabled = False
    config.preprocessing.max_side_px = 400
    model = analyze_image(
        path, tmp_path / "analysis", config,
        manual_scale=0.01, crop_rect=(0, 0, 800, 600),
    )
    assert model.pixels_per_metre == pytest.approx(50.0)
    assert model.metadata["analysis_resize_ratio"] == pytest.approx(0.5)
    assert len(model.rooms) == 3
    assert all(room.evidence_source == "wall_topology" for room in model.rooms)
    xs = [point.x for wall in model.walls for point in (wall.start, wall.end)]
    ys = [point.y for wall in model.walls for point in (wall.start, wall.end)]
    assert max(xs) - min(xs) == pytest.approx(6.0, abs=0.15)
    assert max(ys) - min(ys) == pytest.approx(4.0, abs=0.15)
