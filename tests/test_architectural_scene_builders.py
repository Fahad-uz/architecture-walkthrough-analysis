from __future__ import annotations

from pathlib import Path

from architecture_walkthrough.geometry.models import DoorOpening, FloorPlanModel, Point2D, RoomPolygon, WallSegment, WindowOpening
from architecture_walkthrough.scene.asset_loader import load_asset_registry
from architecture_walkthrough.scene.floor_builder import polygon_floor_mesh
from architecture_walkthrough.scene.glb_validator import validate_glb
from architecture_walkthrough.scene.pbr_materials import load_material_registry
from architecture_walkthrough.scene.simple_glb import export_simple_glb
from architecture_walkthrough.scene.wall_builder import (
    WallOpening,
    opening_from_door,
    opening_from_window,
    split_wall_sections,
)


def test_wall_splits_around_one_door() -> None:
    wall = WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=5, y=0), height_m=3.0)
    door = DoorOpening(center=Point2D(x=2.5, y=0), width_m=1.0, height_m=2.1)

    sections = split_wall_sections(wall, [opening_from_door(wall, door)])

    assert len(sections) == 3
    assert any(section.bottom_m == 2.1 and section.top_m == 3.0 for section in sections)


def test_wall_splits_around_one_window() -> None:
    wall = WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=5, y=0), height_m=3.0)
    window = WindowOpening(center=Point2D(x=2.5, y=0), width_m=1.0, height_m=1.0, sill_height_m=0.9)

    sections = split_wall_sections(wall, [opening_from_window(wall, window)])

    assert len(sections) == 4
    assert any(section.bottom_m == 0.0 and section.top_m == 0.9 for section in sections)
    assert any(section.bottom_m == 1.9 and section.top_m == 3.0 for section in sections)


def test_multiple_openings_reject_overlap() -> None:
    wall = WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=5, y=0), height_m=3.0)
    openings = [
        WallOpening(offset_m=2.0, width_m=1.2, bottom_m=0.0, height_m=2.1, kind="door"),
        WallOpening(offset_m=2.4, width_m=1.2, bottom_m=0.0, height_m=2.1, kind="door"),
    ]

    try:
        split_wall_sections(wall, openings)
    except ValueError as exc:
        assert "overlaps" in str(exc)
    else:
        raise AssertionError("overlapping openings should fail")


def test_concave_floor_triangulation() -> None:
    room = RoomPolygon(
        name="concave",
        points=[
            Point2D(x=0, y=0),
            Point2D(x=3, y=0),
            Point2D(x=3, y=1),
            Point2D(x=1.5, y=1),
            Point2D(x=1.5, y=3),
            Point2D(x=0, y=3),
        ],
    )

    mesh = polygon_floor_mesh(room.points, 0.1, (200, 200, 200, 255))

    assert len(mesh.faces) > 0
    assert mesh.bounds[1][0] == 3
    assert mesh.bounds[1][1] == 3


def test_asset_and_material_registries_parse() -> None:
    assets = load_asset_registry(Path("assets/models/asset_registry.yaml"))
    materials = load_material_registry(Path("assets/textures/material_registry.yaml"))

    assert assets.get_entry("bed_double") is not None
    assert materials.require("painted_wall").roughness > 0


def test_sample_high_detail_floorplan_exports_and_validates(tmp_path: Path) -> None:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_high_detail_floorplan.json"))

    output = export_simple_glb(model, tmp_path / "sample.glb")
    report = validate_glb(output)

    assert report["valid"] is True
    assert report["mesh_count"] >= 10
