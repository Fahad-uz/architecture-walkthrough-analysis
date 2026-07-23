from __future__ import annotations

from pathlib import Path

import trimesh

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.models import (
    ArchitecturalElement,
    BalconyPolygon,
    FloorPlanModel,
    FurniturePlacement,
    Point2D,
    RoomPolygon,
)
from architecture_walkthrough.pipeline import build_model
from architecture_walkthrough.scene.simple_glb import (
    _floor_meshes_for_model,
    _furniture_family,
    _room_floor_coverage,
    export_simple_glb,
)


def test_specific_furniture_categories_win_over_broad_tokens() -> None:
    assert _furniture_family("bedside_table") == "table"
    assert _furniture_family("kitchen_sink") == "sink"
    assert _furniture_family("kitchen_stove") == "stove"
    assert _furniture_family("kitchen_counter") == "counter"
    assert _furniture_family("nightstand") == "table"


def test_incomplete_room_faces_get_a_safe_foundation_floor() -> None:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json")).model_copy(
        update={
            "rooms": [
                RoomPolygon(
                    id="partial",
                    points=[
                        Point2D(x=0.2, y=0.2),
                        Point2D(x=0.8, y=0.2),
                        Point2D(x=0.8, y=0.8),
                        Point2D(x=0.2, y=0.8),
                    ],
                )
            ]
        }
    )
    assert _room_floor_coverage(model) < 0.45
    floor = _floor_meshes_for_model(model)[0]
    wall_xs = [value for wall in model.walls for value in (wall.start.x, wall.end.x)]
    wall_ys = [value for wall in model.walls for value in (wall.start.y, wall.end.y)]
    assert floor.bounds[0][0] <= min(wall_xs)
    assert floor.bounds[1][0] >= max(wall_xs)
    assert floor.bounds[0][1] <= min(wall_ys)
    assert floor.bounds[1][1] >= max(wall_ys)


def test_simple_glb_export_includes_balcony_polygons(tmp_path: Path) -> None:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json")).model_copy(
        update={
            "balconies": [
                BalconyPolygon(
                    id="balcony",
                    points=[
                        Point2D(x=0.0, y=4.0),
                        Point2D(x=2.0, y=4.0),
                        Point2D(x=2.0, y=5.0),
                        Point2D(x=0.0, y=5.0),
                    ],
                )
            ]
        }
    )
    output = export_simple_glb(model, tmp_path / "balcony.glb")
    loaded = trimesh.load(output, force="scene")
    assert any(str(name).startswith("Balcony_") for name in loaded.graph.nodes_geometry)


def test_simple_glb_export_writes_loadable_glb(tmp_path: Path) -> None:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json"))
    output = export_simple_glb(model, tmp_path / "building.glb")
    assert output.exists()
    assert output.stat().st_size > 0
    loaded = trimesh.load(output, force="scene")
    assert len(loaded.geometry) >= 2


def test_build_model_defaults_to_pure_python_glb_export(tmp_path: Path) -> None:
    # force=True: this test exercises the export path; the quality gate has its
    # own coverage in test_validation_gate.py (the legacy fixture has no scale).
    output = build_model(Path("tests/fixtures/sample_floorplan.json"), tmp_path / "building.glb", AppConfig(), force=True)
    assert output.exists()
    assert output.suffix == ".glb"


def test_simple_glb_export_includes_furniture_geometry(tmp_path: Path) -> None:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json")).model_copy(
        update={
            "furniture": [
                FurniturePlacement(
                    category="sofa",
                    center=Point2D(x=2.0, y=1.5),
                    width_m=1.6,
                    depth_m=0.8,
                )
            ]
        }
    )
    output = export_simple_glb(model, tmp_path / "building.glb")
    loaded = trimesh.load(output, force="scene")
    assert any("Furniture" in name for name in loaded.graph.nodes_geometry)


def test_simple_glb_export_builds_multi_part_furniture(tmp_path: Path) -> None:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json")).model_copy(
        update={
            "furniture": [
                FurniturePlacement(
                    category="bed",
                    center=Point2D(x=2.0, y=1.5),
                    width_m=1.8,
                    depth_m=2.2,
                )
            ]
        }
    )
    output = export_simple_glb(model, tmp_path / "building.glb")
    loaded = trimesh.load(output, force="scene")
    furniture_nodes = [name for name in loaded.graph.nodes_geometry if "Furniture" in name]
    assert len(furniture_nodes) >= 4


def test_simple_glb_export_uses_semantic_door_names(tmp_path: Path) -> None:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json"))
    output = export_simple_glb(model, tmp_path / "doors.glb")
    loaded = trimesh.load(output, force="scene")
    nodes = {str(name) for name in loaded.graph.nodes_geometry}
    assert any(name.startswith("DoorLeaf_") for name in nodes)
    assert any(name.startswith("DoorHeader_") for name in nodes)


def test_simple_glb_export_includes_supported_special_elements(tmp_path: Path) -> None:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json")).model_copy(
        update={
            "special_elements": [
                ArchitecturalElement(
                    id="stairs",
                    kind="staircase",
                    center=Point2D(x=0.8, y=1.0),
                    width_m=0.8,
                    depth_m=1.6,
                    metadata={"step_count": 6},
                ),
                ArchitecturalElement(
                    id="lift",
                    kind="lift",
                    center=Point2D(x=2.0, y=1.0),
                    width_m=1.0,
                    depth_m=1.0,
                ),
                ArchitecturalElement(
                    id="counter",
                    kind="kitchen_counter",
                    center=Point2D(x=3.2, y=1.0),
                    width_m=1.0,
                    depth_m=0.55,
                ),
                ArchitecturalElement(
                    id="balcony",
                    kind="balcony",
                    polygon=[
                        Point2D(x=0.4, y=2.2),
                        Point2D(x=1.6, y=2.2),
                        Point2D(x=1.6, y=2.8),
                        Point2D(x=0.4, y=2.8),
                    ],
                ),
            ]
        }
    )
    output = export_simple_glb(model, tmp_path / "specials.glb")
    loaded = trimesh.load(output, force="scene")
    nodes = {str(name) for name in loaded.graph.nodes_geometry}
    for kind in ("staircase", "lift", "kitchen_counter", "balcony"):
        assert any(name.startswith("Special_") and kind in name for name in nodes)
