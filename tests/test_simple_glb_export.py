from __future__ import annotations

import math
from pathlib import Path

import pytest
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
    _furniture_meshes,
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


def test_simple_glb_export_converts_z_up_geometry_to_gltf_y_up(
    tmp_path: Path,
) -> None:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json"))
    output = export_simple_glb(model, tmp_path / "y-up.glb")
    loaded = trimesh.load(output, force="scene")
    transform, geometry_name = loaded.graph.get("Floor_000")
    floor = loaded.geometry[geometry_name].copy()
    floor.apply_transform(transform)

    # The foundation spans the plan in X/Z and is thin on the Y-up axis.
    assert floor.extents[0] > 1.0
    assert floor.extents[2] > 1.0
    assert floor.extents[1] == pytest.approx(0.10, abs=1e-5)
    assert floor.bounds[1][1] == pytest.approx(0.0, abs=1e-5)
    wall_transform, wall_geometry_name = loaded.graph.get("Wall_000_00")
    wall = loaded.geometry[wall_geometry_name].copy()
    wall.apply_transform(wall_transform)
    assert wall.bounds[0][1] >= -1e-5
    assert wall.bounds[1][1] > 2.0
    assert loaded.bounds[1][1] <= max(wall.height_m for wall in model.walls) + 0.5


def test_build_model_defaults_to_pure_python_glb_export(tmp_path: Path) -> None:
    # force=True: this test exercises the export path; the quality gate has its
    # own coverage in test_validation_gate.py (the legacy fixture has no scale).
    output = build_model(
        Path("tests/fixtures/sample_floorplan.json"),
        tmp_path / "building.glb",
        AppConfig(),
        force=True,
    )
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


@pytest.mark.parametrize(
    ("category", "width", "depth", "rotation", "maximum_parts"),
    [
        ("bed", 0.90, 0.80, 0.0, 6),
        ("sofa", 1.80, 0.70, 90.0, 8),
        ("dining_table", 1.10, 0.60, -30.0, 9),
        ("kitchen_counter", 1.20, 0.50, 90.0, 7),
        ("kitchen_sink", 0.60, 0.50, 0.0, 5),
        ("kitchen_stove", 0.60, 0.50, -90.0, 8),
        ("plant", 0.35, 0.55, 0.0, 4),
        ("plant", 5.00, 0.20, 33.0, 4),
        ("plant", 0.20, 5.00, -17.0, 4),
        ("chair", 0.45, 0.45, 25.0, 6),
        ("dining_table", 0.01, 0.01, 12.0, 9),
    ],
)
def test_simple_furniture_is_bounded_and_lightweight(
    category: str,
    width: float,
    depth: float,
    rotation: float,
    maximum_parts: int,
) -> None:
    item = FurniturePlacement(
        category=category,
        center=Point2D(x=5.0, y=4.0),
        width_m=width,
        depth_m=depth,
        rotation_deg=rotation,
    )
    meshes = _furniture_meshes(item)
    assert 1 <= len(meshes) <= maximum_parts

    angle = math.radians(rotation)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    for mesh in meshes:
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        assert mesh.area > 0
        for world_x, world_y, world_z in mesh.vertices:
            assert all(math.isfinite(float(component)) for component in (world_x, world_y, world_z))
            offset_x = float(world_x) - item.center.x
            offset_y = float(world_y) - item.center.y
            local_x = offset_x * cos_angle + offset_y * sin_angle
            local_y = -offset_x * sin_angle + offset_y * cos_angle
            assert abs(local_x) <= width / 2 + 1e-6
            assert abs(local_y) <= depth / 2 + 1e-6

    if category == "kitchen_stove":
        assert sum(len(mesh.faces) for mesh in meshes) < 500


def test_simple_glb_uses_semantic_furniture_part_names(tmp_path: Path) -> None:
    categories = [
        "sofa",
        "bed",
        "dining_table",
        "kitchen_counter",
        "kitchen_sink",
        "kitchen_stove",
        "plant",
    ]
    furniture = [
        FurniturePlacement(
            category=category,
            center=Point2D(x=1.0 + index * 1.1, y=1.5),
            width_m=1.0 if category not in {"kitchen_sink", "kitchen_stove", "plant"} else 0.55,
            depth_m=0.60 if category != "plant" else 0.45,
        )
        for index, category in enumerate(categories)
    ]
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json")).model_copy(
        update={"furniture": furniture}
    )
    output = export_simple_glb(model, tmp_path / "semantic-furniture.glb")
    loaded = trimesh.load(output, force="scene")
    nodes = {str(name) for name in loaded.graph.nodes_geometry}

    for expected in (
        "sofa_Base",
        "sofa_Seat_Cushion_01",
        "bed_Duvet",
        "dining_table_Leg_",
        "kitchen_counter_Front_A",
        "kitchen_sink_Rim",
        "kitchen_stove_Burner_",
        "plant_Leaf_01",
    ):
        assert any(expected in name for name in nodes), expected


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
