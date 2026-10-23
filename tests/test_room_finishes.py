from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time

import numpy as np
from pydantic import ValidationError
import pytest
from shapely.geometry import Polygon
import trimesh

from architecture_walkthrough.geometry.models import (
    DoorOpening, FloorPlanModel, MaterialAssignment, Point2D, RoomPolygon, SlabPolygon,
    WallSegment, WindowOpening,
)
from architecture_walkthrough.geometry.wall_graph import GraphFace, match_faces_to_rooms
from architecture_walkthrough.scene.finish_materials import apply_finish_regions
from architecture_walkthrough.scene.finish_regions import partition_finish_mesh
from architecture_walkthrough.scene.opening_builder import openings_for_wall
from architecture_walkthrough.scene.pbr_materials import build_material_plan, load_material_registry
from architecture_walkthrough.scene.simple_glb import _floor_meshes_for_model, export_simple_glb
from architecture_walkthrough.scene.wall_builder import split_wall_meshes


def room(points, **kwargs):
    return RoomPolygon(points=[Point2D(x=x, y=y) for x, y in points], **kwargs)


def plan(model):
    return build_material_plan(model, load_material_registry(Path("assets/textures/material_registry.yaml")))


def partition(mesh, regions, wall=None):
    vertices, faces, labels, _ = partition_finish_mesh(
        mesh.vertices.tolist(), mesh.faces.tolist(), regions, wall=wall,
    )
    result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert result.is_watertight
    assert result.is_winding_consistent
    assert result.volume == pytest.approx(mesh.volume, abs=1e-8)
    assert result.area == pytest.approx(mesh.area, abs=1e-8)
    np.testing.assert_allclose(result.bounds, mesh.bounds, atol=1e-8)
    return result, labels


def areas(mesh, labels):
    grouped = defaultdict(float)
    for area, label in zip(mesh.area_faces, labels):
        grouped[label] += area
    return grouped


@pytest.mark.parametrize("field", ["floor_material_preset", "wall_material_preset"])
@pytest.mark.parametrize("value", ["", "   ", "a" * 129, 3])
def test_room_finish_fields_reject_invalid_values(field, value):
    with pytest.raises(ValidationError):
        room([(0, 0), (2, 0), (2, 2)], **{field: value})


def test_room_finishes_survive_json_and_face_regeneration(tmp_path):
    original = room([(0, 0), (4, 0), (4, 3), (0, 3)], id="stable", face_id="persistent",
                    floor_material_preset="custom_wood", wall_material_preset="ceramic_tile")
    model = FloorPlanModel(rooms=[original])
    path = tmp_path / "corrected.json"
    model.save_json(path)
    loaded = FloorPlanModel.load_json(path)
    regenerated = match_faces_to_rooms([
        GraphFace(face_id="changed", polygon=Polygon([(0, 0), (4.1, 0), (4.1, 3), (0, 3)]))
    ], loaded.rooms)
    assert regenerated[0].id == "stable"
    assert regenerated[0].face_id == "persistent"
    assert regenerated[0].floor_material_preset == "custom_wood"
    assert regenerated[0].wall_material_preset == "ceramic_tile"
    assert room([(0, 0), (1, 0), (1, 1)]).floor_material_preset is None


def test_resolved_floor_contract_has_direct_legacy_semantic_scene_precedence():
    rooms = [room([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)], id=str(i), name=name)
             for i, name in enumerate(["Bedroom", "Bedroom", "Kitchen", "Living"])]
    rooms[0].floor_material_preset = "marble"
    model = FloorPlanModel(rooms=rooms, material_assignments=[
        MaterialAssignment(target="0", preset="ceramic_tile"),
        MaterialAssignment(target="1", preset="marble"),
    ])
    resolved = plan(model)
    assert resolved["room_floor_presets"] == ["marble", "marble", "ceramic_tile", "marble"]
    assert [region["source"] for region in resolved["floor_regions"]] == [
        "room", "assignment", "semantic", "scene",
    ]
    assert resolved["wall_regions"] == []


@pytest.mark.parametrize("foundation", ["rooms", "slab", "fallback"])
def test_adjacent_floor_finishes_partition_existing_foundation(foundation):
    walls = [WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=4, y=0)),
             WallSegment(start=Point2D(x=4, y=0), end=Point2D(x=4, y=4))]
    rooms = [room([(0, 0), (2, 0), (2, 4), (0, 4)], floor_material_preset="wood"),
             room([(2, 0), (4, 0), (4, 4), (2, 4)], floor_material_preset="ceramic_tile")]
    model = FloorPlanModel(walls=walls, rooms=rooms)
    if foundation == "slab":
        model.slabs = [SlabPolygon(points=[Point2D(x=x, y=y) for x, y in
                                          [(-1, -1), (5, -1), (5, 5), (-1, 5)]], thickness_m=0.3)]
    if foundation == "fallback":
        model.walls[1].end.y = 20
    resolved = plan(model)
    for original in _floor_meshes_for_model(model):
        updated, labels = partition(original, resolved["floor_regions"])
        grouped = areas(updated, labels)
        assert grouped["wood"] == pytest.approx(8)
        assert grouped["ceramic_tile"] == pytest.approx(8)
        # Only the original top is recolored; bottoms/edges retain their finish.
        assert all(normal[2] > 0.999 for normal, label in zip(updated.face_normals, labels) if label)


def test_concave_floor_and_conflicting_overlap_keep_original_material():
    model = FloorPlanModel(rooms=[
        room([(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)], floor_material_preset="wood"),
        room([(0, 0), (2, 0), (2, 2), (0, 2)], floor_material_preset="ceramic_tile"),
    ])
    resolved = plan(model)
    original = trimesh.creation.box(extents=[4, 4, 0.1])
    original.apply_translation([2, 2, -0.05])
    updated, labels = partition(original, resolved["floor_regions"])
    grouped = areas(updated, labels)
    assert grouped["wood"] == pytest.approx(2)
    assert grouped["ceramic_tile"] == pytest.approx(1)
    assert any("overlap" in warning for warning in resolved["warnings"])
    # The shared 3m² is retained as original; every selected triangle is
    # bounded by the actual concave/overlap edges.
    for centroid, label in zip(updated.triangles_center, labels):
        if 0 < centroid[0] < 1 and 0 < centroid[1] < 2 and abs(centroid[2]) < 1e-8:
            assert label is None


def test_long_wall_supports_distinct_rooms_both_sides_and_openings():
    wall = WallSegment(id="shared", start=Point2D(x=0, y=0), end=Point2D(x=8, y=0),
                       thickness_m=0.2, height_m=3)
    model = FloorPlanModel(walls=[wall], rooms=[
        room([(0, 0), (3, 0), (3, 3), (0, 3)], wall_material_preset="wood"),
        room([(3, 0), (8, 0), (8, 3), (3, 3)], wall_material_preset="ceramic_tile"),
        room([(0, -3), (8, -3), (8, 0), (0, 0)], wall_material_preset="painted_wall"),
    ], doors=[DoorOpening(wall_id="shared", center=Point2D(x=2, y=0),
                          start_offset_m=1.5, end_offset_m=2.5, width_m=1, height_m=2)],
       windows=[WindowOpening(wall_id="shared", center=Point2D(x=5, y=0),
                              start_offset_m=4.5, end_offset_m=5.5, width_m=1,
                              sill_height_m=1, height_m=1)])
    resolved = plan(model)
    meshes = split_wall_meshes(wall, openings_for_wall(wall, model.doors, model.windows, 0,
                                                     model.walls), (230, 226, 218, 255))
    grouped = defaultdict(float)
    for original in meshes:
        updated, labels = partition(original, resolved["wall_regions"], wall.model_dump())
        for key, area in areas(updated, labels).items():
            grouped[key] += area
        for center, normal, label in zip(updated.triangles_center, updated.face_normals, labels):
            if label == "wood":
                assert center[0] <= 3 + 1e-8 and normal[1] > 0.999
            if label == "ceramic_tile":
                assert center[0] >= 3 - 1e-8 and normal[1] > 0.999
            if label == "painted_wall":
                assert normal[1] < -0.999
    assert grouped["wood"] == pytest.approx(3 * 3 - 2)
    assert grouped["ceramic_tile"] == pytest.approx(5 * 3 - 1)
    assert grouped["painted_wall"] == pytest.approx(8 * 3 - 2 - 1)


def test_wall_overlap_retains_original_finish_and_warns():
    wall = WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=8, y=0), height_m=3)
    model = FloorPlanModel(walls=[wall], rooms=[
        room([(0, 0), (5, 0), (5, 3), (0, 3)], wall_material_preset="wood"),
        room([(3, 0), (8, 0), (8, 3), (3, 3)], wall_material_preset="ceramic_tile"),
    ])
    resolved = plan(model)
    assert any("overlap" in warning for warning in resolved["warnings"])
    assert [(r["start_m"], r["end_m"]) for r in resolved["wall_regions"]] == [(0, 3), (5, 8)]
    updated, labels = partition(split_wall_meshes(wall, [], (230, 226, 218, 255))[0],
                                resolved["wall_regions"], wall.model_dump())
    assert areas(updated, labels)["wood"] == pytest.approx(9)
    assert areas(updated, labels)["ceramic_tile"] == pytest.approx(9)


def test_simple_export_contains_actual_floor_and_wall_material_primitives(tmp_path):
    wall = WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=4, y=0))
    model = FloorPlanModel(walls=[wall], rooms=[
        room([(0, 0), (2, 0), (2, 3), (0, 3)], floor_material_preset="wood",
             wall_material_preset="ceramic_tile"),
        room([(2, 0), (4, 0), (4, 3), (2, 3)], floor_material_preset="marble"),
    ])
    loaded = trimesh.load(export_simple_glb(model, tmp_path / "finishes.glb"), force="scene")
    names = {getattr(mesh.visual.material, "name", "") for mesh in loaded.geometry.values()
             if isinstance(mesh.visual, trimesh.visual.TextureVisuals)}
    assert {"PBR_wood", "PBR_marble", "PBR_ceramic_tile"} <= names
    resolved = plan(model)
    mesh = _floor_meshes_for_model(model)[0]
    adapter = apply_finish_regions(mesh, resolved["floor_regions"], resolved)
    assert adapter.is_watertight
    assert adapter.volume == pytest.approx(mesh.volume)


def test_no_room_overrides_keeps_material_region_rendering_disabled():
    resolved = plan(FloorPlanModel(rooms=[room([(0, 0), (4, 0), (4, 4)], name="Bedroom")]))
    assert resolved["floor_regions_enabled"] is False
    assert resolved["wall_regions"] == []


def test_wall_concave_border_excludes_embedded_and_perpendicular_corner_strips():
    walls = [WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=6, y=0)),
             WallSegment(start=Point2D(x=0.01, y=0), end=Point2D(x=0.01, y=3)),
             WallSegment(start=Point2D(x=1, y=0.02), end=Point2D(x=1, y=2))]
    model = FloorPlanModel(walls=walls, rooms=[room(
        [(0, 0), (2, 0), (2, 2), (4, 2), (4, 0), (6, 0), (6, 3), (0, 3)],
        wall_material_preset="wood",
    )])
    regions = plan(model)["wall_regions"]
    assert [(r["start_m"], r["end_m"]) for r in regions if r["wall_index"] == 0] == [(0, 2), (4, 6)]
    assert not any(r["wall_index"] in {1, 2} for r in regions)


def test_finish_partition_culls_distant_regions_and_bounds_many_room_meshes():
    original = trimesh.creation.box(extents=[12, 12, 0.1])
    original.apply_translation([6, 6, -0.05])
    regions = [{"points": [[x, y], [x + 2, y], [x + 2, y + 2], [x, y + 2]],
                "preset": "wood" if x % 4 else "marble"}
               for x in range(0, 12, 2) for y in range(0, 12, 2)]
    started = time.perf_counter()
    updated, labels = partition(original, regions)
    assert time.perf_counter() - started < 5
    assert len(updated.faces) < 3000
    assert sum(area for label, area in areas(updated, labels).items() if label) == pytest.approx(144)
    far = [{"points": [[100 + x, y], [101 + x, y], [101 + x, y + 1], [100 + x, y + 1]],
            "preset": "wood"} for x in range(100) for y in range(3)]
    culled, no_labels = partition(original, far)
    assert len(culled.faces) == len(original.faces)
    assert not any(no_labels)


def test_original_face_colors_survive_unpainted_parts():
    original = trimesh.creation.box(extents=[4, 4, 0.1])
    original.apply_translation([2, 2, -0.05])
    original.visual.face_colors = np.array([[index * 15, 100, 180, 255]
                                          for index in range(len(original.faces))], dtype=np.uint8)
    model = FloorPlanModel(rooms=[room([(0, 0), (2, 0), (2, 4), (0, 4)], floor_material_preset="wood")])
    resolved = plan(model)
    _v, _f, labels, sources = partition_finish_mesh(original.vertices.tolist(), original.faces.tolist(),
                                                  resolved["floor_regions"])
    updated = apply_finish_regions(original, resolved["floor_regions"], resolved)
    for index, (label, source) in enumerate(zip(labels, sources)):
        if label is None:
            material = updated.visual.material.materials[updated.visual.face_materials[index]]
            np.testing.assert_array_equal(material.baseColorFactor, original.visual.face_colors[source])


def test_simple_export_rebuilds_stale_plan_and_does_not_recolor_other_rooms(tmp_path):
    from architecture_walkthrough.scene.pbr_materials import MATERIAL_PLAN_METADATA_KEY

    model = FloorPlanModel(walls=[WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=4, y=0))], rooms=[
        room([(0, 0), (2, 0), (2, 3), (0, 3)], floor_material_preset="wood", wall_material_preset="wood"),
        room([(2, 0), (4, 0), (4, 3), (2, 3)], name="Bedroom"),
    ])
    model.metadata[MATERIAL_PLAN_METADATA_KEY] = plan(model)
    model.rooms[0].floor_material_preset = "ceramic_tile"
    model.rooms[0].wall_material_preset = None
    loaded = trimesh.load(export_simple_glb(model, tmp_path / "changed.glb"), force="scene")
    names = {getattr(getattr(mesh.visual, "material", None), "name", "")
             for mesh in loaded.geometry.values()}
    assert "PBR_ceramic_tile" in names
    assert "PBR_wood" not in names
    model.rooms[0].floor_material_preset = None
    loaded = trimesh.load(export_simple_glb(model, tmp_path / "cleared.glb"), force="scene")
    assert not any(str(name).startswith("PBR_") for name in [
        getattr(getattr(mesh.visual, "material", None), "name", "") for mesh in loaded.geometry.values()
    ])


def test_preview_export_uses_configured_registry(tmp_path):
    from architecture_walkthrough.config import AppConfig, TextureSettings
    from architecture_walkthrough.scene.export_glb import export_floorplan_glb

    registry = tmp_path / "custom.yaml"
    registry.write_text("materials:\n  custom_cork:\n    base_color: [0.2, 0.4, 0.6, 1]\n")
    model = FloorPlanModel(walls=[WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=4, y=0))], rooms=[
        room([(0, 0), (4, 0), (4, 3), (0, 3)], floor_material_preset="custom_cork"),
    ])
    config = AppConfig(textures=TextureSettings(registry_path=registry))
    loaded = trimesh.load(export_floorplan_glb(model, tmp_path / "custom.glb", config), force="scene")
    assert "PBR_custom_cork" in {
        getattr(getattr(mesh.visual, "material", None), "name", "") for mesh in loaded.geometry.values()
    }


def test_actual_preview_overlap_with_automatic_room_retains_original_surface(tmp_path):
    model = FloorPlanModel(walls=[
        WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=4, y=0)),
        WallSegment(start=Point2D(x=4, y=0), end=Point2D(x=4, y=3)),
    ], rooms=[
        room([(0, 0), (3, 0), (3, 3), (0, 3)], floor_material_preset="wood"),
        room([(2, 0), (4, 0), (4, 3), (2, 3)], name="Kitchen"),
    ])
    resolved = plan(model)
    assert any("original surface material retained" in warning for warning in resolved["warnings"])
    loaded = trimesh.load(export_simple_glb(model, tmp_path / "overlap.glb"), force="scene")
    wood_area = sum(mesh.area for mesh in loaded.geometry.values()
                    if getattr(getattr(mesh.visual, "material", None), "name", "") == "PBR_wood")
    assert wood_area == pytest.approx(6)
    assert not any(getattr(getattr(mesh.visual, "material", None), "name", "") == "PBR_ceramic_tile"
                   for mesh in loaded.geometry.values())
