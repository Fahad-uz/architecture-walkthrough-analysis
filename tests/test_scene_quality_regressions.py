from __future__ import annotations

import json
from pathlib import Path
import shutil
import struct

import numpy as np
from PIL import Image
import pytest
from shapely.geometry import Point, Polygon

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.models import (
    AssetPlacement, DoorOpening, FloorPlanModel, FurniturePlacement, Point2D,
    WallSegment, WindowOpening,
)
from architecture_walkthrough.scene.blender_generator import generate_glb_with_blender
from architecture_walkthrough.scene.floor_builder import polygon_floor_mesh
from architecture_walkthrough.scene.furniture import render_furniture
from architecture_walkthrough.scene.opening_builder import openings_for_wall
from architecture_walkthrough.scene.pbr_materials import load_material_registry, model_with_material_plan
from architecture_walkthrough.scene.window_builder import window_meshes


@pytest.mark.parametrize("clockwise", [False, True])
def test_concave_floor_is_one_watertight_outward_solid(clockwise: bool) -> None:
    coords = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)]
    if clockwise:
        coords.reverse()
    mesh = polygon_floor_mesh([Point2D(x=x, y=y) for x, y in coords], 0.1, (220, 220, 220, 255))
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume == pytest.approx(0.7)
    top = mesh.triangles_center[:, 2] > -1e-8
    assert np.all(mesh.face_normals[top, 2] > 0.99)
    polygon = Polygon(coords)
    assert all(polygon.covers(Point(x, y)) for x, y, _ in mesh.triangles_center[top])


def test_window_frame_matches_authoritative_interval_not_stale_center() -> None:
    wall = WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=4, y=0))
    window = WindowOpening(center=Point2D(x=3, y=0), width_m=2, start_offset_m=0.5, end_offset_m=1.5, sill_height_m=0)
    glass = window_meshes(wall, window, (80, 80, 80, 255), (140, 180, 200, 100))[0]
    assert glass.bounds[0, 0] == pytest.approx(0.5)
    assert glass.bounds[1, 0] == pytest.approx(1.5)
    assert glass.bounds[0, 2] == pytest.approx(0)


def test_unhosted_door_cuts_only_nearest_anonymous_wall() -> None:
    walls = [WallSegment(start=Point2D(x=0, y=y), end=Point2D(x=4, y=y)) for y in (0, 4)]
    door = DoorOpening(center=Point2D(x=2, y=0))
    assert len(openings_for_wall(walls[0], [door], [], 0, walls)) == 1
    assert not openings_for_wall(walls[1], [door], [], 1, walls)


def test_explicit_asset_placements_render_once_without_mutating_saved_model() -> None:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_high_detail_floorplan.json"))
    before = model.model_dump()
    assert len(render_furniture(model)) == 2
    model.asset_placements.append(AssetPlacement(category="plant", center=Point2D(x=4, y=3), width_m=0.5, depth_m=0.5, height_m=1.4))
    resolved = render_furniture(model)
    assert len(resolved) == 3
    assert resolved[-1].height_m == 1.4
    assert model.model_dump()["furniture"] == before["furniture"]
    empty = FloorPlanModel()
    assert render_furniture(empty) == []
    assert len(render_furniture(FloorPlanModel(asset_placements=model.asset_placements))) == 3
    rug = FurniturePlacement(category="rug", center=resolved[0].center, width_m=resolved[0].width_m, depth_m=resolved[0].depth_m)
    assert len(render_furniture(FloorPlanModel(furniture=[rug], asset_placements=[model.asset_placements[0]]))) == 2


def test_relative_texture_files_reach_renderer_and_missing_maps_fall_back(tmp_path: Path) -> None:
    Image.new("RGB", (2, 2), (100, 120, 140)).save(tmp_path / "color.png")
    registry_file = tmp_path / "materials.yaml"
    registry_file.write_text("materials:\n  wood:\n    base_color_texture: color.png\n    normal_texture: missing.png\n    texture_scale_m: 2.0\n", encoding="utf-8")
    registry = load_material_registry(registry_file)
    model = FloorPlanModel()
    rendered = model_with_material_plan(model, registry)
    material = rendered.metadata["blender_material_plan"]["materials"]["wood"]
    assert material["base_color_texture"] == str((tmp_path / "color.png").resolve())
    assert "normal_texture" not in material
    assert material["texture_scale_m"] == 2.0
    assert model.metadata == {}


@pytest.mark.integration
def test_real_blender_embeds_surface_maps_and_preserves_legacy_doorway(tmp_path: Path) -> None:
    if shutil.which("blender") is None:
        pytest.skip("Blender not on PATH")
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_high_detail_floorplan.json"))
    output = generate_glb_with_blender(model, tmp_path / "textured.glb", AppConfig(), mode="none")
    binary = output.read_bytes()
    size = struct.unpack_from("<I", binary, 12)[0]
    gltf = json.loads(binary[20:20 + size])
    assert all("uri" not in image and "bufferView" in image for image in gltf["images"])
    materials = {material["name"]: material for material in gltf["materials"]}
    for name in ("PBR_wood", "PBR_painted_wall", "PBR_ceramic_tile"):
        material = materials[name]
        assert "baseColorTexture" in material["pbrMetallicRoughness"]
        assert "metallicRoughnessTexture" in material["pbrMetallicRoughness"]
        assert "normalTexture" in material
    assert any(node.get("name", "").startswith("DoorLeaf_") for node in gltf["nodes"])
    assert any(node.get("name") == "Wall_Shell" for node in gltf["nodes"])
    assert all("TEXCOORD_0" in primitive["attributes"] for mesh in gltf["meshes"] for primitive in mesh["primitives"])
    lights = gltf["extensions"]["KHR_lights_punctual"]["lights"]
    assert lights
    for light in lights:
        if light["type"] == "directional":
            assert light["intensity"] == pytest.approx(1.2, rel=0.01)
        elif light["type"] == "point":
            assert 6 <= light["intensity"] <= 18.01


@pytest.mark.integration
def test_baked_walls_keep_surface_material_on_cutaway_tops(tmp_path: Path) -> None:
    if shutil.which("blender") is None:
        pytest.skip("Blender not on PATH")
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_high_detail_floorplan.json"))
    config = AppConfig()
    config.bake.draft_samples = 4
    config.bake.draft_lightmap_px = 128
    output = generate_glb_with_blender(model, tmp_path / "baked.glb", config, mode="draft")
    binary = output.read_bytes()
    size = struct.unpack_from("<I", binary, 12)[0]
    gltf = json.loads(binary[20:20 + size])
    wall = next(node for node in gltf["nodes"] if node.get("name") == "Walls_Joined")
    materials = [gltf["materials"][primitive["material"]]
                 for primitive in gltf["meshes"][wall["mesh"]]["primitives"]]
    assert any("emissiveTexture" in material for material in materials)
    assert any("baseColorTexture" in material.get("pbrMetallicRoughness", {})
               and "emissiveTexture" not in material for material in materials)
    assert all("bufferView" in image for image in gltf["images"])
