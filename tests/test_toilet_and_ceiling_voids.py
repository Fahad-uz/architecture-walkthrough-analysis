from __future__ import annotations

import math
from pathlib import Path
import shutil

import numpy as np
import pytest
import trimesh

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.models import (
    CeilingSettings,
    FloorPlanModel,
    FurniturePlacement,
    Point2D,
    RoomPolygon,
    WallSegment,
)
from architecture_walkthrough.scene.blender_generator import generate_glb_with_blender
from architecture_walkthrough.scene.simple_glb import _furniture_meshes, export_simple_glb


@pytest.mark.parametrize("rotation", [0, 90, -35])
def test_toilet_has_hollow_seat_and_watertight_parts_inside_rotated_footprint(
    rotation: float,
) -> None:
    item = FurniturePlacement(
        category="bathroom_toilet", center=Point2D(x=2, y=3),
        width_m=0.42, depth_m=0.72, rotation_deg=rotation,
    )
    meshes = _furniture_meshes(item)
    parts = {mesh.metadata["part_name"]: mesh for mesh in meshes}
    assert {"Pedestal", "Bowl", "Seat", "Cistern"} <= parts.keys()
    angle = math.radians(rotation)
    inverse = np.array([[math.cos(angle), math.sin(angle)],
                        [-math.sin(angle), math.cos(angle)]])
    for mesh in meshes:
        assert mesh.is_watertight
        assert mesh.is_winding_consistent
        assert mesh.volume > 0
        local_xy = (mesh.vertices[:, :2] - [2, 3]) @ inverse.T
        assert np.max(np.abs(local_xy[:, 0])) <= item.width_m / 2 + 1e-8
        assert np.max(np.abs(local_xy[:, 1])) <= item.depth_m / 2 + 1e-8
        assert mesh.bounds[0, 2] >= -1e-8
    seat = parts["Seat"]
    local_centers = (seat.triangles_center[:, :2] - [2, 3]) @ inverse.T
    local_centers[:, 1] += item.depth_m * 0.12
    radii = np.linalg.norm(local_centers / [item.width_m / 2, item.depth_m * 0.74 / 2], axis=1)
    # No seat triangle caps the cavity; a bathtub-like filled slab fails this.
    assert np.min(radii) > 0.75
    assert parts["Cistern"].bounds[1, 2] > parts["Seat"].bounds[1, 2] + 0.2


def _plan(void_ids: object) -> FloorPlanModel:
    def room(room_id: str, x1: float, x2: float) -> RoomPolygon:
        return RoomPolygon(id=room_id, points=[Point2D(x=x, y=y)
                           for x, y in [(x1, 0), (x2, 0), (x2, 4), (x1, 4)]])

    return FloorPlanModel(
        walls=[WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=5, y=0))],
        rooms=[room("stairs", 0, 1.5), room("living", 1.5, 5)],
        ceiling=CeilingSettings(enabled=True, height_m=2.8),
        metadata={"ceiling_void_room_ids": void_ids},
        furniture=[FurniturePlacement(category="toilet", center=Point2D(x=3, y=2),
                                      width_m=0.42, depth_m=0.72, height_m=0.82,
                                      rotation_deg=90)],
    )


def _assert_exported_scene(path: Path, *, void: bool) -> None:
    scene = trimesh.load(path, force="scene")
    ceiling_names = {str(name) for name in scene.graph.nodes_geometry
                     if str(name).startswith("Ceiling_")}
    assert "Ceiling_001" in ceiling_names
    assert ("Ceiling_000" in ceiling_names) is not void
    furniture = []
    for name in scene.graph.nodes_geometry:
        if "Furniture_" not in str(name) or "_toilet_" not in str(name):
            continue
        transform, geometry_name = scene.graph.get(name)
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        furniture.append(mesh)
    assert len(furniture) >= 4
    vertices = np.vstack([mesh.vertices for mesh in furniture])
    # Both exporters write Y-up and rotate the long axis to X here.
    assert vertices[:, 1].max() == pytest.approx(0.82, abs=1e-5)
    assert vertices[:, 1].min() >= -1e-5
    assert vertices[:, 0].min() >= 3 - 0.72 / 2 - 1e-5
    assert vertices[:, 0].max() <= 3 + 0.72 / 2 + 1e-5
    assert np.ptp(vertices[:, 2]) <= 0.42 + 1e-5


@pytest.mark.parametrize("void_ids", [["stairs"], [], "stairs"])
def test_simple_export_only_omits_explicit_ceiling_void_rooms(
    tmp_path: Path, void_ids: object,
) -> None:
    output = export_simple_glb(_plan(void_ids), tmp_path / "room-ceilings.glb")
    _assert_exported_scene(output, void=void_ids == ["stairs"])


@pytest.mark.integration
@pytest.mark.parametrize("void_ids", [["stairs"], []])
def test_blender_export_preserves_toilet_bounds_and_explicit_ceiling_voids(
    tmp_path: Path, void_ids: object,
) -> None:
    if shutil.which("blender") is None:
        pytest.skip("Blender not on PATH")
    output = generate_glb_with_blender(
        _plan(void_ids), tmp_path / "room-ceilings.glb", AppConfig(), mode="none",
    )
    _assert_exported_scene(output, void=void_ids == ["stairs"])
