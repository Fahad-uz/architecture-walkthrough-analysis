from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh

from architecture_walkthrough.geometry.models import FloorPlanModel, FurniturePlacement, Point2D, WallSegment


def _wall_mesh(wall: WallSegment) -> trimesh.Trimesh:
    length = wall.start.distance_to(wall.end)
    if length <= 0:
        raise ValueError("wall segment length must be positive")
    angle = math.atan2(wall.end.y - wall.start.y, wall.end.x - wall.start.x)
    center_x = (wall.start.x + wall.end.x) / 2
    center_y = (wall.start.y + wall.end.y) / 2
    transform = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    transform[:3, 3] = [center_x, center_y, wall.height_m / 2]
    return trimesh.creation.box(
        extents=[length, wall.thickness_m, wall.height_m],
        transform=transform,
    )


def _bounds(points: list[Point2D]) -> tuple[float, float, float, float]:
    if not points:
        return -2.0, -2.0, 2.0, 2.0
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _floor_mesh(model: FloorPlanModel) -> trimesh.Trimesh:
    points = [point for wall in model.walls for point in (wall.start, wall.end)]
    min_x, min_y, max_x, max_y = _bounds(points)
    padding = 0.5
    width = max(max_x - min_x + padding * 2, 1.0)
    depth = max(max_y - min_y + padding * 2, 1.0)
    transform = np.eye(4)
    transform[:3, 3] = [(min_x + max_x) / 2, (min_y + max_y) / 2, -0.05]
    return trimesh.creation.box(extents=[width, depth, 0.1], transform=transform)


def _furniture_mesh(item: FurniturePlacement) -> trimesh.Trimesh:
    height = 0.45 if item.category.lower() in {"table", "coffee_table", "chair"} else 0.8
    transform = trimesh.transformations.rotation_matrix(math.radians(item.rotation_deg), [0, 0, 1])
    transform[:3, 3] = [item.center.x, item.center.y, height / 2]
    return trimesh.creation.box(
        extents=[item.width_m, item.depth_m, height],
        transform=transform,
    )


def export_simple_glb(model: FloorPlanModel, output_glb: Path) -> Path:
    if not model.walls:
        raise ValueError("cannot export GLB: floorplan contains no wall geometry")
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    meshes: list[trimesh.Trimesh] = [_floor_mesh(model)]
    meshes.extend(_wall_mesh(wall) for wall in model.walls)
    scene = trimesh.Scene()
    scene.add_geometry(meshes[0], node_name="Floor_Slab", geom_name="Floor_Slab")
    for index, mesh in enumerate(meshes[1:]):
        scene.add_geometry(mesh, node_name=f"Wall_{index:03d}", geom_name=f"Wall_{index:03d}")
    for index, item in enumerate(model.furniture):
        scene.add_geometry(
            _furniture_mesh(item),
            node_name=f"Furniture_{index:03d}_{item.category}",
            geom_name=f"Furniture_{index:03d}_{item.category}",
        )
    exported = scene.export(file_type="glb")
    if isinstance(exported, str):
        output_glb.write_text(exported, encoding="utf-8")
    else:
        output_glb.write_bytes(exported)
    return output_glb
