from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh

from architecture_walkthrough.geometry.models import FloorPlanModel, FurniturePlacement, Point2D, WallSegment

RGBA = tuple[int, int, int, int]

COLORS: dict[str, RGBA] = {
    "floor": (218, 205, 185, 255),
    "wall": (230, 226, 218, 255),
    "wall_cap": (105, 108, 108, 255),
    "bed": (116, 116, 63, 255),
    "sofa": (119, 117, 63, 255),
    "chair": (138, 120, 87, 255),
    "dining_table": (118, 72, 42, 255),
    "side_table": (112, 67, 39, 255),
    "coffee_table": (116, 72, 41, 255),
    "kitchen_counter": (128, 34, 25, 255),
    "rug": (146, 76, 58, 255),
    "plant": (40, 118, 45, 255),
    "pillow": (232, 224, 203, 255),
    "fixture": (238, 238, 232, 255),
    "dark": (48, 48, 44, 255),
    "pot": (104, 69, 43, 255),
}


def _paint(mesh: trimesh.Trimesh, color: RGBA) -> trimesh.Trimesh:
    mesh.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(mesh.vertices), 1))
    return mesh


def _box(extents: list[float], center: list[float], color: RGBA, rotation_deg: float = 0.0) -> trimesh.Trimesh:
    transform = trimesh.transformations.rotation_matrix(math.radians(rotation_deg), [0, 0, 1])
    transform[:3, 3] = center
    return _paint(trimesh.creation.box(extents=extents, transform=transform), color)


def _oriented_offset(item: FurniturePlacement, local_x: float, local_y: float) -> tuple[float, float]:
    angle = math.radians(item.rotation_deg)
    return (
        item.center.x + local_x * math.cos(angle) - local_y * math.sin(angle),
        item.center.y + local_x * math.sin(angle) + local_y * math.cos(angle),
    )


def _part(item: FurniturePlacement, local_x: float, local_y: float, width: float, depth: float, height: float, color: RGBA, z: float | None = None) -> trimesh.Trimesh:
    x, y = _oriented_offset(item, local_x, local_y)
    return _box([width, depth, height], [x, y, height / 2 if z is None else z], color, item.rotation_deg)


def _wall_mesh(wall: WallSegment) -> trimesh.Trimesh:
    length = wall.start.distance_to(wall.end)
    if length <= 0:
        raise ValueError("wall segment length must be positive")
    angle = math.atan2(wall.end.y - wall.start.y, wall.end.x - wall.start.x)
    center_x = (wall.start.x + wall.end.x) / 2
    center_y = (wall.start.y + wall.end.y) / 2
    transform = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    transform[:3, 3] = [center_x, center_y, wall.height_m / 2]
    mesh = trimesh.creation.box(
        extents=[length, wall.thickness_m, wall.height_m],
        transform=transform,
    )
    return _paint(mesh, COLORS["wall"])


def _wall_cap_mesh(wall: WallSegment) -> trimesh.Trimesh:
    length = wall.start.distance_to(wall.end)
    angle = math.atan2(wall.end.y - wall.start.y, wall.end.x - wall.start.x)
    center_x = (wall.start.x + wall.end.x) / 2
    center_y = (wall.start.y + wall.end.y) / 2
    transform = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    transform[:3, 3] = [center_x, center_y, wall.height_m + 0.035]
    mesh = trimesh.creation.box(extents=[length, wall.thickness_m * 1.2, 0.07], transform=transform)
    return _paint(mesh, COLORS["wall_cap"])


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
    return _paint(trimesh.creation.box(extents=[width, depth, 0.1], transform=transform), COLORS["floor"])


def _chair_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    width = min(item.width_m, 0.7)
    depth = min(item.depth_m, 0.7)
    return [
        _part(item, 0, 0, width, depth, 0.18, COLORS["chair"], 0.35),
        _part(item, 0, depth * 0.42, width, 0.08, 0.75, COLORS["chair"], 0.45),
    ]


def _table_meshes(item: FurniturePlacement, color: RGBA) -> list[trimesh.Trimesh]:
    top_height = 0.10
    leg_w = min(item.width_m, item.depth_m) * 0.10
    leg_w = max(min(leg_w, 0.12), 0.04)
    meshes = [_part(item, 0, 0, item.width_m, item.depth_m, top_height, color, 0.76)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            meshes.append(_part(item, sx * item.width_m * 0.38, sy * item.depth_m * 0.36, leg_w, leg_w, 0.72, COLORS["dark"], 0.36))
    return meshes


def _sofa_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    arm = min(item.depth_m * 0.18, 0.18)
    return [
        _part(item, 0, 0, item.width_m, item.depth_m, 0.32, COLORS["sofa"], 0.28),
        _part(item, 0, item.depth_m * 0.44, item.width_m, arm, 0.82, COLORS["sofa"], 0.42),
        _part(item, -item.width_m * 0.48, 0, arm, item.depth_m, 0.62, COLORS["sofa"], 0.34),
        _part(item, item.width_m * 0.48, 0, arm, item.depth_m, 0.62, COLORS["sofa"], 0.34),
    ]


def _bed_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    pillow_depth = min(item.depth_m * 0.22, 0.45)
    meshes = [
        _part(item, 0, 0, item.width_m, item.depth_m, 0.35, COLORS["bed"], 0.32),
        _part(item, 0, item.depth_m * 0.18, item.width_m * 0.86, item.depth_m * 0.55, 0.12, COLORS["bed"], 0.58),
        _part(item, -item.width_m * 0.24, item.depth_m * 0.40, item.width_m * 0.34, pillow_depth, 0.12, COLORS["pillow"], 0.62),
        _part(item, item.width_m * 0.24, item.depth_m * 0.40, item.width_m * 0.34, pillow_depth, 0.12, COLORS["pillow"], 0.62),
        _part(item, 0, item.depth_m * 0.50, item.width_m, 0.12, 0.85, COLORS["dark"], 0.42),
    ]
    return meshes


def _counter_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    meshes = [
        _part(item, 0, 0, item.width_m, item.depth_m, 0.9, COLORS["kitchen_counter"], 0.45),
        _part(item, 0, 0, item.width_m, item.depth_m, 0.08, COLORS["fixture"], 0.94),
    ]
    sink_w = min(item.width_m * 0.28, 0.6)
    meshes.append(_part(item, item.width_m * 0.22, 0, sink_w, item.depth_m * 0.5, 0.05, COLORS["dark"], 1.0))
    return meshes


def _fixture_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    return [
        _part(item, 0, 0, item.width_m, item.depth_m, 0.35, COLORS["fixture"], 0.22),
        _part(item, 0, item.depth_m * 0.18, item.width_m * 0.65, item.depth_m * 0.42, 0.16, COLORS["dark"], 0.46),
    ]


def _plant_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    radius = max(min(item.width_m, item.depth_m) * 0.35, 0.08)
    pot = trimesh.creation.cylinder(radius=radius, height=0.35, sections=16)
    pot.apply_translation([item.center.x, item.center.y, 0.175])
    leaf = trimesh.creation.icosphere(subdivisions=2, radius=radius * 1.65)
    leaf.apply_scale([1.0, 1.0, 0.65])
    leaf.apply_translation([item.center.x, item.center.y, 0.62])
    return [_paint(pot, COLORS["pot"]), _paint(leaf, COLORS["plant"])]


def _furniture_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    category = item.category.lower()
    if "bed" in category:
        return _bed_meshes(item)
    if "sofa" in category:
        return _sofa_meshes(item)
    if "chair" in category:
        return _chair_meshes(item)
    if "counter" in category or "kitchen" in category:
        return _counter_meshes(item)
    if "fixture" in category or "toilet" in category or "sink" in category:
        return _fixture_meshes(item)
    if "plant" in category:
        return _plant_meshes(item)
    if "rug" in category:
        return [_part(item, 0, 0, item.width_m, item.depth_m, 0.04, COLORS["rug"], 0.03)]
    if "table" in category:
        return _table_meshes(item, COLORS.get(category, COLORS["dining_table"]))
    return [_part(item, 0, 0, item.width_m, item.depth_m, 0.55, COLORS["chair"], 0.28)]


def export_simple_glb(model: FloorPlanModel, output_glb: Path) -> Path:
    if not model.walls:
        raise ValueError("cannot export GLB: floorplan contains no wall geometry")
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    meshes: list[trimesh.Trimesh] = [_floor_mesh(model)]
    meshes.extend(_wall_mesh(wall) for wall in model.walls)
    wall_cap_meshes = [_wall_cap_mesh(wall) for wall in model.walls]
    scene = trimesh.Scene()
    scene.add_geometry(meshes[0], node_name="Floor_Slab", geom_name="Floor_Slab")
    for index, mesh in enumerate(meshes[1:]):
        scene.add_geometry(mesh, node_name=f"Wall_{index:03d}", geom_name=f"Wall_{index:03d}")
    for index, mesh in enumerate(wall_cap_meshes):
        scene.add_geometry(mesh, node_name=f"Wall_Cap_{index:03d}", geom_name=f"Wall_Cap_{index:03d}")
    for index, item in enumerate(model.furniture):
        for part_index, mesh in enumerate(_furniture_meshes(item)):
            scene.add_geometry(
                mesh,
                node_name=f"Furniture_{index:03d}_{part_index:02d}_{item.category}",
                geom_name=f"Furniture_{index:03d}_{part_index:02d}_{item.category}",
            )
    exported = scene.export(file_type="glb")
    if isinstance(exported, str):
        output_glb.write_text(exported, encoding="utf-8")
    else:
        output_glb.write_bytes(exported)
    return output_glb
