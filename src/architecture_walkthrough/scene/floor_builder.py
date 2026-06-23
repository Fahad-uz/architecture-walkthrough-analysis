from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import triangulate
import trimesh

from architecture_walkthrough.geometry.models import FloorPlanModel, Point2D, RoomPolygon


def _paint(mesh: trimesh.Trimesh, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    mesh.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(mesh.vertices), 1))
    return mesh


def polygon_floor_mesh(
    polygon_points: list[Point2D],
    thickness_m: float,
    color: tuple[int, int, int, int],
) -> trimesh.Trimesh:
    polygon = Polygon([(point.x, point.y) for point in polygon_points])
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError("floor polygon is invalid")
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for triangle in triangulate(polygon):
        clipped = triangle.intersection(polygon)
        if clipped.is_empty or clipped.area <= 0:
            continue
        if clipped.geom_type == "Polygon":
            coords = list(clipped.exterior.coords)[:-1]
            if len(coords) < 3:
                continue
            base = len(vertices)
            vertices.extend([[x, y, 0.0] for x, y in coords])
            vertices.extend([[x, y, -thickness_m] for x, y in coords])
            for index in range(1, len(coords) - 1):
                faces.append([base, base + index, base + index + 1])
                faces.append([base + len(coords), base + len(coords) + index + 1, base + len(coords) + index])
            for index in range(len(coords)):
                nxt = (index + 1) % len(coords)
                faces.append([base + index, base + nxt, base + len(coords) + nxt])
                faces.append([base + index, base + len(coords) + nxt, base + len(coords) + index])
    if not vertices or not faces:
        raise ValueError("floor polygon produced no triangles")
    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=True)
    return _paint(mesh, color)


def room_floor_meshes(
    rooms: list[RoomPolygon],
    thickness_m: float,
    color: tuple[int, int, int, int],
) -> list[trimesh.Trimesh]:
    return [polygon_floor_mesh(room.points, thickness_m, color) for room in rooms]


def fallback_floor_mesh(model: FloorPlanModel, thickness_m: float, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    points = [point for wall in model.walls for point in (wall.start, wall.end)]
    if not points:
        raise ValueError("cannot derive fallback floor without wall points")
    min_x = min(point.x for point in points)
    min_y = min(point.y for point in points)
    max_x = max(point.x for point in points)
    max_y = max(point.y for point in points)
    pad = 0.3
    return polygon_floor_mesh(
        [
            Point2D(x=min_x - pad, y=min_y - pad),
            Point2D(x=max_x + pad, y=min_y - pad),
            Point2D(x=max_x + pad, y=max_y + pad),
            Point2D(x=min_x - pad, y=max_y + pad),
        ],
        thickness_m,
        color,
    )
