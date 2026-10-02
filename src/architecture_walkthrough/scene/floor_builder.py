from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon  # type: ignore[import-untyped]
from shapely.geometry.polygon import orient  # type: ignore[import-untyped]
import trimesh

from architecture_walkthrough.geometry.models import FloorPlanModel, Point2D, RoomPolygon


def _paint(mesh: trimesh.Trimesh, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.tile(np.array(color, dtype=np.uint8), (len(mesh.vertices), 1)),
    )
    return mesh


def polygon_floor_mesh(
    polygon_points: list[Point2D],
    thickness_m: float,
    color: tuple[int, int, int, int],
) -> trimesh.Trimesh:
    polygon = Polygon([(point.x, point.y) for point in polygon_points])
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError("floor polygon is invalid")
    if not np.isfinite(thickness_m) or thickness_m <= 0:
        raise ValueError("floor thickness must be positive")
    # Triangulate one shared top surface, then extrude only its outer boundary.
    # Extruding each clipped triangle separately left duplicate internal walls
    # and inverted top normals, producing visible seams and broken collision.
    coords = np.asarray(orient(polygon, sign=1.0).exterior.coords[:-1], dtype=float)
    remaining = list(range(len(coords)))
    triangles: list[list[int]] = []

    def cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ab, ac = b - a, c - a
        return float(ab[0] * ac[1] - ab[1] * ac[0])

    tolerance = max(float(polygon.area), 1.0) * 1e-12
    while len(remaining) > 3:
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            a, b, c = coords[[previous, current, following]]
            turn = cross(a, b, c)
            if abs(turn) <= tolerance:
                remaining.pop(position)
                break
            if turn < 0:
                continue
            blocked = any(
                min(cross(a, b, coords[other]), cross(b, c, coords[other]), cross(c, a, coords[other])) >= -tolerance
                for other in remaining if other not in {previous, current, following}
            )
            if blocked:
                continue
            triangles.append([previous, current, following])
            remaining.pop(position)
            break
        else:
            raise ValueError("floor polygon could not be triangulated")
    if len(remaining) == 3:
        triangles.append(remaining)
    mesh = trimesh.creation.extrude_triangulation(coords, np.asarray(triangles), thickness_m)
    mesh.apply_translation([0.0, 0.0, -thickness_m])
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
