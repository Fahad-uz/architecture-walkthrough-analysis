from __future__ import annotations

import math

import numpy as np
import trimesh

from architecture_walkthrough.geometry.models import WallSegment, WindowOpening
from architecture_walkthrough.scene.wall_builder import point_offset_on_wall


def _paint(mesh: trimesh.Trimesh, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.tile(np.array(color, dtype=np.uint8), (len(mesh.vertices), 1)),
    )
    return mesh


def window_meshes(
    wall: WallSegment,
    window: WindowOpening,
    frame_color: tuple[int, int, int, int],
    glass_color: tuple[int, int, int, int],
) -> list[trimesh.Trimesh]:
    offset = window.offset_m if window.offset_m is not None else point_offset_on_wall(wall, window.center)
    angle = math.atan2(wall.end.y - wall.start.y, wall.end.x - wall.start.x)
    x = wall.start.x + offset * math.cos(angle)
    y = wall.start.y + offset * math.sin(angle)
    z = window.sill_height_m + window.height_m / 2
    transform = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    transform[:3, 3] = [x, y, z]
    glass = trimesh.creation.box(extents=[window.width_m, 0.025, window.height_m], transform=transform)
    top_bottom = []
    for dz in (-window.height_m / 2, window.height_m / 2):
        t = transform.copy()
        t[:3, 3] = [x, y, z + dz]
        top_bottom.append(trimesh.creation.box(extents=[window.width_m + 0.08, 0.06, 0.06], transform=t))
    sides = []
    for sx in (-window.width_m / 2, window.width_m / 2):
        t = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        t[:3, 3] = [x + sx * math.cos(angle), y + sx * math.sin(angle), z]
        sides.append(trimesh.creation.box(extents=[0.06, 0.06, window.height_m], transform=t))
    return [_paint(glass, glass_color), *[_paint(mesh, frame_color) for mesh in top_bottom + sides]]
