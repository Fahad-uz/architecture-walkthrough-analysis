from __future__ import annotations

import math

import numpy as np
import trimesh

from architecture_walkthrough.geometry.models import DoorOpening, WallSegment
from architecture_walkthrough.scene.wall_builder import point_offset_on_wall


def _paint(mesh: trimesh.Trimesh, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    mesh.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(mesh.vertices), 1))
    return mesh


def door_meshes(wall: WallSegment, door: DoorOpening, color: tuple[int, int, int, int]) -> list[trimesh.Trimesh]:
    offset = door.offset_m if door.offset_m is not None else point_offset_on_wall(wall, door.center)
    angle = math.atan2(wall.end.y - wall.start.y, wall.end.x - wall.start.x)
    x = wall.start.x + offset * math.cos(angle)
    y = wall.start.y + offset * math.sin(angle)
    leaf_angle = angle + math.radians(18)
    transform = trimesh.transformations.rotation_matrix(leaf_angle, [0, 0, 1])
    transform[:3, 3] = [x, y, door.height_m / 2]
    leaf = trimesh.creation.box(extents=[door.width_m, 0.045, door.height_m], transform=transform)
    frame_transform = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    frame_transform[:3, 3] = [x, y, door.height_m + 0.04]
    header = trimesh.creation.box(extents=[door.width_m + 0.16, wall.thickness_m * 1.4, 0.08], transform=frame_transform)
    return [_paint(leaf, color), _paint(header, color)]
