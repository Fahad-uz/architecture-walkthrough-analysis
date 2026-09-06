from __future__ import annotations

import math

import numpy as np
import trimesh

from architecture_walkthrough.geometry.models import DoorOpening, WallSegment
from architecture_walkthrough.scene.wall_builder import point_offset_on_wall


DOOR_LEAF_OPEN_DEG = 90.0


def _paint(mesh: trimesh.Trimesh, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.tile(np.array(color, dtype=np.uint8), (len(mesh.vertices), 1)),
    )
    return mesh


def door_meshes(wall: WallSegment, door: DoorOpening, color: tuple[int, int, int, int]) -> list[trimesh.Trimesh]:
    wall_length = wall.start.distance_to(wall.end)
    interval = door.interval(door.width_m)
    if interval is None:
        offset = point_offset_on_wall(wall, door.center)
        interval = (offset - door.width_m / 2, offset + door.width_m / 2)
    start_m = max(0.0, min(interval))
    end_m = min(wall_length, max(interval))
    width = end_m - start_m
    if width < 0.18:
        return []

    angle = math.atan2(wall.end.y - wall.start.y, wall.end.x - wall.start.x)
    hinge_at_start = (door.hinge_side or "start") == "start"
    hinge_offset = start_m if hinge_at_start else end_m
    leaf_direction = 1.0 if hinge_at_start else -1.0
    swing = 1.0 if (door.swing_side or "left") == "left" else -1.0
    hinge_x = wall.start.x + hinge_offset * math.cos(angle)
    hinge_y = wall.start.y + hinge_offset * math.sin(angle)
    # Keep walkthrough doors fully open. A decorative 25-degree leaf blocked
    # most of the opening and let first-person cameras visually pass through it.
    leaf_angle = angle + leaf_direction * swing * math.radians(DOOR_LEAF_OPEN_DEG)
    x = hinge_x + math.cos(leaf_angle) * leaf_direction * width / 2
    y = hinge_y + math.sin(leaf_angle) * leaf_direction * width / 2
    transform = trimesh.transformations.rotation_matrix(leaf_angle, [0, 0, 1])
    transform[:3, 3] = [x, y, door.height_m / 2]
    leaf = trimesh.creation.box(extents=[max(0.14, width - 0.04), 0.045, door.height_m], transform=transform)

    opening_mid = (start_m + end_m) / 2
    header_x = wall.start.x + opening_mid * math.cos(angle)
    header_y = wall.start.y + opening_mid * math.sin(angle)
    frame_transform = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    frame_transform[:3, 3] = [header_x, header_y, door.height_m + 0.04]
    header = trimesh.creation.box(
        extents=[width + 0.16, wall.thickness_m * 1.4, 0.08],
        transform=frame_transform,
    )
    return [_paint(leaf, color), _paint(header, color)]
