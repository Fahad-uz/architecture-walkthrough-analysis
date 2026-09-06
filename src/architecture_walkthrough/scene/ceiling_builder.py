from __future__ import annotations

from architecture_walkthrough.geometry.models import RoomPolygon
from architecture_walkthrough.scene.floor_builder import polygon_floor_mesh


def ceiling_meshes(rooms: list[RoomPolygon], height_m: float, thickness_m: float, color: tuple[int, int, int, int]):
    meshes = []
    for room in rooms:
        mesh = polygon_floor_mesh(room.points, thickness_m, color)
        mesh.apply_translation([0, 0, height_m])
        meshes.append(mesh)
    return meshes
