from __future__ import annotations

from architecture_walkthrough.geometry.models import WallSegment
from architecture_walkthrough.scene.wall_builder import WallSection, wall_section_mesh, wall_length


def skirting_meshes(walls: list[WallSegment], color: tuple[int, int, int, int]):
    meshes = []
    for wall in walls:
        section = WallSection(start_m=0.0, end_m=wall_length(wall), bottom_m=0.02, top_m=0.12)
        trim_wall = wall.model_copy(update={"thickness_m": wall.thickness_m * 1.25})
        meshes.append(wall_section_mesh(trim_wall, section, color))
    return meshes
