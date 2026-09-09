from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPolygon, Polygon  # type: ignore[import-untyped]
from shapely.ops import unary_union  # type: ignore[import-untyped]
import trimesh

from architecture_walkthrough.geometry.models import (
    ArchitecturalElement,
    FloorPlanModel,
    FurniturePlacement,
    Point2D,
    RoomPolygon,
    WallSegment,
)
from architecture_walkthrough.scene.ceiling_builder import ceiling_meshes
from architecture_walkthrough.scene.furniture import render_furniture
from architecture_walkthrough.scene.door_builder import door_meshes
from architecture_walkthrough.scene.floor_builder import (
    fallback_floor_mesh,
    polygon_floor_mesh,
    room_floor_meshes,
)
from architecture_walkthrough.scene.opening_builder import nearest_wall_index, openings_for_wall
from architecture_walkthrough.scene.trim_builder import skirting_meshes
from architecture_walkthrough.scene.uv_mapping import apply_planar_uv
from architecture_walkthrough.scene.wall_builder import split_wall_meshes
from architecture_walkthrough.scene.window_builder import window_meshes

RGBA = tuple[int, int, int, int]
MIN_ROOM_FLOOR_COVERAGE = 0.45

COLORS: dict[str, RGBA] = {
    "floor": (218, 205, 185, 255),
    "floor_marble": (224, 214, 196, 255),
    "floor_balcony": (154, 90, 55, 255),
    "floor_kitchen": (105, 32, 25, 255),
    "floor_bath": (150, 145, 137, 255),
    "floor_lift": (88, 89, 88, 255),
    "wall": (230, 226, 218, 255),
    "wall_cap": (105, 108, 108, 255),
    "bed": (204, 198, 181, 255),
    "bed_accent": (174, 169, 151, 255),
    "sofa": (61, 103, 128, 255),
    "sofa_cushion": (86, 132, 151, 255),
    "chair": (82, 119, 137, 255),
    "dining_table": (118, 72, 42, 255),
    "side_table": (112, 67, 39, 255),
    "coffee_table": (116, 72, 41, 255),
    "kitchen_counter": (128, 34, 25, 255),
    "rug": (146, 76, 58, 255),
    "plant": (40, 118, 45, 255),
    "pillow": (232, 224, 203, 255),
    "fixture": (238, 238, 232, 255),
    "glass": (125, 172, 190, 180),
    "door": (128, 74, 34, 255),
    "metal": (38, 40, 42, 255),
    "step": (196, 181, 158, 255),
    "dark": (48, 48, 44, 255),
    "pot": (104, 69, 43, 255),
    "wood": (91, 51, 27, 255),
    "wood_light": (151, 100, 56, 255),
    "cabinet_front": (111, 66, 35, 255),
}


def _paint(mesh: trimesh.Trimesh, color: RGBA) -> trimesh.Trimesh:
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.tile(np.array(color, dtype=np.uint8), (len(mesh.vertices), 1)),
    )
    return mesh


def _named(mesh: trimesh.Trimesh, part_name: str) -> trimesh.Trimesh:
    mesh.metadata["part_name"] = part_name
    return mesh


def _safe_name(value: object) -> str:
    cleaned = "".join(
        character if character.isalnum() else "_" for character in str(value or "unknown")
    )
    return cleaned.strip("_") or "unknown"


def _box(
    extents: list[float], center: list[float], color: RGBA, rotation_deg: float = 0.0
) -> trimesh.Trimesh:
    transform = trimesh.transformations.rotation_matrix(math.radians(rotation_deg), [0, 0, 1])
    transform[:3, 3] = center
    return _paint(trimesh.creation.box(extents=extents, transform=transform), color)


def _oriented_offset(
    item: FurniturePlacement, local_x: float, local_y: float
) -> tuple[float, float]:
    angle = math.radians(item.rotation_deg)
    return (
        item.center.x + local_x * math.cos(angle) - local_y * math.sin(angle),
        item.center.y + local_x * math.sin(angle) + local_y * math.cos(angle),
    )


def _part(
    item: FurniturePlacement,
    local_x: float,
    local_y: float,
    width: float,
    depth: float,
    height: float,
    color: RGBA,
    z: float | None = None,
) -> trimesh.Trimesh:
    x, y = _oriented_offset(item, local_x, local_y)
    return _box(
        [width, depth, height], [x, y, height / 2 if z is None else z], color, item.rotation_deg
    )


def _cylinder_part(
    item: FurniturePlacement,
    local_x: float,
    local_y: float,
    width: float,
    depth: float,
    height: float,
    color: RGBA,
    z: float,
    sections: int = 12,
) -> trimesh.Trimesh:
    """Return a footprint-oriented low-poly elliptical cylinder."""

    mesh = trimesh.creation.cylinder(
        radius=0.5,
        height=height,
        sections=max(8, min(24, int(sections))),
    )
    mesh.apply_scale([width, depth, 1.0])
    x, y = _oriented_offset(item, local_x, local_y)
    transform = trimesh.transformations.rotation_matrix(
        math.radians(item.rotation_deg),
        [0, 0, 1],
    )
    transform[:3, 3] = [x, y, z]
    mesh.apply_transform(transform)
    return _paint(mesh, color)


def _ellipsoid_part(
    item: FurniturePlacement,
    local_x: float,
    local_y: float,
    width: float,
    depth: float,
    height: float,
    color: RGBA,
    z: float,
    local_rotation_deg: float = 0.0,
) -> trimesh.Trimesh:
    """Return a smooth, bounded soft-form primitive with only 42 vertices."""

    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    mesh.apply_scale([width / 2, depth / 2, height / 2])
    x, y = _oriented_offset(item, local_x, local_y)
    transform = trimesh.transformations.rotation_matrix(
        math.radians(item.rotation_deg + local_rotation_deg),
        [0, 0, 1],
    )
    transform[:3, 3] = [x, y, z]
    mesh.apply_transform(transform)
    return _paint(mesh, color)


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
    return _paint(
        trimesh.creation.box(extents=[width, depth, 0.1], transform=transform), COLORS["floor"]
    )


def _merged_room_floors(model: FloorPlanModel) -> list[RoomPolygon]:
    polygons = []
    for room in model.rooms:
        polygon = Polygon([(point.x, point.y) for point in room.points])
        if polygon.is_valid and polygon.area > 0.05:
            polygons.append(polygon)
    if not polygons:
        return []
    merged = unary_union(polygons)
    if isinstance(merged, Polygon):
        merged_polygons = [merged]
    elif isinstance(merged, MultiPolygon):
        merged_polygons = list(merged.geoms)
    else:
        return []
    rooms: list[RoomPolygon] = []
    for index, polygon in enumerate(
        sorted(merged_polygons, key=lambda item: item.area, reverse=True)
    ):
        if polygon.area <= 0.05:
            continue
        rooms.append(
            RoomPolygon(
                id=f"floor_union_{index:03d}",
                name="floor",
                points=[
                    Point2D(x=float(x), y=float(y)) for x, y in list(polygon.exterior.coords)[:-1]
                ],
                confidence=0.7,
                evidence_source="room_floor_union",
            )
        )
    return rooms


def _room_floor_coverage(model: FloorPlanModel) -> float:
    """Return how much of the wall envelope is covered by valid room faces.

    Reconstruction can recover a handful of closed rooms while leaving the
    open-plan portion unclosed. Rendering only those faces creates dangerous
    floor voids in walkthrough mode, so low coverage falls back to one stable
    foundation slab.
    """
    wall_points = [point for wall in model.walls for point in (wall.start, wall.end)]
    if not wall_points:
        return 0.0
    min_x, min_y, max_x, max_y = _bounds(wall_points)
    envelope_area = (max_x - min_x) * (max_y - min_y)
    if envelope_area <= 0:
        return 0.0
    polygons = [
        polygon
        for room in model.rooms
        if (polygon := Polygon([(point.x, point.y) for point in room.points])).is_valid
        and polygon.area > 0.05
    ]
    if not polygons:
        return 0.0
    return min(1.0, float(unary_union(polygons).area) / envelope_area)


def _floor_meshes_for_model(model: FloorPlanModel) -> list[trimesh.Trimesh]:
    if model.slabs:
        return [
            polygon_floor_mesh(slab.points, slab.thickness_m, COLORS["floor"])
            for slab in model.slabs
        ]
    merged_floor_rooms = _merged_room_floors(model)
    if merged_floor_rooms and _room_floor_coverage(model) >= MIN_ROOM_FLOOR_COVERAGE:
        return room_floor_meshes(merged_floor_rooms, 0.10, COLORS["floor"])
    return [fallback_floor_mesh(model, 0.10, COLORS["floor"])]


def _wall_index_for_opening(
    model: FloorPlanModel, wall_id: str | None, center: Point2D
) -> int | None:
    if wall_id:
        for index, wall in enumerate(model.walls):
            if wall.id == wall_id:
                return index
        if wall_id.isdigit():
            numeric = int(wall_id)
            if 0 <= numeric < len(model.walls):
                return numeric
    return nearest_wall_index(model.walls, center)


def _chair_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    width = item.width_m
    depth = item.depth_m
    meshes = [
        _named(
            _ellipsoid_part(
                item,
                0,
                -depth * 0.03,
                width * 0.92,
                depth * 0.82,
                0.11,
                COLORS["sofa_cushion"],
                0.45,
            ),
            "Seat",
        ),
        _named(
            _part(
                item,
                0,
                depth * 0.445,
                width * 0.86,
                depth * 0.09,
                0.46,
                COLORS["chair"],
                0.69,
            ),
            "Back",
        ),
    ]
    leg_width = min(0.038, min(width, depth) * 0.09)
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            meshes.append(
                _named(
                    _cylinder_part(
                        item,
                        x_sign * width * 0.37,
                        y_sign * depth * 0.34,
                        leg_width,
                        leg_width,
                        0.40,
                        COLORS["wood"],
                        0.20,
                        sections=10,
                    ),
                    f"Leg_{int(x_sign)}_{int(y_sign)}",
                )
            )
    return meshes


def _table_meshes(item: FurniturePlacement, color: RGBA) -> list[trimesh.Trimesh]:
    coffee = "coffee" in item.category.lower() or "side" in item.category.lower()
    surface_z = 0.42 if coffee else 0.76
    top_height = 0.07
    top_bottom = surface_z - top_height
    leg_w = min(0.08, min(item.width_m, item.depth_m) * 0.08)
    meshes = [
        _named(
            _part(
                item,
                0,
                0,
                item.width_m,
                item.depth_m,
                top_height,
                color,
                surface_z - top_height / 2,
            ),
            "Top",
        )
    ]
    apron_height = min(0.12, top_bottom * 0.22)
    apron_thickness = min(0.035, min(item.width_m, item.depth_m) * 0.08)
    apron_x = max(0.0, item.width_m / 2 - apron_thickness / 2)
    apron_y = max(0.0, item.depth_m / 2 - apron_thickness / 2)
    apron_z = top_bottom - apron_height / 2
    meshes.extend(
        [
            _named(
                _part(
                    item,
                    0,
                    -apron_y,
                    item.width_m * 0.78,
                    apron_thickness,
                    apron_height,
                    COLORS["wood"],
                    apron_z,
                ),
                "Apron_Front",
            ),
            _named(
                _part(
                    item,
                    0,
                    apron_y,
                    item.width_m * 0.78,
                    apron_thickness,
                    apron_height,
                    COLORS["wood"],
                    apron_z,
                ),
                "Apron_Back",
            ),
            _named(
                _part(
                    item,
                    -apron_x,
                    0,
                    apron_thickness,
                    item.depth_m * 0.72,
                    apron_height,
                    COLORS["wood"],
                    apron_z,
                ),
                "Apron_Left",
            ),
            _named(
                _part(
                    item,
                    apron_x,
                    0,
                    apron_thickness,
                    item.depth_m * 0.72,
                    apron_height,
                    COLORS["wood"],
                    apron_z,
                ),
                "Apron_Right",
            ),
        ]
    )
    for sx in (-1, 1):
        for sy in (-1, 1):
            meshes.append(
                _named(
                    _cylinder_part(
                        item,
                        sx * item.width_m * 0.38,
                        sy * item.depth_m * 0.36,
                        leg_w,
                        leg_w,
                        top_bottom,
                        COLORS["wood"],
                        top_bottom / 2,
                        sections=10,
                    ),
                    f"Leg_{sx}_{sy}",
                )
            )
    return meshes


def _sofa_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    width = item.width_m
    depth = item.depth_m
    arm_width = min(width * 0.105, 0.16)
    inner_width = max(width - arm_width * 2.0, width * 0.62)
    cushion_count = 1 if width < 1.15 else 2 if width < 2.10 else 3
    cushion_gap = min(0.025, inner_width * 0.025)
    cushion_width = (inner_width - cushion_gap * (cushion_count - 1)) / cushion_count
    meshes = [
        _named(_part(item, 0, 0, width * 0.92, depth * 0.82, 0.16, COLORS["sofa"], 0.16), "Base"),
        _named(
            _part(item, 0, depth * 0.45, width * 0.88, depth * 0.10, 0.58, COLORS["sofa"], 0.47),
            "Back",
        ),
    ]
    for index in range(cushion_count):
        local_x = -inner_width / 2 + cushion_width / 2 + index * (cushion_width + cushion_gap)
        meshes.extend(
            [
                _named(
                    _ellipsoid_part(
                        item,
                        local_x,
                        -depth * 0.08,
                        cushion_width * 0.96,
                        depth * 0.60,
                        0.17,
                        COLORS["sofa_cushion"],
                        0.325,
                    ),
                    f"Seat_Cushion_{index + 1:02d}",
                ),
                _named(
                    _ellipsoid_part(
                        item,
                        local_x,
                        depth * 0.37,
                        cushion_width * 0.96,
                        depth * 0.17,
                        0.43,
                        COLORS["sofa_cushion"],
                        0.575,
                    ),
                    f"Back_Cushion_{index + 1:02d}",
                ),
            ]
        )
    for side in (-1.0, 1.0):
        meshes.append(
            _named(
                _part(
                    item,
                    side * (width - arm_width) / 2,
                    0,
                    arm_width,
                    depth * 0.88,
                    0.46,
                    COLORS["sofa"],
                    0.31,
                ),
                f"Arm_{int(side)}",
            )
        )
    return meshes


def _bed_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    width = item.width_m
    depth = item.depth_m
    meshes = [
        _named(_part(item, 0, 0, width, depth, 0.18, COLORS["wood"], 0.13), "Frame"),
        _named(
            _part(item, 0, -depth * 0.035, width * 0.93, depth * 0.82, 0.21, COLORS["bed"], 0.315),
            "Mattress",
        ),
        _named(
            _part(
                item,
                0,
                -depth * 0.15,
                width * 0.88,
                depth * 0.54,
                0.065,
                COLORS["bed_accent"],
                0.448,
            ),
            "Duvet",
        ),
    ]
    headboard_depth = depth * 0.07
    meshes.append(
        _named(
            _part(
                item,
                0,
                max(0.0, depth / 2 - headboard_depth / 2),
                width * 0.96,
                headboard_depth,
                0.74,
                COLORS["wood"],
                0.40,
            ),
            "Headboard",
        )
    )
    pillow_count = 2 if width >= 1.05 else 1
    pillow_width = width * (0.36 if pillow_count == 2 else 0.56)
    for pillow_index in range(pillow_count):
        pillow_x = (
            (pillow_index - (pillow_count - 1) / 2) * width * 0.43 if pillow_count > 1 else 0.0
        )
        meshes.append(
            _named(
                _ellipsoid_part(
                    item,
                    pillow_x,
                    depth * 0.29,
                    pillow_width,
                    depth * 0.16,
                    0.12,
                    COLORS["pillow"],
                    0.49,
                ),
                f"Pillow_{pillow_index + 1:02d}",
            )
        )
    return meshes


def _floor_patch_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    category = item.category.lower()
    if "balcony" in category:
        color = COLORS["floor_balcony"]
    elif "kitchen" in category:
        color = COLORS["floor_kitchen"]
    elif "bath" in category:
        color = COLORS["floor_bath"]
    elif "lift" in category:
        color = COLORS["floor_lift"]
    else:
        color = COLORS["floor_marble"]
    return [_part(item, 0, 0, item.width_m, item.depth_m, 0.035, color, 0.02)]


def _railing_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    if "post" in item.category.lower():
        return [_part(item, 0, 0, item.width_m, item.depth_m, 0.95, COLORS["metal"], 0.48)]
    meshes = [
        _part(item, 0, 0, item.width_m, item.depth_m, 0.08, COLORS["metal"], 0.95),
        _part(item, 0, 0, item.width_m, item.depth_m, 0.06, COLORS["metal"], 0.45),
    ]
    post_count = max(2, int(item.width_m / 0.45))
    for index in range(post_count + 1):
        local_x = -item.width_m / 2 + item.width_m * index / post_count
        meshes.append(_part(item, local_x, 0, 0.04, 0.05, 0.9, COLORS["metal"], 0.45))
    return meshes


def _door_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    return [
        _part(item, 0, 0, item.width_m, item.depth_m, 2.0, COLORS["door"], 1.0),
        _part(item, 0, -item.depth_m * 0.38, item.width_m * 1.3, 0.05, 0.05, COLORS["metal"], 1.05),
    ]


def _window_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    return [
        _part(item, 0, 0, item.width_m, item.depth_m, 1.1, COLORS["glass"], 1.45),
        _part(item, 0, 0, item.width_m, item.depth_m * 1.8, 0.08, COLORS["metal"], 0.92),
        _part(item, 0, 0, item.width_m, item.depth_m * 1.8, 0.08, COLORS["metal"], 2.02),
    ]


def _stair_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    return [_part(item, 0, 0, item.width_m, item.depth_m, 0.16, COLORS["step"], 0.08)]


def _wardrobe_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    return [
        _part(item, 0, 0, item.width_m, item.depth_m, 1.8, COLORS["door"], 0.9),
        _part(item, -item.width_m * 0.18, 0, 0.03, item.depth_m * 0.9, 1.65, COLORS["dark"], 0.95),
        _part(item, item.width_m * 0.18, 0, 0.03, item.depth_m * 0.9, 1.65, COLORS["dark"], 0.95),
    ]


def _lamp_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    radius = max(min(item.width_m, item.depth_m) * 0.22, 0.06)
    pole = trimesh.creation.cylinder(radius=radius * 0.22, height=0.45, sections=12)
    pole.apply_translation([item.center.x, item.center.y, 0.42])
    shade = trimesh.creation.cone(radius=radius, height=0.22, sections=20)
    shade.apply_translation([item.center.x, item.center.y, 0.78])
    bulb = trimesh.creation.icosphere(subdivisions=1, radius=radius * 0.35)
    bulb.apply_translation([item.center.x, item.center.y, 0.72])
    return [
        _paint(pole, COLORS["metal"]),
        _paint(shade, (246, 226, 180, 255)),
        _paint(bulb, (255, 238, 180, 255)),
    ]


def _tv_unit_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    return [
        _part(item, 0, 0, item.width_m, item.depth_m, 0.45, COLORS["door"], 0.22),
        _part(item, 0, 0, item.width_m * 0.12, item.depth_m * 0.82, 1.05, COLORS["dark"], 0.88),
    ]


def _counter_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    width = item.width_m
    depth = item.depth_m
    meshes = [
        _named(
            _part(
                item,
                0,
                depth * 0.025,
                width * 0.96,
                depth * 0.90,
                0.84,
                COLORS["kitchen_counter"],
                0.45,
            ),
            "Cabinet",
        ),
        _named(_part(item, 0, 0, width, depth, 0.07, COLORS["fixture"], 0.905), "Worktop"),
        _named(
            _part(item, 0, -depth * 0.425, width * 0.90, depth * 0.08, 0.12, COLORS["dark"], 0.06),
            "ToeKick",
        ),
    ]
    module_count = max(1, min(6, round(width / 0.58)))
    module_span = width * 0.90 / module_count
    front_depth = min(0.025, depth)
    handle_depth = min(0.025, depth)
    fronts: dict[float, list[trimesh.Trimesh]] = {-1.0: [], 1.0: []}
    handles: dict[float, list[trimesh.Trimesh]] = {-1.0: [], 1.0: []}
    for module_index in range(module_count):
        module_x = -width * 0.45 + module_span * (module_index + 0.5)
        for face_sign in (-1.0, 1.0):
            fronts[face_sign].append(
                _part(
                    item,
                    module_x,
                    face_sign * max(0.0, depth / 2 - front_depth / 2),
                    module_span - min(0.014, module_span * 0.08),
                    front_depth,
                    0.62,
                    COLORS["cabinet_front"],
                    0.49,
                )
            )
            handles[face_sign].append(
                _part(
                    item,
                    module_x,
                    face_sign * max(0.0, depth / 2 - handle_depth / 2),
                    min(0.16, module_span * 0.34),
                    handle_depth,
                    0.022,
                    COLORS["metal"],
                    0.72,
                )
            )
    for face_name, face_sign in (("A", -1.0), ("B", 1.0)):
        meshes.append(
            _named(
                trimesh.util.concatenate(fronts[face_sign]),
                f"Front_{face_name}",
            )
        )
        meshes.append(
            _named(
                trimesh.util.concatenate(handles[face_sign]),
                f"Handles_{face_name}",
            )
        )
    return meshes


def _stove_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    width = item.width_m
    depth = item.depth_m
    door_depth = min(0.026, depth)
    door_y = max(0.0, depth / 2 - door_depth / 2)
    meshes = [
        _named(
            _part(
                item,
                0,
                depth * 0.015,
                width * 0.96,
                depth * 0.93,
                0.86,
                COLORS["kitchen_counter"],
                0.43,
            ),
            "Body",
        ),
        _named(_part(item, 0, 0, width, depth, 0.055, COLORS["dark"], 0.895), "Cooktop"),
        _named(
            _part(item, 0, -door_y, width * 0.76, door_depth, 0.46, COLORS["metal"], 0.48),
            "Oven_Door_A",
        ),
        _named(
            _part(item, 0, door_y, width * 0.76, door_depth, 0.46, COLORS["metal"], 0.48),
            "Oven_Door_B",
        ),
    ]
    burner_size = min(width, depth) * 0.22
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            meshes.append(
                _named(
                    _cylinder_part(
                        item,
                        x_sign * width * 0.24,
                        y_sign * depth * 0.23,
                        burner_size,
                        burner_size,
                        0.016,
                        COLORS["fixture"],
                        0.931,
                        sections=12,
                    ),
                    f"Burner_{int(x_sign)}_{int(y_sign)}",
                )
            )
    return meshes


def _sink_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    width = item.width_m
    depth = item.depth_m
    basin_width = width * 0.70
    basin_depth = depth * 0.58
    rim_width = min(width, depth) * 0.075
    faucet_size = min(width, depth) * 0.065
    rim = trimesh.util.concatenate(
        [
            _part(
                item, 0, -depth * 0.355, width * 0.84, rim_width, 0.055, COLORS["fixture"], 0.895
            ),
            _part(item, 0, depth * 0.355, width * 0.84, rim_width, 0.055, COLORS["fixture"], 0.895),
            _part(item, -width * 0.39, 0, rim_width, depth * 0.64, 0.055, COLORS["fixture"], 0.895),
            _part(item, width * 0.39, 0, rim_width, depth * 0.64, 0.055, COLORS["fixture"], 0.895),
        ]
    )
    return [
        _named(
            _part(
                item,
                0,
                depth * 0.015,
                width * 0.96,
                depth * 0.93,
                0.86,
                COLORS["kitchen_counter"],
                0.43,
            ),
            "Cabinet",
        ),
        _named(rim, "Rim"),
        _named(
            _part(item, 0, -depth * 0.015, basin_width, basin_depth, 0.025, COLORS["dark"], 0.875),
            "Basin",
        ),
        _named(
            _cylinder_part(
                item,
                0,
                depth * 0.37,
                faucet_size,
                faucet_size,
                0.25,
                COLORS["metal"],
                1.02,
                sections=10,
            ),
            "Faucet_Base",
        ),
        _named(
            _part(
                item,
                0,
                depth * 0.22,
                faucet_size,
                depth * 0.30,
                faucet_size,
                COLORS["metal"],
                1.135,
            ),
            "Faucet_Spout",
        ),
    ]


def _appliance_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    return [
        _part(item, 0, 0, item.width_m, item.depth_m, 1.7, COLORS["dark"], 0.85),
        _part(item, 0, -item.depth_m * 0.35, item.width_m * 0.8, 0.04, 0.08, COLORS["metal"], 1.35),
    ]


def _fixture_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    return [
        _part(item, 0, 0, item.width_m, item.depth_m, 0.35, COLORS["fixture"], 0.22),
        _part(
            item,
            0,
            item.depth_m * 0.18,
            item.width_m * 0.65,
            item.depth_m * 0.42,
            0.16,
            COLORS["dark"],
            0.46,
        ),
    ]


def _plant_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    width = item.width_m
    depth = item.depth_m
    meshes = [
        _named(
            _cylinder_part(
                item,
                0,
                0,
                width * 0.52,
                depth * 0.52,
                0.32,
                COLORS["pot"],
                0.16,
                sections=12,
            ),
            "Pot",
        )
    ]
    leaves = (
        (-width * 0.15, depth * 0.04, width * 0.38, depth * 0.30, 0.64, -25.0),
        (width * 0.15, depth * 0.08, width * 0.40, depth * 0.28, 0.72, 28.0),
        (0.0, -depth * 0.12, width * 0.44, depth * 0.32, 0.79, 4.0),
    )
    for local_x, local_y, leaf_width, leaf_depth, leaf_z, leaf_rotation in leaves:
        # Rotating an ellipse increases its axis-aligned extent. Scale it to
        # the remaining local footprint so extreme rectangular planters stay
        # inside the geometry already cleared by room fitting.
        angle = math.radians(leaf_rotation)
        radius_x = leaf_width / 2
        radius_y = leaf_depth / 2
        extent_x = math.hypot(radius_x * math.cos(angle), radius_y * math.sin(angle))
        extent_y = math.hypot(radius_x * math.sin(angle), radius_y * math.cos(angle))
        available_x = max(0.0, width / 2 - abs(local_x))
        available_y = max(0.0, depth / 2 - abs(local_y))
        scale = min(
            1.0,
            available_x / max(extent_x, 1e-9),
            available_y / max(extent_y, 1e-9),
        )
        meshes.append(
            _named(
                _ellipsoid_part(
                    item,
                    local_x,
                    local_y,
                    leaf_width * scale,
                    leaf_depth * scale,
                    0.25,
                    COLORS["plant"],
                    leaf_z,
                    local_rotation_deg=leaf_rotation,
                ),
                f"Leaf_{len(meshes):02d}",
            )
        )
    return meshes


def _furniture_family(category: str) -> str:
    """Resolve specific semantic categories before broader room-like tokens."""
    normalized = category.lower().replace("-", "_").replace(" ", "_")
    tokens = {token for token in normalized.split("_") if token}
    if "floor_patch" in normalized:
        return "floor_patch"
    if "railing" in tokens:
        return "railing"
    if {"stair", "stairs", "staircase"} & tokens:
        return "stair"
    if "door" in tokens:
        return "door"
    if "window" in tokens:
        return "window"
    if "lamp" in tokens:
        return "lamp"
    if "tv" in tokens or "television" in tokens:
        return "tv"
    if {"stove", "stovetop", "hob", "cooktop"} & tokens:
        return "stove"
    if "sink" in tokens or normalized.endswith("sink"):
        return "sink"
    if "appliance" in tokens:
        return "appliance"
    # `bedside_table` is a table, not a bed.  Resolve table before bed.
    if (
        "table" in tokens
        or normalized.endswith("table")
        or "nightstand" in tokens
        or "bedside" in tokens
    ):
        return "table"
    if "chair" in tokens:
        return "chair"
    if "bed" in tokens or normalized.startswith("bed_") or normalized.endswith("_bed"):
        return "bed"
    if "sofa" in tokens or "couch" in tokens:
        return "sofa"
    if {"wardrobe", "cabinet", "shelf", "closet"} & tokens:
        return "wardrobe"
    # Kitchen sinks and stoves have already been handled above.
    if "counter" in tokens or "kitchen" in tokens:
        return "counter"
    if {"fixture", "toilet", "bath", "bathtub"} & tokens or "bath" in normalized:
        return "fixture"
    if "plant" in tokens:
        return "plant"
    if "rug" in tokens or "carpet" in tokens:
        return "rug"
    return "generic"


def _furniture_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    category = item.category.lower()
    family = _furniture_family(category)
    if family == "floor_patch":
        return _floor_patch_meshes(item)
    if family == "railing":
        return _railing_meshes(item)
    if family == "stair":
        return _stair_meshes(item)
    if family == "door":
        return _door_meshes(item)
    if family == "window":
        return _window_meshes(item)
    if family == "wardrobe":
        return _wardrobe_meshes(item)
    if family == "lamp":
        return _lamp_meshes(item)
    if family == "tv":
        return _tv_unit_meshes(item)
    if family == "stove":
        return _stove_meshes(item)
    if family == "sink":
        return _sink_meshes(item)
    if family == "appliance":
        return _appliance_meshes(item)
    if family == "bed":
        return _bed_meshes(item)
    if family == "sofa":
        return _sofa_meshes(item)
    if family == "chair":
        return _chair_meshes(item)
    if family == "counter":
        return _counter_meshes(item)
    if family == "fixture":
        return _fixture_meshes(item)
    if family == "plant":
        return _plant_meshes(item)
    if family == "rug":
        return [_part(item, 0, 0, item.width_m, item.depth_m, 0.04, COLORS["rug"], 0.03)]
    if family == "table":
        return _table_meshes(item, COLORS.get(category, COLORS["dining_table"]))
    return [_part(item, 0, 0, item.width_m, item.depth_m, 0.55, COLORS["chair"], 0.28)]


def _special_placement(element: ArchitecturalElement) -> FurniturePlacement:
    """Turn a semantic architectural element into a procedural placement."""
    if element.polygon:
        xs = [point.x for point in element.polygon]
        ys = [point.y for point in element.polygon]
        polygon_center = Point2D(x=(min(xs) + max(xs)) / 2, y=(min(ys) + max(ys)) / 2)
        polygon_width = max(xs) - min(xs)
        polygon_depth = max(ys) - min(ys)
    else:
        polygon_center = Point2D(x=0.0, y=0.0)
        polygon_width = 0.0
        polygon_depth = 0.0
    return FurniturePlacement(
        category=element.kind,
        center=element.center or polygon_center,
        width_m=max(float(element.width_m or polygon_width or 1.0), 0.10),
        depth_m=max(float(element.depth_m or polygon_depth or 1.0), 0.10),
        rotation_deg=element.rotation_deg,
    )


def _staircase_meshes(item: FurniturePlacement, step_count: int) -> list[trimesh.Trimesh]:
    steps = max(4, min(14, step_count))
    step_depth = item.depth_m / steps
    meshes: list[trimesh.Trimesh] = []
    for index in range(steps):
        local_y = -item.depth_m / 2 + step_depth * (index + 0.5)
        height = 0.16 * (index + 1)
        meshes.append(
            _part(
                item,
                0,
                local_y,
                item.width_m,
                step_depth * 0.94,
                height,
                COLORS["step"],
                height / 2,
            )
        )
    return meshes


def _lift_meshes(item: FurniturePlacement) -> list[trimesh.Trimesh]:
    width = max(item.width_m, 1.0)
    depth = max(item.depth_m, 1.0)
    return [
        _part(item, 0, 0, width, depth, 0.08, COLORS["metal"], 0.04),
        _part(item, 0, depth * 0.49, width, 0.08, 2.2, COLORS["metal"], 1.1),
        _part(item, -width * 0.48, 0, 0.08, depth, 2.2, COLORS["metal"], 1.1),
        _part(item, width * 0.48, 0, 0.08, depth, 2.2, COLORS["metal"], 1.1),
    ]


def _special_element_meshes(element: ArchitecturalElement) -> list[trimesh.Trimesh]:
    kind = element.kind.lower().replace("-", "_").replace(" ", "_")
    placement = _special_placement(element)
    if kind in {"stair", "stairs", "staircase"}:
        return _staircase_meshes(placement, int(element.metadata.get("step_count") or 10))
    if kind == "lift":
        return _lift_meshes(placement)
    if kind in {"counter", "kitchen_counter"}:
        return _counter_meshes(placement)
    if kind in {"balcony", "terrace"}:
        if len(element.polygon) >= 3:
            return [polygon_floor_mesh(element.polygon, 0.08, COLORS["floor_balcony"])]
        return _floor_patch_meshes(placement.model_copy(update={"category": f"floor_patch_{kind}"}))
    return []


def export_simple_glb(model: FloorPlanModel, output_glb: Path) -> Path:
    if not model.walls:
        raise ValueError("cannot export GLB: floorplan contains no wall geometry")
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()

    for index, mesh in enumerate(_floor_meshes_for_model(model)):
        scene.add_geometry(
            apply_planar_uv(mesh), node_name=f"Floor_{index:03d}", geom_name=f"Floor_{index:03d}"
        )

    for index, balcony in enumerate(model.balconies):
        mesh = polygon_floor_mesh(balcony.points, 0.08, COLORS["floor_balcony"])
        name = f"Balcony_{index:03d}"
        scene.add_geometry(apply_planar_uv(mesh), node_name=name, geom_name=name)

    wall_cap_meshes = []
    for wall_index, wall in enumerate(model.walls):
        wall_openings = openings_for_wall(wall, model.doors, model.windows, wall_index, model.walls)
        try:
            wall_meshes = split_wall_meshes(wall, wall_openings, COLORS["wall"])
        except ValueError:
            wall_meshes = [_wall_mesh(wall)]
        for section_index, mesh in enumerate(wall_meshes):
            scene.add_geometry(
                apply_planar_uv(mesh),
                node_name=f"Wall_{wall_index:03d}_{section_index:02d}",
                geom_name=f"Wall_{wall_index:03d}_{section_index:02d}",
            )
        wall_cap_meshes.append(_wall_cap_mesh(wall))

    for index, mesh in enumerate(wall_cap_meshes):
        scene.add_geometry(
            mesh, node_name=f"Wall_Cap_{index:03d}", geom_name=f"Wall_Cap_{index:03d}"
        )

    for door_index, door in enumerate(model.doors):
        opening_wall_index = _wall_index_for_opening(model, door.wall_id, door.center)
        if opening_wall_index is None:
            continue
        for part_index, mesh in enumerate(
            door_meshes(model.walls[opening_wall_index], door, COLORS["door"])
        ):
            if part_index == 0:
                name = f"DoorLeaf_{door_index:03d}"
            elif part_index == 1:
                name = f"DoorHeader_{door_index:03d}"
            else:
                name = f"DoorPart_{door_index:03d}_{part_index:02d}"
            scene.add_geometry(mesh, node_name=name, geom_name=name)

    for window_index, window in enumerate(model.windows):
        opening_wall_index = _wall_index_for_opening(model, window.wall_id, window.center)
        if opening_wall_index is None:
            continue
        for part_index, mesh in enumerate(
            window_meshes(
                model.walls[opening_wall_index],
                window,
                COLORS["metal"],
                COLORS["glass"],
            )
        ):
            scene.add_geometry(
                mesh,
                node_name=f"Window_{window_index:03d}_{part_index:02d}",
                geom_name=f"Window_{window_index:03d}_{part_index:02d}",
            )

    if model.ceiling.enabled and model.rooms:
        for index, mesh in enumerate(
            ceiling_meshes(
                model.rooms, model.ceiling.height_m, model.ceiling.thickness_m, COLORS["wall"]
            )
        ):
            scene.add_geometry(
                apply_planar_uv(mesh),
                node_name=f"Ceiling_{index:03d}",
                geom_name=f"Ceiling_{index:03d}",
            )

    for index, mesh in enumerate(skirting_meshes(model.walls, COLORS["wall_cap"])):
        scene.add_geometry(
            mesh, node_name=f"Skirting_{index:03d}", geom_name=f"Skirting_{index:03d}"
        )

    for index, item in enumerate(render_furniture(model)):
        parts = _furniture_meshes(item)
        top = max((float(mesh.bounds[1, 2]) for mesh in parts), default=0.0)
        if item.height_m is not None and top > 0:
            for mesh in parts:
                mesh.apply_scale([1.0, 1.0, item.height_m / top])
        for part_index, mesh in enumerate(parts):
            category = _safe_name(item.category)
            part_name = _safe_name(mesh.metadata.get("part_name") or f"Part_{part_index:02d}")
            name = f"Furniture_{index:03d}_{part_index:02d}_{category}_{part_name}"
            scene.add_geometry(
                mesh,
                node_name=name,
                geom_name=name,
            )

    for index, element in enumerate(model.special_elements):
        kind = element.kind.lower().replace("-", "_").replace(" ", "_")
        for part_index, mesh in enumerate(_special_element_meshes(element)):
            name = f"Special_{index:03d}_{kind}_{part_index:02d}"
            scene.add_geometry(mesh, node_name=name, geom_name=name)

    # Trimesh keeps the source Z-up coordinate system when writing glTF,
    # unlike Blender's exporter. The browser and the rest of this project use
    # glTF's Y-up convention: plan (x, y) maps to world (x, -z), with height
    # on +Y. Apply the same conversion Blender performs before serializing.
    scene.apply_transform(trimesh.transformations.rotation_matrix(math.radians(-90.0), [1, 0, 0]))
    exported = scene.export(file_type="glb")
    if isinstance(exported, str):
        output_glb.write_text(exported, encoding="utf-8")
    else:
        output_glb.write_bytes(exported)
    return output_glb
