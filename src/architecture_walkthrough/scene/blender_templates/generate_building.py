"""Headless Blender generator: floorplan JSON -> lit GLB.

Runs inside Blender 3.4+:
  blender --background --python generate_building.py -- \
      --floorplan plan.json --output building.glb --mode final \
      [--samples N] [--lightmap-px N] [--no-denoise]

Geometry is identical across all bake modes; only lighting treatment differs:
  none  - no bake; punctual lights exported (KHR_lights_punctual fallback)
  draft - fast Cycles bake for iteration
  final - high-quality Cycles bake (the default for real output)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.geometry import tessellate_polygon

# ---------------------------------------------------------------------------
# Arguments


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--floorplan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["final", "draft", "none"], default="final")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--lightmap-px", type=int, default=None)
    parser.add_argument("--no-denoise", action="store_true")
    return parser.parse_args(argv)


ARGS = parse_args()
PLAN = json.loads(Path(ARGS.floorplan).read_text(encoding="utf-8"))
MATERIAL_PLAN = (
    (PLAN.get("metadata") or {}).get("blender_material_plan") or {}
)
if not isinstance(MATERIAL_PLAN, dict):
    MATERIAL_PLAN = {}
MODE = ARGS.mode
SAMPLES = ARGS.samples or (256 if MODE == "final" else 16)
LIGHTMAP_PX = ARGS.lightmap_px or (2048 if MODE == "final" else 512)
DENOISE = not ARGS.no_denoise

WALL_HEIGHT_DEFAULT = 2.8
FLOOR_THICKNESS = 0.1
CEILING_THICKNESS = 0.08
BASEBOARD_HEIGHT = 0.1
DOOR_LEAF_OPEN_DEG = 25.0
MIN_OPENING_WIDTH = 0.18
MIN_ROOM_FLOOR_COVERAGE = 0.45

# ---------------------------------------------------------------------------
# Scene reset


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = DENOISE


def enable_gpu_if_available() -> str:
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is None:
        return "cpu"
    cprefs = prefs.preferences
    for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            cprefs.compute_device_type = backend
        except TypeError:
            continue
        cprefs.get_devices()
        gpus = [d for d in cprefs.devices if d.type != "CPU"]
        if gpus:
            for device in cprefs.devices:
                device.use = device.type != "CPU"
            bpy.context.scene.cycles.device = "GPU"
            return backend.lower()
    return "cpu"


# ---------------------------------------------------------------------------
# Materials


def make_pbr(
    name: str,
    rgba,
    roughness: float = 0.7,
    metallic: float = 0.0,
    transmission: float = 0.0,
    alpha_mode: str = "OPAQUE",
    double_sided: bool = True,
) -> bpy.types.Material:
    normalized_alpha_mode = str(alpha_mode or "OPAQUE").upper()
    if normalized_alpha_mode not in {"OPAQUE", "MASK", "BLEND"}:
        normalized_alpha_mode = "OPAQUE"
    effective_rgba = tuple(rgba)
    if normalized_alpha_mode == "OPAQUE":
        effective_rgba = (*effective_rgba[:3], 1.0)

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = effective_rgba
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = float(effective_rgba[3])
        if normalized_alpha_mode == "MASK":
            alpha_value = material.node_tree.nodes.new("ShaderNodeValue")
            alpha_value.name = "glTF Alpha"
            alpha_value.outputs[0].default_value = float(effective_rgba[3])
            alpha_clip = material.node_tree.nodes.new("ShaderNodeMath")
            alpha_clip.name = "glTF Alpha Clip"
            alpha_clip.operation = "GREATER_THAN"
            alpha_clip.inputs[1].default_value = 0.5
            material.node_tree.links.new(
                alpha_value.outputs[0],
                alpha_clip.inputs[0],
            )
            material.node_tree.links.new(
                alpha_clip.outputs[0],
                bsdf.inputs["Alpha"],
            )
    material.diffuse_color = effective_rgba
    material.use_backface_culling = not double_sided
    if transmission > 0:
        # Blender 4+/5 renamed Transmission to Transmission Weight.
        key = "Transmission Weight" if "Transmission Weight" in bsdf.inputs else "Transmission"
        bsdf.inputs[key].default_value = transmission
    try:
        if normalized_alpha_mode == "MASK":
            material.blend_method = "CLIP"
            material.alpha_threshold = 0.5
        elif normalized_alpha_mode == "BLEND":
            material.blend_method = "BLEND"
        else:
            material.blend_method = "OPAQUE"
    except (AttributeError, TypeError):
        pass
    return material


MATERIALS: dict[str, bpy.types.Material] = {}


def material_key(value: object) -> str:
    cleaned = "".join(
        char if char.isalnum() else "_"
        for char in str(value or "").strip().lower()
    )
    return "_".join(part for part in cleaned.split("_") if part)


def scalar_material(name: str, payload: dict) -> bpy.types.Material:
    raw_color = payload.get("base_color", [0.8, 0.8, 0.8, 1.0])
    try:
        color = tuple(
            max(0.0, min(1.0, float(component)))
            for component in raw_color
        )
    except (TypeError, ValueError):
        color = (0.8, 0.8, 0.8, 1.0)
    if len(color) != 4:
        color = (0.8, 0.8, 0.8, 1.0)
    try:
        roughness = max(0.0, min(1.0, float(payload.get("roughness", 0.65))))
        metallic = max(0.0, min(1.0, float(payload.get("metallic", 0.0))))
    except (TypeError, ValueError):
        roughness, metallic = 0.65, 0.0
    return make_pbr(
        f"PBR_{material_key(name) or 'fallback'}",
        color,
        roughness=roughness,
        metallic=metallic,
        alpha_mode=str(payload.get("alpha_mode") or "OPAQUE").upper(),
        double_sided=bool(payload.get("double_sided", False)),
    )


def material_for_preset(
    preset: object,
    fallback_role: str,
) -> bpy.types.Material:
    key = material_key(preset)
    if key:
        resolved = MATERIALS.get(f"preset:{key}")
        if resolved is not None:
            return resolved
    return MATERIALS[fallback_role]


def planned_default(role: str, fallback_role: str) -> bpy.types.Material:
    defaults = MATERIAL_PLAN.get("defaults") or {}
    preset = defaults.get(role) if isinstance(defaults, dict) else None
    return material_for_preset(preset, fallback_role)


def planned_indexed(
    collection: str,
    index: int,
    default_role: str,
    fallback_role: str,
) -> bpy.types.Material:
    presets = MATERIAL_PLAN.get(collection) or []
    preset = presets[index] if isinstance(presets, list) and index < len(presets) else None
    if preset is not None:
        return material_for_preset(preset, fallback_role)
    return planned_default(default_role, fallback_role)


def build_materials() -> None:
    style = PLAN.get("style") or {}
    floor_preset = str(style.get("floor_material") or "wood").lower()
    if "marble" in floor_preset or "tile" in floor_preset:
        floor_name = "Floor_Stone"
        floor_color = (0.72, 0.74, 0.75, 1.0)
        floor_roughness = 0.32
    else:
        floor_name = "Floor_Wood"
        floor_color = (0.52, 0.34, 0.20, 1.0)
        floor_roughness = 0.45

    MATERIALS["wall"] = make_pbr("Wall_Plaster", (0.90, 0.89, 0.86, 1.0), roughness=0.82)
    MATERIALS["floor"] = make_pbr(floor_name, floor_color, roughness=floor_roughness)
    MATERIALS["ceiling"] = make_pbr("Ceiling_Paint", (0.92, 0.92, 0.90, 1.0), roughness=0.9)
    MATERIALS["baseboard"] = make_pbr("Baseboard_Paint", (0.94, 0.93, 0.90, 1.0), roughness=0.5)
    MATERIALS["door"] = make_pbr("Door_Wood", (0.42, 0.26, 0.14, 1.0), roughness=0.4)
    MATERIALS["frame"] = make_pbr("Frame_Paint", (0.93, 0.92, 0.89, 1.0), roughness=0.45)
    MATERIALS["glass"] = make_pbr("Window_Glass", (0.8, 0.9, 0.95, 1.0), roughness=0.05, transmission=1.0)
    MATERIALS["upholstery"] = make_pbr("Furniture_Upholstery", (0.24, 0.43, 0.55, 1.0), roughness=0.78)
    MATERIALS["upholstery_cushion"] = make_pbr(
        "Furniture_Upholstery_Cushion",
        (0.34, 0.54, 0.63, 1.0),
        roughness=0.84,
    )
    MATERIALS["fabric_light"] = make_pbr("Furniture_Fabric_Light", (0.82, 0.78, 0.69, 1.0), roughness=0.86)
    MATERIALS["fabric_accent"] = make_pbr(
        "Furniture_Fabric_Accent",
        (0.69, 0.66, 0.58, 1.0),
        roughness=0.9,
    )
    MATERIALS["wood"] = make_pbr("Furniture_Wood", (0.39, 0.22, 0.11, 1.0), roughness=0.52)
    MATERIALS["wood_light"] = make_pbr("Furniture_Wood_Light", (0.62, 0.42, 0.23, 1.0), roughness=0.50)
    MATERIALS["counter"] = make_pbr("Counter_Stone", (0.68, 0.69, 0.66, 1.0), roughness=0.28)
    MATERIALS["cabinet"] = make_pbr("Cabinet_Wood", (0.34, 0.18, 0.09, 1.0), roughness=0.48)
    MATERIALS["metal"] = make_pbr("Metal_Dark", (0.12, 0.14, 0.16, 1.0), roughness=0.24, metallic=0.72)
    MATERIALS["fixture"] = make_pbr("Fixture_Ceramic", (0.92, 0.94, 0.93, 1.0), roughness=0.22)
    MATERIALS["rug"] = make_pbr("Rug_Fabric", (0.46, 0.20, 0.16, 1.0), roughness=0.92)
    MATERIALS["plant"] = make_pbr("Plant_Leaves", (0.12, 0.38, 0.17, 1.0), roughness=0.8)
    MATERIALS["pot"] = make_pbr("Plant_Pot", (0.38, 0.20, 0.12, 1.0), roughness=0.72)
    planned_materials = MATERIAL_PLAN.get("materials") or {}
    if isinstance(planned_materials, dict):
        for preset_name, payload in sorted(planned_materials.items()):
            if not isinstance(payload, dict):
                continue
            key = material_key(preset_name)
            if key:
                MATERIALS[f"preset:{key}"] = scalar_material(key, payload)


# ---------------------------------------------------------------------------
# Geometry helpers


def link(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.collection.objects.link(obj)
    return obj


def new_box(name: str, size: Vector, location: Vector, z_rotation: float, material: bpy.types.Material) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    hx, hy, hz = size.x / 2, size.y / 2, size.z / 2
    verts = [
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
    ]
    faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.rotation_euler = (0.0, 0.0, z_rotation)
    obj.data.materials.append(material)
    return link(obj)


def wall_vec(wall: dict) -> tuple[Vector, Vector, float, float]:
    start = Vector((wall["start"]["x"], wall["start"]["y"], 0.0))
    end = Vector((wall["end"]["x"], wall["end"]["y"], 0.0))
    direction = end - start
    length = direction.length
    angle = math.atan2(direction.y, direction.x)
    return start, end, length, angle


def opening_interval(opening: dict, wall_length: float | None = None) -> tuple[float, float] | None:
    start, end = opening.get("start_offset_m"), opening.get("end_offset_m")
    if start is not None and end is not None:
        interval = (float(start), float(end))
    else:
        offset, width = opening.get("offset_m"), float(opening.get("width_m") or 0.9)
        if offset is None:
            return None
        interval = (float(offset) - width / 2, float(offset) + width / 2)
    start_m, end_m = sorted(interval)
    if wall_length is not None:
        start_m = max(0.0, start_m)
        end_m = min(float(wall_length), end_m)
    if end_m - start_m < MIN_OPENING_WIDTH:
        print(
            f"[generate_building] skipping invalid opening {opening.get('id') or '<unnamed>'}: "
            f"{start_m:.3f}-{end_m:.3f}m"
        )
        return None
    return start_m, end_m


def openings_on_wall(wall_id: str) -> tuple[list[dict], list[dict]]:
    doors = [d for d in PLAN.get("doors", []) if d.get("wall_id") == wall_id]
    windows = [w for w in PLAN.get("windows", []) if w.get("wall_id") == wall_id]
    return doors, windows


def build_wall(wall: dict, index: int) -> bpy.types.Object:
    start, _end, length, angle = wall_vec(wall)
    thickness = float(wall.get("thickness_m") or 0.12)
    height = float(wall.get("height_m") or WALL_HEIGHT_DEFAULT)
    center = start + Vector((math.cos(angle), math.sin(angle), 0.0)) * (length / 2)
    wall_material = planned_indexed(
        "wall_presets",
        index,
        "wall",
        "wall",
    )
    obj = new_box(
        f"Wall_{index:03d}",
        Vector((length, thickness, height)),
        Vector((center.x, center.y, height / 2)),
        angle,
        wall_material,
    )
    doors, windows = openings_on_wall(wall.get("id") or "")
    cutters: list[bpy.types.Object] = []
    for opening in doors:
        interval = opening_interval(opening, length)
        if interval is None:
            continue
        o_start, o_end = interval
        width = o_end - o_start
        door_height = float(opening.get("height_m") or 2.1)
        local = (o_start + o_end) / 2
        position = start + Vector((math.cos(angle), math.sin(angle), 0.0)) * local
        cutters.append(
            new_box(
                f"Cut_Door_{index:03d}_{len(cutters)}",
                Vector((width, thickness * 3, door_height)),
                Vector((position.x, position.y, door_height / 2)),
                angle,
                wall_material,
            )
        )
    for opening in windows:
        interval = opening_interval(opening, length)
        if interval is None:
            continue
        o_start, o_end = interval
        width = o_end - o_start
        window_height = float(opening.get("height_m") or 1.2)
        sill = float(opening.get("sill_height_m") or 0.9)
        local = (o_start + o_end) / 2
        position = start + Vector((math.cos(angle), math.sin(angle), 0.0)) * local
        cutters.append(
            new_box(
                f"Cut_Window_{index:03d}_{len(cutters)}",
                Vector((width, thickness * 3, window_height)),
                Vector((position.x, position.y, sill + window_height / 2)),
                angle,
                wall_material,
            )
        )
    for cutter in cutters:
        cutter.display_type = "WIRE"
        cutter.hide_render = True
        modifier = obj.modifiers.new("Opening", "BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.solver = "EXACT"
        modifier.object = cutter
    if cutters:
        # Explicit selection works on both Blender 3.4 (the Debian/Docker
        # baseline) and current Blender. It avoids relying on newer context
        # override behavior for operator polling.
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        for modifier in list(obj.modifiers):
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        for cutter in cutters:
            bpy.data.objects.remove(cutter, do_unlink=True)
    return obj


def build_polygon_slab(name: str, points: list[dict], z: float, thickness: float, material: bpy.types.Material) -> bpy.types.Object | None:
    coords = [Vector((p["x"], p["y"], 0.0)) for p in points]
    if len(coords) < 3:
        return None
    triangles = tessellate_polygon([coords])
    if not triangles:
        return None
    mesh = bpy.data.meshes.new(name)
    top = [(c.x, c.y, z) for c in coords]
    bottom = [(c.x, c.y, z - thickness) for c in coords]
    verts = top + bottom
    n = len(coords)
    faces: list[tuple[int, ...]] = []
    for tri in triangles:
        faces.append(tuple(tri))
        faces.append(tuple(reversed([i + n for i in tri])))
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, j + n, i + n))
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(material)
    return link(obj)


def polygon_area(points: list[dict]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            float(point["x"]) * float(points[(index + 1) % len(points)]["y"])
            - float(points[(index + 1) % len(points)]["x"]) * float(point["y"])
            for index, point in enumerate(points)
        )
    ) / 2


def room_floor_coverage(rooms: list[dict], walls: list[dict]) -> float:
    if not rooms or not walls:
        return 0.0
    xs = [float(value) for wall in walls for value in (wall["start"]["x"], wall["end"]["x"])]
    ys = [float(value) for wall in walls for value in (wall["start"]["y"], wall["end"]["y"])]
    envelope_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    if envelope_area <= 0:
        return 0.0
    # Valid reconstructed room faces do not overlap. Capping still keeps a
    # malformed payload from reporting impossible coverage above one.
    room_area = sum(polygon_area(room.get("points") or []) for room in rooms)
    return min(1.0, room_area / envelope_area)


def build_floors(rooms: list[dict], walls: list[dict]) -> None:
    slabs = PLAN.get("slabs") or []
    if slabs:
        for index, slab in enumerate(slabs):
            build_polygon_slab(
                f"Floor_{index:03d}",
                slab.get("points") or [],
                0.0,
                float(slab.get("thickness_m") or FLOOR_THICKNESS),
                planned_indexed(
                    "slab_presets",
                    index,
                    "floor",
                    "floor",
                ),
            )
        return
    if rooms and room_floor_coverage(rooms, walls) >= MIN_ROOM_FLOOR_COVERAGE:
        for index, room in enumerate(rooms):
            build_polygon_slab(
                f"Floor_{index:03d}",
                room.get("points") or [],
                0.0,
                FLOOR_THICKNESS,
                planned_indexed(
                    "room_floor_presets",
                    index,
                    "floor",
                    "floor",
                ),
            )
        return
    if not walls:
        return
    xs = [float(value) for wall in walls for value in (wall["start"]["x"], wall["end"]["x"])]
    ys = [float(value) for wall in walls for value in (wall["start"]["y"], wall["end"]["y"])]
    pad = 0.3
    envelope = [
        {"x": min(xs) - pad, "y": min(ys) - pad},
        {"x": max(xs) + pad, "y": min(ys) - pad},
        {"x": max(xs) + pad, "y": max(ys) + pad},
        {"x": min(xs) - pad, "y": max(ys) + pad},
    ]
    build_polygon_slab(
        "Floor_000",
        envelope,
        0.0,
        FLOOR_THICKNESS,
        planned_default("floor", "floor"),
    )


def build_baseboards(wall: dict, index: int) -> list[bpy.types.Object]:
    start, _end, length, angle = wall_vec(wall)
    thickness = float(wall.get("thickness_m") or 0.12) * 1.15
    doors, _windows = openings_on_wall(wall.get("id") or "")
    blocked: list[tuple[float, float]] = []
    for door in doors:
        interval = opening_interval(door, length)
        if interval:
            blocked.append(interval)
    blocked.sort()
    wall_material = planned_indexed(
        "wall_presets",
        index,
        "wall",
        "baseboard",
    )
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for b_start, b_end in blocked:
        if b_start > cursor:
            spans.append((cursor, b_start))
        cursor = max(cursor, b_end)
    if cursor < length:
        spans.append((cursor, length))
    objs = []
    for span_index, (s, e) in enumerate(spans):
        if e - s < 0.05:
            continue
        mid = (s + e) / 2
        position = start + Vector((math.cos(angle), math.sin(angle), 0.0)) * mid
        objs.append(
            new_box(
                f"Baseboard_{index:03d}_{span_index}",
                Vector((e - s, thickness, BASEBOARD_HEIGHT)),
                Vector((position.x, position.y, BASEBOARD_HEIGHT / 2)),
                angle,
                wall_material,
            )
        )
    return objs


def build_door_assets(door: dict, walls_by_id: dict, index: int) -> None:
    wall = walls_by_id.get(door.get("wall_id"))
    if wall is None:
        return
    start, _end, length, angle = wall_vec(wall)
    interval = opening_interval(door, length)
    if interval is None:
        return
    o_start, o_end = interval
    width = o_end - o_start
    height = float(door.get("height_m") or 2.1)
    thickness = float(wall.get("thickness_m") or 0.12)
    direction = Vector((math.cos(angle), math.sin(angle), 0.0))
    door_material = planned_indexed(
        "door_presets",
        index,
        "door",
        "door",
    )
    # Frame: two jambs and a header hugging the cut.
    for side_offset in (o_start + 0.03, o_end - 0.03):
        position = start + direction * side_offset
        new_box(
            f"DoorJamb_{index:03d}_{side_offset:.2f}",
            Vector((0.06, thickness * 1.15, height)),
            Vector((position.x, position.y, height / 2)),
            angle,
            door_material,
        )
    header_mid = start + direction * ((o_start + o_end) / 2)
    new_box(
        f"DoorHeader_{index:03d}",
        Vector((width, thickness * 1.15, 0.08)),
        Vector((header_mid.x, header_mid.y, height + 0.04)),
        angle,
        door_material,
    )
    # Leaf hinged at hinge_side, slightly open.
    hinge_side = door.get("hinge_side") or "start"
    swing = 1.0 if (door.get("swing_side") or "left") == "left" else -1.0
    hinge_offset = o_start if hinge_side == "start" else o_end
    leaf_direction = 1.0 if hinge_side == "start" else -1.0
    hinge_point = start + direction * hinge_offset
    leaf_angle = angle + leaf_direction * swing * math.radians(DOOR_LEAF_OPEN_DEG)
    leaf_center = hinge_point + Vector((math.cos(leaf_angle), math.sin(leaf_angle), 0.0)) * (leaf_direction * width / 2)
    new_box(
        f"DoorLeaf_{index:03d}",
        Vector((width - 0.04, 0.045, height - 0.04)),
        Vector((leaf_center.x, leaf_center.y, (height - 0.04) / 2)),
        leaf_angle,
        door_material,
    )


def build_window_assets(window: dict, walls_by_id: dict, index: int) -> None:
    wall = walls_by_id.get(window.get("wall_id"))
    if wall is None:
        return
    start, _end, length, angle = wall_vec(wall)
    interval = opening_interval(window, length)
    if interval is None:
        return
    o_start, o_end = interval
    width = o_end - o_start
    height = float(window.get("height_m") or 1.2)
    sill = float(window.get("sill_height_m") or 0.9)
    thickness = float(wall.get("thickness_m") or 0.12)
    direction = Vector((math.cos(angle), math.sin(angle), 0.0))
    mid = start + direction * ((o_start + o_end) / 2)
    z_mid = sill + height / 2
    frame_material = planned_indexed(
        "window_frame_presets",
        index,
        "window_frame",
        "frame",
    )
    # Frame border.
    for offset, size in (
        (Vector((0, 0, -height / 2)), Vector((width, thickness * 1.1, 0.06))),
        (Vector((0, 0, height / 2)), Vector((width, thickness * 1.1, 0.06))),
    ):
        new_box(
            f"WindowFrame_{index:03d}_{offset.z:.2f}",
            size,
            Vector((mid.x, mid.y, z_mid)) + offset,
            angle,
            frame_material,
        )
    for side in (o_start + 0.03, o_end - 0.03):
        position = start + direction * side
        new_box(
            f"WindowJamb_{index:03d}_{side:.2f}",
            Vector((0.06, thickness * 1.1, height)),
            Vector((position.x, position.y, z_mid)),
            angle,
            frame_material,
        )
    # Keep the transmission-capable glass shader until the scalar registry
    # contract can express transmission without degrading window realism.
    new_box(
        f"WindowGlass_{index:03d}",
        Vector((width - 0.05, 0.02, height - 0.05)),
        Vector((mid.x, mid.y, z_mid)),
        angle,
        MATERIALS["glass"],
    )


# ---------------------------------------------------------------------------
# Procedural furniture and architectural elements


def safe_name(value: object) -> str:
    cleaned = "".join(char if str(char).isalnum() else "_" for char in str(value or "unknown"))
    return cleaned.strip("_") or "unknown"


def oriented_xy(item: dict, local_x: float, local_y: float) -> tuple[float, float]:
    center = item.get("center") or {}
    angle = math.radians(float(item.get("rotation_deg") or 0.0))
    return (
        float(center.get("x") or 0.0) + local_x * math.cos(angle) - local_y * math.sin(angle),
        float(center.get("y") or 0.0) + local_x * math.sin(angle) + local_y * math.cos(angle),
    )


def asset_box(
    prefix: str,
    item: dict,
    part: str,
    local_x: float,
    local_y: float,
    width: float,
    depth: float,
    height: float,
    z_center: float,
    material: str,
    bevel_width: float | None = None,
    bevel_segments: int = 2,
) -> bpy.types.Object:
    x, y = oriented_xy(item, local_x, local_y)
    obj = new_box(
        f"{prefix}_{safe_name(part)}",
        Vector((max(0.025, width), max(0.025, depth), max(0.025, height))),
        Vector((x, y, z_center)),
        math.radians(float(item.get("rotation_deg") or 0.0)),
        MATERIALS[material],
    )
    bevel = obj.modifiers.new("Soft_Edges", "BEVEL")
    shortest_side = max(0.025, min(width, depth, height))
    requested_bevel = (
        min(0.025, max(0.004, shortest_side * 0.08))
        if bevel_width is None
        else max(0.002, float(bevel_width))
    )
    bevel.width = min(requested_bevel, shortest_side * 0.42)
    bevel.segments = max(1, int(bevel_segments))
    return obj


def asset_cylinder(
    prefix: str,
    item: dict,
    part: str,
    local_x: float,
    local_y: float,
    width: float,
    depth: float,
    height: float,
    z_center: float,
    material: str,
    *,
    segments: int = 16,
    top_scale: float = 1.0,
    local_rotation_deg: float = 0.0,
) -> bpy.types.Object:
    """Build a context-free elliptical cylinder or frustum."""
    width = max(0.025, float(width))
    depth = max(0.025, float(depth))
    height = max(0.025, float(height))
    segments = max(8, int(segments))
    top_scale = max(0.05, float(top_scale))
    bottom_rx, bottom_ry = width / 2, depth / 2
    top_rx, top_ry = bottom_rx * top_scale, bottom_ry * top_scale
    hz = height / 2
    verts = []
    for z, rx, ry in ((-hz, bottom_rx, bottom_ry), (hz, top_rx, top_ry)):
        verts.extend(
            (
                rx * math.cos(math.tau * index / segments),
                ry * math.sin(math.tau * index / segments),
                z,
            )
            for index in range(segments)
        )
    faces = [
        tuple(reversed(range(segments))),
        tuple(range(segments, segments * 2)),
    ]
    for index in range(segments):
        next_index = (index + 1) % segments
        faces.append(
            (
                index,
                next_index,
                segments + next_index,
                segments + index,
            )
        )

    name = f"{prefix}_{safe_name(part)}"
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = len(polygon.vertices) == 4
    x, y = oriented_xy(item, local_x, local_y)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector((x, y, z_center))
    obj.rotation_euler = (
        0.0,
        0.0,
        math.radians(float(item.get("rotation_deg") or 0.0) + local_rotation_deg),
    )
    obj.data.materials.append(MATERIALS[material])
    link(obj)
    bevel = obj.modifiers.new("Soft_Rims", "BEVEL")
    bevel.width = min(0.008, min(width, depth, height) * 0.12)
    bevel.segments = 2
    return obj


def asset_ellipsoid(
    prefix: str,
    item: dict,
    part: str,
    local_x: float,
    local_y: float,
    width: float,
    depth: float,
    height: float,
    z_center: float,
    material: str,
    *,
    longitude_segments: int = 16,
    latitude_segments: int = 8,
    local_rotation_deg: float = 0.0,
) -> bpy.types.Object:
    """Build a smooth low-poly ellipsoid without relying on bpy operators."""
    width = max(0.025, float(width))
    depth = max(0.025, float(depth))
    height = max(0.025, float(height))
    longitude_segments = max(8, int(longitude_segments))
    latitude_segments = max(4, int(latitude_segments))
    rx, ry, rz = width / 2, depth / 2, height / 2
    verts = [(0.0, 0.0, -rz)]
    for latitude in range(1, latitude_segments):
        phi = -math.pi / 2 + math.pi * latitude / latitude_segments
        ring_radius = math.cos(phi)
        verts.extend(
            (
                rx * ring_radius * math.cos(math.tau * longitude / longitude_segments),
                ry * ring_radius * math.sin(math.tau * longitude / longitude_segments),
                rz * math.sin(phi),
            )
            for longitude in range(longitude_segments)
        )
    top_index = len(verts)
    verts.append((0.0, 0.0, rz))

    faces = []
    first_ring = 1
    for longitude in range(longitude_segments):
        next_longitude = (longitude + 1) % longitude_segments
        faces.append((0, first_ring + next_longitude, first_ring + longitude))
    for latitude in range(latitude_segments - 2):
        lower_ring = 1 + latitude * longitude_segments
        upper_ring = lower_ring + longitude_segments
        for longitude in range(longitude_segments):
            next_longitude = (longitude + 1) % longitude_segments
            faces.append(
                (
                    lower_ring + longitude,
                    lower_ring + next_longitude,
                    upper_ring + next_longitude,
                    upper_ring + longitude,
                )
            )
    last_ring = 1 + (latitude_segments - 2) * longitude_segments
    for longitude in range(longitude_segments):
        next_longitude = (longitude + 1) % longitude_segments
        faces.append((top_index, last_ring + longitude, last_ring + next_longitude))

    name = f"{prefix}_{safe_name(part)}"
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    x, y = oriented_xy(item, local_x, local_y)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector((x, y, z_center))
    obj.rotation_euler = (
        0.0,
        0.0,
        math.radians(float(item.get("rotation_deg") or 0.0) + local_rotation_deg),
    )
    obj.data.materials.append(MATERIALS[material])
    return link(obj)


def build_table(item: dict, prefix: str, width: float, depth: float, coffee: bool = False) -> None:
    top_z = 0.42 if coffee else 0.76
    asset_box(
        prefix,
        item,
        "Top",
        0,
        0,
        width,
        depth,
        0.07,
        top_z,
        "wood_light",
        bevel_width=0.035,
        bevel_segments=3,
    )
    leg_height = top_z - 0.08
    leg_width = max(0.035, min(width, depth) * 0.08)
    apron_height = min(0.12, leg_height * 0.22)
    apron_z = top_z - 0.07 - apron_height / 2
    asset_box(prefix, item, "Apron_Front", 0, -depth * 0.40, width * 0.78, 0.035, apron_height, apron_z, "wood")
    asset_box(prefix, item, "Apron_Back", 0, depth * 0.40, width * 0.78, 0.035, apron_height, apron_z, "wood")
    asset_box(prefix, item, "Apron_Left", -width * 0.42, 0, 0.035, depth * 0.72, apron_height, apron_z, "wood")
    asset_box(prefix, item, "Apron_Right", width * 0.42, 0, 0.035, depth * 0.72, apron_height, apron_z, "wood")
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            asset_cylinder(
                prefix,
                item,
                f"Leg_{int(x_sign)}_{int(y_sign)}",
                x_sign * width * 0.38,
                y_sign * depth * 0.36,
                leg_width,
                leg_width,
                leg_height,
                leg_height / 2,
                "wood",
                segments=12,
                top_scale=1.28,
            )


def asset_family(category: str) -> str:
    """Resolve specific semantic categories before broad room-like tokens."""
    normalized = category.lower().replace("-", "_").replace(" ", "_")
    tokens = {token for token in normalized.split("_") if token}
    if "sink" in tokens or normalized.endswith("sink"):
        return "sink"
    if {"stove", "stovetop", "hob", "cooktop"} & tokens:
        return "stove"
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
    if "counter" in tokens or "kitchen" in tokens:
        return "counter"
    if {"bath", "bathtub", "toilet", "fixture"} & tokens or "bath" in normalized:
        return "fixture"
    if "rug" in tokens or "carpet" in tokens:
        return "rug"
    if "tv" in tokens or "television" in tokens:
        return "tv"
    if "plant" in tokens:
        return "plant"
    return "generic"


def build_procedural_asset(item: dict, prefix: str, category: str) -> None:
    category = category.lower()
    family = asset_family(category)
    width = max(0.20, float(item.get("width_m") or 0.60))
    depth = max(0.20, float(item.get("depth_m") or 0.60))

    if family == "bed":
        # Keep every visible part inside the grounded footprint. Expanding a
        # detected bed in the renderer can push it through an adjacent wall.
        width, depth = max(width, 0.55), max(depth, 0.85)
        asset_box(prefix, item, "Frame", 0, 0, width, depth, 0.18, 0.13, "wood", bevel_width=0.018)
        asset_box(
            prefix,
            item,
            "Mattress",
            0,
            -depth * 0.035,
            width * 0.93,
            depth * 0.82,
            0.21,
            0.315,
            "fabric_light",
            bevel_width=0.055,
            bevel_segments=3,
        )
        asset_box(
            prefix,
            item,
            "Duvet",
            0,
            -depth * 0.15,
            width * 0.88,
            depth * 0.54,
            0.065,
            0.448,
            "fabric_accent",
            bevel_width=0.03,
            bevel_segments=3,
        )
        asset_box(
            prefix,
            item,
            "Headboard",
            0,
            depth * 0.465,
            width * 0.96,
            max(0.045, depth * 0.07),
            0.74,
            0.40,
            "wood",
            bevel_width=0.018,
        )
        pillow_count = 2 if width >= 1.05 else 1
        pillow_width = width * (0.36 if pillow_count == 2 else 0.56)
        for pillow_index in range(pillow_count):
            pillow_x = (
                (pillow_index - (pillow_count - 1) / 2) * width * 0.43
                if pillow_count > 1
                else 0.0
            )
            asset_ellipsoid(
                prefix,
                item,
                f"Pillow_{pillow_index + 1:02d}",
                pillow_x,
                depth * 0.29,
                pillow_width,
                depth * 0.16,
                0.12,
                0.49,
                "fabric_light",
            )
        return

    if family == "sofa":
        # Small sofa footprints often represent single modules in a sectional.
        # Preserve those modules and let cushion count communicate their scale.
        width, depth = max(width, 0.55), max(depth, 0.34)
        arm_width = max(0.065, min(width * 0.105, 0.16))
        inner_width = max(width - arm_width * 2.0, width * 0.62)
        cushion_count = 1 if width < 1.15 else 2 if width < 2.10 else 3
        cushion_gap = min(0.025, inner_width * 0.025)
        cushion_width = (
            inner_width - cushion_gap * (cushion_count - 1)
        ) / cushion_count
        asset_box(
            prefix,
            item,
            "Base",
            0,
            0,
            width * 0.92,
            depth * 0.82,
            0.16,
            0.16,
            "upholstery",
            bevel_width=0.025,
        )
        for cushion_index in range(cushion_count):
            cushion_x = (
                -inner_width / 2
                + cushion_width / 2
                + cushion_index * (cushion_width + cushion_gap)
            )
            asset_box(
                prefix,
                item,
                f"Seat_Cushion_{cushion_index + 1:02d}",
                cushion_x,
                -depth * 0.08,
                cushion_width * 0.96,
                depth * 0.60,
                0.17,
                0.35,
                "upholstery_cushion",
                bevel_width=0.05,
                bevel_segments=3,
            )
            asset_box(
                prefix,
                item,
                f"Back_Cushion_{cushion_index + 1:02d}",
                cushion_x,
                depth * 0.37,
                cushion_width * 0.96,
                depth * 0.17,
                0.43,
                0.61,
                "upholstery_cushion",
                bevel_width=0.05,
                bevel_segments=3,
            )
        for side in (-1.0, 1.0):
            asset_box(
                prefix,
                item,
                f"Arm_{int(side)}",
                side * (width - arm_width) / 2,
                0,
                arm_width,
                depth * 0.88,
                0.46,
                0.31,
                "upholstery",
                bevel_width=0.035,
                bevel_segments=3,
            )
            for y_sign in (-1.0, 1.0):
                asset_cylinder(
                    prefix,
                    item,
                    f"Foot_{int(side)}_{int(y_sign)}",
                    side * width * 0.39,
                    y_sign * depth * 0.31,
                    0.045,
                    0.045,
                    0.09,
                    0.045,
                    "wood",
                    segments=10,
                    top_scale=1.18,
                )
        return

    if family == "chair":
        width, depth = max(width, 0.42), max(depth, 0.42)
        asset_box(prefix, item, "Seat", 0, 0, width, depth, 0.10, 0.45, "upholstery")
        asset_box(prefix, item, "Back", 0, depth * 0.44, width, 0.07, 0.52, 0.70, "upholstery")
        for x_sign in (-1.0, 1.0):
            for y_sign in (-1.0, 1.0):
                asset_box(prefix, item, f"Leg_{int(x_sign)}_{int(y_sign)}", x_sign * width * 0.40, y_sign * depth * 0.38, 0.04, 0.04, 0.42, 0.21, "wood")
        return

    if family == "table":
        coffee = "coffee" in category or "side" in category
        width = max(width, 0.38 if coffee else 0.55)
        depth = max(depth, 0.32 if coffee else 0.45)
        build_table(item, prefix, width, depth, coffee=coffee)
        return

    if family == "wardrobe":
        width, depth = max(width, 0.70), max(depth, 0.38)
        asset_box(prefix, item, "Carcass", 0, 0, width, depth, 1.85, 0.925, "cabinet")
        asset_box(prefix, item, "Door_Left", -width * 0.24, -depth * 0.51, width * 0.46, 0.035, 1.66, 0.93, "wood_light")
        asset_box(prefix, item, "Door_Right", width * 0.24, -depth * 0.51, width * 0.46, 0.035, 1.66, 0.93, "wood_light")
        return

    if family == "stove":
        width, depth = max(width, 0.48), max(depth, 0.44)
        asset_box(prefix, item, "Cooktop", 0, 0, width, depth, 0.06, 0.92, "metal")
        for x_sign in (-1.0, 1.0):
            for y_sign in (-1.0, 1.0):
                asset_box(prefix, item, f"Burner_{int(x_sign)}_{int(y_sign)}", x_sign * width * 0.23, y_sign * depth * 0.23, width * 0.22, depth * 0.22, 0.018, 0.96, "fixture")
        return

    if family == "sink":
        width, depth = max(width, 0.50), max(depth, 0.42)
        asset_box(prefix, item, "Rim", 0, 0, width, depth, 0.08, 0.90, "fixture")
        asset_box(prefix, item, "Basin", 0, 0, width * 0.72, depth * 0.64, 0.12, 0.84, "metal")
        return

    if family == "counter":
        width, depth = max(width, 0.60), max(depth, 0.52)
        asset_box(prefix, item, "Cabinet", 0, 0, width, depth, 0.86, 0.43, "cabinet")
        asset_box(prefix, item, "Worktop", 0, 0, width * 1.04, depth * 1.06, 0.07, 0.895, "counter")
        return

    if family == "fixture":
        width, depth = max(width, 0.58), max(depth, 1.10 if "bath" in category else 0.62)
        asset_box(prefix, item, "Body", 0, 0, width, depth, 0.38, 0.22, "fixture")
        asset_box(prefix, item, "Inset", 0, -depth * 0.05, width * 0.68, depth * 0.67, 0.08, 0.43, "metal")
        return

    if family == "rug":
        asset_box(prefix, item, "Rug", 0, 0, width, depth, 0.025, 0.017, "rug")
        return

    if family == "tv":
        width, depth = max(width, 0.75), max(depth, 0.25)
        asset_box(prefix, item, "Console", 0, 0, width, depth, 0.42, 0.21, "wood")
        asset_box(prefix, item, "Screen", 0, 0, width * 0.78, 0.035, 0.78, 0.92, "metal")
        return

    if family == "plant":
        asset_box(prefix, item, "Pot", 0, 0, width * 0.55, depth * 0.55, 0.35, 0.175, "pot")
        asset_box(prefix, item, "Leaves", 0, 0, width, depth, 0.70, 0.72, "plant")
        return

    asset_box(prefix, item, "Body", 0, 0, width, depth, 0.55, 0.275, "wood_light")


def build_furniture(item: dict, index: int) -> None:
    category = safe_name(item.get("category") or "unknown")
    build_procedural_asset(item, f"Furniture_{index:03d}_{category}", category)


def build_special_element(element: dict, index: int) -> None:
    kind = safe_name(element.get("kind") or "unknown")
    prefix = f"Special_{index:03d}_{kind}"
    polygon = element.get("polygon") or []
    if kind in {"balcony", "terrace"} and len(polygon) >= 3:
        build_polygon_slab(
            prefix,
            polygon,
            0.02,
            0.08,
            planned_default("balcony", "counter"),
        )
        return
    if kind in {"stair", "stairs", "staircase"}:
        item = dict(element)
        item.setdefault("center", {"x": 0.0, "y": 0.0})
        width = max(0.70, float(item.get("width_m") or 1.0))
        depth = max(1.20, float(item.get("depth_m") or 2.0))
        steps = max(4, min(14, int((item.get("metadata") or {}).get("step_count") or 10)))
        for step in range(steps):
            step_depth = depth / steps
            y = -depth / 2 + step_depth * (step + 0.5)
            height = 0.16 * (step + 1)
            asset_box(prefix, item, f"Step_{step:02d}", 0, y, width, step_depth * 0.94, height, height / 2, "counter")
        return
    if kind == "lift":
        item = dict(element)
        item.setdefault("center", {"x": 0.0, "y": 0.0})
        width = max(1.0, float(item.get("width_m") or 1.4))
        depth = max(1.0, float(item.get("depth_m") or 1.4))
        asset_box(prefix, item, "Floor", 0, 0, width, depth, 0.08, 0.04, "metal")
        asset_box(prefix, item, "Rear", 0, depth * 0.49, width, 0.08, 2.2, 1.1, "metal")
        for side in (-1.0, 1.0):
            asset_box(prefix, item, f"Side_{int(side)}", side * width * 0.48, 0, 0.08, depth, 2.2, 1.1, "metal")
        return
    if kind in {"kitchen_counter", "counter", "shelf"} and element.get("center"):
        build_procedural_asset(element, prefix, kind)


# ---------------------------------------------------------------------------
# Lighting


def build_lighting() -> None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    sky = nodes.new("ShaderNodeTexSky")
    if hasattr(sky, "sky_type"):
        try:
            sky.sky_type = "NISHITA"
        except TypeError:
            pass
    if hasattr(sky, "sun_elevation"):
        sky.sun_elevation = math.radians(40)
        sky.sun_rotation = math.radians(135)
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = 0.9
    output = nodes.new("ShaderNodeOutputWorld")
    links.new(sky.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])

    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 2.5
    sun_data.angle = math.radians(2.0)
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(135))
    link(sun)

    for index, room in enumerate(PLAN.get("rooms", [])):
        points = room.get("points", [])
        if len(points) < 3:
            continue
        cx = sum(p["x"] for p in points) / len(points)
        cy = sum(p["y"] for p in points) / len(points)
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]
        area = abs((max(xs) - min(xs)) * (max(ys) - min(ys)))
        light_data = bpy.data.lights.new(f"RoomLight_{index:03d}", "POINT")
        # Blender point-light power is measured in watts. The old 25-120 W
        # range left enclosed rooms and their baked floors almost black.
        light_data.energy = max(300.0, min(1200.0, area * 55.0))
        light_data.color = (1.0, 0.82, 0.66)
        light_data.shadow_soft_size = 0.35
        light = bpy.data.objects.new(f"RoomLight_{index:03d}", light_data)
        height = float(PLAN.get("walls", [{}])[0].get("height_m") or WALL_HEIGHT_DEFAULT)
        light.location = (cx, cy, height - 0.35)
        link(light)


def tune_realtime_lights_for_export() -> None:
    """Keep portable glTF lighting useful without exporting bake-level power.

    The room lights need substantial power while Cycles bakes enclosed rooms,
    but those same values can clip badly in realtime viewers.  Baked modes also
    retain procedural furniture and door leaves as ordinary PBR materials, so
    omitting lights from the GLB leaves those semantic assets almost black.
    """
    for obj in bpy.context.scene.objects:
        if obj.type != "LIGHT":
            continue
        light = obj.data
        if light.type == "SUN":
            light.energy = min(float(light.energy), 1.5)
            continue
        if light.type != "POINT":
            continue
        if MODE == "none":
            light.energy = max(18.0, min(55.0, float(light.energy) * 0.05))
        else:
            light.energy = max(12.0, min(35.0, float(light.energy) * 0.035))


# ---------------------------------------------------------------------------
# Lightmap baking


def _make_sole_active(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def join_objects(names_prefixes: list[str], joined_name: str) -> bpy.types.Object | None:
    targets = [o for o in bpy.context.scene.objects if o.type == "MESH" and any(o.name.startswith(p) for p in names_prefixes)]
    if not targets:
        return None
    bpy.ops.object.select_all(action="DESELECT")
    for target in targets:
        target.select_set(True)
    bpy.context.view_layer.objects.active = targets[0]
    if len(targets) > 1:
        bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = joined_name
    return joined


def bake_object_lightmap(obj: bpy.types.Object, image_name: str, resolution: int) -> bpy.types.Image:
    lightmap = bpy.data.images.new(image_name, resolution, resolution, alpha=False, float_buffer=False)
    uv_layer = obj.data.uv_layers.new(name="LightmapUV")
    obj.data.uv_layers.active = uv_layer
    _make_sole_active(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    try:
        bpy.ops.uv.lightmap_pack(PREF_CONTEXT="ALL_FACES", PREF_MARGIN_DIV=0.2)
    except Exception:
        bpy.ops.uv.smart_project(island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")
    for slot in obj.material_slots:
        material = slot.material
        if material is None or not material.use_nodes:
            continue
        nodes = material.node_tree.nodes
        image_node = nodes.new("ShaderNodeTexImage")
        image_node.image = lightmap
        image_node.name = f"BakeTarget_{material.name}"
        uv_node = nodes.new("ShaderNodeUVMap")
        uv_node.uv_map = "LightmapUV"
        material.node_tree.links.new(uv_node.outputs["UV"], image_node.inputs["Vector"])
        nodes.active = image_node
    _make_sole_active(obj)
    bpy.ops.object.bake(type="COMBINED", uv_layer="LightmapUV", use_clear=True)
    return lightmap


def rewire_to_baked(obj: bpy.types.Object, lightmap: bpy.types.Image) -> None:
    """After baking, materials become emissive lightmap lookups so the GLB
    carries offline lighting that any glTF viewer displays as-is."""
    for slot in obj.material_slots:
        source = slot.material
        if source is None:
            continue
        baked = bpy.data.materials.new(f"{source.name}_Baked_{obj.name}")
        baked.use_nodes = True
        nodes = baked.node_tree.nodes
        links = baked.node_tree.links
        nodes.clear()
        image_node = nodes.new("ShaderNodeTexImage")
        image_node.image = lightmap
        uv_node = nodes.new("ShaderNodeUVMap")
        uv_node.uv_map = "LightmapUV"
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 1.15
        output = nodes.new("ShaderNodeOutputMaterial")
        links.new(uv_node.outputs["UV"], image_node.inputs["Vector"])
        links.new(image_node.outputs["Color"], emission.inputs["Color"])
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
        slot.material = baked


def run_bake() -> None:
    groups = [
        (["Wall_"], "Walls_Joined", LIGHTMAP_PX),
        (["Floor_"], "Floors_Joined", LIGHTMAP_PX),
        (["Ceiling_"], "Ceilings_Joined", max(512, LIGHTMAP_PX // 2)),
        # Door leaves stay separate so the walkthrough can exclude them from
        # its static collision mesh by their stable DoorLeaf_* names.
        (["Baseboard_", "DoorJamb_", "DoorHeader_", "WindowFrame_", "WindowJamb_"], "Trim_Joined", max(512, LIGHTMAP_PX // 2)),
    ]
    for prefixes, joined_name, resolution in groups:
        joined = join_objects(prefixes, joined_name)
        if joined is None:
            continue
        print(f"[generate_building] baking {joined_name} at {resolution}px")
        try:
            lightmap = bake_object_lightmap(joined, f"Lightmap_{joined_name}", resolution)
        except Exception as exc:  # GPU init can fail late (e.g. outdated driver)
            if bpy.context.scene.cycles.device == "GPU":
                print(f"[generate_building] GPU bake failed ({exc}); retrying on CPU")
                bpy.context.scene.cycles.device = "CPU"
                lightmap = bake_object_lightmap(joined, f"Lightmap_{joined_name}", resolution)
            else:
                raise
        lightmap.pack()
        rewire_to_baked(joined, lightmap)


# ---------------------------------------------------------------------------
# Main


def main() -> None:
    import time as _time

    started = _time.time()
    reset_scene()
    device = enable_gpu_if_available() if MODE != "none" else "n/a"
    print(f"[generate_building] mode={MODE} samples={SAMPLES} lightmap={LIGHTMAP_PX} device={device}")
    build_materials()

    walls = PLAN.get("walls", [])
    walls_by_id = {w.get("id"): w for w in walls if w.get("id")}
    for index, wall in enumerate(walls):
        build_wall(wall, index)
        build_baseboards(wall, index)

    rooms = PLAN.get("rooms", [])
    build_floors(rooms, walls)
    ceiling = PLAN.get("ceiling") or {}
    if rooms and ceiling.get("enabled", False):
        ceiling_height = float(ceiling.get("height_m") or WALL_HEIGHT_DEFAULT)
        ceiling_thickness = float(ceiling.get("thickness_m") or CEILING_THICKNESS)
        for index, room in enumerate(rooms):
            build_polygon_slab(
                f"Ceiling_{index:03d}",
                room.get("points", []),
                ceiling_height + ceiling_thickness,
                ceiling_thickness,
                material_for_preset(
                    MATERIAL_PLAN.get("ceiling_preset"),
                    "ceiling",
                ),
            )
    for index, balcony in enumerate(PLAN.get("balconies") or []):
        build_polygon_slab(
            f"Balcony_{index:03d}",
            balcony.get("points") or [],
            0.02,
            0.08,
            planned_indexed(
                "balcony_floor_presets",
                index,
                "balcony",
                "counter",
            ),
        )

    for index, door in enumerate(PLAN.get("doors", [])):
        build_door_assets(door, walls_by_id, index)
    for index, window in enumerate(PLAN.get("windows", [])):
        build_window_assets(window, walls_by_id, index)

    for index, item in enumerate(PLAN.get("furniture", [])):
        build_furniture(item, index)
    for index, element in enumerate(PLAN.get("special_elements", [])):
        build_special_element(element, index)

    build_lighting()

    if MODE != "none":
        run_bake()

    # Export controlled lights in every mode.  Static architectural groups may
    # be emissive lightmaps in draft/final, but semantic assets are deliberately
    # kept as separate PBR meshes for interaction and still require illumination.
    tune_realtime_lights_for_export()

    output = Path(ARGS.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "filepath": str(output),
        "export_format": "GLB",
        "export_apply": True,
        "export_lights": True,
    }
    bpy.ops.export_scene.gltf(**export_kwargs)
    # Build report: makes the preset that actually ran visible to the app/UI,
    # so "which bake was this?" is never ambiguous again.
    report = {
        "mode": MODE,
        "samples": SAMPLES if MODE != "none" else 0,
        "lightmap_px": LIGHTMAP_PX if MODE != "none" else 0,
        "denoise": DENOISE,
        "device": bpy.context.scene.cycles.device.lower() if MODE != "none" else "n/a",
        "device_backend": device,
        "duration_seconds": round(_time.time() - started, 1),
    }
    output.with_suffix(".bake.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[generate_building] wrote {output} ({report})")


main()
