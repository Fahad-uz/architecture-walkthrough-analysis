"""Headless Blender generator: floorplan JSON -> lit GLB.

Runs inside Blender (5.x):
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
MODE = ARGS.mode
SAMPLES = ARGS.samples or (256 if MODE == "final" else 16)
LIGHTMAP_PX = ARGS.lightmap_px or (2048 if MODE == "final" else 512)
DENOISE = not ARGS.no_denoise

WALL_HEIGHT_DEFAULT = 2.8
FLOOR_THICKNESS = 0.1
CEILING_THICKNESS = 0.08
BASEBOARD_HEIGHT = 0.1
DOOR_LEAF_OPEN_DEG = 25.0

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


def make_pbr(name: str, rgba, roughness: float = 0.7, metallic: float = 0.0, transmission: float = 0.0) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if transmission > 0:
        # Blender 4+/5 renamed Transmission to Transmission Weight.
        key = "Transmission Weight" if "Transmission Weight" in bsdf.inputs else "Transmission"
        bsdf.inputs[key].default_value = transmission
        material.blend_method = "BLEND"
    return material


MATERIALS: dict[str, bpy.types.Material] = {}


def build_materials() -> None:
    MATERIALS["wall"] = make_pbr("Wall_Plaster", (0.87, 0.85, 0.80, 1.0), roughness=0.85)
    MATERIALS["floor"] = make_pbr("Floor_Wood", (0.48, 0.32, 0.19, 1.0), roughness=0.45)
    MATERIALS["ceiling"] = make_pbr("Ceiling_Paint", (0.92, 0.92, 0.90, 1.0), roughness=0.9)
    MATERIALS["baseboard"] = make_pbr("Baseboard_Paint", (0.94, 0.93, 0.90, 1.0), roughness=0.5)
    MATERIALS["door"] = make_pbr("Door_Wood", (0.42, 0.26, 0.14, 1.0), roughness=0.4)
    MATERIALS["frame"] = make_pbr("Frame_Paint", (0.93, 0.92, 0.89, 1.0), roughness=0.45)
    MATERIALS["glass"] = make_pbr("Window_Glass", (0.8, 0.9, 0.95, 1.0), roughness=0.05, transmission=1.0)


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


def opening_interval(opening: dict) -> tuple[float, float] | None:
    start, end = opening.get("start_offset_m"), opening.get("end_offset_m")
    if start is not None and end is not None:
        return float(start), float(end)
    offset, width = opening.get("offset_m"), float(opening.get("width_m") or 0.9)
    if offset is not None:
        return float(offset) - width / 2, float(offset) + width / 2
    return None


def openings_on_wall(wall_id: str) -> tuple[list[dict], list[dict]]:
    doors = [d for d in PLAN.get("doors", []) if d.get("wall_id") == wall_id]
    windows = [w for w in PLAN.get("windows", []) if w.get("wall_id") == wall_id]
    return doors, windows


def build_wall(wall: dict, index: int) -> bpy.types.Object:
    start, _end, length, angle = wall_vec(wall)
    thickness = float(wall.get("thickness_m") or 0.12)
    height = float(wall.get("height_m") or WALL_HEIGHT_DEFAULT)
    center = start + Vector((math.cos(angle), math.sin(angle), 0.0)) * (length / 2)
    obj = new_box(
        f"Wall_{index:03d}",
        Vector((length, thickness, height)),
        Vector((center.x, center.y, height / 2)),
        angle,
        MATERIALS["wall"],
    )
    doors, windows = openings_on_wall(wall.get("id") or "")
    cutters: list[bpy.types.Object] = []
    for opening in doors:
        interval = opening_interval(opening)
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
                MATERIALS["wall"],
            )
        )
    for opening in windows:
        interval = opening_interval(opening)
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
                MATERIALS["wall"],
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
        with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
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


def build_baseboards(wall: dict, index: int) -> list[bpy.types.Object]:
    start, _end, length, angle = wall_vec(wall)
    thickness = float(wall.get("thickness_m") or 0.12) * 1.15
    doors, _windows = openings_on_wall(wall.get("id") or "")
    blocked: list[tuple[float, float]] = []
    for door in doors:
        interval = opening_interval(door)
        if interval:
            blocked.append(interval)
    blocked.sort()
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
                MATERIALS["baseboard"],
            )
        )
    return objs


def build_door_assets(door: dict, walls_by_id: dict, index: int) -> None:
    wall = walls_by_id.get(door.get("wall_id"))
    interval = opening_interval(door)
    if wall is None or interval is None:
        return
    start, _end, _length, angle = wall_vec(wall)
    o_start, o_end = interval
    width = o_end - o_start
    height = float(door.get("height_m") or 2.1)
    thickness = float(wall.get("thickness_m") or 0.12)
    direction = Vector((math.cos(angle), math.sin(angle), 0.0))
    # Frame: two jambs and a header hugging the cut.
    for side_offset in (o_start + 0.03, o_end - 0.03):
        position = start + direction * side_offset
        new_box(
            f"DoorJamb_{index:03d}_{side_offset:.2f}",
            Vector((0.06, thickness * 1.15, height)),
            Vector((position.x, position.y, height / 2)),
            angle,
            MATERIALS["frame"],
        )
    header_mid = start + direction * ((o_start + o_end) / 2)
    new_box(
        f"DoorHeader_{index:03d}",
        Vector((width, thickness * 1.15, 0.08)),
        Vector((header_mid.x, header_mid.y, height + 0.04)),
        angle,
        MATERIALS["frame"],
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
        MATERIALS["door"],
    )


def build_window_assets(window: dict, walls_by_id: dict, index: int) -> None:
    wall = walls_by_id.get(window.get("wall_id"))
    interval = opening_interval(window)
    if wall is None or interval is None:
        return
    start, _end, _length, angle = wall_vec(wall)
    o_start, o_end = interval
    width = o_end - o_start
    height = float(window.get("height_m") or 1.2)
    sill = float(window.get("sill_height_m") or 0.9)
    thickness = float(wall.get("thickness_m") or 0.12)
    direction = Vector((math.cos(angle), math.sin(angle), 0.0))
    mid = start + direction * ((o_start + o_end) / 2)
    z_mid = sill + height / 2
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
            MATERIALS["frame"],
        )
    for side in (o_start + 0.03, o_end - 0.03):
        position = start + direction * side
        new_box(
            f"WindowJamb_{index:03d}_{side:.2f}",
            Vector((0.06, thickness * 1.1, height)),
            Vector((position.x, position.y, z_mid)),
            angle,
            MATERIALS["frame"],
        )
    new_box(
        f"WindowGlass_{index:03d}",
        Vector((width - 0.05, 0.02, height - 0.05)),
        Vector((mid.x, mid.y, z_mid)),
        angle,
        MATERIALS["glass"],
    )


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
    background.inputs["Strength"].default_value = 0.6
    output = nodes.new("ShaderNodeOutputWorld")
    links.new(sky.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])

    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 3.5
    sun_data.angle = math.radians(1.0)
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
        light_data.energy = max(25.0, min(120.0, area * 8.0))
        light_data.shadow_soft_size = 0.2
        light = bpy.data.objects.new(f"RoomLight_{index:03d}", light_data)
        height = float(PLAN.get("walls", [{}])[0].get("height_m") or WALL_HEIGHT_DEFAULT)
        light.location = (cx, cy, height - 0.35)
        link(light)


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
        (["Baseboard_", "DoorJamb_", "DoorHeader_", "DoorLeaf_", "WindowFrame_", "WindowJamb_"], "Trim_Joined", max(512, LIGHTMAP_PX // 2)),
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
    device = enable_gpu_if_available()
    print(f"[generate_building] mode={MODE} samples={SAMPLES} lightmap={LIGHTMAP_PX} device={device}")
    build_materials()

    walls = PLAN.get("walls", [])
    walls_by_id = {w.get("id"): w for w in walls if w.get("id")}
    for index, wall in enumerate(walls):
        build_wall(wall, index)
        build_baseboards(wall, index)

    rooms = PLAN.get("rooms", [])
    if rooms:
        for index, room in enumerate(rooms):
            build_polygon_slab(f"Floor_{index:03d}", room.get("points", []), 0.0, FLOOR_THICKNESS, MATERIALS["floor"])
            height = float(walls[0].get("height_m") or WALL_HEIGHT_DEFAULT) if walls else WALL_HEIGHT_DEFAULT
            build_polygon_slab(
                f"Ceiling_{index:03d}", room.get("points", []), height + CEILING_THICKNESS, CEILING_THICKNESS, MATERIALS["ceiling"]
            )
    elif walls:
        xs = [v for w in walls for v in (w["start"]["x"], w["end"]["x"])]
        ys = [v for w in walls for v in (w["start"]["y"], w["end"]["y"])]
        pad = 0.3
        rect = [
            {"x": min(xs) - pad, "y": min(ys) - pad},
            {"x": max(xs) + pad, "y": min(ys) - pad},
            {"x": max(xs) + pad, "y": max(ys) + pad},
            {"x": min(xs) - pad, "y": max(ys) + pad},
        ]
        build_polygon_slab("Floor_000", rect, 0.0, FLOOR_THICKNESS, MATERIALS["floor"])

    for index, door in enumerate(PLAN.get("doors", [])):
        build_door_assets(door, walls_by_id, index)
    for index, window in enumerate(PLAN.get("windows", [])):
        build_window_assets(window, walls_by_id, index)

    build_lighting()

    if MODE != "none":
        run_bake()

    output = Path(ARGS.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "filepath": str(output),
        "export_format": "GLB",
        "export_apply": True,
        "export_lights": MODE == "none",
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
