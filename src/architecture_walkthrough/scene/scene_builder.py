from __future__ import annotations

import json
from pathlib import Path

from architecture_walkthrough.geometry.models import FloorPlanModel
from architecture_walkthrough.scene.furniture import blender_furniture_function_script
from architecture_walkthrough.scene.lighting import LightingSettings, blender_lighting_script
from architecture_walkthrough.scene.materials import blender_material_script


def build_blender_script(
    floorplan: FloorPlanModel,
    output_glb: Path,
    preview_image: Path | None = None,
    blend_file: Path | None = None,
    lighting: LightingSettings | None = None,
    render_frames_dir: Path | None = None,
    frame_count: int = 120,
) -> str:
    lighting = lighting or LightingSettings()
    preview_line = ""
    if preview_image:
        preview_line = f'bpy.ops.render.render(write_still=True); bpy.data.images["Render Result"].save_render(r"{preview_image}")'
    render_line = ""
    if render_frames_dir:
        render_line = f"""
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = {frame_count}
bpy.context.scene.render.filepath = r"{render_frames_dir / 'frame_'}"
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(animation=True)
"""
    blend_line = f'bpy.ops.wm.save_as_mainfile(filepath=r"{blend_file}")' if blend_file else ""
    data = json.dumps(floorplan.model_dump(mode="json"))
    return f"""
import bpy
import math

floorplan = {data}

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
bpy.context.scene.unit_settings.system = "METRIC"
bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items] else "BLENDER_EEVEE"
bpy.context.scene.render.resolution_x = 960
bpy.context.scene.render.resolution_y = 540

{blender_material_script()}
wall_mat = make_mat("Wall_Matte_White", (0.82, 0.82, 0.78, 1))
floor_mat = make_mat("Floor_Neutral", (0.42, 0.38, 0.32, 1))
furniture_mat = make_mat("Furniture_Placeholder", (0.25, 0.36, 0.48, 1))

{blender_furniture_function_script()}

def wall_length(wall):
    dx = wall["end"]["x"] - wall["start"]["x"]
    dy = wall["end"]["y"] - wall["start"]["y"]
    return math.sqrt(dx * dx + dy * dy)

def add_wall(wall, index):
    length = wall_length(wall)
    cx = (wall["start"]["x"] + wall["end"]["x"]) / 2
    cy = (wall["start"]["y"] + wall["end"]["y"]) / 2
    angle = math.atan2(wall["end"]["y"] - wall["start"]["y"], wall["end"]["x"] - wall["start"]["x"])
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, wall["height_m"] / 2))
    obj = bpy.context.object
    obj.name = f"Wall_{{index:03d}}"
    obj.dimensions = (length, wall["thickness_m"], wall["height_m"])
    obj.rotation_euler[2] = angle
    obj.data.materials.append(wall_mat)
    return obj

def scene_bounds():
    pts = []
    for wall in floorplan["walls"]:
        pts.append((wall["start"]["x"], wall["start"]["y"]))
        pts.append((wall["end"]["x"], wall["end"]["y"]))
    if not pts:
        return (-2, -2, 2, 2)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs)-1, min(ys)-1, max(xs)+1, max(ys)+1)

minx, miny, maxx, maxy = scene_bounds()
bpy.ops.mesh.primitive_cube_add(size=1, location=((minx + maxx) / 2, (miny + maxy) / 2, -0.05))
floor = bpy.context.object
floor.name = "Floor_Slab"
floor.dimensions = (maxx - minx, maxy - miny, 0.1)
floor.data.materials.append(floor_mat)

for idx, wall in enumerate(floorplan["walls"]):
    add_wall(wall, idx)
for item in floorplan.get("furniture", []):
    add_placeholder_furniture(item, furniture_mat)

{blender_lighting_script(lighting)}

cam_data = bpy.data.cameras.new("Walkthrough_Camera")
cam = bpy.data.objects.new("Walkthrough_Camera", cam_data)
bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam
waypoints = floorplan.get("camera_waypoints", [])
if waypoints:
    for index, waypoint in enumerate(waypoints):
        pos = waypoint["position"]
        frame = 1 + int(index * max(1, ({frame_count} - 1) / max(1, len(waypoints) - 1)))
        cam.location = (pos["x"], pos["y"], 1.65)
        look_at = waypoint.get("look_at") or (waypoints[index + 1]["position"] if index + 1 < len(waypoints) else pos)
        dx = look_at["x"] - pos["x"]
        dy = look_at["y"] - pos["y"]
        cam.rotation_euler = (math.radians(75), 0, math.atan2(dy, dx) - math.radians(90))
        cam.keyframe_insert(data_path="location", frame=frame)
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)
else:
    cam.location = ((minx + maxx) / 2, miny - 1.5, 1.65)
cam.rotation_euler = (math.radians(70), 0, 0)

{blend_line}
bpy.ops.export_scene.gltf(filepath=r"{output_glb}", export_format="GLB")
{preview_line}
{render_line}
"""
