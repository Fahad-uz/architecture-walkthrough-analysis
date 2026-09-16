from __future__ import annotations

import json
from pathlib import Path

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.models import FloorPlanModel
from architecture_walkthrough.scene.blender_generator import TEMPLATE
from architecture_walkthrough.scene.lighting import LightingSettings
from architecture_walkthrough.scene.pbr_materials import (
    load_material_registry,
    model_with_material_plan,
)


def build_blender_script(
    floorplan: FloorPlanModel,
    output_glb: Path,
    preview_image: Path | None = None,
    blend_file: Path | None = None,
    lighting: LightingSettings | None = None,
    render_frames_dir: Path | None = None,
    frame_count: int = 120,
    config: AppConfig | None = None,
) -> str:
    """Build the offline-render script around the production scene generator.

    Model export and video rendering used to have separate geometry paths. The
    video path produced uncut wall boxes and placeholder furniture, so its
    frames could not match the browser GLB. This wrapper executes the same
    production template in no-bake mode, then adds only the animated camera and
    render settings.
    """
    lighting = lighting or LightingSettings()
    runtime_config = config or AppConfig()
    registry = load_material_registry(runtime_config.textures.registry_path)
    render_model = model_with_material_plan(floorplan, registry)
    plan_data = json.dumps(render_model.model_dump(mode="json"))
    render_block = ""
    if render_frames_dir is not None:
        render_block = f"""
scene.frame_start = 1
scene.frame_end = {max(1, frame_count)}
scene.render.filepath = {str(render_frames_dir / "frame_")!r}
scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(animation=True)
"""
    preview_block = ""
    if preview_image is not None:
        preview_block = f"""
scene.render.filepath = {str(preview_image)!r}
scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
"""
    blend_block = ""
    if blend_file is not None:
        blend_block = f"bpy.ops.wm.save_as_mainfile(filepath={str(blend_file)!r})"

    return f"""
import bpy
import json
import math
import sys
from pathlib import Path
from mathutils import Vector

floorplan = json.loads({plan_data!r})
output_glb = Path({str(output_glb)!r})
template_path = Path({str(TEMPLATE.resolve())!r})
render_input = output_glb.with_suffix(".render_input.json")
render_input.parent.mkdir(parents=True, exist_ok=True)
render_input.write_text(json.dumps(floorplan), encoding="utf-8")

saved_argv = list(sys.argv)
sys.argv = [
    str(template_path),
    "--",
    "--floorplan",
    str(render_input),
    "--output",
    str(output_glb),
    "--mode",
    "none",
]
template_globals = {{"__name__": "__main__", "__file__": str(template_path)}}
exec(compile(template_path.read_text(encoding="utf-8"), str(template_path), "exec"), template_globals)
sys.argv = saved_argv

scene = bpy.context.scene
for light in bpy.data.lights:
    if "bake_energy" in light:
        light.energy = light["bake_energy"]
available_engines = {{
    item.identifier
    for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
}}
scene.render.engine = (
    "BLENDER_EEVEE_NEXT"
    if "BLENDER_EEVEE_NEXT" in available_engines
    else "BLENDER_EEVEE"
)
scene.render.resolution_x = 960
scene.render.resolution_y = 540
scene.render.resolution_percentage = 100
scene.view_settings.exposure = {lighting.exposure}

cam_data = bpy.data.cameras.new("Walkthrough_Camera")
cam_data.lens = 24
cam_data.clip_start = 0.05
cam = bpy.data.objects.new("Walkthrough_Camera", cam_data)
bpy.context.collection.objects.link(cam)
scene.camera = cam
if hasattr(bpy.context.preferences.edit, "keyframe_new_interpolation_type"):
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"

waypoints = floorplan.get("camera_waypoints") or []
if waypoints:
    for index, waypoint in enumerate(waypoints):
        position = waypoint["position"]
        frame = 1 + int(
            index * max(1, ({max(1, frame_count)} - 1) / max(1, len(waypoints) - 1))
        )
        cam.location = (float(position["x"]), float(position["y"]), 1.65)
        if waypoint.get("look_at"):
            target = waypoint["look_at"]
        elif index + 1 < len(waypoints):
            target = waypoints[index + 1]["position"]
        elif index:
            target = waypoints[index - 1]["position"]
        else:
            target = {{"x": float(position["x"]), "y": float(position["y"]) + 1.0}}
        direction = Vector(
            (
                float(target["x"]) - float(position["x"]),
                float(target["y"]) - float(position["y"]),
                -0.10,
            )
        )
        if direction.length < 1e-6:
            direction = Vector((0.0, 1.0, -0.10))
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert(data_path="location", frame=frame)
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)
else:
    wall_points = [
        (float(point["x"]), float(point["y"]))
        for wall in floorplan.get("walls", [])
        for point in (wall["start"], wall["end"])
    ]
    if wall_points:
        xs = [point[0] for point in wall_points]
        ys = [point[1] for point in wall_points]
        center = Vector(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 1.2))
        cam.location = (center.x, min(ys) - 2.0, 1.65)
        cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    else:
        cam.location = (0.0, -3.0, 1.65)
        cam.rotation_euler = Vector((0.0, 1.0, -0.1)).to_track_quat("-Z", "Y").to_euler()

{blend_block}
{preview_block}
{render_block}
"""
