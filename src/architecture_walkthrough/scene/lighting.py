from __future__ import annotations

from architecture_walkthrough.geometry.models import LightingSettings
from architecture_walkthrough.scene.lighting_values import effective_lighting

__all__ = ["LightingSettings", "effective_lighting", "blender_lighting_script"]


def blender_lighting_script(settings: LightingSettings) -> str:
    effective = effective_lighting(settings.model_dump())
    return f"""
world = bpy.context.scene.world or bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.color = (0.78, 0.82, 0.86)
bpy.context.scene.view_settings.exposure = {settings.exposure}
sun = bpy.data.lights.new("Sun", "SUN")
sun.energy = {1.8 * effective['power_factor']}
sun.color = {effective['color_scale']!r}
sun_obj = bpy.data.objects.new("Sun", sun)
bpy.context.collection.objects.link(sun_obj)
sun_obj.rotation_euler = (math.radians(45), 0, math.radians(35))
area = bpy.data.lights.new("Interior_Key_Area", "AREA")
area.energy = {settings.intensity * settings.brightness}
area.color = {effective['color_scale']!r}
area.size = 5.0
area_obj = bpy.data.objects.new("Interior_Key_Area", area)
bpy.context.collection.objects.link(area_obj)
area_obj.location = (0, 0, 4)
"""
