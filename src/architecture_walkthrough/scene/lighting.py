from __future__ import annotations

from pydantic import BaseModel, Field


class LightingSettings(BaseModel):
    preset: str = "neutral_studio"
    color_temperature: int = Field(default=4500, ge=1000, le=12000)
    intensity: float = Field(default=450.0, ge=0.0)
    exposure: float = Field(default=0.0, ge=-5.0, le=5.0)


def blender_lighting_script(settings: LightingSettings) -> str:
    return f"""
world = bpy.context.scene.world or bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.color = (0.78, 0.82, 0.86)
bpy.context.scene.view_settings.exposure = {settings.exposure}
sun = bpy.data.lights.new("Sun", "SUN")
sun.energy = 1.8
sun_obj = bpy.data.objects.new("Sun", sun)
bpy.context.collection.objects.link(sun_obj)
sun_obj.rotation_euler = (math.radians(45), 0, math.radians(35))
area = bpy.data.lights.new("Interior_Key_Area", "AREA")
area.energy = {settings.intensity}
area.size = 5.0
area_obj = bpy.data.objects.new("Interior_Key_Area", area)
bpy.context.collection.objects.link(area_obj)
area_obj.location = (0, 0, 4)
"""
