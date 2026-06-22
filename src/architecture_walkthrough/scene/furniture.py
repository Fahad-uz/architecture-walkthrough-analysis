from __future__ import annotations


def blender_furniture_function_script() -> str:
    return """
def add_placeholder_furniture(item, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(item["center"]["x"], item["center"]["y"], 0.35))
    obj = bpy.context.object
    obj.name = "Furniture_" + item.get("category", "unknown")
    obj.dimensions = (item["width_m"], item["depth_m"], 0.7)
    obj.rotation_euler[2] = math.radians(item.get("rotation_deg", 0))
    obj.data.materials.append(mat)
    return obj
"""
