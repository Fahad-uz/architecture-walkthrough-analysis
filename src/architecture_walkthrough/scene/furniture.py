from __future__ import annotations

import re

from architecture_walkthrough.geometry.models import FloorPlanModel, FurniturePlacement


def render_furniture(model: FloorPlanModel) -> list[FurniturePlacement]:
    """Resolve explicitly placed assets without duplicating legacy furniture.

    Both lists can describe the same measured object. Only matching categories
    and footprints coalesce; a rug and table at the same location stay distinct.
    The saved, user-edited model is never modified.
    """
    result = list(model.furniture)

    def key(item: FurniturePlacement) -> tuple:
        tokens = re.split(r"[^a-z0-9]+", item.category.lower())
        family = next((token for token in tokens if token in {
            "bed", "sofa", "couch", "table", "chair", "plant", "rug", "carpet",
            "wardrobe", "cabinet", "sink", "stove",
        }), item.category.lower())
        family = {"couch": "sofa", "carpet": "rug"}.get(family, family)
        return (family, *(round(value, 5) for value in (
            item.center.x, item.center.y, item.width_m, item.depth_m,
            item.rotation_deg % 360,
        )))

    occupied = {key(item): index for index, item in enumerate(result)}
    for asset in model.asset_placements:
        item = FurniturePlacement.model_validate(asset.model_dump())
        identity = key(item)
        if identity not in occupied:
            occupied[identity] = len(result)
            result.append(item)
        elif item.height_m is not None:
            index = occupied[identity]
            result[index] = result[index].model_copy(update={"height_m": item.height_m})
    return result


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
