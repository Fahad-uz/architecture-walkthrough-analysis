from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from architecture_walkthrough.geometry.models import (
        AssetPlacement,
        FloorPlanModel,
        FurniturePlacement,
    )


def furniture_footprint_key(
    item: FurniturePlacement | AssetPlacement,
) -> tuple[str, float, float, float, float, float]:
    """Identify legacy mirrors without merging different colocated categories."""
    tokens = re.split(r"[^a-z0-9]+", item.category.lower())
    family = next((token for token in tokens if token in {
        "bed", "sofa", "couch", "table", "chair", "plant", "rug", "carpet",
        "wardrobe", "cabinet", "sink", "stove",
    }), item.category.lower())
    family = {"couch": "sofa", "carpet": "rug"}.get(family, family)
    return (
        family, round(item.center.x, 5), round(item.center.y, 5),
        round(item.width_m, 5), round(item.depth_m, 5), round(item.rotation_deg % 360, 5),
    )


def link_legacy_furniture_assets(
    furniture: list[FurniturePlacement],
    assets: list[AssetPlacement],
) -> tuple[list[FurniturePlacement], list[AssetPlacement]]:
    """Link exact legacy mirrors before the editor can move or delete an item.

    A persisted link also records removal: assets whose furniture no longer
    exists must not become independent objects again. Unknown origins remain
    unknown, and objects without a unique mirror need no invented identity.
    Copies keep normalization from mutating placements supplied by a caller.
    """
    result = list(furniture)
    linked_assets = list(assets)
    candidates: dict[tuple[str, float, float, float, float, float], list[int]] = {}
    for index, item in enumerate(result):
        candidates.setdefault(furniture_footprint_key(item), []).append(index)
    used = {item.id for item in result if item.id}
    used.update(asset.furniture_id for asset in assets if asset.furniture_id)
    for asset_index, asset in enumerate(assets):
        if asset.furniture_id is not None:
            continue
        matches = candidates.get(furniture_footprint_key(asset), [])
        if len(matches) != 1:
            continue
        index = matches[0]
        item = result[index]
        if item.id is None:
            identifier = f"legacy_furniture_{index:03d}"
            suffix = 1
            while identifier in used:
                identifier = f"legacy_furniture_{index:03d}_{suffix}"
                suffix += 1
            used.add(identifier)
            item = item.model_copy(update={"id": identifier})
        # Historically the detailed mirror supplied the rendered height. Store
        # that effective legacy value once; subsequent inspector edits win.
        if item.provenance == "unknown" and asset.height_m is not None:
            item = item.model_copy(update={"height_m": asset.height_m})
        result[index] = item
        linked_assets[asset_index] = asset.model_copy(update={"furniture_id": item.id})
    return result, linked_assets


def resolve_furniture(model: FloorPlanModel) -> list[FurniturePlacement]:
    """Resolve the authoritative arrangement for rendering and route planning.

    Linked assets retain their saved metadata but take geometry from furniture.
    Their height fills only an unspecified furniture height. Unlinked legacy
    assets retain exact-footprint coalescing and its detailed-height fallback.
    """
    from architecture_walkthrough.geometry.models import FurniturePlacement

    result = list(model.furniture)
    identifiers = {item.id: index for index, item in enumerate(result) if item.id}
    occupied = {furniture_footprint_key(item): index for index, item in enumerate(result)}
    for asset in model.asset_placements:
        if asset.furniture_id is not None:
            index = identifiers.get(asset.furniture_id)
            if index is not None and result[index].height_m is None and asset.height_m is not None:
                result[index] = result[index].model_copy(update={"height_m": asset.height_m})
            continue
        item = FurniturePlacement.model_validate(asset.model_dump())
        identity = furniture_footprint_key(item)
        if identity not in occupied:
            occupied[identity] = len(result)
            result.append(item)
        elif item.height_m is not None:
            index = occupied[identity]
            result[index] = result[index].model_copy(update={"height_m": item.height_m})
    return result
