from __future__ import annotations

import json
from pathlib import Path
import struct

import pytest
from shapely.geometry import Point

from architecture_walkthrough.geometry.models import (
    AssetPlacement,
    FloorPlanModel,
    FurniturePlacement,
    Point2D,
)
from architecture_walkthrough.scene.furniture import render_furniture
from architecture_walkthrough.scene.pbr_materials import MaterialRegistry, model_with_material_plan
from architecture_walkthrough.scene.simple_glb import export_simple_glb
from architecture_walkthrough.walkthrough.path_planner import _build_free_space


def _legacy_model() -> FloorPlanModel:
    return FloorPlanModel.model_validate({
        "walls": [{"start": {"x": 0, "y": 0}, "end": {"x": 6, "y": 0}}],
        "rooms": [{"points": [
            {"x": 0, "y": 0}, {"x": 6, "y": 0}, {"x": 6, "y": 5}, {"x": 0, "y": 5},
        ]}],
        "furniture": [{
            "category": "table", "center": {"x": 2, "y": 2},
            "width_m": 1, "depth_m": 1, "height_m": .7,
        }],
        "asset_placements": [{
            "category": "table_dining", "center": {"x": 2, "y": 2},
            "width_m": 1, "depth_m": 1, "height_m": .9, "asset_preset": "marble_table",
        }],
    })


def test_legacy_mirror_link_preserves_effective_height_and_caller_inputs() -> None:
    furniture = FurniturePlacement(
        category="sofa", center=Point2D(x=2, y=2), width_m=1.5, depth_m=.8, height_m=.7,
    )
    asset = AssetPlacement.model_validate({
        **furniture.model_dump(), "category": "sofa_3_seat", "height_m": 1.05,
        "asset_preset": "leather_sofa",
    })
    model = FloorPlanModel(furniture=[furniture], asset_placements=[asset])
    assert model.furniture[0].id is not None
    assert model.asset_placements[0].furniture_id == model.furniture[0].id
    assert model.furniture[0].provenance == "unknown"
    assert model.furniture[0].height_m == render_furniture(model)[0].height_m == 1.05
    assert model.asset_placements[0].asset_preset == "leather_sofa"
    assert furniture.id is None and furniture.height_m == .7
    assert asset.furniture_id is None
    reloaded = FloorPlanModel.model_validate_json(model.model_dump_json())
    assert reloaded.furniture == model.furniture
    assert reloaded.asset_placements == model.asset_placements


@pytest.mark.parametrize("provenance", ["user", "detected"])
def test_modern_explicit_height_wins_when_first_linking_a_mirror(provenance: str) -> None:
    payload = _legacy_model().model_dump()
    payload["furniture"][0].update({"provenance": provenance, "height_m": 1.2})
    payload["asset_placements"][0]["furniture_id"] = None
    model = FloorPlanModel.model_validate(payload)
    assert render_furniture(model)[0].height_m == 1.2


@pytest.mark.parametrize("deleted", [False, True])
def test_saved_edits_reach_both_exporters_and_collision_space_without_ghost_assets(
    tmp_path: Path, deleted: bool,
) -> None:
    model = _legacy_model()
    original_asset = model.asset_placements[0].model_dump()
    if deleted:
        model.furniture.clear()
    else:
        item = model.furniture[0]
        item.center = Point2D(x=4, y=3)
        item.width_m = 1.4
        item.depth_m = .8
        item.rotation_deg = 37
        item.height_m = 1.2
    path = tmp_path / "edited.json"
    model.save_json(path)
    reloaded = FloorPlanModel.load_json(path)
    before_render = reloaded.model_dump()
    resolved = render_furniture(reloaded)
    assert resolved == reloaded.furniture
    assert len(resolved) == (0 if deleted else 1)
    assert reloaded.asset_placements[0].model_dump() == original_asset
    assert model_with_material_plan(reloaded, MaterialRegistry()).furniture == resolved

    glb = export_simple_glb(reloaded, tmp_path / "edited.glb").read_bytes()
    json_size = struct.unpack_from("<I", glb, 12)[0]
    document = json.loads(glb[20:20 + json_size])
    furniture_groups = {
        node["name"].split("_")[1]
        for node in document["nodes"] if node.get("name", "").startswith("Furniture_")
    }
    assert furniture_groups == (set() if deleted else {"000"})

    free_space, _ = _build_free_space(reloaded, clearance_m=.1, grid_size_m=.1)
    assert free_space.covers(Point(2, 2)), "the legacy asset's old footprint must be free"
    assert free_space.covers(Point(4, 3)) == deleted
    assert reloaded.model_dump() == before_render


def test_linked_asset_height_only_fills_unspecified_height_after_edits() -> None:
    model = _legacy_model()
    model.furniture[0].height_m = None
    model.furniture[0].center = Point2D(x=4, y=3)
    resolved = render_furniture(model)
    assert len(resolved) == 1
    assert resolved[0].height_m == .9
    assert resolved[0].center == Point2D(x=4, y=3)
    assert model.furniture[0].height_m is None


def test_deleted_link_is_not_reassigned_to_new_furniture_at_the_old_footprint() -> None:
    model = _legacy_model()
    deleted_identifier = model.furniture[0].id
    payload = model.model_dump()
    payload["furniture"][0].update({"id": None, "height_m": 1.2, "provenance": "user"})
    replacement = FloorPlanModel.model_validate(payload)
    assert replacement.furniture[0].id is None
    assert replacement.asset_placements[0].furniture_id == deleted_identifier
    assert render_furniture(replacement) == replacement.furniture
    assert render_furniture(replacement)[0].height_m == 1.2


def test_unlinked_standalone_assets_and_different_colocated_categories_survive() -> None:
    model = _legacy_model()
    asset = model.asset_placements[0].model_copy(update={"furniture_id": None})
    rug = FurniturePlacement(
        category="rug", center=asset.center, width_m=asset.width_m, depth_m=asset.depth_m,
    )
    standalone = FloorPlanModel(furniture=[rug], asset_placements=[asset])
    assert standalone.furniture[0].id is None
    assert standalone.asset_placements[0].furniture_id is None
    assert [item.category for item in render_furniture(standalone)] == ["rug", "table_dining"]
    assets_only = FloorPlanModel(asset_placements=[asset])
    assert [item.category for item in render_furniture(assets_only)] == ["table_dining"]


def test_legacy_link_ids_do_not_collide_with_existing_or_deleted_identifiers() -> None:
    model = _legacy_model()
    payload = model.model_dump()
    payload["furniture"][0]["id"] = None
    payload["asset_placements"][0]["furniture_id"] = None
    payload["asset_placements"].append({
        **payload["asset_placements"][0], "furniture_id": "legacy_furniture_000",
        "center": {"x": 1, "y": 1},
    })
    normalized = FloorPlanModel.model_validate(payload)
    assert normalized.furniture[0].id == "legacy_furniture_000_1"
    assert normalized.asset_placements[0].furniture_id == "legacy_furniture_000_1"
    assert len(render_furniture(normalized)) == 1


def test_multiple_detailed_mirrors_share_the_single_unambiguous_furniture_identity() -> None:
    payload = _legacy_model().model_dump()
    payload["asset_placements"][0]["furniture_id"] = None
    payload["asset_placements"].append({
        **payload["asset_placements"][0], "asset_preset": "alternate_table", "height_m": 1.1,
    })
    model = FloorPlanModel.model_validate(payload)
    assert {asset.furniture_id for asset in model.asset_placements} == {model.furniture[0].id}
    assert render_furniture(model)[0].height_m == 1.1
    assert len(render_furniture(model)) == 1
    assert {asset.asset_preset for asset in model.asset_placements} == {
        "marble_table", "alternate_table",
    }
    model.furniture.clear()
    assert render_furniture(model) == []


def test_ambiguous_legacy_mirrors_do_not_guess_a_furniture_association() -> None:
    payload = _legacy_model().model_dump()
    payload["furniture"][0]["id"] = None
    payload["furniture"].append({**payload["furniture"][0], "height_m": 1.1})
    payload["asset_placements"][0]["furniture_id"] = None
    model = FloorPlanModel.model_validate(payload)
    assert all(item.id is None for item in model.furniture)
    assert model.asset_placements[0].furniture_id is None
    assert model.asset_placements[0].asset_preset == "marble_table"
    assert [item.height_m for item in model.furniture] == [.9, 1.1]
