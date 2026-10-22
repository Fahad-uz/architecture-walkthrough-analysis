from __future__ import annotations

from pathlib import Path

from architecture_walkthrough.geometry.models import FloorPlanModel, FurniturePlacement
from architecture_walkthrough.pipeline import prepare_walkthrough_floorplan


def test_legacy_furniture_does_not_acquire_an_invented_origin() -> None:
    item = FurniturePlacement.model_validate({
        "category": "sofa", "center": {"x": 2, "y": 2}, "width_m": 1.8, "depth_m": .8,
    })
    assert item.id is None
    assert item.provenance == "unknown"


def test_user_arrangement_and_identity_survive_save_and_walkthrough_preparation(tmp_path: Path) -> None:
    model = FloorPlanModel.model_validate({
        "pixels_per_metre": 100,
        "rooms": [{"id": "living", "name": "Living", "points": [
            {"x": 0, "y": 0}, {"x": 5, "y": 0}, {"x": 5, "y": 5}, {"x": 0, "y": 5},
        ]}],
        "furniture": [{
            "id": "user-sofa", "provenance": "user", "category": "sofa",
            "center": {"x": 2.1, "y": 3.2}, "width_m": 1.9, "depth_m": .85,
            "height_m": .78, "rotation_deg": 37,
        }],
    })
    source = tmp_path / "floorplan.corrected.json"
    output = tmp_path / "walkthrough.json"
    model.save_json(source)
    prepared = prepare_walkthrough_floorplan(source, output)
    reloaded = FloorPlanModel.load_json(output)
    assert reloaded.furniture == model.furniture == prepared.furniture
    assert reloaded.rooms == model.rooms
    assert reloaded.walls == model.walls
    assert reloaded.furniture[0].id == "user-sofa"
    assert reloaded.furniture[0].provenance == "user"


def test_empty_furnishing_stays_empty_during_preparation(tmp_path: Path) -> None:
    source = tmp_path / "floorplan.corrected.json"
    FloorPlanModel.model_validate({"rooms": [{"id": "empty", "points": [
        {"x": 0, "y": 0}, {"x": 5, "y": 0}, {"x": 5, "y": 5}, {"x": 0, "y": 5},
    ]}]}).save_json(source)
    prepared = prepare_walkthrough_floorplan(source, tmp_path / "walkthrough.json")
    assert prepared.furniture == []
