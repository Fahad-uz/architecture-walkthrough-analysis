from __future__ import annotations

from pathlib import Path

from architecture_walkthrough.geometry.models import FloorPlanModel
from architecture_walkthrough.scene.scene_builder import build_blender_script


def test_blender_script_embeds_floorplan_as_json_string() -> None:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json"))
    script = build_blender_script(model, Path("building.glb"))
    assert "import json" in script
    assert "floorplan = json.loads(" in script
    assert '"external": true' in script
