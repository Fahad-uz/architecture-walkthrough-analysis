from __future__ import annotations

from pathlib import Path

import pytest

from architecture_walkthrough.geometry.models import FloorPlanModel
from architecture_walkthrough.scene.blender_runner import run_blender_script
from architecture_walkthrough.scene.scene_builder import build_blender_script


def test_blender_script_embeds_floorplan_as_json_string() -> None:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json"))
    script = build_blender_script(model, Path("building.glb"))
    compile(script, "generated_walkthrough.py", "exec")
    assert "import json" in script
    assert "floorplan = json.loads(" in script
    assert '"external": true' in script
    assert "generate_building.py" in script
    assert "template_path.read_text" in script
    assert "add_placeholder_furniture" not in script
    assert "BLENDER_EEVEE_NEXT" in script and "BLENDER_EEVEE" in script


@pytest.mark.integration
def test_blender_walkthrough_script_builds_the_production_scene(tmp_path: Path) -> None:
    import shutil

    if shutil.which("blender") is None:
        pytest.skip("Blender not on PATH")
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json"))
    output = tmp_path / "walkthrough.glb"
    preview = tmp_path / "preview.png"
    script_path = tmp_path / "walkthrough.py"
    script_path.write_text(
        build_blender_script(model, output, preview_image=preview),
        encoding="utf-8",
    )

    run_blender_script("blender", script_path, timeout_seconds=120)

    assert output.exists() and output.stat().st_size > 10_000
    assert preview.exists() and preview.stat().st_size > 1_000
