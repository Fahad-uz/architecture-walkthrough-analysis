from __future__ import annotations

from pathlib import Path

import pytest

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.models import FloorPlanModel
from architecture_walkthrough.scene.blender_generator import (
    TEMPLATE,
    blender_generate_command,
    generate_glb_with_blender,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_template_exists_and_compiles() -> None:
    source = TEMPLATE.read_text(encoding="utf-8")
    compile(source, str(TEMPLATE), "exec")  # syntax check without bpy
    # Geometry must be identical across modes: bake gating happens after build.
    assert 'if MODE != "none":' in source
    assert "EXACT" in source  # boolean solver
    assert "KHR" in source or "export_lights" in source


def test_command_includes_mode_and_exit_code() -> None:
    command = blender_generate_command(
        "blender", Path("plan.json"), Path("out.glb"), "draft", 16, 512, denoise=True
    )
    assert "--python-exit-code" in command
    assert "--mode" in command and "draft" in command
    assert "--no-denoise" not in command
    no_denoise = blender_generate_command(
        "blender", Path("plan.json"), Path("out.glb"), "final", 256, 2048, denoise=False
    )
    assert "--no-denoise" in no_denoise


def test_unknown_bake_mode_is_rejected(tmp_path: Path) -> None:
    model = FloorPlanModel.load_json(FIXTURES / "sample_two_bedroom_flat.json")
    with pytest.raises(ValueError, match="unknown bake mode"):
        generate_glb_with_blender(model, tmp_path / "out.glb", AppConfig(), mode="ultra")


@pytest.mark.integration
def test_blender_generates_glb_in_none_mode(tmp_path: Path) -> None:
    import shutil

    if shutil.which("blender") is None:
        pytest.skip("Blender not on PATH")
    model = FloorPlanModel.load_json(FIXTURES / "sample_two_bedroom_flat.json")
    output = generate_glb_with_blender(model, tmp_path / "building.glb", AppConfig(), mode="none")
    assert output.exists() and output.stat().st_size > 10_000
