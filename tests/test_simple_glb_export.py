from __future__ import annotations

from pathlib import Path

import trimesh

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.pipeline import build_model
from architecture_walkthrough.scene.simple_glb import export_simple_glb
from architecture_walkthrough.geometry.models import FloorPlanModel


def test_simple_glb_export_writes_loadable_glb(tmp_path: Path) -> None:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json"))
    output = export_simple_glb(model, tmp_path / "building.glb")
    assert output.exists()
    assert output.stat().st_size > 0
    loaded = trimesh.load(output, force="scene")
    assert len(loaded.geometry) >= 2


def test_build_model_defaults_to_pure_python_glb_export(tmp_path: Path) -> None:
    output = build_model(Path("tests/fixtures/sample_floorplan.json"), tmp_path / "building.glb", AppConfig())
    assert output.exists()
    assert output.suffix == ".glb"
