from __future__ import annotations

from pathlib import Path

import trimesh

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.pipeline import build_model
from architecture_walkthrough.scene.simple_glb import export_simple_glb
from architecture_walkthrough.geometry.models import FloorPlanModel, FurniturePlacement, Point2D


def test_simple_glb_export_writes_loadable_glb(tmp_path: Path) -> None:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json"))
    output = export_simple_glb(model, tmp_path / "building.glb")
    assert output.exists()
    assert output.stat().st_size > 0
    loaded = trimesh.load(output, force="scene")
    assert len(loaded.geometry) >= 2


def test_build_model_defaults_to_pure_python_glb_export(tmp_path: Path) -> None:
    # force=True: this test exercises the export path; the quality gate has its
    # own coverage in test_validation_gate.py (the legacy fixture has no scale).
    output = build_model(Path("tests/fixtures/sample_floorplan.json"), tmp_path / "building.glb", AppConfig(), force=True)
    assert output.exists()
    assert output.suffix == ".glb"


def test_simple_glb_export_includes_furniture_geometry(tmp_path: Path) -> None:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json")).model_copy(
        update={
            "furniture": [
                FurniturePlacement(
                    category="sofa",
                    center=Point2D(x=2.0, y=1.5),
                    width_m=1.6,
                    depth_m=0.8,
                )
            ]
        }
    )
    output = export_simple_glb(model, tmp_path / "building.glb")
    loaded = trimesh.load(output, force="scene")
    assert any("Furniture" in name for name in loaded.graph.nodes_geometry)


def test_simple_glb_export_builds_multi_part_furniture(tmp_path: Path) -> None:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json")).model_copy(
        update={
            "furniture": [
                FurniturePlacement(
                    category="bed",
                    center=Point2D(x=2.0, y=1.5),
                    width_m=1.8,
                    depth_m=2.2,
                )
            ]
        }
    )
    output = export_simple_glb(model, tmp_path / "building.glb")
    loaded = trimesh.load(output, force="scene")
    furniture_nodes = [name for name in loaded.graph.nodes_geometry if "Furniture" in name]
    assert len(furniture_nodes) >= 4
