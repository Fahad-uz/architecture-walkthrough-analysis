from __future__ import annotations

from pathlib import Path

import pytest

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.models import ArchitecturalElement, FloorPlanModel, FurniturePlacement, Point2D
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


def test_template_keeps_scene_semantics_in_the_blender_export() -> None:
    source = TEMPLATE.read_text(encoding="utf-8")
    assert 'ceiling.get("enabled", False)' in source
    assert 'PLAN.get("furniture", [])' in source
    assert 'PLAN.get("special_elements", [])' in source
    assert "build_procedural_asset" in source
    assert "asset_family" in source
    assert "room_floor_coverage" in source
    assert "MIN_ROOM_FLOOR_COVERAGE" in source
    assert "tune_realtime_lights_for_export" in source
    assert '"export_lights": True' in source
    assert "opening_interval(opening, length)" in source
    assert "Blender 3.4+" in source
    assert "temp_override" not in source
    trim_group = next(line for line in source.splitlines() if '"Trim_Joined"' in line)
    assert '"DoorLeaf_"' not in trim_group


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
    import json
    import shutil
    import struct

    import trimesh

    if shutil.which("blender") is None:
        pytest.skip("Blender not on PATH")
    model = FloorPlanModel.load_json(FIXTURES / "sample_two_bedroom_flat.json")
    model = model.model_copy(
        update={
            "furniture": [
                FurniturePlacement(
                    category="bed",
                    center=Point2D(x=4.7, y=5.8),
                    width_m=1.5,
                    depth_m=2.0,
                ),
                FurniturePlacement(
                    category="sofa",
                    center=Point2D(x=1.5, y=5.7),
                    width_m=1.8,
                    depth_m=0.8,
                ),
                FurniturePlacement(
                    category="bedside_table",
                    center=Point2D(x=3.9, y=5.6),
                    width_m=0.55,
                    depth_m=0.45,
                ),
                FurniturePlacement(
                    category="kitchen_sink",
                    center=Point2D(x=8.4, y=3.7),
                    width_m=0.60,
                    depth_m=0.50,
                ),
                FurniturePlacement(
                    category="kitchen_stove",
                    center=Point2D(x=7.5, y=3.7),
                    width_m=0.60,
                    depth_m=0.50,
                ),
            ],
            "special_elements": [
                ArchitecturalElement(
                    id="stair_fixture",
                    kind="staircase",
                    center=Point2D(x=8.4, y=2.1),
                    width_m=1.0,
                    depth_m=2.0,
                )
            ],
        }
    )
    output = generate_glb_with_blender(model, tmp_path / "building.glb", AppConfig(), mode="none")
    assert output.exists() and output.stat().st_size > 10_000
    scene = trimesh.load(output, force="scene")
    nodes = {str(name) for name in scene.graph.nodes_geometry}
    assert any(name.startswith("Furniture_") for name in nodes)
    assert any(name.startswith("Special_") for name in nodes)
    assert any(name.startswith("DoorLeaf_") for name in nodes)
    assert any("bedside_table_Top" in name for name in nodes)
    assert any("kitchen_sink_Rim" in name for name in nodes)
    assert any("kitchen_stove_Cooktop" in name for name in nodes)
    assert not any(name.startswith("Ceiling_") for name in nodes)
    binary = output.read_bytes()
    assert b"KHR_lights_punctual" in binary
    json_length, _ = struct.unpack_from("<II", binary, 12)
    gltf = json.loads(binary[20 : 20 + json_length])
    exported_lights = gltf["extensions"]["KHR_lights_punctual"]["lights"]
    assert exported_lights
    assert max(float(light["intensity"]) for light in exported_lights) <= 6_000
