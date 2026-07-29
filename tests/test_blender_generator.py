from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import architecture_walkthrough.scene.blender_generator as blender_generator
from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.models import (
    ArchitecturalElement,
    CeilingSettings,
    FloorPlanModel,
    FurniturePlacement,
    MaterialAssignment,
    Point2D,
)
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
    assert "asset_cylinder" in source
    assert "asset_ellipsoid" in source
    assert source.count("mesh.from_pydata") >= 4
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


def test_blender_input_embeds_a_schema_safe_material_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = FloorPlanModel.load_json(
        FIXTURES / "sample_high_detail_floorplan.json"
    )
    config = AppConfig()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        blender_generator,
        "require_executable",
        lambda _executable, _purpose: "blender",
    )

    def fake_run(command: list[str], timeout_seconds: int):
        floorplan_path = Path(command[command.index("--floorplan") + 1])
        output_path = Path(command[command.index("--output") + 1])
        payload = json.loads(floorplan_path.read_text(encoding="utf-8"))
        captured["payload"] = payload
        captured["timeout"] = timeout_seconds
        FloorPlanModel.model_validate(payload)
        output_path.write_bytes(b"glTF")
        return SimpleNamespace(stderr="", stdout="")

    monkeypatch.setattr(blender_generator, "run_subprocess", fake_run)

    output = generate_glb_with_blender(
        model,
        tmp_path / "building.glb",
        config,
        mode="none",
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    material_plan = payload["metadata"]["blender_material_plan"]
    assert material_plan["room_floor_presets"] == ["wood", "ceramic_tile"]
    assert "blender_material_plan" not in model.metadata
    assert output.read_bytes() == b"glTF"


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
                FurniturePlacement(
                    category="dining_table",
                    center=Point2D(x=5.9, y=3.1),
                    width_m=1.60,
                    depth_m=0.80,
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
    assert any("bed_Duvet" in name for name in nodes)
    assert sum("bed_Pillow_" in name for name in nodes) == 2
    assert sum("sofa_Seat_Cushion_" in name for name in nodes) == 2
    assert sum("sofa_Back_Cushion_" in name for name in nodes) == 2
    assert any("dining_table_Top" in name for name in nodes)
    assert sum("dining_table_Leg_" in name for name in nodes) == 4
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


@pytest.mark.integration
def test_blender_binds_material_presets_to_architectural_surfaces(
    tmp_path: Path,
) -> None:
    import json
    import shutil
    import struct

    import trimesh

    if shutil.which("blender") is None:
        pytest.skip("Blender not on PATH")
    registry_path = tmp_path / "materials.yaml"
    registry_path.write_text(
        Path("assets/textures/material_registry.yaml").read_text(encoding="utf-8")
        + "\n"
        + "  masked_panel:\n"
        + "    base_color: [0.2, 0.4, 0.6, 0.4]\n"
        + "    alpha_mode: MASK\n"
        + "  opaque_panel:\n"
        + "    base_color: [0.6, 0.4, 0.2, 0.4]\n"
        + "    alpha_mode: OPAQUE\n",
        encoding="utf-8",
    )
    config = AppConfig()
    config = config.model_copy(
        update={
            "textures": config.textures.model_copy(
                update={"registry_path": registry_path}
            )
        }
    )
    model = FloorPlanModel.load_json(FIXTURES / "sample_two_bedroom_flat.json")
    model = model.model_copy(
        update={
            "walls": [
                model.walls[0].model_copy(update={"material_preset": "metal"}),
                model.walls[1].model_copy(
                    update={"material_preset": "masked_panel"}
                ),
                model.walls[2].model_copy(
                    update={"material_preset": "opaque_panel"}
                ),
                *model.walls[3:],
            ],
            "material_assignments": [
                MaterialAssignment(
                    target="room_living",
                    preset="glass",
                )
            ],
            "ceiling": CeilingSettings(
                enabled=True,
                material_preset="painted_wall",
            ),
        }
    )

    output = generate_glb_with_blender(
        model,
        tmp_path / "materials.glb",
        config,
        mode="none",
    )
    scene = trimesh.load(output, force="scene")
    nodes = {str(name) for name in scene.graph.nodes_geometry}

    def material_for(prefix: str) -> str:
        node = next(name for name in nodes if name.startswith(prefix))
        _transform, geometry_name = scene.graph[node]
        material = scene.geometry[geometry_name].visual.material
        return str(material.name)

    assert material_for("Wall_000") == "PBR_metal"
    assert material_for("Wall_001") == "PBR_masked_panel"
    assert material_for("Wall_002") == "PBR_opaque_panel"
    assert material_for("Wall_003") == "PBR_painted_wall"
    assert material_for("Floor_000") == "PBR_glass"
    assert material_for("Floor_001") == "PBR_wood"
    assert material_for("DoorJamb_000") == "PBR_wood"
    assert material_for("DoorLeaf_000") == "PBR_wood"
    assert material_for("WindowFrame_000") == "PBR_metal"
    assert material_for("WindowGlass_000") == "Window_Glass"
    assert material_for("Ceiling_000") == "PBR_painted_wall"

    binary = output.read_bytes()
    json_length, _ = struct.unpack_from("<II", binary, 12)
    gltf = json.loads(binary[20 : 20 + json_length])
    materials = {material["name"]: material for material in gltf["materials"]}
    assert materials["PBR_metal"]["pbrMetallicRoughness"]["metallicFactor"] == pytest.approx(0.85)
    assert materials["PBR_metal"].get("doubleSided", False) is False
    assert materials["PBR_masked_panel"]["alphaMode"] == "MASK"
    assert materials["PBR_masked_panel"]["pbrMetallicRoughness"]["baseColorFactor"][3] == pytest.approx(0.4)
    assert "alphaMode" not in materials["PBR_opaque_panel"]
    assert materials["PBR_opaque_panel"]["pbrMetallicRoughness"]["baseColorFactor"][3] == pytest.approx(1.0)
    assert materials["PBR_glass"]["alphaMode"] == "BLEND"
    assert materials["PBR_glass"]["doubleSided"] is True
    assert materials["Window_Glass"]["doubleSided"] is True
