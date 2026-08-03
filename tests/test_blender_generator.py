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


def _face_points_away_from(
    vertices: list[tuple[float, float, float]],
    face: tuple[int, ...],
    center: tuple[float, float, float],
) -> bool:
    a, b, c = (vertices[index] for index in face[:3])
    edge_ab = tuple(b[axis] - a[axis] for axis in range(3))
    edge_ac = tuple(c[axis] - a[axis] for axis in range(3))
    normal = (
        edge_ab[1] * edge_ac[2] - edge_ab[2] * edge_ac[1],
        edge_ab[2] * edge_ac[0] - edge_ab[0] * edge_ac[2],
        edge_ab[0] * edge_ac[1] - edge_ab[1] * edge_ac[0],
    )
    face_center = tuple(
        sum(vertices[index][axis] for index in face) / len(face)
        for axis in range(3)
    )
    return sum(
        normal[axis] * (face_center[axis] - center[axis])
        for axis in range(3)
    ) > 0


def test_template_exists_and_compiles() -> None:
    source = TEMPLATE.read_text(encoding="utf-8")
    compile(source, str(TEMPLATE), "exec")  # syntax check without bpy
    # Geometry must be identical across modes: bake gating happens after build.
    assert 'if MODE != "none":' in source
    assert "EXACT" in source  # boolean solver
    assert "KHR" in source or "export_lights" in source


def test_box_faces_point_outward_for_cycles_baking() -> None:
    """Back-facing boxes produce empty COMBINED lightmaps in Cycles."""
    import ast

    source = TEMPLATE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    new_box = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "new_box"
    )
    faces_node = next(
        statement.value
        for statement in new_box.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "faces"
            for target in statement.targets
        )
    )
    faces = ast.literal_eval(faces_node)
    vertices = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ]

    assert all(_face_points_away_from(vertices, face, (0.0, 0.0, 0.0)) for face in faces)


def test_polygon_slab_faces_point_outward_for_cycles_baking() -> None:
    import ast

    source = TEMPLATE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    build_slab = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_polygon_slab"
    )
    face_loops = [
        statement
        for statement in build_slab.body
        if isinstance(statement, ast.For)
        and isinstance(statement.target, ast.Name)
        and statement.target.id in {"tri", "i"}
    ]
    face_program = ast.fix_missing_locations(ast.Module(body=face_loops, type_ignores=[]))
    for clockwise, loop in (
        (False, [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]),
        (True, [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)]),
    ):
        namespace: dict[str, object] = {
            "triangles": [(0, 1, 2)],
            "clockwise": clockwise,
            "n": 3,
            "faces": [],
        }
        exec(compile(face_program, str(TEMPLATE), "exec"), namespace)
        faces = namespace["faces"]
        assert isinstance(faces, list)
        vertices = [
            *((x, y, 1.0) for x, y in loop),
            *((x, y, 0.0) for x, y in loop),
        ]
        assert all(
            _face_points_away_from(vertices, face, (1 / 3, 1 / 3, 0.5))
            for face in faces
        )


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
    assert "DOOR_LEAF_OPEN_DEG = 90.0" in source
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
                FurniturePlacement(
                    category="kitchen_counter",
                    center=Point2D(x=6.7, y=4.4),
                    width_m=1.20,
                    depth_m=0.55,
                ),
                FurniturePlacement(
                    category="plant",
                    center=Point2D(x=6.7, y=5.4),
                    width_m=0.45,
                    depth_m=0.45,
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
    assert any(name.endswith("sofa_Back") for name in nodes)
    assert any("dining_table_Top" in name for name in nodes)
    assert sum("dining_table_Leg_" in name for name in nodes) == 4
    assert any("kitchen_sink_Rim" in name for name in nodes)
    assert any("kitchen_sink_Basin" in name for name in nodes)
    assert any("kitchen_sink_Faucet_Spout" in name for name in nodes)
    assert any("kitchen_stove_Cooktop" in name for name in nodes)
    assert sum("kitchen_stove_Burner_" in name for name in nodes) == 4
    assert any("kitchen_counter_ToeKick" in name for name in nodes)
    assert sum("kitchen_counter_Front_" in name for name in nodes) == 4
    assert sum("kitchen_counter_Handle_" in name for name in nodes) == 4
    assert any("plant_Pot" in name for name in nodes)
    assert sum("plant_Leaf_" in name for name in nodes) == 4
    assert not any(name.startswith("Ceiling_") for name in nodes)
    binary = output.read_bytes()
    assert b"KHR_lights_punctual" in binary
    json_length, _ = struct.unpack_from("<II", binary, 12)
    gltf = json.loads(binary[20 : 20 + json_length])
    exported_lights = gltf["extensions"]["KHR_lights_punctual"]["lights"]
    assert exported_lights
    assert max(float(light["intensity"]) for light in exported_lights) <= 6_000


@pytest.mark.integration
def test_blender_furniture_stays_inside_grounded_footprints(tmp_path: Path) -> None:
    import math
    import shutil
    import subprocess

    blender = shutil.which("blender")
    if blender is None:
        pytest.skip("Blender not on PATH")

    model = FloorPlanModel.load_json(FIXTURES / "sample_two_bedroom_flat.json")
    furniture = [
        FurniturePlacement(
            category="bed",
            center=Point2D(x=1.3, y=1.2),
            width_m=0.90,
            depth_m=0.80,
        ),
        FurniturePlacement(
            category="sofa",
            center=Point2D(x=3.0, y=1.2),
            width_m=0.90,
            depth_m=0.40,
            rotation_deg=90.0,
        ),
        FurniturePlacement(
            category="dining_table",
            center=Point2D(x=4.8, y=1.2),
            width_m=1.10,
            depth_m=0.60,
            rotation_deg=-30.0,
        ),
        FurniturePlacement(
            category="kitchen_counter",
            center=Point2D(x=6.7, y=1.2),
            width_m=1.20,
            depth_m=0.50,
            rotation_deg=90.0,
        ),
        FurniturePlacement(
            category="kitchen_sink",
            center=Point2D(x=1.3, y=4.2),
            width_m=0.60,
            depth_m=0.50,
        ),
        FurniturePlacement(
            category="kitchen_stove",
            center=Point2D(x=3.0, y=4.2),
            width_m=0.60,
            depth_m=0.50,
            rotation_deg=-90.0,
        ),
        FurniturePlacement(
            category="plant",
            center=Point2D(x=4.5, y=4.2),
            width_m=0.45,
            depth_m=0.45,
        ),
        FurniturePlacement(
            category="chair",
            center=Point2D(x=5.8, y=4.2),
            width_m=0.45,
            depth_m=0.45,
            rotation_deg=25.0,
        ),
    ]
    model = model.model_copy(update={"furniture": furniture})
    floorplan_path = tmp_path / "footprint-plan.json"
    output_path = tmp_path / "footprint-building.glb"
    manifest_path = tmp_path / "furniture-manifest.json"
    probe_path = tmp_path / "probe-furniture.py"
    floorplan_path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    probe_path.write_text(
        f"""
import json
import runpy
import sys
from pathlib import Path

from mathutils import Vector

template = Path({str(TEMPLATE.resolve())!r})
floorplan = Path({str(floorplan_path)!r})
output = Path({str(output_path)!r})
manifest = Path({str(manifest_path)!r})
sys.argv = [
    str(template),
    "--",
    "--floorplan",
    str(floorplan),
    "--output",
    str(output),
    "--mode",
    "none",
]
runpy.run_path(str(template), run_name="__main__")

import bpy

parts = {{}}
for obj in bpy.data.objects:
    if obj.type != "MESH" or not obj.name.startswith("Furniture_"):
        continue
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    parts[obj.name] = {{
        "corners": [[float(point.x), float(point.y), float(point.z)] for point in corners],
        "vertices": len(obj.data.vertices),
        "polygons": len(obj.data.polygons),
        "minimum_polygon_area": min(
            (float(polygon.area) for polygon in obj.data.polygons),
            default=0.0,
        ),
    }}
manifest.write_text(json.dumps(parts, indent=2), encoding="utf-8")
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            blender,
            "--background",
            "--python-exit-code",
            "1",
            "--python",
            str(probe_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-8_000:]
    parts = json.loads(manifest_path.read_text(encoding="utf-8"))

    tolerance = 1e-5
    for index, placement in enumerate(furniture):
        prefix = f"Furniture_{index:03d}_"
        placement_parts = {
            name: payload for name, payload in parts.items() if name.startswith(prefix)
        }
        assert placement_parts, prefix
        angle = math.radians(placement.rotation_deg)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        for name, payload in placement_parts.items():
            for world_x, world_y, _world_z in payload["corners"]:
                assert all(
                    math.isfinite(component)
                    for component in (world_x, world_y, _world_z)
                ), name
                offset_x = world_x - placement.center.x
                offset_y = world_y - placement.center.y
                local_x = offset_x * cos_angle + offset_y * sin_angle
                local_y = -offset_x * sin_angle + offset_y * cos_angle
                assert abs(local_x) <= placement.width_m / 2 + tolerance, name
                assert abs(local_y) <= placement.depth_m / 2 + tolerance, name

    curved_parts = {
        name: payload
        for name, payload in parts.items()
        if any(
            token in name
            for token in ("Pillow_", "dining_table_Leg_", "Burner_", "plant_Leaf_")
        )
    }
    assert curved_parts
    assert all(payload["vertices"] > 8 for payload in curved_parts.values())
    assert all(payload["polygons"] > 6 for payload in curved_parts.values())
    assert all(
        payload["minimum_polygon_area"] > 1e-9
        for payload in curved_parts.values()
    )


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
