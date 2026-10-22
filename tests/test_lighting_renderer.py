from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from architecture_walkthrough.geometry.models import FloorPlanModel, LightingSettings
from architecture_walkthrough.scene.blender_generator import TEMPLATE
from architecture_walkthrough.scene.lighting_values import effective_lighting
from architecture_walkthrough.scene.scene_builder import build_blender_script


class _Socket:
    def __init__(self):
        self.default_value = None


class _Sockets(dict):
    def __missing__(self, key):
        socket = _Socket()
        self[key] = socket
        return socket


class _Nodes(list):
    def new(self, kind):
        node = SimpleNamespace(type=kind, inputs=_Sockets(), outputs=_Sockets())
        self.append(node)
        return node


class _Links(list):
    def new(self, output, input_):
        self.append((output, input_))


class _Light(dict):
    def __init__(self, name, kind, energy=0.0):
        super().__init__()
        self.name = name
        self.type = kind
        self.energy = energy
        self.color = (1.0, 1.0, 1.0)


class _Lights(list):
    def new(self, name, kind):
        light = _Light(name, kind)
        self.append(light)
        return light


class _Worlds(list):
    def new(self, name):
        world = SimpleNamespace(
            name=name,
            use_nodes=False,
            node_tree=SimpleNamespace(nodes=_Nodes(), links=_Links()),
        )
        self.append(world)
        return world


def _lighting_runtime(settings: dict, mode: str = "none") -> dict:
    """Run production lighting functions with Blender's data API replaced by records."""
    scene = SimpleNamespace(objects=[], world=None, view_settings=SimpleNamespace(exposure=0.0))
    bpy = SimpleNamespace(
        context=SimpleNamespace(scene=scene),
        data=SimpleNamespace(
            worlds=_Worlds(),
            lights=_Lights(),
            objects=SimpleNamespace(
                new=lambda name, data: SimpleNamespace(name=name, data=data, type="LIGHT"),
            ),
        ),
    )
    namespace = {
        "__name__": "lighting_under_test",
        "bpy": bpy,
        "math": math,
        "LIGHTING": effective_lighting(settings),
        "PLAN": {
            "lighting": copy.deepcopy(settings),
            "rooms": [{"points": [
                {"x": 0.0, "y": 0.0}, {"x": 2.0, "y": 0.0},
                {"x": 2.0, "y": 2.0}, {"x": 0.0, "y": 2.0},
            ]}],
            "walls": [{"height_m": 2.8}],
        },
        "MODE": mode,
        "WALL_HEIGHT_DEFAULT": 2.8,
        "link": scene.objects.append,
    }
    names = {
        "polygon_area", "point_in_room", "room_light_positions",
        "build_lighting", "tune_realtime_lights_for_export",
    }
    tree = ast.parse(TEMPLATE.read_text(encoding="utf-8"))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(TEMPLATE), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("brightness", [0.25, 1.0, 2.0])
@pytest.mark.parametrize("temperature", [3000, 4500, 6500])
def test_scene_light_power_tone_and_positions_follow_saved_settings(
    brightness: float, temperature: int,
) -> None:
    settings = {"brightness": brightness, "color_temperature": temperature}
    runtime = _lighting_runtime(settings)
    before = copy.deepcopy(runtime["PLAN"])
    runtime["build_lighting"]()
    scene = runtime["bpy"].context.scene
    lights = {obj.name: obj for obj in scene.objects}
    sun, room = lights["Sun"], lights["RoomLight_000_00"]
    assert sun.data.energy == pytest.approx(0.1 * brightness)
    assert sun.data.color == pytest.approx(effective_lighting(settings)["color_scale"])
    assert room.data.energy == pytest.approx(24.0 * brightness)
    assert room.data.color == pytest.approx(effective_lighting(settings)["light_color"])
    assert room.location == (1.0, 1.0, 2.5)
    background = next(
        node for node in scene.world.node_tree.nodes if node.type == "ShaderNodeBackground"
    )
    assert background.inputs["Strength"].default_value == pytest.approx(0.04 * brightness)
    assert runtime["PLAN"] == before


def test_zero_intensity_preserves_dark_scene_through_export() -> None:
    runtime = _lighting_runtime({"intensity": 0.0, "brightness": 2.0})
    runtime["build_lighting"]()
    runtime["tune_realtime_lights_for_export"]()
    lights = runtime["bpy"].context.scene.objects
    assert lights
    assert all(obj.data.energy == 0.0 and obj.data["bake_energy"] == 0.0 for obj in lights)


@pytest.mark.parametrize("mode", ["none", "draft", "final"])
@pytest.mark.parametrize("factor", [0.0, 0.25, 1.0, 2.0, 3.0])
def test_export_applies_brightness_after_mode_clamps(mode: str, factor: float) -> None:
    settings = {"brightness": 1.0, "intensity": 450.0 * factor}
    runtime = _lighting_runtime(settings, mode)
    scene = runtime["bpy"].context.scene
    lights = []
    for index, base_energy in enumerate((1.0, 12.0, 24.0, 80.0, 200.0)):
        light = _Light(f"point_{index}", "POINT", base_energy * factor)
        light["base_energy"] = base_energy
        lights.append(light)
    sun = _Light("Sun", "SUN", 0.1 * factor)
    sun["base_energy"] = 0.1
    for light in [*lights, sun]:
        scene.objects.append(SimpleNamespace(type="LIGHT", data=light))
    scene.objects.append(SimpleNamespace(type="MESH"))

    runtime["tune_realtime_lights_for_export"]()

    for base_energy, light in zip((1.0, 12.0, 24.0, 80.0, 200.0), lights):
        minimum, maximum, scale = (6.0, 18.0, 0.30) if mode == "none" else (4.0, 12.0, 0.20)
        baseline_candela = max(minimum, min(maximum, base_energy * scale))
        assert light.energy == pytest.approx(baseline_candela * 4.0 * math.pi / 683.0 * factor)
        assert light["bake_energy"] == pytest.approx(base_energy * factor)
    assert sun.energy == pytest.approx(1.2 / 683.0 * factor)
    assert sun["bake_energy"] == pytest.approx(0.1 * factor)


def test_world_multiplies_sky_color_and_neutral_preserves_original_sky() -> None:
    for temperature in (4500, 3000, 6500):
        runtime = _lighting_runtime({"color_temperature": temperature})
        runtime["build_lighting"]()
        world = runtime["bpy"].context.scene.world
        sky = next(node for node in world.node_tree.nodes if node.type == "ShaderNodeTexSky")
        background = next(
            node for node in world.node_tree.nodes if node.type == "ShaderNodeBackground"
        )
        multiply = next(
            node for node in world.node_tree.nodes
            if node.type == "ShaderNodeMixRGB" and node.blend_type == "MULTIPLY"
        )
        assert (sky.outputs["Color"], multiply.inputs[1]) in world.node_tree.links
        assert (multiply.outputs["Color"], background.inputs["Color"]) in world.node_tree.links
        assert multiply.inputs[0].default_value == 1.0
        assert multiply.inputs[2].default_value[:3] == pytest.approx(
            effective_lighting({"color_temperature": temperature})["color_scale"]
        )
        if temperature == 4500:
            assert multiply.inputs[2].default_value == (1.0, 1.0, 1.0, 1.0)


def _embedded_floorplan(script: str) -> dict:
    tree = ast.parse(script)
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "floorplan" for target in node.targets)
    )
    assert isinstance(assignment.value, ast.Call)
    return json.loads(ast.literal_eval(assignment.value.args[0]))


@pytest.mark.parametrize("explicit_override", [False, True])
def test_render_script_uses_saved_lighting_unless_explicitly_overridden(
    explicit_override: bool,
) -> None:
    model = FloorPlanModel(
        lighting=LightingSettings(brightness=0.5, color_temperature=3000, exposure=-1.0),
    )
    override = LightingSettings(brightness=1.5, color_temperature=6500, exposure=1.0)
    expected = override if explicit_override else model.lighting
    before = model.model_dump(mode="json")
    override_before = override.model_dump(mode="json")

    script = build_blender_script(
        model, Path("building.glb"), lighting=override if explicit_override else None,
    )

    assert _embedded_floorplan(script)["lighting"] == expected.model_dump(mode="json")
    exposure_assignment = next(
        node for node in ast.walk(ast.parse(script))
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "exposure" for target in node.targets)
    )
    assert ast.literal_eval(exposure_assignment.value) == expected.exposure
    assert model.model_dump(mode="json") == before
    assert override.model_dump(mode="json") == override_before
