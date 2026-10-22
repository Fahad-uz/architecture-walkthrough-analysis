from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from architecture_walkthrough.geometry.models import FloorPlanModel, LightingSettings
from architecture_walkthrough.scene.lighting import LightingSettings as SceneLightingSettings
from architecture_walkthrough.scene.lighting_values import effective_lighting


def test_lighting_settings_are_shared_and_defaults_are_independent() -> None:
    assert SceneLightingSettings is LightingSettings
    first, second = FloorPlanModel(), FloorPlanModel()
    assert first.lighting.model_dump() == {
        "preset": "neutral_studio",
        "color_temperature": 4500,
        "intensity": 450.0,
        "exposure": 0.0,
        "brightness": 1.0,
    }
    first.lighting.brightness = 1.5
    assert second.lighting.brightness == 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("brightness", 0.249),
        ("brightness", 2.001),
        ("intensity", -0.001),
        ("intensity", 4500.001),
        ("color_temperature", 999),
        ("color_temperature", 12001),
        ("color_temperature", 4500.5),
        ("exposure", -5.001),
        ("exposure", 5.001),
    ],
)
def test_lighting_rejects_out_of_range_values(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        LightingSettings.model_validate({field: value})


@pytest.mark.parametrize("field", ["brightness", "intensity", "color_temperature", "exposure"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_lighting_rejects_non_finite_values(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        FloorPlanModel.model_validate({"lighting": {field: value}})


@pytest.mark.parametrize(
    "values",
    [
        {"brightness": 0.25, "intensity": 0.0, "color_temperature": 1000, "exposure": -5.0},
        {"brightness": 2.0, "intensity": 4500.0, "color_temperature": 12000, "exposure": 5.0},
    ],
)
def test_lighting_range_endpoints_are_supported(values: dict[str, float]) -> None:
    settings = LightingSettings.model_validate(values)
    assert all(getattr(settings, field) == value for field, value in values.items())


def test_old_floorplan_without_lighting_loads_and_saves_with_defaults(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "sample_floorplan.json"
    old_payload = json.loads(fixture.read_text(encoding="utf-8"))
    old_payload.pop("lighting", None)
    old_path = tmp_path / "old.json"
    old_path.write_text(json.dumps(old_payload), encoding="utf-8")
    model = FloorPlanModel.load_json(old_path)

    assert model.lighting == LightingSettings()
    saved_path = tmp_path / "saved.json"
    model.save_json(saved_path)
    saved = FloorPlanModel.load_json(saved_path)
    assert saved == model
    assert saved.walls == model.walls
    assert json.loads(saved_path.read_text(encoding="utf-8"))["lighting"] == (
        LightingSettings().model_dump(mode="json")
    )


def test_saved_lighting_retains_legacy_settings_and_new_controls(tmp_path: Path) -> None:
    model = FloorPlanModel(
        lighting=LightingSettings(
            preset="legacy_custom_preset",
            brightness=1.75,
            color_temperature=3000,
            intensity=900.0,
            exposure=-0.75,
        ),
    )
    path = tmp_path / "lighting.json"
    model.save_json(path)
    assert FloorPlanModel.load_json(path) == model


def test_effective_defaults_preserve_original_light_color_and_power() -> None:
    implicit = effective_lighting({})
    explicit = effective_lighting(LightingSettings().model_dump(mode="json"))
    assert implicit == explicit
    assert implicit["power_factor"] == 1.0
    assert tuple(implicit["light_color"]) == (1.0, 0.9, 0.78)
    assert implicit["exposure"] == 0.0


@pytest.mark.parametrize(
    ("brightness", "intensity", "expected"),
    [(0.25, 450.0, 0.25), (2.0, 450.0, 2.0), (1.5, 900.0, 3.0), (2.0, 0.0, 0.0)],
)
def test_effective_power_combines_brightness_and_legacy_intensity(
    brightness: float, intensity: float, expected: float,
) -> None:
    values = effective_lighting({"brightness": brightness, "intensity": intensity})
    assert values["power_factor"] == expected


def test_color_temperature_changes_tone_without_changing_power_or_exposure() -> None:
    values = [
        effective_lighting({"color_temperature": temperature, "exposure": 0.5})
        for temperature in (3000, 4500, 6500)
    ]
    red_to_blue = [value["light_color"][0] / value["light_color"][2] for value in values]
    assert red_to_blue[0] > red_to_blue[1] > red_to_blue[2]
    assert all(value["power_factor"] == 1.0 for value in values)
    assert all(value["exposure"] == 0.5 for value in values)
    assert all(0.0 < channel <= 1.0 for value in values for channel in value["light_color"])


def test_effective_lighting_does_not_mutate_input() -> None:
    settings = LightingSettings(brightness=1.5, color_temperature=6500).model_dump(mode="json")
    original = copy.deepcopy(settings)
    assert effective_lighting(settings) == effective_lighting(settings)
    assert settings == original
