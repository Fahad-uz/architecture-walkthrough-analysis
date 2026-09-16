from __future__ import annotations

import math
from pathlib import Path
import shutil

import numpy as np
import pytest
import trimesh

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.models import ArchitecturalElement, FloorPlanModel, Point2D
from architecture_walkthrough.scene.blender_generator import generate_glb_with_blender
from architecture_walkthrough.scene.simple_glb import export_simple_glb
from architecture_walkthrough.scene.stair_builder import stair_parts, stair_placement


def test_straight_stair_treads_are_contiguous_and_reach_requested_height() -> None:
    parts = stair_parts(0.6, 1.2, {"step_count": 6, "total_height_m": 0.9})
    assert len(parts) == 6
    assert parts[0]["y"] - parts[0]["depth"] / 2 == pytest.approx(-0.6)
    assert parts[-1]["y"] + parts[-1]["depth"] / 2 == pytest.approx(0.6)
    assert [part["height"] for part in parts] == pytest.approx([0.15 * i for i in range(1, 7)])
    for part, following in zip(parts, parts[1:]):
        assert part["y"] + part["depth"] / 2 == pytest.approx(
            following["y"] - following["depth"] / 2
        )
    assert all(part["width"] == 0.6 for part in parts)


def test_dogleg_has_two_opposite_flights_connected_at_half_height() -> None:
    parts = stair_parts(2.2, 3.2, {
        "layout": "dogleg", "steps_per_flight": 7, "total_height_m": 3.0,
        "landing_depth_m": 1.1, "flight_gap_m": 0.1,
    })
    landing = parts[0]
    lower = [part for part in parts if part["name"].startswith("Flight_1")]
    upper = [part for part in parts if part["name"].startswith("Flight_2")]
    assert len(lower) == len(upper) == 7
    assert landing["height"] == lower[-1]["height"] == pytest.approx(1.5)
    assert landing["width"] == 2.2
    assert upper[-1]["height"] == pytest.approx(3.0)
    assert upper[0]["height"] - landing["height"] == pytest.approx(3.0 / 14)
    assert lower[0]["y"] < lower[-1]["y"]
    assert upper[0]["y"] > upper[-1]["y"]
    assert lower[-1]["y"] + lower[-1]["depth"] / 2 == pytest.approx(
        landing["y"] - landing["depth"] / 2
    )
    assert upper[0]["y"] + upper[0]["depth"] / 2 == pytest.approx(
        landing["y"] - landing["depth"] / 2
    )
    for part in parts:
        assert abs(part["x"]) + part["width"] / 2 <= 1.1 + 1e-9
        assert abs(part["y"]) + part["depth"] / 2 <= 1.6 + 1e-9
        assert part["z"] - part["height"] / 2 == pytest.approx(0)


def test_rotated_polygon_only_stair_uses_local_footprint_dimensions() -> None:
    angle = math.radians(27)
    polygon = [
        {"x": 4 + x * math.cos(angle) - y * math.sin(angle),
         "y": 5 + x * math.sin(angle) + y * math.cos(angle)}
        for x, y in [(-0.4, -1.1), (0.4, -1.1), (0.4, 1.1), (-0.4, 1.1)]
    ]
    item = stair_placement({"polygon": polygon, "rotation_deg": 27})
    assert item["center"] == pytest.approx({"x": 4, "y": 5})
    assert item["width_m"] == pytest.approx(0.8)
    assert item["depth_m"] == pytest.approx(2.2)


@pytest.mark.parametrize("metadata", [
    {"step_count": 0}, {"step_count": 65}, {"step_count": 1.5}, {"step_count": True},
    {"total_height_m": 0}, {"total_height_m": float("nan")},
    {"layout": "dogleg", "landing_depth_m": 0},
    {"layout": "dogleg", "landing_depth_m": 3.2},
    {"layout": "dogleg", "flight_gap_m": 2.2},
])
def test_invalid_stair_settings_are_rejected(metadata: dict) -> None:
    with pytest.raises(ValueError):
        stair_parts(2.2, 3.2, metadata)


def _stair_model() -> FloorPlanModel:
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json"))
    model.special_elements = [
        ArchitecturalElement(
            id="dogleg", kind="staircase", center=Point2D(x=2.0, y=1.5),
            width_m=2.2, depth_m=3.2, rotation_deg=90,
            metadata={"layout": "dogleg", "steps_per_flight": 7, "total_height_m": 3.0,
                      "landing_depth_m": 1.1, "flight_gap_m": 0.1},
        ),
        ArchitecturalElement(
            id="straight", kind="stairs", center=Point2D(x=0.5, y=1.0),
            width_m=0.6, depth_m=1.2, rotation_deg=-27,
            metadata={"step_count": 6, "total_height_m": 0.9},
        ),
    ]
    return model


def _assert_exported_stairs(output: Path, model: FloorPlanModel) -> None:
    scene = trimesh.load(output, force="scene")
    for index, element in enumerate(model.special_elements):
        prefix = f"Special_{index:03d}_{element.kind}"
        matching = [node for node in scene.graph.nodes_geometry if str(node).startswith(prefix)]
        assert len(matching) == (15 if index == 0 else 6)
        all_vertices = []
        for node in matching:
            transform, geometry = scene.graph.get(node)
            mesh = scene.geometry[geometry].copy()
            mesh.apply_transform(transform)
            # Exporters use glTF Y-up: convert back into the XY plan coordinates.
            vertices = np.column_stack((mesh.vertices[:, 0], -mesh.vertices[:, 2], mesh.vertices[:, 1]))
            all_vertices.append(vertices)
            assert np.isfinite(vertices).all()
            assert mesh.is_winding_consistent
            assert mesh.volume > 0
            assert vertices[:, 2].min() == pytest.approx(0, abs=1e-5)
        points = np.concatenate(all_vertices)
        assert element.center is not None
        offsets = points[:, :2] - [element.center.x, element.center.y]
        angle = math.radians(element.rotation_deg)
        local = offsets @ np.array([[math.cos(angle), -math.sin(angle)],
                                    [math.sin(angle), math.cos(angle)]])
        assert np.ptp(local[:, 0]) == pytest.approx(element.width_m, abs=1e-5)
        assert np.ptp(local[:, 1]) == pytest.approx(element.depth_m, abs=1e-5)
        assert local.min(axis=0) + local.max(axis=0) == pytest.approx([0, 0], abs=1e-5)
        assert points[:, 2].max() == pytest.approx(element.metadata["total_height_m"], abs=1e-5)


def test_preview_export_preserves_both_stair_layouts(tmp_path: Path) -> None:
    model = _stair_model()
    _assert_exported_stairs(export_simple_glb(model, tmp_path / "stairs.glb"), model)


@pytest.mark.integration
def test_blender_export_preserves_both_stair_layouts(tmp_path: Path) -> None:
    if shutil.which("blender") is None:
        pytest.skip("Blender not on PATH")
    model = _stair_model()
    output = generate_glb_with_blender(model, tmp_path / "stairs.glb", AppConfig(), mode="none")
    _assert_exported_stairs(output, model)
