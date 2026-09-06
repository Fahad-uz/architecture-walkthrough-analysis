from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from architecture_walkthrough.geometry.models import (
    BalconyPolygon,
    CeilingSettings,
    DoorOpening,
    FloorPlanModel,
    MaterialAssignment,
    Point2D,
    RoomPolygon,
    SceneStyleSettings,
    SlabPolygon,
    WallSegment,
    WindowOpening,
)
from architecture_walkthrough.scene.pbr_materials import (
    MaterialRegistry,
    PBRMaterialPreset,
    build_material_plan,
    load_material_registry,
    model_with_material_plan,
)


def _room(
    *,
    room_id: str,
    face_id: str,
    name: str | None,
) -> RoomPolygon:
    return RoomPolygon(
        id=room_id,
        face_id=face_id,
        name=name,
        points=[
            Point2D(x=0, y=0),
            Point2D(x=1, y=0),
            Point2D(x=1, y=1),
        ],
    )


def _default_registry() -> MaterialRegistry:
    return load_material_registry(Path("assets/textures/material_registry.yaml"))


def test_material_registry_normalizes_names_and_emits_scalars_only(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "materials.yaml"
    registry_path.write_text(
        "materials:\n"
        "  ' Painted Wall ':\n"
        "    base_color: [0.1, 0.2, 0.3, 1.0]\n"
        "    base_color_texture: wall.png\n"
        "    roughness: 0.7\n"
        "    metallic: 0.1\n"
        "    alpha_mode: blend\n"
        "    double_sided: true\n",
        encoding="utf-8",
    )

    registry = load_material_registry(registry_path)
    scalar = registry.scalar_payload()

    assert registry.require("PAINTED-WALL").name == "painted_wall"
    assert scalar == {
        "painted_wall": {
            "base_color": [0.1, 0.2, 0.3, 1.0],
            "roughness": 0.7,
            "metallic": 0.1,
            "alpha_mode": "BLEND",
            "double_sided": True,
        }
    }
    assert "wall.png" not in json.dumps(scalar)


def test_material_registry_rejects_missing_or_empty_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="material registry not found"):
        load_material_registry(tmp_path / "missing.yaml")

    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("materials: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="defines no materials"):
        load_material_registry(empty_path)


def test_room_material_resolution_prioritizes_id_then_face_then_name() -> None:
    rooms = [
        _room(room_id="room-id", face_id="face-a", name="Bedroom"),
        _room(room_id="room-b", face_id="face-id", name="Bedroom"),
        _room(room_id="room-c", face_id="face-c", name="Bedroom"),
        _room(room_id="room-d", face_id="face-d", name="Master Bedroom"),
        _room(room_id="room-e", face_id="face-e", name="Kitchen"),
        _room(room_id="room-f", face_id="face-f", name="Living"),
    ]
    model = FloorPlanModel(
        rooms=rooms,
        material_assignments=[
            MaterialAssignment(target="BEDROOM", preset="ceramic_tile"),
            MaterialAssignment(target="face-a", preset="marble"),
            MaterialAssignment(target="room-id", preset="wood"),
            MaterialAssignment(target="face-id", preset="marble"),
        ],
    )

    plan = build_material_plan(model, _default_registry())

    assert plan["room_floor_presets"] == [
        "wood",
        "marble",
        "ceramic_tile",
        "wood",
        "ceramic_tile",
        "marble",
    ]


def test_material_plan_includes_all_surface_selections() -> None:
    model = FloorPlanModel(
        walls=[
            WallSegment(
                start=Point2D(x=0, y=0),
                end=Point2D(x=2, y=0),
                material_preset="metal",
            ),
            WallSegment(
                start=Point2D(x=2, y=0),
                end=Point2D(x=2, y=2),
            ),
        ],
        doors=[
            DoorOpening(center=Point2D(x=1, y=0)),
        ],
        windows=[
            WindowOpening(center=Point2D(x=2, y=1)),
        ],
        rooms=[
            _room(room_id="room", face_id="face", name="Study"),
        ],
        slabs=[
            SlabPolygon(
                id="slab-a",
                name="slab",
                points=[
                    Point2D(x=0, y=0),
                    Point2D(x=2, y=0),
                    Point2D(x=2, y=2),
                ],
                material_preset="ceramic_tile",
            )
        ],
        balconies=[
            BalconyPolygon(
                id="balcony-a",
                name="Balcony",
                points=[
                    Point2D(x=0, y=0),
                    Point2D(x=2, y=0),
                    Point2D(x=2, y=1),
                ],
            )
        ],
        ceiling=CeilingSettings(enabled=True, material_preset="painted_wall"),
        style=SceneStyleSettings(
            wall_material="painted_wall",
            floor_material="marble",
            door_material="wood",
            window_frame_material="metal",
        ),
    )

    plan = build_material_plan(model, _default_registry())

    assert plan["defaults"] == {
        "wall": "painted_wall",
        "floor": "marble",
        "door": "wood",
        "window_frame": "metal",
        "ceiling": "painted_wall",
        "balcony": "ceramic_tile",
    }
    assert plan["wall_presets"] == ["metal", "painted_wall"]
    assert plan["room_floor_presets"] == ["wood"]
    assert plan["door_presets"] == ["wood"]
    assert plan["window_frame_presets"] == ["metal"]
    assert plan["slab_presets"] == ["ceramic_tile"]
    assert plan["balcony_floor_presets"] == ["ceramic_tile"]
    assert plan["ceiling_preset"] == "painted_wall"
    json.dumps(plan)


def test_unknown_or_missing_presets_use_a_json_safe_fallback() -> None:
    registry = MaterialRegistry(
        materials={
            "marble": PBRMaterialPreset(name="marble"),
        }
    )
    model = FloorPlanModel(
        walls=[
            WallSegment(
                start=Point2D(x=0, y=0),
                end=Point2D(x=1, y=0),
                material_preset="missing-wall",
            )
        ],
        rooms=[
            _room(room_id="room", face_id="face", name="Kitchen"),
        ],
        style=SceneStyleSettings(
            wall_material="missing-style-wall",
            floor_material="marble",
            door_material="missing-door",
            window_frame_material="missing-frame",
        ),
        ceiling=CeilingSettings(material_preset="missing-ceiling"),
    )

    plan = build_material_plan(model, registry)

    assert plan["defaults"] == {
        "wall": "fallback",
        "floor": "marble",
        "door": "fallback",
        "window_frame": "fallback",
        "ceiling": "fallback",
        "balcony": "marble",
    }
    assert plan["wall_presets"] == ["fallback"]
    assert plan["room_floor_presets"] == ["marble"]
    assert plan["materials"]["fallback"]["base_color"] == [0.8, 0.8, 0.8, 1.0]
    assert plan["warnings"]
    json.dumps(plan)


def test_material_plan_embedding_does_not_mutate_the_source_model() -> None:
    model = FloorPlanModel(
        rooms=[_room(room_id="bedroom", face_id="face", name="Bedroom")],
        metadata={"source": "fixture"},
    )

    rendered = model_with_material_plan(model, _default_registry())

    assert "blender_material_plan" not in model.metadata
    assert rendered.metadata["source"] == "fixture"
    assert rendered.metadata["blender_material_plan"]["room_floor_presets"] == [
        "wood"
    ]


def test_material_plan_embedding_logs_unknown_preset_fallbacks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = FloorPlanModel(
        walls=[
            WallSegment(
                start=Point2D(x=0, y=0),
                end=Point2D(x=1, y=0),
                material_preset="missing-finish",
            )
        ]
    )

    with caplog.at_level(
        logging.WARNING,
        logger="architecture_walkthrough.scene.pbr_materials",
    ):
        model_with_material_plan(model, _default_registry())

    assert "requested unknown material 'missing_finish'" in caplog.text
