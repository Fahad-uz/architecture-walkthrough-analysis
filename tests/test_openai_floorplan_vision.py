from __future__ import annotations

from architecture_walkthrough.ai.floorplan_vision import (
    AIFurnitureHint,
    AIOpeningHint,
    AIWallHint,
    FloorPlanVisionHints,
    NormalizedPoint,
    OpenAIFloorPlanVisionAnalyzer,
    hints_to_floorplan_geometry,
)
from architecture_walkthrough.config import AISettings, AppConfig
import pytest


def test_openai_analyzer_is_disabled_without_flag(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    analyzer = OpenAIFloorPlanVisionAnalyzer(AISettings(openai_enabled=False))
    assert not analyzer.is_available()


def test_hints_convert_to_floorplan_geometry() -> None:
    hints = FloorPlanVisionHints(
        walls=[
            AIWallHint(
                start=NormalizedPoint(x=0.1, y=0.2),
                end=NormalizedPoint(x=0.9, y=0.2),
                confidence=0.9,
                external=True,
            )
        ],
        openings=[
            AIOpeningHint(kind="door", center=NormalizedPoint(x=0.5, y=0.2), confidence=0.9),
            AIOpeningHint(kind="window", center=NormalizedPoint(x=0.3, y=0.2), confidence=0.9),
        ],
        furniture=[
            AIFurnitureHint(
                category="bed",
                center=NormalizedPoint(x=0.5, y=0.8),
                width=0.2,
                depth=0.3,
                confidence=0.9,
            )
        ],
    )
    config = AppConfig()
    walls, rooms, doors, windows, furniture = hints_to_floorplan_geometry(
        hints,
        image_width_px=1000,
        image_height_px=500,
        pixels_per_metre=100,
        config=config,
    )
    assert len(walls) == 1
    assert walls[0].external is True
    assert walls[0].start.x == 1.0
    assert walls[0].end.x == 9.0
    assert walls[0].start.y == 4.0
    assert rooms == []
    assert len(doors) == 1
    assert len(windows) == 1
    assert len(furniture) == 1
    assert furniture[0].category == "bed"
    assert furniture[0].center.y == pytest.approx(1.0)
