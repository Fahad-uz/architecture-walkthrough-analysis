from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from architecture_walkthrough.ai.floorplan_vision import (
    AIFurnitureHint,
    AIOpeningHint,
    AIWallHint,
    FloorPlanVisionHints,
    GeminiFloorPlanVisionAnalyzer,
    GeminiFloorPlanVisionError,
    NormalizedPoint,
    hints_to_floorplan_geometry,
)
from architecture_walkthrough.config import AISettings, AppConfig


def test_gemini_analyzer_is_disabled_without_flag(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    analyzer = GeminiFloorPlanVisionAnalyzer(AISettings(gemini_enabled=False))
    assert not analyzer.is_available()


def test_gemini_analyzer_reports_missing_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    image_path = tmp_path / "plan.png"
    image_path.write_bytes(b"not-used")
    analyzer = GeminiFloorPlanVisionAnalyzer(AISettings(gemini_enabled=True))

    result = analyzer.analyze_with_diagnostics(image_path)

    assert result.attempted is True
    assert result.succeeded is False
    assert result.error is not None
    assert "GEMINI_API_KEY" in result.error


def test_gemini_analyzer_can_require_success(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    image_path = tmp_path / "plan.png"
    image_path.write_bytes(b"not-used")
    analyzer = GeminiFloorPlanVisionAnalyzer(AISettings(gemini_enabled=True))

    with pytest.raises(GeminiFloorPlanVisionError, match="GEMINI_API_KEY"):
        analyzer.analyze_with_diagnostics(image_path, require_success=True)


def test_gemini_request_applies_bounded_transport_and_retry_budget(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    class FakeModels:
        def generate_content(self, **_kwargs):
            return SimpleNamespace(text=FloorPlanVisionHints().model_dump_json())

    class FakeClient:
        def __init__(self, *, api_key, http_options) -> None:
            captured["api_key"] = api_key
            captured["http_options"] = http_options
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            captured["closed"] = True

    def fake_backoff(operation, *, attempts, base_delay_seconds):
        captured["attempts"] = attempts
        captured["base_delay_seconds"] = base_delay_seconds
        return operation()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("google.genai.Client", FakeClient)
    monkeypatch.setattr("architecture_walkthrough.ai.retry.call_with_backoff", fake_backoff)
    image_path = tmp_path / "plan.png"
    image_path.write_bytes(b"test-image")
    analyzer = GeminiFloorPlanVisionAnalyzer(
        AISettings(
            gemini_request_timeout_seconds=8,
            gemini_retry_attempts=2,
            gemini_retry_base_delay_seconds=1.5,
        )
    )

    assert analyzer._request_hints(image_path) == FloorPlanVisionHints()
    options = captured["http_options"]
    assert options.timeout == 8_000
    assert options.retry_options.attempts == 1
    assert captured["attempts"] == 2
    assert captured["base_delay_seconds"] == 1.5
    assert captured["closed"] is True


@pytest.mark.parametrize(("succeeds_on_retry", "expected_success"), [(True, True), (False, False)])
def test_gemini_timeout_retries_once_and_remains_a_soft_failure(
    monkeypatch,
    tmp_path,
    succeeds_on_retry: bool,
    expected_success: bool,
) -> None:
    calls = {"count": 0, "closed": False}

    class FakeModels:
        def generate_content(self, **_kwargs):
            calls["count"] += 1
            if calls["count"] == 1 or not succeeds_on_retry:
                raise httpx.ReadTimeout("")
            return SimpleNamespace(text=FloorPlanVisionHints().model_dump_json())

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            calls["closed"] = True

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("google.genai.Client", FakeClient)
    image_path = tmp_path / "plan.png"
    image_path.write_bytes(b"test-image")
    analyzer = GeminiFloorPlanVisionAnalyzer(
        AISettings(
            gemini_request_timeout_seconds=5,
            gemini_retry_attempts=2,
            gemini_retry_base_delay_seconds=0,
        )
    )

    result = analyzer.analyze_with_diagnostics(image_path)

    assert result.attempted is True
    assert result.succeeded is expected_success
    assert calls == {"count": 2, "closed": True}
    if expected_success:
        assert result.hints == FloorPlanVisionHints()
    else:
        assert result.hints is None
        assert "Gemini floor-plan analysis failed" in (result.error or "")


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
