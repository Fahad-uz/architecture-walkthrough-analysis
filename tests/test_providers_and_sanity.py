from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from architecture_walkthrough.ai.sanity_check import (
    GeminiLayoutSanityChecker,
    SanityWarning,
    _layout_summary,
)
from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.models import FloorPlanModel, Point2D, WallSegment
from architecture_walkthrough.vision.providers import (
    CubiCasaGeometryProvider,
    WallBandGeometryProvider,
    build_geometry_provider,
)


def test_default_provider_is_wallband_cv() -> None:
    provider = build_geometry_provider(AppConfig())
    assert isinstance(provider, WallBandGeometryProvider)
    assert provider.name == "wallband_cv"
    assert provider.accuracy_tier == "medium"


def test_cubicasa_provider_requires_implementation_flag(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pkl"
    checkpoint.write_bytes(b"fake")
    config = AppConfig()
    config.ai.segmentation_checkpoint = checkpoint
    # Even with a checkpoint on disk, the provider stays off until inference lands.
    assert not CubiCasaGeometryProvider.is_available(config)
    assert isinstance(build_geometry_provider(config), WallBandGeometryProvider)


def _model() -> FloorPlanModel:
    return FloorPlanModel(
        pixels_per_metre=100.0,
        walls=[WallSegment(id="w0", start=Point2D(x=0, y=0), end=Point2D(x=4, y=0))],
    )


def test_sanity_checker_disabled_reports_not_attempted() -> None:
    config = AppConfig()
    config.ai.gemini_sanity_check_enabled = False
    result = GeminiLayoutSanityChecker(config.ai).check(Path("missing.png"), _model(), 400, 300)
    assert not result.attempted
    assert result.error


def test_sanity_checker_without_key_is_soft_failure(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = GeminiLayoutSanityChecker(AppConfig().ai).check(Path("missing.png"), _model(), 400, 300)
    assert result.attempted and not result.succeeded
    assert result.warnings == []


def test_sanity_checker_parses_mocked_warnings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    image = tmp_path / "plan.png"
    image.write_bytes(b"png")
    warnings = [
        SanityWarning(kind="missed_door", description="door arc near kitchen", x=0.4, y=0.6, confidence=0.8)
    ]
    with patch.object(GeminiLayoutSanityChecker, "_request", return_value=warnings):
        result = GeminiLayoutSanityChecker(AppConfig().ai).check(image, _model(), 400, 300)
    assert result.succeeded
    assert result.warnings[0].kind == "missed_door"


def test_layout_summary_is_normalized_json() -> None:
    payload = json.loads(_layout_summary(_model(), image_width_px=400, image_height_px=300))
    assert payload["walls"][0]["start"] == {"x": 0.0, "y": 1.0}
    assert payload["walls"][0]["end"]["x"] == 1.0
