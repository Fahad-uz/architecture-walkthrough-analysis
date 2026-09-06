from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
import trimesh

from architecture_walkthrough.ai.floorplan_vision import FloorPlanVisionAnalysis
from architecture_walkthrough.ai.sanity_check import GeminiLayoutSanityChecker
from architecture_walkthrough.config import AISettings, AppConfig, LimitSettings
from architecture_walkthrough.pipeline import _runtime_ai_config, convert_image_to_glb


def _write_test_plan(image_path: Path) -> None:
    image = Image.new("RGB", (240, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 220, 160), outline="black", width=8)
    draw.line((120, 20, 120, 160), fill="black", width=5)
    image.save(image_path)


def test_convert_image_to_glb_from_png(tmp_path: Path) -> None:
    image_path = tmp_path / "plan.png"
    _write_test_plan(image_path)

    output = convert_image_to_glb(
        image_path,
        tmp_path / "building.glb",
        AppConfig(ai=AISettings(gemini_enabled=False)),
        manual_scale=0.05,
        work_dir=tmp_path / "work",
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert (tmp_path / "work" / "floorplan.json").exists()
    loaded = trimesh.load(output, force="scene")
    assert len(loaded.geometry) >= 2


def test_semantic_provider_failure_skips_redundant_sanity_call(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "plan.png"
    _write_test_plan(image_path)
    failed_analysis = FloorPlanVisionAnalysis(
        attempted=True,
        succeeded=False,
        error="request timed out",
    )
    monkeypatch.setattr(
        "architecture_walkthrough.pipeline._floorplan_vision_analysis",
        lambda *_args, **_kwargs: (failed_analysis, False),
    )

    def unexpected_sanity_call(*_args, **_kwargs):
        raise AssertionError("sanity provider should not be called after semantic failure")

    monkeypatch.setattr(GeminiLayoutSanityChecker, "check", unexpected_sanity_call)

    output = convert_image_to_glb(
        image_path,
        tmp_path / "building.glb",
        AppConfig(ai=AISettings(gemini_enabled=True)),
        manual_scale=0.05,
        work_dir=tmp_path / "work",
    )

    assert output.exists()


def test_gemini_budget_clamps_or_skips_for_short_worker_limits() -> None:
    clamped, error = _runtime_ai_config(
        AppConfig(
            ai=AISettings(gemini_enabled=True),
            limits=LimitSettings(processing_timeout_seconds=37),
        )
    )

    assert error is None
    assert clamped.ai.gemini_request_timeout_seconds == 6

    unchanged, error = _runtime_ai_config(
        AppConfig(
            ai=AISettings(gemini_enabled=True),
            limits=LimitSettings(processing_timeout_seconds=30),
        )
    )

    assert unchanged.ai.gemini_request_timeout_seconds == 20
    assert error is not None
    assert "insufficient time" in error
