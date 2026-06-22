from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from architecture_walkthrough.api.app import UPLOAD_PAGE, create_app
from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.pipeline import convert_image_to_glb


def test_image_upload_pipeline_generates_glb_artifact(tmp_path: Path) -> None:
    image_path = tmp_path / "plan.png"
    image = Image.new("RGB", (240, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 220, 160), outline="black", width=8)
    image.save(image_path)

    output = convert_image_to_glb(image_path, tmp_path / "building.glb", AppConfig(), work_dir=tmp_path / "work")
    assert output.exists()
    assert output.stat().st_size > 0


def test_app_exposes_upload_and_download_routes() -> None:
    app = create_app()
    routes = {getattr(route, "path", "") for route in app.routes}
    assert "/" in routes
    assert "/jobs" in routes
    assert "/jobs/{job_id}/artifacts/{artifact_name}" in routes
    assert "Create GLB" in UPLOAD_PAGE
