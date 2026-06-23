from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
import trimesh

from architecture_walkthrough.config import AISettings, AppConfig
from architecture_walkthrough.pipeline import convert_image_to_glb


def test_convert_image_to_glb_from_png(tmp_path: Path) -> None:
    image_path = tmp_path / "plan.png"
    image = Image.new("RGB", (240, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 220, 160), outline="black", width=8)
    draw.line((120, 20, 120, 160), fill="black", width=5)
    image.save(image_path)

    output = convert_image_to_glb(
        image_path,
        tmp_path / "building.glb",
            AppConfig(ai=AISettings(openai_enabled=False)),
        manual_scale=0.05,
        work_dir=tmp_path / "work",
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert (tmp_path / "work" / "floorplan.json").exists()
    loaded = trimesh.load(output, force="scene")
    assert len(loaded.geometry) >= 2
