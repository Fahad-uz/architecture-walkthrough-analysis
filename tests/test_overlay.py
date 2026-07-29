from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from architecture_walkthrough.geometry.models import (
    BalconyPolygon,
    FloorPlanModel,
    Point2D,
    RoomPolygon,
)
from architecture_walkthrough.vision.overlay import write_analysis_overlay


def test_analysis_overlay_draws_balconies_separately_from_rooms(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plan.png"
    assert cv2.imwrite(
        str(source),
        np.full((200, 240, 3), 255, dtype=np.uint8),
    )
    model = FloorPlanModel(
        pixels_per_metre=20,
        rooms=[
            RoomPolygon(
                id="room",
                name="LIVING",
                points=[
                    Point2D(x=1, y=1),
                    Point2D(x=5, y=1),
                    Point2D(x=5, y=5),
                    Point2D(x=1, y=5),
                ],
            )
        ],
        balconies=[
            BalconyPolygon(
                id="balcony",
                name="BALCONY",
                points=[
                    Point2D(x=1, y=5),
                    Point2D(x=5, y=5),
                    Point2D(x=5, y=6),
                    Point2D(x=1, y=6),
                ],
            )
        ],
    )

    svg_path, png_path = write_analysis_overlay(
        source,
        model,
        tmp_path / "overlay.svg",
        tmp_path / "overlay.png",
    )

    svg = svg_path.read_text(encoding="utf-8")
    assert '<g id="rooms">' in svg
    assert '<g id="balconies">' in svg
    assert "BALCONY" in svg
    assert png_path is not None and png_path.stat().st_size > 0
