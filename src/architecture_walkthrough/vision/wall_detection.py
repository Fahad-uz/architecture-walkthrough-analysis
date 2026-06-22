from __future__ import annotations

from pathlib import Path

import cv2

from architecture_walkthrough.geometry.models import Point2D, WallSegment


def detect_wall_lines(edge_image_path: Path, thickness_m: float = 0.12, height_m: float = 3.0) -> list[WallSegment]:
    image = cv2.imread(str(edge_image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read edge image: {edge_image_path}")
    lines = cv2.HoughLinesP(image, 1, 3.14159 / 180, threshold=60, minLineLength=30, maxLineGap=8)
    if lines is None:
        return []
    return [
        WallSegment(
            start=Point2D(x=float(x1), y=float(y1)),
            end=Point2D(x=float(x2), y=float(y2)),
            thickness_m=thickness_m,
            height_m=height_m,
        )
        for [[x1, y1, x2, y2]] in lines
    ]
